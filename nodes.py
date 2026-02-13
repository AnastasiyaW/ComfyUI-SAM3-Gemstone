import os
import time
import logging
import torch
import numpy as np
from PIL import Image

import comfy.model_management as mm
import folder_paths

# ---------------------------------------------------------------------------
# Logging — verbose, no silence
# ---------------------------------------------------------------------------
logger = logging.getLogger("SAM3-Gemstone")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "[SAM3-Gemstone] %(levelname)s %(funcName)s:%(lineno)d — %(message)s"
    ))
    logger.addHandler(_h)

# ---------------------------------------------------------------------------
# Model directory — ComfyUI/models/sam3/
# ---------------------------------------------------------------------------
SAM3_MODEL_DIR = os.path.join(folder_paths.models_dir, "sam3")
os.makedirs(SAM3_MODEL_DIR, exist_ok=True)

SAM3_HF_REPO = "facebook/sam3"
SAM3_CHECKPOINT = "sam3.pt"
SAM3_CONFIG = "config.json"

# ---------------------------------------------------------------------------
# Gemstone-specific text prompts
# ---------------------------------------------------------------------------
GEMSTONE_PROMPTS = [
    "gemstone", "precious stone", "cut gemstone", "polished gemstone",
    "faceted gemstone", "diamond", "ruby", "sapphire", "emerald",
    "amethyst", "topaz", "opal", "garnet", "tourmaline", "aquamarine",
    "tanzanite", "alexandrite", "spinel", "peridot", "citrine",
    "morganite", "kunzite", "zircon", "chrysoberyl", "iolite",
    "raw crystal", "mineral specimen", "cabochon", "brilliant cut stone",
    "translucent stone", "transparent crystal", "jewelry stone",
    "shiny reflective stone",
]

# ---------------------------------------------------------------------------
# Global model cache
# ---------------------------------------------------------------------------
_sam3_cache: dict = {}


def _get_dtype(precision: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]


# ============================================================================
#  1.  LOADER NODE
# ============================================================================
class SAM3GemstoneModelLoader:
    """Load SAM 3 model. CUDA only, no CPU fallback."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hf_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "hf_... (required for gated facebook/sam3)",
                }),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "compile_model": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("SAM3_MODEL",)
    RETURN_NAMES = ("sam3_model",)
    FUNCTION = "load"
    CATEGORY = "SAM3-Gemstone"

    @staticmethod
    def _ensure_checkpoint(hf_token: str) -> str:
        ckpt_path = os.path.join(SAM3_MODEL_DIR, SAM3_CHECKPOINT)
        cfg_path = os.path.join(SAM3_MODEL_DIR, SAM3_CONFIG)

        logger.info("Checking checkpoint: %s", ckpt_path)
        logger.info("Checking config:     %s", cfg_path)

        if os.path.isfile(ckpt_path) and os.path.isfile(cfg_path):
            size_gb = os.path.getsize(ckpt_path) / (1024**3)
            logger.info("Checkpoint found: %.2f GB", size_gb)
            assert size_gb > 1.0, (
                f"sam3.pt is only {size_gb:.2f} GB — looks corrupted or incomplete. "
                f"Expected ~3.3 GB. Delete and re-download."
            )
            return ckpt_path

        # Need to download
        from huggingface_hub import hf_hub_download

        token = hf_token.strip()
        assert token, (
            "HuggingFace token is REQUIRED to download facebook/sam3 (gated repo). "
            "Get a token at https://huggingface.co/settings/tokens and request access at "
            "https://huggingface.co/facebook/sam3"
        )

        logger.info("Downloading %s from %s ...", SAM3_CHECKPOINT, SAM3_HF_REPO)
        hf_hub_download(repo_id=SAM3_HF_REPO, filename=SAM3_CHECKPOINT,
                        local_dir=SAM3_MODEL_DIR, token=token)

        logger.info("Downloading %s ...", SAM3_CONFIG)
        hf_hub_download(repo_id=SAM3_HF_REPO, filename=SAM3_CONFIG,
                        local_dir=SAM3_MODEL_DIR, token=token)

        for fname in ("tokenizer.json", "tokenizer_config.json",
                      "vocab.json", "merges.txt", "special_tokens_map.json",
                      "processor_config.json"):
            target = os.path.join(SAM3_MODEL_DIR, fname)
            if not os.path.isfile(target):
                logger.info("Downloading %s ...", fname)
                hf_hub_download(repo_id=SAM3_HF_REPO, filename=fname,
                                local_dir=SAM3_MODEL_DIR, token=token)

        logger.info("All files downloaded to %s", SAM3_MODEL_DIR)
        return ckpt_path

    def load(self, hf_token: str, precision: str, compile_model: bool):
        t0 = time.time()

        device = mm.get_torch_device()
        dtype = _get_dtype(precision)

        logger.info("=" * 60)
        logger.info("LOAD START  device=%s  dtype=%s  compile=%s", device, dtype, compile_model)
        logger.info("=" * 60)

        # --- HARD: CUDA only ---
        assert device.type == "cuda", (
            f"SAM3-Gemstone requires CUDA. Got device={device}. "
            f"No CPU fallback. Check GPU availability."
        )

        prop = torch.cuda.get_device_properties(device)
        logger.info("GPU: %s  (compute %d.%d, %.1f GB VRAM)",
                    prop.name, prop.major, prop.minor,
                    prop.total_mem / (1024**3))

        cache_key = f"sam3_{precision}_{compile_model}"
        if cache_key in _sam3_cache:
            logger.info("Returning cached model (key=%s)", cache_key)
            return (_sam3_cache[cache_key],)

        # --- H100 / Ampere+ fast-math ---
        if prop.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            logger.info("TF32 + cuDNN benchmark enabled (compute >= 8.0)")

        # --- Checkpoint ---
        ckpt_path = self._ensure_checkpoint(hf_token)

        # --- BPE tokenizer ---
        import pkg_resources
        bpe_path = pkg_resources.resource_filename("sam3", "assets/bpe_simple_vocab_16e6.txt.gz")
        assert os.path.isfile(bpe_path), (
            f"BPE tokenizer NOT FOUND at {bpe_path}. "
            f"Install full sam3 package or copy bpe_simple_vocab_16e6.txt.gz to that path."
        )
        logger.info("BPE tokenizer: %s", bpe_path)

        # --- Build model ---
        logger.info("Building SAM3 image model...")
        from sam3.model_builder import build_sam3_image_model

        model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=str(device),
            checkpoint_path=ckpt_path,
            load_from_HF=False,
            compile=compile_model,
        )
        logger.info("Model built. Moving to dtype=%s ...", dtype)

        model = model.to(dtype=dtype)
        model.eval()

        # --- Verify model is on CUDA ---
        first_param = next(model.parameters())
        logger.info("Model first param: device=%s  dtype=%s", first_param.device, first_param.dtype)
        assert first_param.is_cuda, (
            f"Model ended up on {first_param.device}, expected CUDA. Something is wrong."
        )

        pipe = {"model": model, "device": device, "dtype": dtype}
        _sam3_cache[cache_key] = pipe

        elapsed = time.time() - t0
        logger.info("LOAD DONE in %.1fs", elapsed)
        return (pipe,)


# ============================================================================
#  2.  SEGMENTATION NODE
# ============================================================================
class SAM3GemstoneSegmentation:
    """Segment gemstones using SAM 3 text prompts. CUDA only."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model": ("SAM3_MODEL",),
                "image": ("IMAGE",),
                "prompt_preset": (["custom"] + GEMSTONE_PROMPTS, {"default": "gemstone"}),
                "custom_prompt": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "e.g. 'blue sapphire on velvet'",
                }),
                "score_threshold": ("FLOAT", {
                    "default": 0.35, "min": 0.01, "max": 1.0, "step": 0.01,
                    "display": "slider",
                }),
                "max_detections": ("INT", {
                    "default": 64, "min": 1, "max": 256, "step": 1,
                }),
                "mask_expansion": ("INT", {
                    "default": 0, "min": -50, "max": 50, "step": 1,
                }),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "multi_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "One prompt per line for multi-class segmentation",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("overlay_image", "gemstone_mask", "cropped_gems")
    FUNCTION = "segment"
    CATEGORY = "SAM3-Gemstone"

    @staticmethod
    def _expand_mask(mask: torch.Tensor, pixels: int) -> torch.Tensor:
        if pixels == 0:
            return mask
        kernel_size = abs(pixels) * 2 + 1
        pad = abs(pixels)
        m = mask.unsqueeze(0).unsqueeze(0).float()
        if pixels > 0:
            m = torch.nn.functional.max_pool2d(m, kernel_size=kernel_size, stride=1, padding=pad)
        else:
            m = -torch.nn.functional.max_pool2d(-m, kernel_size=kernel_size, stride=1, padding=pad)
        return m.squeeze(0).squeeze(0).clamp(0, 1)

    def segment(
        self,
        sam3_model: dict,
        image: torch.Tensor,
        prompt_preset: str,
        custom_prompt: str,
        score_threshold: float,
        max_detections: int,
        mask_expansion: int,
        keep_model_loaded: bool,
        multi_prompt: str = "",
    ):
        t0 = time.time()

        model = sam3_model["model"]
        device = sam3_model["device"]
        dtype = sam3_model["dtype"]

        logger.info("=" * 60)
        logger.info("SEGMENT START  device=%s  dtype=%s", device, dtype)
        logger.info("Image shape: %s  dtype: %s", image.shape, image.dtype)
        logger.info("=" * 60)

        # --- Resolve prompts ---
        prompts: list[str] = []
        if multi_prompt.strip():
            prompts = [p.strip() for p in multi_prompt.strip().splitlines() if p.strip()]
        elif prompt_preset == "custom":
            prompts = [custom_prompt.strip() or "gemstone"]
        else:
            prompts = [prompt_preset]

        logger.info("Prompts (%d): %s", len(prompts), prompts)

        # --- Ensure model on CUDA ---
        model.to(device)

        from sam3.model.sam3_image_processor import Sam3Processor
        processor = Sam3Processor(model)

        B, H, W, C = image.shape
        logger.info("Batch=%d  H=%d  W=%d  C=%d", B, H, W, C)

        all_masks = []
        all_overlays = []
        all_cropped = []

        for b_idx in range(B):
            t_batch = time.time()
            logger.info("--- Batch %d/%d ---", b_idx + 1, B)

            np_img = (image[b_idx].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(np_img)
            logger.debug("PIL image: %s  mode=%s", pil_img.size, pil_img.mode)

            # --- set_image ---
            t_si = time.time()
            state = processor.set_image(pil_img)
            logger.info("set_image: %.2fs", time.time() - t_si)
            logger.debug("state type: %s  keys: %s", type(state).__name__,
                         list(state.keys()) if isinstance(state, dict) else "N/A")

            combined_mask = torch.zeros((H, W), dtype=torch.float32, device="cpu")

            for p_idx, prompt_text in enumerate(prompts):
                t_prompt = time.time()
                logger.info("  Prompt %d/%d: '%s'", p_idx + 1, len(prompts), prompt_text)

                with torch.inference_mode():
                    output = processor.set_text_prompt(state=state, prompt=prompt_text)

                logger.debug("  output keys: %s", list(output.keys()))

                masks_raw = output["masks"]
                scores_raw = output["scores"]

                logger.debug("  masks type=%s  scores type=%s",
                             type(masks_raw).__name__, type(scores_raw).__name__)

                # --- Convert masks ---
                if isinstance(masks_raw, torch.Tensor):
                    masks_t = masks_raw.cpu().float()
                    logger.debug("  masks tensor shape: %s", masks_t.shape)
                else:
                    masks_t = torch.as_tensor(np.array(masks_raw), dtype=torch.float32)
                    logger.debug("  masks converted from list, shape: %s", masks_t.shape)

                if isinstance(scores_raw, torch.Tensor):
                    scores_t = scores_raw.cpu().float()
                else:
                    scores_t = torch.as_tensor(np.array(scores_raw), dtype=torch.float32)

                logger.debug("  scores: %s", scores_t)

                if masks_t.ndim == 2:
                    masks_t = masks_t.unsqueeze(0)
                    scores_t = scores_t.unsqueeze(0) if scores_t.ndim == 0 else scores_t

                # --- Filter ---
                keep_idx = (scores_t >= score_threshold).nonzero(as_tuple=True)[0]
                logger.info("  Total detections: %d  Above threshold (%.2f): %d",
                            masks_t.shape[0], score_threshold, len(keep_idx))

                if len(keep_idx) > max_detections:
                    top_scores, top_idx = scores_t[keep_idx].topk(max_detections)
                    keep_idx = keep_idx[top_idx]
                    logger.info("  Clamped to max_detections=%d", max_detections)

                for i, idx in enumerate(keep_idx):
                    m = masks_t[idx]
                    score_val = scores_t[idx].item()
                    logger.debug("    det %d: score=%.3f  mask shape=%s  min=%.3f max=%.3f",
                                 i, score_val, m.shape, m.min().item(), m.max().item())
                    if m.shape != (H, W):
                        logger.debug("    resizing mask %s -> (%d, %d)", m.shape, H, W)
                        m = torch.nn.functional.interpolate(
                            m.unsqueeze(0).unsqueeze(0), size=(H, W),
                            mode="bilinear", align_corners=False,
                        ).squeeze(0).squeeze(0)
                    if mask_expansion != 0:
                        m = self._expand_mask(m, mask_expansion)
                    combined_mask = torch.max(combined_mask, m)

                logger.info("  Prompt done: %.2fs", time.time() - t_prompt)

            combined_mask = combined_mask.clamp(0, 1)
            mask_coverage = (combined_mask > 0.5).float().mean().item() * 100
            logger.info("  Combined mask coverage: %.1f%%", mask_coverage)

            # --- Overlay ---
            img_tensor = image[b_idx].cpu().clone()
            overlay = img_tensor.clone()
            gem_color = torch.tensor([0.0, 1.0, 0.3], dtype=torch.float32)
            alpha = 0.35
            mask_3d = combined_mask.unsqueeze(-1)
            overlay = overlay * (1 - mask_3d * alpha) + gem_color * mask_3d * alpha
            overlay = overlay.clamp(0, 1)

            # --- Cropped ---
            cropped_rgb = img_tensor * mask_3d + (1 - mask_3d) * 1.0
            cropped_rgb = cropped_rgb.clamp(0, 1)

            all_masks.append(combined_mask)
            all_overlays.append(overlay)
            all_cropped.append(cropped_rgb)

            logger.info("  Batch %d done: %.2fs", b_idx + 1, time.time() - t_batch)

        # --- Offload ---
        if not keep_model_loaded:
            logger.info("Offloading model from GPU...")
            model.to(mm.unet_offload_device())
            mm.soft_empty_cache()

        mask_batch = torch.stack(all_masks, dim=0)
        overlay_batch = torch.stack(all_overlays, dim=0)
        cropped_batch = torch.stack(all_cropped, dim=0)

        elapsed = time.time() - t0
        logger.info("SEGMENT DONE in %.1fs  output shapes: overlay=%s mask=%s cropped=%s",
                    elapsed, overlay_batch.shape, mask_batch.shape, cropped_batch.shape)

        return (overlay_batch, mask_batch, cropped_batch)


# ============================================================================
#  3.  MULTI-PROMPT BUILDER NODE
# ============================================================================
class SAM3GemstonePromptBuilder:
    """Build a multi-line prompt list from gemstone category toggles."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "diamonds": ("BOOLEAN", {"default": True}),
                "rubies": ("BOOLEAN", {"default": True}),
                "sapphires": ("BOOLEAN", {"default": True}),
                "emeralds": ("BOOLEAN", {"default": True}),
                "amethysts": ("BOOLEAN", {"default": False}),
                "opals": ("BOOLEAN", {"default": False}),
                "topazes": ("BOOLEAN", {"default": False}),
                "garnets": ("BOOLEAN", {"default": False}),
                "tourmalines": ("BOOLEAN", {"default": False}),
                "pearls": ("BOOLEAN", {"default": False}),
                "generic_gemstone": ("BOOLEAN", {"default": True}),
                "raw_crystals": ("BOOLEAN", {"default": False}),
                "custom_additions": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Additional prompts, one per line",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("multi_prompt",)
    FUNCTION = "build"
    CATEGORY = "SAM3-Gemstone"

    _MAP = {
        "diamonds": "diamond", "rubies": "ruby", "sapphires": "sapphire",
        "emeralds": "emerald", "amethysts": "amethyst", "opals": "opal",
        "topazes": "topaz", "garnets": "garnet", "tourmalines": "tourmaline",
        "pearls": "pearl", "generic_gemstone": "gemstone", "raw_crystals": "raw crystal",
    }

    def build(self, custom_additions: str = "", **kwargs):
        lines = [self._MAP[k] for k, v in kwargs.items() if v and k in self._MAP]
        if custom_additions.strip():
            lines.extend([l.strip() for l in custom_additions.splitlines() if l.strip()])
        logger.info("PromptBuilder: %s", lines)
        return ("\n".join(lines),)


# ============================================================================
#  4.  MASK POST-PROCESSOR NODE
# ============================================================================
class SAM3GemstoneMaskPostProcess:
    """Refine gemstone mask: smooth edges, threshold, expand/erode."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "smooth_radius": ("INT", {"default": 3, "min": 0, "max": 31, "step": 1}),
                "binary_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider",
                }),
                "expand_pixels": ("INT", {"default": 0, "min": -50, "max": 50, "step": 1}),
                "invert": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("refined_mask",)
    FUNCTION = "process"
    CATEGORY = "SAM3-Gemstone"

    def process(self, mask: torch.Tensor, smooth_radius: int,
                binary_threshold: float, expand_pixels: int, invert: bool):
        logger.info("MaskPostProcess: input shape=%s  smooth=%d  thresh=%.2f  expand=%d  invert=%s",
                    mask.shape, smooth_radius, binary_threshold, expand_pixels, invert)

        result = mask.clone().float()

        if smooth_radius > 0:
            ks = smooth_radius * 2 + 1
            sigma = smooth_radius / 2.0
            coords = torch.arange(ks, dtype=torch.float32) - smooth_radius
            kernel_1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
            kernel_1d = kernel_1d / kernel_1d.sum()
            kernel_2d = kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)
            kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)

            pad = smooth_radius
            for b in range(result.shape[0]):
                m = result[b].unsqueeze(0).unsqueeze(0)
                m = torch.nn.functional.pad(m, (pad, pad, pad, pad), mode="reflect")
                m = torch.nn.functional.conv2d(m, kernel_2d)
                result[b] = m.squeeze(0).squeeze(0)

        if binary_threshold > 0:
            result = (result >= binary_threshold).float()

        if expand_pixels != 0:
            for b in range(result.shape[0]):
                result[b] = SAM3GemstoneSegmentation._expand_mask(result[b], expand_pixels)

        if invert:
            result = 1.0 - result

        logger.info("MaskPostProcess done: output shape=%s", result.shape)
        return (result.clamp(0, 1),)


# ============================================================================
#  5.  BATCH STATS NODE
# ============================================================================
class SAM3GemstoneStats:
    """Output basic statistics about detected gemstones."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT")
    RETURN_NAMES = ("stats_text", "gem_count", "coverage_pct")
    FUNCTION = "compute"
    CATEGORY = "SAM3-Gemstone"
    OUTPUT_NODE = True

    def compute(self, mask: torch.Tensor, image: torch.Tensor):
        B, H, W = mask.shape
        total_pixels = H * W
        logger.info("Stats: B=%d  H=%d  W=%d", B, H, W)

        lines = []
        total_gems = 0
        total_coverage = 0.0

        for b in range(B):
            m = mask[b]
            binary = (m > 0.5).float()
            coverage = binary.sum().item() / total_pixels * 100

            labeled = self._label_components(binary)
            n_gems = int(labeled.max().item())

            total_gems += n_gems
            total_coverage += coverage

            line = f"[Batch {b}] Gems: {n_gems} | Coverage: {coverage:.1f}% | Resolution: {W}x{H}"
            logger.info("  %s", line)
            lines.append(line)

        avg_coverage = total_coverage / B if B > 0 else 0.0
        lines.insert(0, f"=== SAM3 Gemstone Stats ({B} images) ===")
        lines.append(f"Total gems: {total_gems} | Avg coverage: {avg_coverage:.1f}%")

        return ("\n".join(lines), total_gems, round(avg_coverage, 2))

    @staticmethod
    def _label_components(binary: torch.Tensor) -> torch.Tensor:
        arr = binary.cpu().numpy().astype(np.uint8)
        H, W = arr.shape
        labels = np.zeros_like(arr, dtype=np.int32)
        label_id = 0
        for y in range(H):
            for x in range(W):
                if arr[y, x] == 1 and labels[y, x] == 0:
                    label_id += 1
                    stack = [(y, x)]
                    while stack:
                        cy, cx = stack.pop()
                        if cy < 0 or cy >= H or cx < 0 or cx >= W:
                            continue
                        if arr[cy, cx] == 0 or labels[cy, cx] != 0:
                            continue
                        labels[cy, cx] = label_id
                        stack.extend([(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)])
        return torch.from_numpy(labels)


# ============================================================================
#  MAPPINGS
# ============================================================================
NODE_CLASS_MAPPINGS = {
    "SAM3GemstoneModelLoader": SAM3GemstoneModelLoader,
    "SAM3GemstoneSegmentation": SAM3GemstoneSegmentation,
    "SAM3GemstonePromptBuilder": SAM3GemstonePromptBuilder,
    "SAM3GemstoneMaskPostProcess": SAM3GemstoneMaskPostProcess,
    "SAM3GemstoneStats": SAM3GemstoneStats,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3GemstoneModelLoader": "SAM3 Gemstone Model Loader",
    "SAM3GemstoneSegmentation": "SAM3 Gemstone Segmentation",
    "SAM3GemstonePromptBuilder": "SAM3 Gemstone Prompt Builder",
    "SAM3GemstoneMaskPostProcess": "SAM3 Gemstone Mask PostProcess",
    "SAM3GemstoneStats": "SAM3 Gemstone Stats",
}
