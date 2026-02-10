import os
import torch
import numpy as np
from PIL import Image

import comfy.model_management as mm
import folder_paths

# ---------------------------------------------------------------------------
# Model directory — ComfyUI/models/sam3/
# ---------------------------------------------------------------------------
SAM3_MODEL_DIR = os.path.join(folder_paths.models_dir, "sam3")
os.makedirs(SAM3_MODEL_DIR, exist_ok=True)

SAM3_HF_REPO = "facebook/sam3"
SAM3_CHECKPOINT = "sam3.pt"        # 3.45 GB — единственный чекпоинт (0.9B, максимальное качество)
SAM3_CONFIG = "config.json"

# ---------------------------------------------------------------------------
# Gemstone-specific text prompts ranked by expected segmentation quality.
# Users can pick from the dropdown or supply a custom prompt.
# ---------------------------------------------------------------------------
GEMSTONE_PROMPTS = [
    "gemstone",
    "precious stone",
    "cut gemstone",
    "polished gemstone",
    "faceted gemstone",
    "diamond",
    "ruby",
    "sapphire",
    "emerald",
    "amethyst",
    "topaz",
    "opal",
    "garnet",
    "tourmaline",
    "aquamarine",
    "tanzanite",
    "alexandrite",
    "spinel",
    "peridot",
    "citrine",
    "morganite",
    "kunzite",
    "zircon",
    "chrysoberyl",
    "iolite",
    "raw crystal",
    "mineral specimen",
    "cabochon",
    "brilliant cut stone",
    "translucent stone",
    "transparent crystal",
    "jewelry stone",
    "shiny reflective stone",
]

# ---------------------------------------------------------------------------
# Global model cache – avoids reloading between queue runs
# ---------------------------------------------------------------------------
_sam3_cache: dict = {}


def _get_dtype(precision: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]


# ============================================================================
#  1.  LOADER NODE
# ============================================================================
class SAM3GemstoneModelLoader:
    """Load SAM 3 model (0.9B params — maximum quality, single checkpoint).

    Auto-downloads sam3.pt (3.45 GB) from HuggingFace on first run.
    HF token is required because facebook/sam3 is a gated repo.
    """

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
                "compile_model": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("SAM3_MODEL",)
    RETURN_NAMES = ("sam3_model",)
    FUNCTION = "load"
    CATEGORY = "SAM3-Gemstone"

    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_checkpoint(hf_token: str) -> str:
        """Return local path to sam3.pt, downloading if missing."""
        ckpt_path = os.path.join(SAM3_MODEL_DIR, SAM3_CHECKPOINT)
        cfg_path = os.path.join(SAM3_MODEL_DIR, SAM3_CONFIG)

        if os.path.isfile(ckpt_path) and os.path.isfile(cfg_path):
            print(f"[SAM3-Gemstone] Checkpoint found: {ckpt_path}")
            return ckpt_path

        from huggingface_hub import hf_hub_download

        token = hf_token.strip() or None

        print(f"[SAM3-Gemstone] Downloading {SAM3_CHECKPOINT} from {SAM3_HF_REPO} ...")
        hf_hub_download(
            repo_id=SAM3_HF_REPO,
            filename=SAM3_CHECKPOINT,
            local_dir=SAM3_MODEL_DIR,
            token=token,
        )
        print(f"[SAM3-Gemstone] Downloading {SAM3_CONFIG} from {SAM3_HF_REPO} ...")
        hf_hub_download(
            repo_id=SAM3_HF_REPO,
            filename=SAM3_CONFIG,
            local_dir=SAM3_MODEL_DIR,
            token=token,
        )
        # tokenizer files needed by SAM 3 text encoder
        for fname in ("tokenizer.json", "tokenizer_config.json",
                      "vocab.json", "merges.txt", "special_tokens_map.json",
                      "processor_config.json"):
            target = os.path.join(SAM3_MODEL_DIR, fname)
            if not os.path.isfile(target):
                print(f"[SAM3-Gemstone] Downloading {fname} ...")
                hf_hub_download(
                    repo_id=SAM3_HF_REPO,
                    filename=fname,
                    local_dir=SAM3_MODEL_DIR,
                    token=token,
                )

        print(f"[SAM3-Gemstone] All files downloaded to {SAM3_MODEL_DIR}")
        return ckpt_path

    # ------------------------------------------------------------------
    def load(self, hf_token: str, precision: str, compile_model: bool):
        device = mm.get_torch_device()
        dtype = _get_dtype(precision)

        cache_key = f"sam3_{precision}_{compile_model}"
        if cache_key in _sam3_cache:
            return (_sam3_cache[cache_key],)

        # ---- H100 / Ampere+ fast-math ----------------------------------
        if device.type == "cuda":
            prop = torch.cuda.get_device_properties(device)
            if prop.major >= 8:  # Ampere (A100) = 8, Hopper (H100) = 9
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True

        # ---- Auto-download checkpoint ----------------------------------
        ckpt_path = self._ensure_checkpoint(hf_token)

        # ---- Build the image model from local checkpoint ---------------
        from sam3.model_builder import build_sam3_image_model

        compile_mode = "max-autotune" if compile_model else None
        model = build_sam3_image_model(
            checkpoint_path=ckpt_path,
            compile_mode=compile_mode,
        )

        # Move to device & dtype
        model = model.to(device=device, dtype=dtype)
        model.eval()

        pipe = {
            "model": model,
            "device": device,
            "dtype": dtype,
        }
        _sam3_cache[cache_key] = pipe
        return (pipe,)


# ============================================================================
#  2.  SEGMENTATION NODE  (text-prompted, gemstone-oriented)
# ============================================================================
class SAM3GemstoneSegmentation:
    """Segment gemstones in an image using SAM 3 text prompts.

    Outputs:
      - IMAGE  : original image with segmented regions highlighted
      - MASK   : binary/soft mask of detected gemstones
      - IMAGE  : cropped gemstones on transparent background (RGBA preview)
    """

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
                    "default": 0.35,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                }),
                "max_detections": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 256,
                    "step": 1,
                }),
                "mask_expansion": ("INT", {
                    "default": 0,
                    "min": -50,
                    "max": 50,
                    "step": 1,
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

    # ----- helpers --------------------------------------------------------

    @staticmethod
    def _expand_mask(mask: torch.Tensor, pixels: int) -> torch.Tensor:
        """Dilate (>0) or erode (<0) a binary mask by *pixels*."""
        if pixels == 0:
            return mask
        kernel_size = abs(pixels) * 2 + 1
        pad = abs(pixels)
        m = mask.unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]
        if pixels > 0:
            m = torch.nn.functional.max_pool2d(
                m, kernel_size=kernel_size, stride=1, padding=pad,
            )
        else:
            m = -torch.nn.functional.max_pool2d(
                -m, kernel_size=kernel_size, stride=1, padding=pad,
            )
        return m.squeeze(0).squeeze(0).clamp(0, 1)

    # ----- main -----------------------------------------------------------

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
        model = sam3_model["model"]
        device = sam3_model["device"]
        dtype = sam3_model["dtype"]

        # ---- resolve prompt(s) ------------------------------------------
        prompts: list[str] = []
        if multi_prompt.strip():
            prompts = [p.strip() for p in multi_prompt.strip().splitlines() if p.strip()]
        elif prompt_preset == "custom":
            prompts = [custom_prompt.strip() or "gemstone"]
        else:
            prompts = [prompt_preset]

        # ---- ensure model is on GPU -------------------------------------
        model.to(device)

        from sam3.model.sam3_image_processor import Sam3Processor
        processor = Sam3Processor(model)

        B, H, W, C = image.shape
        all_masks: list[torch.Tensor] = []        # per-batch masks  [H,W]
        all_overlays: list[torch.Tensor] = []      # per-batch overlay [H,W,3]
        all_cropped: list[torch.Tensor] = []       # per-batch cropped [H,W,4]

        for b_idx in range(B):
            # ComfyUI IMAGE [B,H,W,C] float32 0-1 -> PIL RGB
            np_img = (image[b_idx].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(np_img)

            # --- run SAM 3 for each prompt --------------------------------
            state = processor.set_image(pil_img)

            combined_mask = torch.zeros((H, W), dtype=torch.float32, device="cpu")

            for prompt_text in prompts:
                with torch.inference_mode():
                    output = processor.set_text_prompt(state=state, prompt=prompt_text)

                masks = output["masks"]    # list / tensor of masks
                scores = output["scores"]  # confidence per detection

                if isinstance(masks, torch.Tensor):
                    masks_t = masks.cpu().float()
                else:
                    masks_t = torch.as_tensor(np.array(masks), dtype=torch.float32)

                if isinstance(scores, torch.Tensor):
                    scores_t = scores.cpu().float()
                else:
                    scores_t = torch.as_tensor(np.array(scores), dtype=torch.float32)

                if masks_t.ndim == 2:
                    masks_t = masks_t.unsqueeze(0)
                    scores_t = scores_t.unsqueeze(0) if scores_t.ndim == 0 else scores_t

                # filter by score & max_detections
                keep_idx = (scores_t >= score_threshold).nonzero(as_tuple=True)[0]
                if len(keep_idx) > max_detections:
                    top_scores, top_idx = scores_t[keep_idx].topk(max_detections)
                    keep_idx = keep_idx[top_idx]

                for idx in keep_idx:
                    m = masks_t[idx]
                    if m.shape != (H, W):
                        m = torch.nn.functional.interpolate(
                            m.unsqueeze(0).unsqueeze(0),
                            size=(H, W),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0).squeeze(0)
                    if mask_expansion != 0:
                        m = self._expand_mask(m, mask_expansion)
                    combined_mask = torch.max(combined_mask, m)

            combined_mask = combined_mask.clamp(0, 1)

            # ---- build overlay image ------------------------------------
            img_tensor = image[b_idx].cpu().clone()  # [H,W,3]
            overlay = img_tensor.clone()
            gem_color = torch.tensor([0.0, 1.0, 0.3], dtype=torch.float32)
            alpha = 0.35
            mask_3d = combined_mask.unsqueeze(-1)  # [H,W,1]
            overlay = overlay * (1 - mask_3d * alpha) + gem_color * mask_3d * alpha
            overlay = overlay.clamp(0, 1)

            # ---- build cropped gems (RGBA as 4-channel IMAGE) -----------
            cropped_rgba = torch.zeros((H, W, 4), dtype=torch.float32)
            cropped_rgba[:, :, :3] = img_tensor
            cropped_rgba[:, :, 3] = combined_mask
            # Keep only 3 channels for ComfyUI IMAGE (alpha-composite on white)
            cropped_rgb = img_tensor * mask_3d + (1 - mask_3d) * 1.0  # white bg
            cropped_rgb = cropped_rgb.clamp(0, 1)

            all_masks.append(combined_mask)
            all_overlays.append(overlay)
            all_cropped.append(cropped_rgb)

        # ---- offload model if requested ---------------------------------
        if not keep_model_loaded:
            model.to(mm.unet_offload_device())
            mm.soft_empty_cache()

        mask_batch = torch.stack(all_masks, dim=0)         # [B,H,W]
        overlay_batch = torch.stack(all_overlays, dim=0)   # [B,H,W,3]
        cropped_batch = torch.stack(all_cropped, dim=0)    # [B,H,W,3]

        return (overlay_batch, mask_batch, cropped_batch)


# ============================================================================
#  3.  MULTI-PROMPT BUILDER NODE  (convenience helper)
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
        "diamonds": "diamond",
        "rubies": "ruby",
        "sapphires": "sapphire",
        "emeralds": "emerald",
        "amethysts": "amethyst",
        "opals": "opal",
        "topazes": "topaz",
        "garnets": "garnet",
        "tourmalines": "tourmaline",
        "pearls": "pearl",
        "generic_gemstone": "gemstone",
        "raw_crystals": "raw crystal",
    }

    def build(self, custom_additions: str = "", **kwargs):
        lines = [self._MAP[k] for k, v in kwargs.items() if v and k in self._MAP]
        if custom_additions.strip():
            lines.extend([l.strip() for l in custom_additions.splitlines() if l.strip()])
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
                "smooth_radius": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 31,
                    "step": 1,
                }),
                "binary_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                }),
                "expand_pixels": ("INT", {
                    "default": 0,
                    "min": -50,
                    "max": 50,
                    "step": 1,
                }),
                "invert": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("refined_mask",)
    FUNCTION = "process"
    CATEGORY = "SAM3-Gemstone"

    def process(self, mask: torch.Tensor, smooth_radius: int,
                binary_threshold: float, expand_pixels: int, invert: bool):
        result = mask.clone().float()

        # Gaussian smooth
        if smooth_radius > 0:
            ks = smooth_radius * 2 + 1
            sigma = smooth_radius / 2.0
            # build 1-d gaussian kernel
            coords = torch.arange(ks, dtype=torch.float32) - smooth_radius
            kernel_1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
            kernel_1d = kernel_1d / kernel_1d.sum()
            kernel_2d = kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)
            kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # [1,1,ks,ks]

            pad = smooth_radius
            for b in range(result.shape[0]):
                m = result[b].unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                m = torch.nn.functional.pad(m, (pad, pad, pad, pad), mode="reflect")
                m = torch.nn.functional.conv2d(m, kernel_2d)
                result[b] = m.squeeze(0).squeeze(0)

        # Binary threshold
        if binary_threshold > 0:
            result = (result >= binary_threshold).float()

        # Expand / erode
        if expand_pixels != 0:
            for b in range(result.shape[0]):
                result[b] = SAM3GemstoneSegmentation._expand_mask(result[b], expand_pixels)

        if invert:
            result = 1.0 - result

        return (result.clamp(0, 1),)


# ============================================================================
#  5.  BATCH STATS NODE  (quality metrics)
# ============================================================================
class SAM3GemstoneStats:
    """Output basic statistics about detected gemstones for QA."""

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

        lines = []
        total_gems = 0
        total_coverage = 0.0

        for b in range(B):
            m = mask[b]
            binary = (m > 0.5).float()
            coverage = binary.sum().item() / total_pixels * 100

            # Count connected components (simple row-scan approximation)
            labeled = self._label_components(binary)
            n_gems = int(labeled.max().item())

            total_gems += n_gems
            total_coverage += coverage

            lines.append(
                f"[Batch {b}] Gems: {n_gems} | Coverage: {coverage:.1f}% | "
                f"Resolution: {W}x{H}"
            )

        avg_coverage = total_coverage / B if B > 0 else 0.0
        lines.insert(0, f"=== SAM3 Gemstone Stats ({B} images) ===")
        lines.append(f"Total gems: {total_gems} | Avg coverage: {avg_coverage:.1f}%")

        return ("\n".join(lines), total_gems, round(avg_coverage, 2))

    @staticmethod
    def _label_components(binary: torch.Tensor) -> torch.Tensor:
        """Simple 4-connected component labeling on CPU."""
        arr = binary.cpu().numpy().astype(np.uint8)
        H, W = arr.shape
        labels = np.zeros_like(arr, dtype=np.int32)
        label_id = 0
        for y in range(H):
            for x in range(W):
                if arr[y, x] == 1 and labels[y, x] == 0:
                    label_id += 1
                    # BFS flood fill
                    stack = [(y, x)]
                    while stack:
                        cy, cx = stack.pop()
                        if cy < 0 or cy >= H or cx < 0 or cx >= W:
                            continue
                        if arr[cy, cx] == 0 or labels[cy, cx] != 0:
                            continue
                        labels[cy, cx] = label_id
                        stack.extend([(cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)])
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
