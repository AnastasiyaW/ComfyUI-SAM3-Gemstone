import os
import time
import logging
import torch
import numpy as np
import cv2
from PIL import Image

import comfy.model_management as mm
import folder_paths
from torchvision.ops import nms

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
# Model directory
# ---------------------------------------------------------------------------
SAM3_MODEL_DIR = os.path.join(folder_paths.models_dir, "sam3")
os.makedirs(SAM3_MODEL_DIR, exist_ok=True)
SAM3_CHECKPOINT = "sam3.pt"

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

_sam3_cache: dict = {}


def _get_dtype(precision: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]


# ============================================================================
#  1.  LOADER NODE
# ============================================================================
class SAM3GemstoneModelLoader:
    """Load SAM 3 model. CUDA only, no CPU fallback. Local checkpoint only."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "compile_model": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("SAM3_MODEL",)
    RETURN_NAMES = ("sam3_model",)
    FUNCTION = "load"
    CATEGORY = "SAM3-Gemstone"

    def load(self, precision: str, compile_model: bool):
        t0 = time.time()
        device = mm.get_torch_device()
        dtype = _get_dtype(precision)

        logger.info("=" * 60)
        logger.info("LOAD START  device=%s  dtype=%s  compile=%s", device, dtype, compile_model)
        logger.info("=" * 60)

        assert device.type == "cuda", (
            f"SAM3-Gemstone requires CUDA. Got device={device}. No CPU fallback."
        )

        prop = torch.cuda.get_device_properties(device)
        vram_bytes = getattr(prop, "total_memory", None) or getattr(prop, "total_mem", 0)
        logger.info("GPU: %s  (compute %d.%d, %.1f GB VRAM)",
                    prop.name, prop.major, prop.minor, vram_bytes / (1024**3))

        cache_key = f"sam3_{precision}_{compile_model}"
        if cache_key in _sam3_cache:
            logger.info("Returning cached model (key=%s)", cache_key)
            return (_sam3_cache[cache_key],)

        # H100 / Ampere+ fast-math
        if prop.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            logger.info("TF32 + cuDNN benchmark enabled (compute >= 8.0)")

        # Checkpoint
        ckpt_path = os.path.join(SAM3_MODEL_DIR, SAM3_CHECKPOINT)
        assert os.path.isfile(ckpt_path), (
            f"Checkpoint NOT FOUND: {ckpt_path}. "
            f"Place sam3.pt (~3.3 GB) into {SAM3_MODEL_DIR}/"
        )
        size_gb = os.path.getsize(ckpt_path) / (1024**3)
        logger.info("Checkpoint: %s (%.2f GB)", ckpt_path, size_gb)
        assert size_gb > 1.0, (
            f"sam3.pt is only {size_gb:.2f} GB — corrupted? Expected ~3.3 GB."
        )

        # BPE tokenizer
        import pkg_resources
        bpe_path = pkg_resources.resource_filename("sam3", "assets/bpe_simple_vocab_16e6.txt.gz")
        assert os.path.isfile(bpe_path), (
            f"BPE tokenizer NOT FOUND: {bpe_path}. "
            f"Copy bpe_simple_vocab_16e6.txt.gz into sam3 package assets."
        )
        logger.info("BPE tokenizer: %s", bpe_path)

        # Build model
        logger.info("Building SAM3 image model...")
        from sam3.model_builder import build_sam3_image_model

        model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=str(device),
            checkpoint_path=ckpt_path,
            load_from_HF=False,
            compile=compile_model,
        )
        logger.info("Model built. Moving to device=%s dtype=%s ...", device, dtype)
        model = model.to(device=device, dtype=dtype)
        model.eval()

        first_param = next(model.parameters())
        logger.info("Model param check: device=%s  dtype=%s", first_param.device, first_param.dtype)
        assert first_param.is_cuda, f"Model on {first_param.device}, expected CUDA."

        pipe = {"model": model, "device": device, "dtype": dtype}
        _sam3_cache[cache_key] = pipe

        logger.info("LOAD DONE in %.1fs", time.time() - t0)
        return (pipe,)


# ============================================================================
#  2.  SEGMENTATION NODE — Quality pipeline
# ============================================================================
class SAM3GemstoneSegmentation:
    """Segment gemstones with maximum quality. Multi-prompt + NMS + SAHI + filtering.
    Outputs a UNIFIED mask of ALL detected gemstones."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model": ("SAM3_MODEL",),
                "image": ("IMAGE",),
                "score_threshold": ("FLOAT", {
                    "default": 0.30, "min": 0.01, "max": 1.0, "step": 0.01,
                    "display": "slider",
                }),
                "nms_iou_threshold": ("FLOAT", {
                    "default": 0.50, "min": 0.10, "max": 0.95, "step": 0.05,
                    "display": "slider",
                }),
                "min_solidity": ("FLOAT", {
                    "default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05,
                    "display": "slider",
                }),
                "min_area_pct": ("FLOAT", {
                    "default": 0.005, "min": 0.0, "max": 5.0, "step": 0.005,
                }),
                "max_area_pct": ("FLOAT", {
                    "default": 15.0, "min": 1.0, "max": 100.0, "step": 1.0,
                }),
                "max_detections": ("INT", {
                    "default": 128, "min": 1, "max": 512, "step": 1,
                }),
                "mask_expansion": ("INT", {
                    "default": 0, "min": -50, "max": 50, "step": 1,
                }),
                "enable_sahi": ("BOOLEAN", {"default": True}),
                "sahi_tile_size": ("INT", {
                    "default": 1024, "min": 512, "max": 2048, "step": 128,
                }),
                "sahi_overlap": ("FLOAT", {
                    "default": 0.3, "min": 0.1, "max": 0.5, "step": 0.05,
                }),
                "morph_close_size": ("INT", {
                    "default": 3, "min": 0, "max": 15, "step": 2,
                }),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "multi_prompt": ("STRING", {
                    "default": "diamond",
                    "multiline": True,
                    "placeholder": "One prompt per line. 'diamond' works best.",
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

    @staticmethod
    def _get_tiles(H: int, W: int, tile_size: int, overlap: float) -> list:
        step = int(tile_size * (1 - overlap))
        tiles = []
        seen = set()
        for y in range(0, H, step):
            for x in range(0, W, step):
                x2 = min(x + tile_size, W)
                y2 = min(y + tile_size, H)
                x1 = max(0, x2 - tile_size)
                y1 = max(0, y2 - tile_size)
                key = (x1, y1, x2, y2)
                if key not in seen:
                    seen.add(key)
                    tiles.append(key)
        return tiles

    @staticmethod
    def _check_solidity(mask_np: np.ndarray, min_solidity: float) -> bool:
        contours, _ = cv2.findContours(
            mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return False
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return False
        solidity = cv2.contourArea(cnt) / hull_area
        return solidity >= min_solidity

    @staticmethod
    def _morph_close(mask_t: torch.Tensor, kernel_size: int) -> torch.Tensor:
        if kernel_size <= 0:
            return mask_t
        mask_np = (mask_t.cpu().numpy() * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel, iterations=1)
        return torch.from_numpy(closed.astype(np.float32) / 255.0)

    def _run_prompts(self, processor, state, prompts):
        """Run multiple prompts on cached image state. Returns (logits, boxes, scores) lists."""
        all_logits = []
        all_boxes = []
        all_scores = []

        for p_idx, prompt_text in enumerate(prompts):
            t_p = time.time()
            with torch.inference_mode():
                result = processor.set_text_prompt(prompt=prompt_text, state=state)

            masks_logits = result["masks_logits"]  # [N, 1, H, W] float
            boxes = result["boxes"]                 # [N, 4] xyxy
            scores = result["scores"]               # [N]

            n = scores.shape[0] if scores.ndim > 0 else (1 if scores.numel() > 0 else 0)
            logger.info("  Prompt %d/%d '%s': %d detections (%.2fs)",
                        p_idx + 1, len(prompts), prompt_text, n, time.time() - t_p)

            if n > 0:
                logits_cpu = masks_logits.squeeze(1).cpu()  # [N, H, W]
                boxes_cpu = boxes.cpu()
                scores_cpu = scores.cpu()
                if logits_cpu.ndim == 2:
                    logits_cpu = logits_cpu.unsqueeze(0)
                if boxes_cpu.ndim == 1:
                    boxes_cpu = boxes_cpu.unsqueeze(0)
                if scores_cpu.ndim == 0:
                    scores_cpu = scores_cpu.unsqueeze(0)

                for i in range(n):
                    all_logits.append(logits_cpu[i])
                    all_boxes.append(boxes_cpu[i])
                    all_scores.append(scores_cpu[i])
                    logger.debug("    det %d: score=%.3f", i, scores_cpu[i].item())

            processor.reset_all_prompts(state)

        return all_logits, all_boxes, all_scores

    # ----- main -----------------------------------------------------------

    def segment(
        self,
        sam3_model: dict,
        image: torch.Tensor,
        score_threshold: float,
        nms_iou_threshold: float,
        min_solidity: float,
        min_area_pct: float,
        max_area_pct: float,
        max_detections: int,
        mask_expansion: int,
        enable_sahi: bool,
        sahi_tile_size: int,
        sahi_overlap: float,
        morph_close_size: int,
        keep_model_loaded: bool,
        multi_prompt: str = "diamond",
    ):
        t0 = time.time()
        model = sam3_model["model"]
        device = sam3_model["device"]
        dtype = sam3_model["dtype"]

        logger.info("=" * 60)
        logger.info("SEGMENT START  device=%s  dtype=%s", device, dtype)
        logger.info("Image shape: %s  dtype: %s", image.shape, image.dtype)
        logger.info("=" * 60)

        # Resolve prompts
        prompts = [p.strip() for p in multi_prompt.strip().splitlines() if p.strip()]
        if not prompts:
            prompts = ["diamond"]
        logger.info("Prompts (%d): %s", len(prompts), prompts)

        # Build processor with LOW threshold — we filter ourselves
        from sam3.model.sam3_image_processor import Sam3Processor
        processor = Sam3Processor(model, confidence_threshold=0.01)

        model.to(device)
        B, H, W, C = image.shape
        logger.info("Batch=%d  H=%d  W=%d", B, H, W)

        all_masks_out = []
        all_overlays_out = []
        all_cropped_out = []

        for b_idx in range(B):
            t_batch = time.time()
            logger.info("--- Batch %d/%d ---", b_idx + 1, B)

            # 1. PIL conversion
            np_img = (image[b_idx].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(np_img)

            # 2. set_image — cache vision features
            t_si = time.time()
            state = processor.set_image(pil_img)
            logger.info("set_image: %.2fs", time.time() - t_si)

            # 3. Multi-prompt on full image
            logger.info("[Full-frame] Running %d prompts...", len(prompts))
            all_logits, all_boxes, all_scores = self._run_prompts(processor, state, prompts)
            logger.info("[Full-frame] Total detections: %d", len(all_scores))

            # 4. SAHI tiling (optional, for large images)
            need_sahi = enable_sahi and max(H, W) > sahi_tile_size * 2
            if need_sahi:
                tiles = self._get_tiles(H, W, sahi_tile_size, sahi_overlap)
                logger.info("[SAHI] Image %dx%d > threshold, using %d tiles (%dx%d, overlap=%.1f)",
                            W, H, len(tiles), sahi_tile_size, sahi_tile_size, sahi_overlap)

                for t_idx, (tx1, ty1, tx2, ty2) in enumerate(tiles):
                    tile_np = np_img[ty1:ty2, tx1:tx2]
                    tile_pil = Image.fromarray(tile_np)
                    tile_h, tile_w = ty2 - ty1, tx2 - tx1

                    tile_state = processor.set_image(tile_pil)
                    t_logits, t_boxes, t_scores = self._run_prompts(processor, tile_state, prompts)

                    # Remap to full image coords
                    for i in range(len(t_scores)):
                        # Box remap
                        box = t_boxes[i].clone()
                        box[0] += tx1
                        box[1] += ty1
                        box[2] += tx1
                        box[3] += ty1
                        all_boxes.append(box)
                        all_scores.append(t_scores[i])

                        # Mask remap — pad tile logit into full image
                        full_logit = torch.zeros(H, W, dtype=torch.float32)
                        tile_logit = t_logits[i]
                        # tile_logit is [tile_h, tile_w] — might differ slightly due to processor resize
                        if tile_logit.shape[0] != tile_h or tile_logit.shape[1] != tile_w:
                            tile_logit = torch.nn.functional.interpolate(
                                tile_logit.unsqueeze(0).unsqueeze(0),
                                size=(tile_h, tile_w), mode="bilinear", align_corners=False,
                            ).squeeze(0).squeeze(0)
                        full_logit[ty1:ty2, tx1:tx2] = tile_logit
                        all_logits.append(full_logit)

                    if (t_idx + 1) % 5 == 0:
                        logger.info("[SAHI] Tile %d/%d done", t_idx + 1, len(tiles))

                logger.info("[SAHI] Total detections after tiling: %d", len(all_scores))

            # 5. NMS deduplication
            if len(all_scores) == 0:
                logger.warning("No detections at all!")
                unified_mask = torch.zeros(H, W, dtype=torch.float32)
            else:
                boxes_t = torch.stack(all_boxes).to(device)
                scores_t = torch.stack(all_scores).to(device)

                # Filter by score_threshold first
                above = scores_t >= score_threshold
                indices_above = above.nonzero(as_tuple=True)[0]
                logger.info("[Filter] Score >= %.2f: %d / %d",
                            score_threshold, len(indices_above), len(scores_t))

                if len(indices_above) == 0:
                    logger.warning("All detections below score threshold!")
                    unified_mask = torch.zeros(H, W, dtype=torch.float32)
                else:
                    boxes_filtered = boxes_t[indices_above]
                    scores_filtered = scores_t[indices_above]

                    keep_nms = nms(boxes_filtered, scores_filtered, nms_iou_threshold)
                    kept_global = indices_above[keep_nms].cpu()
                    logger.info("[NMS] iou=%.2f: %d -> %d detections",
                                nms_iou_threshold, len(indices_above), len(keep_nms))

                    # 6. Quality filtering
                    image_area = H * W
                    surviving = []
                    for rank, gi in enumerate(kept_global):
                        gi = gi.item()
                        logit = all_logits[gi]
                        binary_np = (logit.numpy() > 0.5).astype(np.uint8)
                        mask_area = binary_np.sum()
                        area_pct = mask_area / image_area * 100
                        score_val = all_scores[gi].item()

                        # Area filter
                        if area_pct < min_area_pct:
                            logger.debug("    [SKIP] det %d: area %.4f%% < min %.4f%%",
                                         gi, area_pct, min_area_pct)
                            continue
                        if area_pct > max_area_pct:
                            logger.debug("    [SKIP] det %d: area %.1f%% > max %.1f%%",
                                         gi, area_pct, max_area_pct)
                            continue

                        # Solidity filter
                        if min_solidity > 0 and not self._check_solidity(binary_np, min_solidity):
                            logger.debug("    [SKIP] det %d: solidity below %.2f", gi, min_solidity)
                            continue

                        surviving.append(gi)
                        logger.debug("    [KEEP] det %d: score=%.3f area=%.3f%%",
                                     gi, score_val, area_pct)

                    logger.info("[Quality] %d -> %d detections after area+solidity",
                                len(keep_nms), len(surviving))

                    # 7. Top-K
                    if len(surviving) > max_detections:
                        surv_scores = torch.tensor([all_scores[i].item() for i in surviving])
                        _, topk_idx = surv_scores.topk(max_detections)
                        surviving = [surviving[i] for i in topk_idx]
                        logger.info("[Top-K] Clamped to %d detections", max_detections)

                    # 8. Soft max-union merge
                    if len(surviving) == 0:
                        logger.warning("No detections survived filtering!")
                        unified_mask = torch.zeros(H, W, dtype=torch.float32)
                    else:
                        unified_logits = torch.zeros(H, W, dtype=torch.float32)
                        for gi in surviving:
                            logit = all_logits[gi]  # [H, W] float [0, 1]
                            unified_logits = torch.max(unified_logits, logit)
                        unified_mask = (unified_logits > 0.5).float()
                        coverage = unified_mask.mean().item() * 100
                        logger.info("[Merge] %d masks merged, coverage: %.1f%%",
                                    len(surviving), coverage)

            # 9. Mask expansion
            if mask_expansion != 0 and unified_mask.sum() > 0:
                unified_mask = self._expand_mask(unified_mask, mask_expansion)

            # 10. Morphological closing
            if morph_close_size > 0 and unified_mask.sum() > 0:
                unified_mask = self._morph_close(unified_mask, morph_close_size)

            # 11. Generate outputs
            img_tensor = image[b_idx].cpu().clone()
            mask_3d = unified_mask.unsqueeze(-1)

            # Overlay: green tint on detected regions
            overlay = img_tensor.clone()
            gem_color = torch.tensor([0.0, 1.0, 0.3], dtype=torch.float32)
            alpha = 0.35
            overlay = overlay * (1 - mask_3d * alpha) + gem_color * mask_3d * alpha
            overlay = overlay.clamp(0, 1)

            # Cropped: gems on white background
            cropped = img_tensor * mask_3d + (1 - mask_3d) * 1.0
            cropped = cropped.clamp(0, 1)

            all_masks_out.append(unified_mask)
            all_overlays_out.append(overlay)
            all_cropped_out.append(cropped)

            logger.info("Batch %d done: %.2fs", b_idx + 1, time.time() - t_batch)

        # Offload
        if not keep_model_loaded:
            logger.info("Offloading model from GPU...")
            model.to(mm.unet_offload_device())
            mm.soft_empty_cache()

        mask_batch = torch.stack(all_masks_out, dim=0)
        overlay_batch = torch.stack(all_overlays_out, dim=0)
        cropped_batch = torch.stack(all_cropped_out, dim=0)

        logger.info("SEGMENT DONE in %.1fs  shapes: overlay=%s mask=%s cropped=%s",
                    time.time() - t0, overlay_batch.shape, mask_batch.shape, cropped_batch.shape)

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
        logger.info("MaskPostProcess: shape=%s smooth=%d thresh=%.2f expand=%d invert=%s",
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

        logger.info("MaskPostProcess done: shape=%s", result.shape)
        return (result.clamp(0, 1),)


# ============================================================================
#  5.  STATS NODE
# ============================================================================
class SAM3GemstoneStats:
    """Output statistics about detected gemstones."""

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
        logger.info("Stats: B=%d H=%d W=%d", B, H, W)

        lines = []
        total_gems = 0
        total_coverage = 0.0

        for b in range(B):
            binary = (mask[b] > 0.5).cpu().numpy().astype(np.uint8)
            coverage = binary.sum() / total_pixels * 100
            num_labels, _ = cv2.connectedComponents(binary, connectivity=4)
            n_gems = num_labels - 1  # subtract background label

            total_gems += n_gems
            total_coverage += coverage

            line = f"[Batch {b}] Gems: {n_gems} | Coverage: {coverage:.1f}% | {W}x{H}"
            logger.info("  %s", line)
            lines.append(line)

        avg_coverage = total_coverage / B if B > 0 else 0.0
        lines.insert(0, f"=== SAM3 Gemstone Stats ({B} images) ===")
        lines.append(f"Total gems: {total_gems} | Avg coverage: {avg_coverage:.1f}%")

        return ("\n".join(lines), total_gems, round(avg_coverage, 2))


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
