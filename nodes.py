from __future__ import annotations

import os
import time
import types
import logging
import torch
import numpy as np
import cv2
from PIL import Image

import comfy.model_management as mm
import comfy.model_patcher
import folder_paths
from torchvision.ops import nms

# ---------------------------------------------------------------------------
# Logging — INFO by default, DEBUG via env SAM3_LOG_LEVEL=DEBUG
# ---------------------------------------------------------------------------
logger = logging.getLogger("SAM3-Gemstone")
logger.setLevel(getattr(logging, os.environ.get("SAM3_LOG_LEVEL", "INFO")))
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "[SAM3-Gemstone] %(levelname)s %(funcName)s:%(lineno)d — %(message)s"
    ))
    logger.addHandler(_h)

# ---------------------------------------------------------------------------
# Model directory + cache
# ---------------------------------------------------------------------------
SAM3_MODEL_DIR = os.path.join(folder_paths.models_dir, "sam3")
os.makedirs(SAM3_MODEL_DIR, exist_ok=True)
SAM3_CHECKPOINT = "sam3.pt"

ZIM_MODEL_DIR = os.path.join(folder_paths.models_dir, "zim")
os.makedirs(ZIM_MODEL_DIR, exist_ok=True)
ZIM_BACKBONE = "vit_b"
ZIM_CKPT_SUBDIR = "zim_vit_b_2043"

# Cache: stores {"patcher": ModelPatcher, "processor": Sam3Processor}
_sam3_cache: dict = {}
_zim_cache: dict = {}


# ---------------------------------------------------------------------------
# Multi-prompt stone detection — ported from jewelry_segmenter.py
# SAM3 works best with specific noun phrases, not generic "gemstone"
# ---------------------------------------------------------------------------
STONE_PROMPTS = [
    # Core (catch most stones)
    "gemstone", "diamond", "precious stone",
    # Big 4
    "emerald", "ruby", "sapphire",
    # Popular semi-precious
    "amethyst", "topaz", "opal", "garnet", "tourmaline",
    "aquamarine", "tanzanite", "morganite", "peridot",
    # Cut shapes (catch stones by shape, not type)
    "marquise cut stone", "pear cut stone", "oval gemstone",
    "round brilliant stone", "cushion cut stone",
    # Generic fallback
    "crystal", "transparent stone", "faceted stone",
]
# Subset for SAHI tiles (speed optimization — full prompt list is too slow per tile)
TILE_PROMPTS = ["diamond", "gemstone"]


# ---------------------------------------------------------------------------
# Shared helper: load ZIM model (used by SAM3Gemstone + ZIMRefineMask)
# ---------------------------------------------------------------------------
def _load_zim_model() -> "ZimPredictor":
    """Load or return cached ZIM predictor."""
    if "zim" in _zim_cache:
        logger.info("[ZIM] Returning cached predictor")
        return _zim_cache["zim"]

    t0 = time.time()
    logger.info("[ZIM] Loading model (backbone=%s)...", ZIM_BACKBONE)

    from zim_anything import zim_model_registry, ZimPredictor

    ckpt_dir = os.path.join(ZIM_MODEL_DIR, ZIM_CKPT_SUBDIR)
    encoder_path = os.path.join(ckpt_dir, "encoder.onnx")
    decoder_path = os.path.join(ckpt_dir, "decoder.onnx")

    if not os.path.isfile(encoder_path):
        raise RuntimeError(
            f"ZIM encoder NOT FOUND: {encoder_path}. "
            f"Download from HuggingFace naver-iv/zim-anything-vitb"
        )
    if not os.path.isfile(decoder_path):
        raise RuntimeError(
            f"ZIM decoder NOT FOUND: {decoder_path}. "
            f"Download from HuggingFace naver-iv/zim-anything-vitb"
        )

    enc_mb = os.path.getsize(encoder_path) / (1024 ** 2)
    dec_mb = os.path.getsize(decoder_path) / (1024 ** 2)
    logger.info("[ZIM] Encoder: %.1f MB, Decoder: %.1f MB", enc_mb, dec_mb)

    zim_model = zim_model_registry[ZIM_BACKBONE](checkpoint=ckpt_dir)
    if torch.cuda.is_available():
        zim_model.cuda()
    predictor = ZimPredictor(zim_model)

    _zim_cache["zim"] = predictor
    logger.info("[ZIM] Loaded in %.1fs", time.time() - t0)
    return predictor


# ============================================================================
#  SINGLE NODE — SAM3 Gemstone Segmentation
# ============================================================================
class SAM3Gemstone:
    """SAM3 gemstone segmentation node.
    Loads model (cached via ModelPatcher), detects gemstones via text prompts,
    uses SAM3's own masks. Connect output to ZIMRefineMask for edge refinement.
    ComfyUI auto-offloads SAM3 when other models need VRAM."""

    # Default prompt lists (editable by user)
    DEFAULT_FULL_PROMPTS = "\n".join(STONE_PROMPTS)
    DEFAULT_TILE_PROMPTS = "\n".join(TILE_PROMPTS)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "full_image_prompts": ("STRING", {
                    "default": cls.DEFAULT_FULL_PROMPTS,
                    "multiline": True,
                    "placeholder": "Prompts for full-image pass (one per line)",
                    "tooltip": "All prompts run on full image. Default: 22 gem types + cut shapes.",
                }),
                "tile_prompts": ("STRING", {
                    "default": cls.DEFAULT_TILE_PROMPTS,
                    "multiline": True,
                    "placeholder": "Prompts for SAHI tile pass (one per line)",
                    "tooltip": "Subset of prompts for tiles (speed). Default: diamond + gemstone.",
                }),
                "sahi_tile_size": ("INT", {
                    "default": 1024, "min": 256, "max": 2048, "step": 128,
                    "tooltip": "SAM3 internally resizes to 1008×1008. "
                               "Use 1024 for native resolution per tile (best quality). "
                               "Smaller tiles = more tiles but same internal resolution.",
                }),
                "mask_mode": (["sam3_mask", "bbox_fill", "bbox_tight"], {
                    "default": "sam3_mask",
                    "tooltip": "sam3_mask = SAM3 predicted masks (best quality), "
                               "bbox_fill = filled bounding boxes (fast fallback), "
                               "bbox_tight = 80% inner bbox (tighter than full bbox)",
                }),
                "mask_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Alpha threshold for sam3_mask mode: 0.0 = soft sigmoid, "
                               "0.5 = crisp binary (best for small gems)",
                }),
                "score_threshold": ("FLOAT", {
                    "default": 0.10, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Minimum confidence score to keep a detection",
                }),
                "nms_iou_threshold": ("FLOAT", {
                    "default": 0.50, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "NMS IoU overlap threshold — lower = more aggressive dedup",
                }),
            },
            "optional": {
                "max_detections": ("INT", {
                    "default": 256, "min": 16, "max": 1024, "step": 16,
                    "tooltip": "Max gems to segment with ZIM. Lower = faster. "
                               "512 for dense pavé jewelry, 128 for simple rings.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("overlay", "mask", "cropped", "stats_text", "gem_count", "coverage_pct")
    FUNCTION = "run"
    CATEGORY = "SAM3-Gemstone"
    OUTPUT_NODE = True

    # =====================================================================
    #  SAM3 Model loading — returns ModelPatcher + processor
    # =====================================================================
    def _load_model(self) -> dict:
        t0 = time.time()
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        logger.info("=" * 60)
        logger.info("LOAD START  device=%s  offload=%s", device, offload_device)

        if device.type != "cuda":
            raise RuntimeError(
                f"SAM3-Gemstone requires CUDA. Got device={device}. No CPU fallback."
            )

        prop = torch.cuda.get_device_properties(device)
        vram_bytes = getattr(prop, "total_memory", None) or getattr(prop, "total_mem", 0)
        logger.info("GPU: %s  (compute %d.%d, %.1f GB VRAM)",
                    prop.name, prop.major, prop.minor, vram_bytes / (1024 ** 3))

        cache_key = "sam3"
        if cache_key in _sam3_cache:
            logger.info("Returning cached model (key=%s)", cache_key)
            return _sam3_cache[cache_key]

        # H100 / Ampere+ fast-math
        if prop.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            logger.info("TF32 + cuDNN benchmark enabled (compute >= 8.0)")

        # Checkpoint
        ckpt_path = os.path.join(SAM3_MODEL_DIR, SAM3_CHECKPOINT)
        if not os.path.isfile(ckpt_path):
            raise RuntimeError(
                f"Checkpoint NOT FOUND: {ckpt_path}. "
                f"Place sam3.pt (~3.3 GB) into {SAM3_MODEL_DIR}/"
            )
        size_gb = os.path.getsize(ckpt_path) / (1024 ** 3)
        logger.info("Checkpoint: %s (%.2f GB)", ckpt_path, size_gb)
        if size_gb < 1.0:
            raise RuntimeError(
                f"sam3.pt is only {size_gb:.2f} GB — corrupted? Expected ~3.3 GB."
            )

        # BPE tokenizer
        import importlib.resources as _ir
        try:
            bpe_path = str(_ir.files("sam3").joinpath("assets/bpe_simple_vocab_16e6.txt.gz"))
        except Exception:
            import pkg_resources
            bpe_path = pkg_resources.resource_filename("sam3", "assets/bpe_simple_vocab_16e6.txt.gz")
        if not os.path.isfile(bpe_path):
            raise RuntimeError(
                f"BPE tokenizer NOT FOUND: {bpe_path}. "
                f"Copy bpe_simple_vocab_16e6.txt.gz into sam3 package assets."
            )
        logger.info("BPE tokenizer: %s", bpe_path)

        # Build model — initially on CPU for ModelPatcher flow
        logger.info("Building SAM3 image model...")
        from sam3.model_builder import build_sam3_image_model

        model = build_sam3_image_model(
            bpe_path=bpe_path,
            device="cpu",
            checkpoint_path=ckpt_path,
            load_from_HF=False,
            compile=False,
        )
        model = model.to(dtype=torch.float32)
        model.eval()

        # Monkey-patch Sam3Image.device — it's a read-only @property but
        # ModelPatcher needs to set model.device during offload/load cycles.
        # Add a setter that writes to _device (which the getter already reads).
        cls = type(model)
        device_attr = getattr(cls, "device", None)
        if isinstance(device_attr, property) and device_attr.fset is None:
            old_prop = cls.device
            cls.device = property(old_prop.fget, lambda self, val: setattr(self, "_device", val))
            logger.info("Patched Sam3Image.device: added setter for ModelPatcher compatibility")
        elif isinstance(device_attr, property) and device_attr.fset is not None:
            logger.info("Sam3Image.device already has a setter — no patch needed")
        else:
            logger.warning("Sam3Image.device is not a property — monkey-patch skipped, ModelPatcher may fail")

        logger.info("Model built on CPU, wrapping in ModelPatcher...")

        # Wrap in ModelPatcher — ComfyUI manages GPU offload
        patcher = comfy.model_patcher.ModelPatcher(
            model,
            load_device=device,
            offload_device=offload_device,
        )
        logger.info("ModelPatcher created: size=%.1f MB, load=%s, offload=%s",
                    patcher.model_size() / (1024 ** 2), device, offload_device)

        # Build processor + monkey-patch (processor references model, works on whatever device model is on)
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model.box_ops import box_cxcywh_to_xyxy

        processor = Sam3Processor(model, confidence_threshold=0.15)

        @torch.inference_mode()
        def _forward_grounding_no_presence(self_proc, state):
            outputs = self_proc.model.forward_grounding(
                backbone_out=state["backbone_out"],
                find_input=self_proc.find_stage,
                geometric_prompt=state["geometric_prompt"],
                find_target=None,
            )
            out_bbox = outputs["pred_boxes"]
            out_logits = outputs["pred_logits"]
            out_masks = outputs["pred_masks"]

            # Raw sigmoid scores WITHOUT presence_logit_dec multiplier
            out_probs = out_logits.sigmoid().squeeze(-1)

            presence_raw = outputs["presence_logit_dec"].sigmoid().item()
            logger.debug("  presence_logit_dec sigmoid = %.4f (NOT applied)", presence_raw)

            keep = out_probs > self_proc.confidence_threshold
            out_probs = out_probs[keep]
            out_bbox = out_bbox[keep]
            out_masks = out_masks[keep]

            boxes = box_cxcywh_to_xyxy(out_bbox)
            img_h = state["original_height"]
            img_w = state["original_width"]
            scale_fct = torch.tensor([img_w, img_h, img_w, img_h], device=self_proc.device)
            boxes = boxes * scale_fct[None, :]

            # Store low-res masks (model decoder output, NOT interpolated to full image).
            # Upscale only after NMS to save memory.
            state["boxes"] = boxes
            state["scores"] = out_probs
            state["masks_lowres"] = out_masks  # shape (N, mask_h, mask_w)
            return state

        if hasattr(processor, '_forward_grounding'):
            processor._forward_grounding = types.MethodType(_forward_grounding_no_presence, processor)
            logger.info("Patched Sam3Processor: presence_logit_dec multiplier DISABLED")
        else:
            logger.warning("Sam3Processor has no _forward_grounding — "
                           "presence_logit_dec bypass skipped (sam3 API may have changed)")

        pipe = {"patcher": patcher, "processor": processor}
        _sam3_cache[cache_key] = pipe

        logger.info("LOAD DONE in %.1fs", time.time() - t0)
        return pipe

    # =====================================================================
    #  Helpers
    # =====================================================================
    @staticmethod
    def _get_tiles(H: int, W: int, tile_size: int, overlap: float) -> list[tuple[int, int, int, int]]:
        step = int(tile_size * (1 - overlap))
        seen: dict[tuple[int, int, int, int], None] = {}
        for y in range(0, H, step):
            for x in range(0, W, step):
                x2 = min(x + tile_size, W)
                y2 = min(y + tile_size, H)
                x1 = max(0, x2 - tile_size)
                y1 = max(0, y2 - tile_size)
                seen.setdefault((x1, y1, x2, y2), None)
        return list(seen.keys())

    def _run_prompts(self, processor, state: dict, prompts: list[str],
                     max_per_prompt: int = 50) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Run multiple prompts on cached image state.

        Returns (boxes, scores, masks_lowres) lists.
        masks_lowres are compact model-resolution masks (not interpolated to full image).
        """
        all_boxes = []
        all_scores = []
        all_masks = []

        for p_idx, prompt_text in enumerate(prompts):
            mm.throw_exception_if_processing_interrupted()
            t_p = time.time()
            with torch.inference_mode():
                result = processor.set_text_prompt(prompt=prompt_text, state=state)

            boxes = result["boxes"]
            scores = result["scores"]
            masks_lr = result.get("masks_lowres")  # (N, mh, mw) or None

            n = scores.shape[0] if scores.ndim > 0 else (1 if scores.numel() > 0 else 0)

            if n > 0:
                boxes_cpu = boxes.cpu()
                scores_cpu = scores.cpu()
                if boxes_cpu.ndim == 1:
                    boxes_cpu = boxes_cpu.unsqueeze(0)
                if scores_cpu.ndim == 0:
                    scores_cpu = scores_cpu.unsqueeze(0)

                masks_cpu = masks_lr.cpu() if masks_lr is not None else None
                if masks_cpu is not None and masks_cpu.ndim == 2:
                    masks_cpu = masks_cpu.unsqueeze(0)

                # Keep only top-K by score
                if n > max_per_prompt:
                    topk_scores, topk_idx = scores_cpu.topk(max_per_prompt)
                    boxes_cpu = boxes_cpu[topk_idx]
                    scores_cpu = topk_scores
                    if masks_cpu is not None:
                        masks_cpu = masks_cpu[topk_idx]
                    kept = max_per_prompt
                else:
                    kept = n

                for i in range(kept):
                    all_boxes.append(boxes_cpu[i])
                    all_scores.append(scores_cpu[i])
                    if masks_cpu is not None:
                        all_masks.append(masks_cpu[i])

                logger.info("  Prompt %d/%d '%s': %d raw → %d kept (%.2fs)",
                            p_idx + 1, len(prompts), prompt_text, n, kept, time.time() - t_p)
            else:
                logger.info("  Prompt %d/%d '%s': 0 detections (%.2fs)",
                            p_idx + 1, len(prompts), prompt_text, time.time() - t_p)

            processor.reset_all_prompts(state)

        return all_boxes, all_scores, all_masks

    def _compute_stats(self, mask: torch.Tensor, H: int, W: int) -> tuple[int, float]:
        binary = (mask > 0.5).cpu().numpy().astype(np.uint8)
        total_pixels = H * W
        coverage = binary.sum() / total_pixels * 100
        num_labels, _ = cv2.connectedComponents(binary, connectivity=4)
        n_gems = num_labels - 1
        return n_gems, round(coverage, 2)

    # =====================================================================
    #  Main execution
    # =====================================================================
    def run(
        self,
        image: torch.Tensor,
        full_image_prompts: str,
        tile_prompts: str,
        sahi_tile_size: int,
        mask_mode: str = "sam3_mask",
        mask_threshold: float = 0.5,
        score_threshold: float = 0.10,
        nms_iou_threshold: float = 0.50,
        max_detections: int = 256,
        use_zim_refinement: bool = False,  # backward compat — ignored
    ):
        t0 = time.time()

        min_area_pct = 0.001
        max_area_pct = 15.0
        enable_sahi = True
        sahi_overlap = 0.3
        # --- Load / get cached model + processor ---
        pipe = self._load_model()
        patcher = pipe["patcher"]
        processor = pipe["processor"]

        # Tell ComfyUI to load SAM3 to GPU (auto-evicts Flux/CLIP/VAE if needed)
        logger.info("Requesting ComfyUI to load SAM3 model to GPU...")
        mm.load_model_gpu(patcher)
        device = patcher.load_device
        logger.info("SAM3 model on GPU (device=%s)", device)

        # Update processor device reference (model may have moved)
        processor.device = device

        logger.info("=" * 60)
        logger.info("SEGMENT START  device=%s", device)
        logger.info("Image shape: %s  dtype: %s", image.shape, image.dtype)
        logger.info("=" * 60)

        # Parse user-editable prompts
        prompts = [p.strip() for p in full_image_prompts.strip().splitlines() if p.strip()]
        t_prompts = [p.strip() for p in tile_prompts.strip().splitlines() if p.strip()]
        if not prompts:
            prompts = STONE_PROMPTS
        if not t_prompts:
            t_prompts = TILE_PROMPTS
        tile_prompts = t_prompts
        logger.info("Full-image prompts (%d), Tile prompts (%d)", len(prompts), len(tile_prompts))

        B, H, W, C = image.shape
        logger.info("Batch=%d  H=%d  W=%d", B, H, W)

        all_masks_out = []
        all_overlays_out = []
        all_cropped_out = []
        total_gems = 0
        total_coverage = 0.0
        stats_lines = []

        try:
            for b_idx in range(B):
                t_batch = time.time()
                logger.info("--- Batch %d/%d ---", b_idx + 1, B)

                with torch.inference_mode():
                    unified_mask = self._process_single_image(
                        image[b_idx], processor, device, prompts, tile_prompts, H, W,
                        score_threshold, nms_iou_threshold,
                        min_area_pct, max_area_pct, max_detections,
                        enable_sahi, sahi_tile_size, sahi_overlap,
                        mask_threshold, mask_mode,
                    )

                # Stats
                n_gems, cov_pct = self._compute_stats(unified_mask, H, W)
                total_gems += n_gems
                total_coverage += cov_pct
                stats_lines.append(f"[Batch {b_idx}] Gems: {n_gems} | Coverage: {cov_pct}% | {W}x{H}")
                logger.info("  Stats: %d gems, %.1f%% coverage", n_gems, cov_pct)

                # Generate outputs
                img_tensor = image[b_idx].cpu().clone()
                mask_3d = unified_mask.unsqueeze(-1)

                overlay = img_tensor.clone()
                gem_color = torch.tensor([0.0, 1.0, 0.3], dtype=torch.float32)
                alpha = 0.35
                overlay = overlay * (1 - mask_3d * alpha) + gem_color * mask_3d * alpha
                overlay = overlay.clamp(0, 1)

                cropped = img_tensor * mask_3d + (1 - mask_3d) * 1.0
                cropped = cropped.clamp(0, 1)

                all_masks_out.append(unified_mask)
                all_overlays_out.append(overlay)
                all_cropped_out.append(cropped)

                logger.info("Batch %d done: %.2fs", b_idx + 1, time.time() - t_batch)

        finally:
            # ComfyUI manages offload via ModelPatcher — no manual cleanup needed
            # Just clear CUDA cache to free fragmented memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mask_batch = torch.stack(all_masks_out, dim=0)
        overlay_batch = torch.stack(all_overlays_out, dim=0)
        cropped_batch = torch.stack(all_cropped_out, dim=0)

        avg_coverage = total_coverage / B if B > 0 else 0.0
        stats_lines.insert(0, f"=== SAM3 Gemstone ({B} images) ===")
        stats_lines.append(f"Total gems: {total_gems} | Avg coverage: {avg_coverage:.1f}%")
        stats_text = "\n".join(stats_lines)

        logger.info("SEGMENT DONE in %.1fs  shapes: overlay=%s mask=%s cropped=%s",
                    time.time() - t0, overlay_batch.shape, mask_batch.shape, cropped_batch.shape)

        return (overlay_batch, mask_batch, cropped_batch, stats_text, total_gems, round(avg_coverage, 2))

    def _process_single_image(
        self, img_tensor_single: torch.Tensor, processor, device: torch.device,
        prompts: list[str], tile_prompts: list[str], H: int, W: int,
        score_threshold: float, nms_iou_threshold: float,
        min_area_pct: float, max_area_pct: float, max_detections: int,
        enable_sahi: bool, sahi_tile_size: int, sahi_overlap: float,
        mask_threshold: float = 0.5, mask_mode: str = "sam3_mask",
    ) -> torch.Tensor:
        """Process one image using SAM3's own masks.

        Pipeline: SAM3 prompts (full-frame + SAHI tiles) → boxes + scores + low-res masks
                  → NMS → area filter → upscale surviving masks → merge.
        mask_mode: 'sam3_mask' = use SAM3 predicted masks cropped to bbox
                   'bbox_fill' = fill entire bbox (fast, coarse)
                   'bbox_tight' = fill 80% inner bbox (tighter)
        """
        np_img = (img_tensor_single.cpu().numpy() * 255).astype(np.uint8)

        all_boxes = []
        all_scores = []
        all_masks = []

        # Full-frame pass — all prompts on full image
        pil_img = Image.fromarray(np_img)
        t_si = time.time()
        state = processor.set_image(pil_img)
        logger.info("set_image (full-frame): %.2fs", time.time() - t_si)
        logger.info("[Full-frame] Running %d prompts...", len(prompts))
        ff_boxes, ff_scores, ff_masks = self._run_prompts(processor, state, prompts)
        all_boxes.extend(ff_boxes)
        all_scores.extend(ff_scores)
        # Store masks with tile offset info for later upscale
        for m in ff_masks:
            all_masks.append((m, 0, 0, W, H))  # (mask_lowres, tx1, ty1, tx2, ty2)
        logger.info("[Full-frame] Total detections: %d", len(ff_scores))
        del state

        # SAHI tiling
        if enable_sahi:
            tiles = self._get_tiles(H, W, sahi_tile_size, sahi_overlap)
            logger.info("[SAHI] %d tiles (%dx%d, overlap=%.1f), tile_prompts=%d",
                        len(tiles), sahi_tile_size, sahi_tile_size, sahi_overlap,
                        len(tile_prompts))

            for t_idx, (tx1, ty1, tx2, ty2) in enumerate(tiles):
                mm.throw_exception_if_processing_interrupted()
                tile_pil = Image.fromarray(np_img[ty1:ty2, tx1:tx2])
                tile_state = processor.set_image(tile_pil)
                t_boxes, t_scores, t_masks = self._run_prompts(processor, tile_state, tile_prompts)
                del tile_state

                # Remap tile boxes to full image coordinates
                for i in range(len(t_scores)):
                    box = t_boxes[i].clone()
                    box[0] += tx1; box[1] += ty1; box[2] += tx1; box[3] += ty1
                    all_boxes.append(box)
                    all_scores.append(t_scores[i])
                # Store masks with tile origin for later upscale into full image
                for m in t_masks:
                    all_masks.append((m, tx1, ty1, tx2, ty2))

                if (t_idx + 1) % 10 == 0:
                    logger.info("[SAHI] Tile %d/%d done", t_idx + 1, len(tiles))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("[SAHI] Total detections after tiling: %d", len(all_scores))

        # --- NMS deduplication (boxes only — fast) ---
        if len(all_scores) == 0:
            logger.warning("No detections at all!")
            return torch.zeros(H, W, dtype=torch.float32)

        boxes_t = torch.stack(all_boxes)
        scores_t = torch.stack(all_scores)

        above = scores_t >= score_threshold
        indices_above = above.nonzero(as_tuple=True)[0]
        logger.info("[Filter] Score >= %.2f: %d / %d",
                    score_threshold, len(indices_above), len(scores_t))

        if len(indices_above) == 0:
            logger.warning("All detections below score threshold!")
            return torch.zeros(H, W, dtype=torch.float32)

        boxes_filtered = boxes_t[indices_above]
        scores_filtered = scores_t[indices_above]

        keep_nms = nms(boxes_filtered.to(device), scores_filtered.to(device), nms_iou_threshold)
        kept_indices = indices_above[keep_nms.cpu()]
        logger.info("[NMS] iou=%.2f: %d -> %d detections",
                    nms_iou_threshold, len(indices_above), len(keep_nms))

        # --- Area filter by bbox ---
        image_area = H * W
        surviving_indices = []
        for idx in kept_indices:
            idx_val = idx.item()
            box = all_boxes[idx_val]
            bw = (box[2] - box[0]).item()
            bh = (box[3] - box[1]).item()
            box_area_pct = (bw * bh) / image_area * 100
            if box_area_pct < min_area_pct or box_area_pct > max_area_pct:
                continue
            surviving_indices.append(idx_val)

        logger.info("[Area filter] %d -> %d detections", len(keep_nms), len(surviving_indices))

        # Top-K by score
        if len(surviving_indices) > max_detections:
            surviving_indices.sort(key=lambda i: all_scores[i].item(), reverse=True)
            surviving_indices = surviving_indices[:max_detections]
            logger.info("[Top-K] Clamped to %d detections", max_detections)

        if not surviving_indices:
            logger.warning("No detections survived filtering!")
            return torch.zeros(H, W, dtype=torch.float32)

        logger.info("[Surviving] %d detections → mask merge (mode=%s)", len(surviving_indices), mask_mode)

        # --- Merge masks based on mode ---
        unified_mask = torch.zeros(H, W, dtype=torch.float32)
        has_masks = len(all_masks) == len(all_boxes) and mask_mode == "sam3_mask"

        t_merge = time.time()

        if mask_mode in ("bbox_fill", "bbox_tight"):
            # Simple bbox-based masks — fast, no SAM3 mask upscaling needed
            shrink = 0.0 if mask_mode == "bbox_fill" else 0.1  # 10% shrink for tight
            for idx_val in surviving_indices:
                box = all_boxes[idx_val]
                bx1, by1 = int(box[0].item()), int(box[1].item())
                bx2, by2 = int(box[2].item()), int(box[3].item())
                if shrink > 0:
                    bw, bh = bx2 - bx1, by2 - by1
                    sx, sy = int(bw * shrink), int(bh * shrink)
                    bx1, by1 = bx1 + sx, by1 + sy
                    bx2, by2 = bx2 - sx, by2 - sy
                bx1 = max(0, bx1); by1 = max(0, by1)
                bx2 = min(W, bx2); by2 = min(H, by2)
                if bx2 > bx1 and by2 > by1:
                    unified_mask[by1:by2, bx1:bx2] = 1.0
            logger.info("[Merge] %d boxes filled (mode=%s) in %.2fs",
                        len(surviving_indices), mask_mode, time.time() - t_merge)

        elif has_masks:
            # SAM3 predicted masks — upscale, crop to bbox, merge
            _logged_shape = False
            n_flooded = 0  # masks that cover >90% of bbox (likely garbage)
            for idx_val in surviving_indices:
                mask_lr, tx1, ty1, tx2, ty2 = all_masks[idx_val]
                tile_h, tile_w = ty2 - ty1, tx2 - tx1

                if not _logged_shape:
                    logger.info("[Merge] mask_lowres shape: %s, tile: %dx%d",
                                list(mask_lr.shape), tile_w, tile_h)
                    _logged_shape = True

                # Upscale low-res mask to tile/full-frame size
                mask_up = torch.nn.functional.interpolate(
                    mask_lr.unsqueeze(0).unsqueeze(0).float(),
                    size=(tile_h, tile_w),
                    mode="bilinear", align_corners=False,
                ).squeeze(0).squeeze(0).sigmoid()

                if mask_threshold > 0.0:
                    mask_up = (mask_up > mask_threshold).float()

                # Crop mask to detection bbox + MINIMAL padding.
                # SAM3 mask covers the entire tile/image — we only want the
                # region around this specific detection's bounding box.
                box = all_boxes[idx_val]
                bx1, by1 = int(box[0].item()), int(box[1].item())
                bx2, by2 = int(box[2].item()), int(box[3].item())
                bbox_w, bbox_h = bx2 - bx1, by2 - by1
                # Minimal padding: 3px or 5% of bbox size
                pad = max(3, int(max(bbox_w, bbox_h) * 0.05))

                # Clamp to image bounds
                rx1 = max(0, bx1 - pad)
                ry1 = max(0, by1 - pad)
                rx2 = min(W, bx2 + pad)
                ry2 = min(H, by2 + pad)

                # Convert to tile-local coords for indexing mask_up
                lx1 = max(0, min(rx1 - tx1, tile_w))
                ly1 = max(0, min(ry1 - ty1, tile_h))
                lx2 = max(0, min(rx2 - tx1, tile_w))
                ly2 = max(0, min(ry2 - ty1, tile_h))

                if lx2 > lx1 and ly2 > ly1:
                    cropped = mask_up[ly1:ly2, lx1:lx2]
                    # Ensure target region matches cropped size exactly
                    # (bbox may clip at tile edge vs image edge differently)
                    th = min(cropped.shape[0], ry2 - ry1)
                    tw = min(cropped.shape[1], rx2 - rx1)
                    cropped = cropped[:th, :tw]
                    ry2_safe = ry1 + th
                    rx2_safe = rx1 + tw

                    # Quality check: if mask covers >90% of bbox, it's likely
                    # a "flooded" mask (SAM3 low-res artifact). Use ellipse instead.
                    fill_ratio = cropped.mean().item()
                    if fill_ratio > 0.90:
                        n_flooded += 1
                        # Fall back to elliptical fill inside bbox
                        yy, xx = torch.meshgrid(
                            torch.linspace(-1, 1, th),
                            torch.linspace(-1, 1, tw),
                            indexing="ij",
                        )
                        ellipse = ((xx ** 2 + yy ** 2) <= 1.0).float()
                        unified_mask[ry1:ry2_safe, rx1:rx2_safe] = torch.max(
                            unified_mask[ry1:ry2_safe, rx1:rx2_safe], ellipse)
                    else:
                        unified_mask[ry1:ry2_safe, rx1:rx2_safe] = torch.max(
                            unified_mask[ry1:ry2_safe, rx1:rx2_safe], cropped)

            logger.info("[Merge] %d masks upscaled+merged (%d flooded→ellipse) in %.2fs",
                        len(surviving_indices), n_flooded, time.time() - t_merge)
        else:
            # Fallback: box-fill if no masks available
            logger.warning("[Merge] No masks available — using box fill")
            for idx_val in surviving_indices:
                box = all_boxes[idx_val]
                x1, y1 = int(box[0].item()), int(box[1].item())
                x2, y2 = int(box[2].item()), int(box[3].item())
                unified_mask[max(0,y1):min(H,y2), max(0,x1):min(W,x2)] = 1.0

        coverage = unified_mask.mean().item() * 100
        logger.info("[Done] %d objects, coverage: %.1f%%", len(surviving_indices), coverage)

        return unified_mask



# ============================================================================
#  STANDALONE NODE — ZIM Refine Mask
# ============================================================================
class ZIMRefineMask:
    """Takes an image and a rough mask, refines edges using ZIM alpha matting.
    Original mask is the BASE (interior always filled).
    ZIM only replaces pixels in the edge band around each object contour.
    If ZIM can't find the object → fallback to original mask edges."""

    # Presets: chain_mode=True → tiny min_object_area for chain links
    PRESET_NORMAL = {"min_object_area": 50, "crop_padding": 30, "large_object_pct": 5.0,
                     "band_outer": 10, "zim_min_coverage": 0.10}
    PRESET_CHAIN = {"min_object_area": 5, "crop_padding": 15, "large_object_pct": 3.0,
                    "band_outer": 6, "zim_min_coverage": 0.05}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "chain_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable for chains/small links. Uses tiny min_object_area and tighter settings.",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("refined_mask", "overlay")
    FUNCTION = "run"
    CATEGORY = "SAM3-Gemstone"

    @staticmethod
    def _mask_to_points(obj_mask_u8: np.ndarray, bx1: int, by1: int, bx2: int, by2: int,
                        H: int, W: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Extract rich point prompts from binary mask for ZIM.

        Foreground: centroid + 4 quadrant points inside bbox (5 points).
        Background: 4 points outside bbox with adaptive margin.
        More fg points = stronger soft Gaussian attention in ZIM's 64×64 space.
        Returns (point_coords Nx2, point_labels N) or (None, None).
        """
        ys, xs = np.where(obj_mask_u8 > 0)
        if len(ys) == 0:
            return None, None

        # Foreground: centroid + 4 quadrant points
        cx = float(xs.mean())
        cy = float(ys.mean())
        bbox_w = bx2 - bx1
        bbox_h = by2 - by1
        qx = bbox_w * 0.25
        qy = bbox_h * 0.25

        fg_coords = [
            [cx, cy],
            [cx - qx, cy - qy],
            [cx + qx, cy - qy],
            [cx - qx, cy + qy],
            [cx + qx, cy + qy],
        ]

        # Background: adaptive margin based on bbox size
        bg_margin = max(5, int(max(bbox_w, bbox_h) * 0.1))
        mid_x = (bx1 + bx2) / 2
        mid_y = (by1 + by2) / 2
        bg_coords = [
            [max(0, bx1 - bg_margin), mid_y],
            [min(W - 1, bx2 + bg_margin), mid_y],
            [mid_x, max(0, by1 - bg_margin)],
            [mid_x, min(H - 1, by2 + bg_margin)],
        ]

        coords = fg_coords + bg_coords
        labels = [1] * len(fg_coords) + [0] * len(bg_coords)
        return np.array(coords, dtype=np.float32), np.array(labels, dtype=np.int32)

    def run(self, image, mask, chain_mode):
        t0 = time.time()
        B, H, W, C = image.shape

        # Select preset
        p = self.PRESET_CHAIN if chain_mode else self.PRESET_NORMAL
        min_object_area = p["min_object_area"]
        crop_padding = p["crop_padding"]
        large_object_pct = p["large_object_pct"]
        band_outer = p["band_outer"]
        zim_min_coverage = p["zim_min_coverage"]

        logger.info("=" * 60)
        logger.info("[ZIM-Refine] START  B=%d  H=%d  W=%d  chain_mode=%s  min_area=%d  band_outer=%d",
                    B, H, W, chain_mode, min_object_area, band_outer)

        predictor = _load_zim_model()

        all_masks_out = []
        all_overlays_out = []

        for b_idx in range(B):
            t_batch = time.time()
            np_img = (image[b_idx].cpu().numpy() * 255).astype(np.uint8)

            if mask.ndim == 3:
                mask_np = (mask[b_idx].cpu().numpy() > 0.5).astype(np.uint8)
            elif mask.ndim == 2:
                mask_np = (mask.cpu().numpy() > 0.5).astype(np.uint8)
            else:
                mask_np = (mask[b_idx].cpu().numpy() > 0.5).astype(np.uint8)

            num_labels, labels = cv2.connectedComponents(mask_np, connectivity=8)
            n_objects = num_labels - 1
            logger.info("[ZIM-Refine] Batch %d: %d objects", b_idx, n_objects)

            if n_objects == 0:
                all_masks_out.append(torch.zeros(H, W, dtype=torch.float32))
                all_overlays_out.append(image[b_idx].cpu().clone())
                continue

            image_area = H * W
            small_objects = []
            large_objects = []
            skipped_tiny = 0

            for label_id in range(1, num_labels):
                ys, xs = np.where(labels == label_id)
                obj_area = len(ys)
                if obj_area < min_object_area:
                    skipped_tiny += 1
                    continue
                x1, y1 = int(xs.min()), int(ys.min())
                x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
                area_pct = obj_area / image_area * 100
                if area_pct >= large_object_pct:
                    large_objects.append((label_id, x1, y1, x2, y2))
                else:
                    small_objects.append((label_id, x1, y1, x2, y2))

            logger.info("[ZIM-Refine] %d small + %d large (skipped %d tiny < %d px)",
                        len(small_objects), len(large_objects), skipped_tiny, min_object_area)

            # ---------------------------------------------------------------
            # Strategy: ALWAYS crop around each object for ZIM.
            # ZIM internally resizes to 1024x1024 — on a 4000px image a
            # 10px gemstone becomes ~2.5px, invisible. By cropping around
            # the bbox+padding, even tiny objects fill enough of ZIM's
            # input resolution for pixel-perfect alpha edges.
            # Mask-derived point prompts (centroid=fg, box corners=bg)
            # guide ZIM to the exact object.
            # ---------------------------------------------------------------
            unified_alpha = np.zeros((H, W), dtype=np.float32)

            zim_refined = 0
            zim_fallback = 0

            all_objects = small_objects + large_objects
            if all_objects:
                t_refine = time.time()

                for idx, (label_id, x1, y1, x2, y2) in enumerate(all_objects):
                    mm.throw_exception_if_processing_interrupted()
                    obj_mask_u8 = (labels == label_id).astype(np.uint8)

                    # Crop around object with padding — maximizes resolution for ZIM
                    cx1 = max(0, x1 - crop_padding)
                    cy1 = max(0, y1 - crop_padding)
                    cx2 = min(W, x2 + crop_padding)
                    cy2 = min(H, y2 + crop_padding)

                    crop = np_img[cy1:cy2, cx1:cx2].copy()
                    predictor.set_image(crop)

                    local_box = np.array([x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1], dtype=np.float32)

                    # Build point prompts in crop-local coordinates
                    local_mask = obj_mask_u8[cy1:cy2, cx1:cx2]
                    crop_h, crop_w = cy2 - cy1, cx2 - cx1
                    pt_coords, pt_labels = self._mask_to_points(
                        local_mask, x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1, crop_h, crop_w)

                    if pt_coords is not None:
                        masks_out, iou_scores, _ = predictor.predict(
                            point_coords=pt_coords, point_labels=pt_labels,
                            box=local_box, multimask_output=True)
                    else:
                        masks_out, iou_scores, _ = predictor.predict(
                            box=local_box, multimask_output=True)

                    # Select best mask by predicted IoU — use raw ZIM sigmoid
                    best_idx = int(np.argmax(iou_scores))
                    crop_alpha = masks_out[best_idx].astype(np.float32)

                    full_zim_alpha = np.zeros((H, W), dtype=np.float32)
                    full_zim_alpha[cy1:cy2, cx1:cx2] = crop_alpha

                    # Check if ZIM found something meaningful in this object's area
                    obj_coverage = (full_zim_alpha * obj_mask_u8).sum() / max(obj_mask_u8.sum(), 1)

                    if obj_coverage < zim_min_coverage:
                        zim_fallback += 1
                        unified_alpha = np.maximum(unified_alpha, obj_mask_u8.astype(np.float32))
                        logger.debug("  [ZIM] obj %d: coverage %.1f%% → FALLBACK",
                                     label_id, obj_coverage * 100)
                        continue

                    # Use ZIM alpha directly — constrain to dilated object zone
                    kern = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (band_outer * 2 + 1, band_outer * 2 + 1))
                    obj_zone = cv2.dilate(obj_mask_u8, kern, iterations=1).astype(np.float32)
                    unified_alpha = np.maximum(unified_alpha, full_zim_alpha * obj_zone)
                    zim_refined += 1

                    if (idx + 1) % 50 == 0:
                        logger.info("[ZIM-Refine] %d/%d done", idx + 1, len(all_objects))

                logger.info("[ZIM-Refine] %d objects: %.2fs", len(all_objects), time.time() - t_refine)

            # Tiny objects that were skipped — add their original mask
            for label_id in range(1, num_labels):
                ys, xs = np.where(labels == label_id)
                if len(ys) < min_object_area:
                    unified_alpha = np.maximum(
                        unified_alpha, (labels == label_id).astype(np.float32))

            logger.info("[ZIM-Refine] Refined: %d, Fallback: %d", zim_refined, zim_fallback)

            refined_mask = torch.from_numpy(unified_alpha).clamp(0, 1)

            img_tensor = image[b_idx].cpu().clone()
            mask_3d = refined_mask.unsqueeze(-1)
            overlay = img_tensor.clone()
            gem_color = torch.tensor([0.0, 1.0, 0.3], dtype=torch.float32)
            overlay = overlay * (1 - mask_3d * 0.35) + gem_color * mask_3d * 0.35
            overlay = overlay.clamp(0, 1)

            all_masks_out.append(refined_mask)
            all_overlays_out.append(overlay)
            logger.info("[ZIM-Refine] Batch %d: %.2fs", b_idx, time.time() - t_batch)

        mask_batch = torch.stack(all_masks_out, dim=0)
        overlay_batch = torch.stack(all_overlays_out, dim=0)

        logger.info("[ZIM-Refine] DONE in %.1fs", time.time() - t0)
        return (mask_batch, overlay_batch)


# ============================================================================
#  Gemstone Inpaint Crop — crop around mask bbox with safe padding
# ============================================================================
class GemstoneInpaintCrop:
    """Crop image+mask around the mask bounding box with safe padding.
    Padding is clamped to available space — no errors if image is too small.
    Returns bbox_data dict for GemstoneInpaintStitch to paste back."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "padding": ("INT", {
                    "default": 32, "min": 0, "max": 500, "step": 1,
                    "tooltip": "Extra padding around mask region (clamped to image bounds)",
                }),
                "invert_mask": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "GEMSTONE_BBOX", "STRING")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "masked_composite", "bbox_data", "info")
    FUNCTION = "crop"
    CATEGORY = "SAM3-Gemstone"

    def crop(self, image, mask, padding, invert_mask):
        t0 = time.time()
        B, H, W, C = image.shape

        if mask.ndim == 4:
            mask_2d = mask[0, :, :, 0]
        elif mask.ndim == 3:
            mask_2d = mask[0]
        else:
            mask_2d = mask

        if mask_2d.shape[0] != H or mask_2d.shape[1] != W:
            mask_2d = torch.nn.functional.interpolate(
                mask_2d.unsqueeze(0).unsqueeze(0).float(),
                size=(H, W), mode="bilinear", align_corners=False,
            ).squeeze(0).squeeze(0)

        if invert_mask:
            mask_2d = 1.0 - mask_2d

        mask_binary = (mask_2d > 0.01).float()
        nonzero = torch.nonzero(mask_binary, as_tuple=False)

        if nonzero.shape[0] == 0:
            logger.info("[InpaintCrop] Empty mask — returning full image")
            bbox_x, bbox_y = 0, 0
            bbox_w, bbox_h = W, H
        else:
            y_min = nonzero[:, 0].min().item()
            y_max = nonzero[:, 0].max().item()
            x_min = nonzero[:, 1].min().item()
            x_max = nonzero[:, 1].max().item()

            y_min = max(0, y_min - padding)
            y_max = min(H - 1, y_max + padding)
            x_min = max(0, x_min - padding)
            x_max = min(W - 1, x_max + padding)

            bbox_x = x_min
            bbox_y = y_min
            bbox_w = x_max - x_min + 1
            bbox_h = y_max - y_min + 1

        logger.info("[InpaintCrop] BBox: (%d,%d) %dx%d  padding=%d  image=%dx%d",
                    bbox_x, bbox_y, bbox_w, bbox_h, padding, W, H)

        cropped_image = image[:, bbox_y:bbox_y + bbox_h, bbox_x:bbox_x + bbox_w, :]
        cropped_mask_2d = mask_2d[bbox_y:bbox_y + bbox_h, bbox_x:bbox_x + bbox_w]

        if B > 1:
            cropped_mask = cropped_mask_2d.unsqueeze(0).expand(B, -1, -1)
        else:
            cropped_mask = cropped_mask_2d.unsqueeze(0)

        mask_expanded = cropped_mask.unsqueeze(-1).expand(-1, -1, -1, C)
        masked_composite = cropped_image * mask_expanded

        bbox_data = {
            "x": bbox_x,
            "y": bbox_y,
            "width": bbox_w,
            "height": bbox_h,
            "original_width": W,
            "original_height": H,
        }

        info = f"Crop: {bbox_w}x{bbox_h} at ({bbox_x},{bbox_y}) | Orig: {W}x{H} | Pad: {padding}"

        logger.info("[InpaintCrop] Done in %.2fs", time.time() - t0)
        return (cropped_image, cropped_mask, masked_composite, bbox_data, info)


# ============================================================================
#  Gemstone Inpaint Stitch — paste crop back into original
# ============================================================================
class GemstoneInpaintStitch:
    """Paste processed crop back into original image using bbox_data from GemstoneInpaintCrop.
    Supports hard replace or feathered blend."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "processed_crop": ("IMAGE",),
                "bbox_data": ("GEMSTONE_BBOX",),
                "blend_mode": (["replace", "feather"], {"default": "replace"}),
                "feather_radius": ("INT", {
                    "default": 8, "min": 0, "max": 100, "step": 1,
                    "tooltip": "Edge feathering pixels (feather mode only)",
                }),
            },
            "optional": {
                "blend_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result",)
    FUNCTION = "stitch"
    CATEGORY = "SAM3-Gemstone"

    def stitch(self, original_image, processed_crop, bbox_data, blend_mode, feather_radius, blend_mask=None):
        t0 = time.time()
        B, H, W, C = original_image.shape
        bx = bbox_data["x"]
        by = bbox_data["y"]
        bw = bbox_data["width"]
        bh = bbox_data["height"]

        logger.info("[InpaintStitch] Pasting %dx%d at (%d,%d) mode=%s", bw, bh, bx, by, blend_mode)

        crop = processed_crop
        if crop.shape[1] != bh or crop.shape[2] != bw:
            crop = torch.nn.functional.interpolate(
                crop.permute(0, 3, 1, 2).float(),
                size=(bh, bw), mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1)

        if crop.shape[3] != C:
            if crop.shape[3] > C:
                crop = crop[:, :, :, :C]
            else:
                pad = torch.ones(B, bh, bw, C - crop.shape[3], dtype=crop.dtype, device=crop.device)
                crop = torch.cat([crop, pad], dim=3)

        result = original_image.clone()

        if blend_mode == "replace":
            result[:, by:by + bh, bx:bx + bw, :] = crop
        else:
            if blend_mask is not None:
                if blend_mask.ndim == 3:
                    alpha = blend_mask[0]
                elif blend_mask.ndim == 4:
                    alpha = blend_mask[0, :, :, 0]
                else:
                    alpha = blend_mask
                if alpha.shape[0] != bh or alpha.shape[1] != bw:
                    logger.info("[InpaintStitch] Resizing blend_mask from %dx%d to %dx%d",
                                alpha.shape[1], alpha.shape[0], bw, bh)
                    alpha = torch.nn.functional.interpolate(
                        alpha.unsqueeze(0).unsqueeze(0).float(),
                        size=(bh, bw), mode="bilinear", align_corners=False,
                    ).squeeze(0).squeeze(0)
            else:
                alpha = torch.ones(bh, bw, dtype=torch.float32)
                if feather_radius > 0:
                    alpha_np = (alpha.numpy() * 255).astype(np.uint8)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                       (feather_radius * 2 + 1, feather_radius * 2 + 1))
                    eroded = cv2.erode(alpha_np, kernel, iterations=1)
                    blurred = cv2.GaussianBlur(eroded, (feather_radius * 2 + 1, feather_radius * 2 + 1), 0)
                    alpha = torch.from_numpy(blurred.astype(np.float32) / 255.0)

            alpha_3d = alpha.unsqueeze(-1)
            for b in range(B):
                orig_region = result[b, by:by + bh, bx:bx + bw, :]
                result[b, by:by + bh, bx:bx + bw, :] = (
                    orig_region * (1 - alpha_3d) + crop[b] * alpha_3d
                )

        logger.info("[InpaintStitch] Done in %.2fs", time.time() - t0)
        return (result,)


# ============================================================================
#  MAPPINGS
# ============================================================================
NODE_CLASS_MAPPINGS = {
    "SAM3Gemstone": SAM3Gemstone,
    "ZIMRefineMask": ZIMRefineMask,
    "GemstoneInpaintCrop": GemstoneInpaintCrop,
    "GemstoneInpaintStitch": GemstoneInpaintStitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3Gemstone": "SAM3 Gemstone",
    "ZIMRefineMask": "ZIM Refine Mask",
    "GemstoneInpaintCrop": "Gemstone Inpaint Crop",
    "GemstoneInpaintStitch": "Gemstone Inpaint Stitch",
}
