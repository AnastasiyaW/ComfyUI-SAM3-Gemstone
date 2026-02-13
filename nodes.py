import os
import time
import types
import logging
import torch
import numpy as np
import cv2
from PIL import Image

import comfy.model_management as mm
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

_sam3_cache: dict = {}
_zim_cache: dict = {}

ZIM_LARGE_OBJECT_PCT = 5.0


# ---------------------------------------------------------------------------
# Shared helper: load ZIM model (used by SAM3Gemstone + ZIMRefineMask)
# ---------------------------------------------------------------------------
def _load_zim_model():
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
    """All-in-one SAM3 gemstone segmentation node.
    Loads model (cached), segments gemstones, optionally refines edges with ZIM,
    outputs mask + overlay + stats."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "diamond",
                    "multiline": True,
                    "placeholder": "One prompt per line. 'diamond' works best.",
                }),
                "score_threshold": ("FLOAT", {
                    "default": 0.10, "min": 0.01, "max": 1.0, "step": 0.01,
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
                "force_sahi": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Always tile even small images (recommended for small objects like gems)",
                }),
                "sahi_tile_size": ("INT", {
                    "default": 512, "min": 256, "max": 2048, "step": 128,
                }),
                "sahi_overlap": ("FLOAT", {
                    "default": 0.3, "min": 0.1, "max": 0.5, "step": 0.05,
                }),
                "morph_close_size": ("INT", {
                    "default": 3, "min": 0, "max": 15, "step": 2,
                }),
                "use_zim_refinement": ("BOOLEAN", {"default": False}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "compile_model": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("overlay", "mask", "cropped", "stats_text", "gem_count", "coverage_pct")
    FUNCTION = "run"
    CATEGORY = "SAM3-Gemstone"
    OUTPUT_NODE = True

    # =====================================================================
    #  SAM3 Model loading (cached)
    # =====================================================================
    def _load_model(self, compile_model: bool):
        t0 = time.time()
        device = mm.get_torch_device()

        logger.info("=" * 60)
        logger.info("LOAD START  device=%s  compile=%s", device, compile_model)

        if device.type != "cuda":
            raise RuntimeError(
                f"SAM3-Gemstone requires CUDA. Got device={device}. No CPU fallback."
            )

        prop = torch.cuda.get_device_properties(device)
        vram_bytes = getattr(prop, "total_memory", None) or getattr(prop, "total_mem", 0)
        logger.info("GPU: %s  (compute %d.%d, %.1f GB VRAM)",
                    prop.name, prop.major, prop.minor, vram_bytes / (1024 ** 3))

        cache_key = f"sam3_{compile_model}"
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
        logger.info("Model built. Moving to device=%s fp32...", device)
        model = model.to(device=device, dtype=torch.float32)
        model.eval()

        first_param = next(model.parameters())
        logger.info("Model param check: device=%s  dtype=%s", first_param.device, first_param.dtype)
        if not first_param.is_cuda:
            raise RuntimeError(f"Model on {first_param.device}, expected CUDA.")

        # Build and cache processor + monkey-patch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model.box_ops import box_cxcywh_to_xyxy
        from torch.nn.functional import interpolate as F_interpolate

        processor = Sam3Processor(model, confidence_threshold=0.01)

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
            out_masks = out_masks[keep]
            out_bbox = out_bbox[keep]

            boxes = box_cxcywh_to_xyxy(out_bbox)
            img_h = state["original_height"]
            img_w = state["original_width"]
            scale_fct = torch.tensor([img_w, img_h, img_w, img_h], device=self_proc.device)
            boxes = boxes * scale_fct[None, :]

            out_masks = F_interpolate(
                out_masks.unsqueeze(1), (img_h, img_w),
                mode="bilinear", align_corners=False,
            ).sigmoid()

            state["masks_logits"] = out_masks
            state["masks"] = out_masks > 0.5
            state["boxes"] = boxes
            state["scores"] = out_probs
            return state

        processor._forward_grounding = types.MethodType(_forward_grounding_no_presence, processor)
        logger.info("Patched Sam3Processor: presence_logit_dec multiplier DISABLED")

        pipe = {"model": model, "device": device, "processor": processor}
        _sam3_cache[cache_key] = pipe

        logger.info("LOAD DONE in %.1fs", time.time() - t0)
        return pipe

    # =====================================================================
    #  ZIM refinement — process surviving detections
    # =====================================================================
    def _refine_with_zim(self, zim_predictor, np_img, surviving, all_boxes, all_logits, H, W):
        t0 = time.time()
        image_area = H * W
        refined_logits = {}

        small_indices = []
        large_indices = []
        for gi in surviving:
            box = all_boxes[gi]
            bw = (box[2] - box[0]).item()
            bh = (box[3] - box[1]).item()
            box_area_pct = (bw * bh) / image_area * 100
            if box_area_pct >= ZIM_LARGE_OBJECT_PCT:
                large_indices.append(gi)
            else:
                small_indices.append(gi)

        logger.info("[ZIM] %d small + %d large objects to refine", len(small_indices), len(large_indices))

        if small_indices:
            t_enc = time.time()
            zim_predictor.set_image(np_img)
            logger.info("[ZIM] Full-image encoder: %.2fs", time.time() - t_enc)

            for idx, gi in enumerate(small_indices):
                box = all_boxes[gi]
                bbox_np = np.array([box[0].item(), box[1].item(),
                                    box[2].item(), box[3].item()], dtype=np.float32)
                masks, scores, _ = zim_predictor.predict(box=bbox_np, multimask_output=False)
                refined_logits[gi] = torch.from_numpy(masks[0].astype(np.float32))

                if (idx + 1) % 50 == 0:
                    logger.info("[ZIM] Small: %d/%d done", idx + 1, len(small_indices))

            logger.info("[ZIM] %d small objects: %.2fs", len(small_indices), time.time() - t_enc)

        if large_indices:
            t_large = time.time()
            pad = 20

            for gi in large_indices:
                box = all_boxes[gi]
                x1 = max(0, int(box[0].item()) - pad)
                y1 = max(0, int(box[1].item()) - pad)
                x2 = min(W, int(box[2].item()) + pad)
                y2 = min(H, int(box[3].item()) + pad)

                crop = np_img[y1:y2, x1:x2].copy()
                zim_predictor.set_image(crop)

                local_box = np.array([
                    box[0].item() - x1, box[1].item() - y1,
                    box[2].item() - x1, box[3].item() - y1,
                ], dtype=np.float32)

                masks, scores, _ = zim_predictor.predict(box=local_box, multimask_output=False)

                full_alpha = np.zeros((H, W), dtype=np.float32)
                full_alpha[y1:y2, x1:x2] = masks[0].astype(np.float32)
                refined_logits[gi] = torch.from_numpy(full_alpha)

            logger.info("[ZIM] %d large objects: %.2fs", len(large_indices), time.time() - t_large)

        logger.info("[ZIM] Total refinement: %.2fs for %d objects", time.time() - t0, len(surviving))
        return refined_logits

    # =====================================================================
    #  Helpers
    # =====================================================================
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
        return cv2.contourArea(cnt) / hull_area >= min_solidity

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

            masks_logits = result["masks_logits"]
            boxes = result["boxes"]
            scores = result["scores"]

            n = scores.shape[0] if scores.ndim > 0 else (1 if scores.numel() > 0 else 0)
            logger.info("  Prompt %d/%d '%s': %d detections (%.2fs)",
                        p_idx + 1, len(prompts), prompt_text, n, time.time() - t_p)

            if n > 0:
                logits_cpu = masks_logits.squeeze(1).cpu()
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

    def _compute_stats(self, mask: torch.Tensor, H: int, W: int) -> tuple:
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
        prompt: str,
        score_threshold: float,
        nms_iou_threshold: float,
        min_solidity: float,
        min_area_pct: float,
        max_area_pct: float,
        max_detections: int,
        mask_expansion: int,
        enable_sahi: bool,
        force_sahi: bool,
        sahi_tile_size: int,
        sahi_overlap: float,
        morph_close_size: int,
        use_zim_refinement: bool,
        keep_model_loaded: bool,
        compile_model: bool,
    ):
        t0 = time.time()

        # --- Free VRAM from other models (Flux, CLIP, VAE, etc.) ---
        logger.info("Requesting VRAM cleanup before SAM3...")
        mm.soft_empty_cache()

        # --- Load / get cached model + processor ---
        pipe = self._load_model(compile_model)
        model = pipe["model"]
        device = pipe["device"]
        processor = pipe["processor"]

        # Ensure model is on GPU (no-op if already there)
        if not next(model.parameters()).is_cuda:
            model.to(device)

        # --- Load ZIM if requested ---
        zim_predictor = None
        if use_zim_refinement:
            try:
                zim_predictor = _load_zim_model()
            except Exception as e:
                logger.error("[ZIM] Failed to load: %s — continuing without refinement", e)

        logger.info("=" * 60)
        logger.info("SEGMENT START  device=%s  zim=%s", device, zim_predictor is not None)
        logger.info("Image shape: %s  dtype: %s", image.shape, image.dtype)
        logger.info("=" * 60)

        # Resolve prompts
        prompts = [p.strip() for p in prompt.strip().splitlines() if p.strip()]
        if not prompts:
            prompts = ["diamond"]
        logger.info("Prompts (%d): %s", len(prompts), prompts)

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
                        image[b_idx], processor, device, prompts, H, W,
                        score_threshold, nms_iou_threshold, min_solidity,
                        min_area_pct, max_area_pct, max_detections, mask_expansion,
                        enable_sahi, force_sahi, sahi_tile_size, sahi_overlap,
                        morph_close_size, zim_predictor,
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
            # Cleanup on success or error
            if not keep_model_loaded:
                logger.info("Offloading model from GPU...")
                model.to(mm.unet_offload_device())
                mm.soft_empty_cache()

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
        self, img_tensor_single, processor, device, prompts, H, W,
        score_threshold, nms_iou_threshold, min_solidity,
        min_area_pct, max_area_pct, max_detections, mask_expansion,
        enable_sahi, force_sahi, sahi_tile_size, sahi_overlap,
        morph_close_size, zim_predictor,
    ):
        """Process one image from batch. Returns unified_mask (H, W) float tensor."""

        np_img = (img_tensor_single.cpu().numpy() * 255).astype(np.uint8)

        need_sahi = enable_sahi and (force_sahi or max(H, W) > sahi_tile_size * 2)

        all_logits = []
        all_boxes = []
        all_scores = []

        # Full-frame pass (skip if force tiling)
        if not (need_sahi and force_sahi):
            pil_img = Image.fromarray(np_img)
            t_si = time.time()
            state = processor.set_image(pil_img)
            logger.info("set_image (full-frame): %.2fs", time.time() - t_si)
            logger.info("[Full-frame] Running %d prompts...", len(prompts))
            all_logits, all_boxes, all_scores = self._run_prompts(processor, state, prompts)
            logger.info("[Full-frame] Total detections: %d", len(all_scores))
            # Free full-frame backbone features
            del state
        else:
            logger.info("[SAHI] force_sahi=True, SKIPPING full-frame pass (saves VRAM)")

        # SAHI tiling
        if need_sahi:
            tiles = self._get_tiles(H, W, sahi_tile_size, sahi_overlap)
            logger.info("[SAHI] %d tiles (%dx%d, overlap=%.1f)",
                        len(tiles), sahi_tile_size, sahi_tile_size, sahi_overlap)

            for t_idx, (tx1, ty1, tx2, ty2) in enumerate(tiles):
                tile_np = np_img[ty1:ty2, tx1:tx2]
                tile_pil = Image.fromarray(tile_np)
                tile_h, tile_w = ty2 - ty1, tx2 - tx1

                tile_state = processor.set_image(tile_pil)
                t_logits, t_boxes, t_scores = self._run_prompts(processor, tile_state, prompts)

                # Free tile backbone features immediately
                del tile_state

                # Remap to full image coords
                for i in range(len(t_scores)):
                    box = t_boxes[i].clone()
                    box[0] += tx1
                    box[1] += ty1
                    box[2] += tx1
                    box[3] += ty1
                    all_boxes.append(box)
                    all_scores.append(t_scores[i])

                    # Store compact: (tile_logit, tile_coords) — expand later
                    tile_logit = t_logits[i]
                    if tile_logit.shape[0] != tile_h or tile_logit.shape[1] != tile_w:
                        tile_logit = torch.nn.functional.interpolate(
                            tile_logit.unsqueeze(0).unsqueeze(0),
                            size=(tile_h, tile_w), mode="bilinear", align_corners=False,
                        ).squeeze(0).squeeze(0)
                    # Store as (logit, region) to avoid HxW allocation per detection
                    all_logits.append((tile_logit, tx1, ty1, tx2, ty2))

                if (t_idx + 1) % 5 == 0:
                    logger.info("[SAHI] Tile %d/%d done", t_idx + 1, len(tiles))

            # Periodic CUDA cache cleanup during tiling
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("[SAHI] Total detections after tiling: %d", len(all_scores))

        # NMS deduplication
        if len(all_scores) == 0:
            logger.warning("No detections at all!")
            return torch.zeros(H, W, dtype=torch.float32)

        boxes_t = torch.stack(all_boxes)
        scores_t = torch.stack(all_scores)

        # NMS on CPU to avoid GPU transfer overhead for small tensors
        above = scores_t >= score_threshold
        indices_above = above.nonzero(as_tuple=True)[0]
        logger.info("[Filter] Score >= %.2f: %d / %d",
                    score_threshold, len(indices_above), len(scores_t))

        if len(indices_above) == 0:
            logger.warning("All detections below score threshold!")
            return torch.zeros(H, W, dtype=torch.float32)

        boxes_filtered = boxes_t[indices_above]
        scores_filtered = scores_t[indices_above]

        # NMS on GPU (torchvision nms is faster on CUDA)
        keep_nms = nms(boxes_filtered.to(device), scores_filtered.to(device), nms_iou_threshold)
        kept_global = indices_above[keep_nms.cpu()].cpu()
        logger.info("[NMS] iou=%.2f: %d -> %d detections",
                    nms_iou_threshold, len(indices_above), len(keep_nms))

        # Quality filtering
        image_area = H * W
        surviving = []
        for gi in kept_global:
            gi = gi.item()
            logit = self._expand_logit(all_logits[gi], H, W)
            binary_np = (logit.numpy() > 0.5).astype(np.uint8)
            mask_area = binary_np.sum()
            area_pct = mask_area / image_area * 100
            score_val = all_scores[gi].item()

            if area_pct < min_area_pct:
                logger.debug("    [SKIP] det %d: area %.4f%% < min", gi, area_pct)
                continue
            if area_pct > max_area_pct:
                logger.debug("    [SKIP] det %d: area %.1f%% > max", gi, area_pct)
                continue
            if min_solidity > 0 and not self._check_solidity(binary_np, min_solidity):
                logger.debug("    [SKIP] det %d: solidity below %.2f", gi, min_solidity)
                continue

            surviving.append(gi)
            logger.debug("    [KEEP] det %d: score=%.3f area=%.3f%%", gi, score_val, area_pct)

        logger.info("[Quality] %d -> %d detections after area+solidity",
                    len(keep_nms), len(surviving))

        # Top-K
        if len(surviving) > max_detections:
            surv_scores = torch.tensor([all_scores[i].item() for i in surviving])
            _, topk_idx = surv_scores.topk(max_detections)
            surviving = [surviving[i] for i in topk_idx]
            logger.info("[Top-K] Clamped to %d detections", max_detections)

        # ZIM refinement
        if zim_predictor is not None and len(surviving) > 0:
            try:
                # Need full-size logits for ZIM
                for gi in surviving:
                    all_logits[gi] = self._expand_logit(all_logits[gi], H, W)
                refined = self._refine_with_zim(
                    zim_predictor, np_img, surviving, all_boxes, all_logits, H, W
                )
                for gi, refined_logit in refined.items():
                    all_logits[gi] = refined_logit
                logger.info("[ZIM] Replaced %d/%d logits", len(refined), len(surviving))
            except Exception as e:
                logger.error("[ZIM] Refinement failed: %s — using SAM3 masks", e)

        # Soft max-union merge
        if len(surviving) == 0:
            logger.warning("No detections survived filtering!")
            return torch.zeros(H, W, dtype=torch.float32)

        unified_logits = torch.zeros(H, W, dtype=torch.float32)
        for gi in surviving:
            logit = self._expand_logit(all_logits[gi], H, W)
            unified_logits = torch.max(unified_logits, logit)
        unified_mask = (unified_logits > 0.5).float()
        coverage = unified_mask.mean().item() * 100
        logger.info("[Merge] %d masks merged, coverage: %.1f%%", len(surviving), coverage)

        # Mask expansion
        if mask_expansion != 0 and unified_mask.sum() > 0:
            unified_mask = self._expand_mask(unified_mask, mask_expansion)

        # Morphological closing
        if morph_close_size > 0 and unified_mask.sum() > 0:
            unified_mask = self._morph_close(unified_mask, morph_close_size)

        return unified_mask

    @staticmethod
    def _expand_logit(logit_or_tuple, H, W):
        """Expand compact tile logit to full image size, or return as-is if already full."""
        if isinstance(logit_or_tuple, tuple):
            tile_logit, tx1, ty1, tx2, ty2 = logit_or_tuple
            full = torch.zeros(H, W, dtype=torch.float32)
            full[ty1:ty2, tx1:tx2] = tile_logit
            return full
        return logit_or_tuple


# ============================================================================
#  STANDALONE NODE — ZIM Refine Mask
# ============================================================================
class ZIMRefineMask:
    """Takes an image and a rough mask, refines edges using ZIM alpha matting."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "min_object_area": ("INT", {
                    "default": 100, "min": 10, "max": 100000, "step": 10,
                    "tooltip": "Minimum object area in pixels to refine (skip tiny noise)",
                }),
                "crop_padding": ("INT", {
                    "default": 30, "min": 0, "max": 200, "step": 5,
                    "tooltip": "Pixels of padding around each object bbox for ZIM crop",
                }),
                "large_object_pct": ("FLOAT", {
                    "default": 5.0, "min": 1.0, "max": 50.0, "step": 1.0,
                    "tooltip": "Objects above this % of image area get individual crops",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("refined_mask", "overlay")
    FUNCTION = "run"
    CATEGORY = "SAM3-Gemstone"

    def run(self, image, mask, min_object_area, crop_padding, large_object_pct):
        t0 = time.time()
        B, H, W, C = image.shape
        logger.info("=" * 60)
        logger.info("[ZIM-Refine] START  B=%d  H=%d  W=%d", B, H, W)

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

            for label_id in range(1, num_labels):
                ys, xs = np.where(labels == label_id)
                obj_area = len(ys)
                if obj_area < min_object_area:
                    continue
                x1, y1 = int(xs.min()), int(ys.min())
                x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
                area_pct = obj_area / image_area * 100
                if area_pct >= large_object_pct:
                    large_objects.append((label_id, x1, y1, x2, y2))
                else:
                    small_objects.append((label_id, x1, y1, x2, y2))

            logger.info("[ZIM-Refine] %d small + %d large (skipped %d tiny)",
                        len(small_objects), len(large_objects),
                        n_objects - len(small_objects) - len(large_objects))

            unified_alpha = np.zeros((H, W), dtype=np.float32)

            if small_objects:
                t_small = time.time()
                predictor.set_image(np_img)

                for idx, (label_id, x1, y1, x2, y2) in enumerate(small_objects):
                    bbox_np = np.array([x1, y1, x2, y2], dtype=np.float32)
                    masks_out, scores, _ = predictor.predict(box=bbox_np, multimask_output=False)
                    alpha = masks_out[0].astype(np.float32)
                    obj_region = (labels == label_id)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    obj_expanded = cv2.dilate(obj_region.astype(np.uint8), kernel, iterations=2)
                    unified_alpha = np.maximum(unified_alpha, alpha * obj_expanded)

                    if (idx + 1) % 50 == 0:
                        logger.info("[ZIM-Refine] Small: %d/%d done", idx + 1, len(small_objects))

                logger.info("[ZIM-Refine] %d small: %.2fs", len(small_objects), time.time() - t_small)

            if large_objects:
                t_large = time.time()
                for label_id, x1, y1, x2, y2 in large_objects:
                    cx1 = max(0, x1 - crop_padding)
                    cy1 = max(0, y1 - crop_padding)
                    cx2 = min(W, x2 + crop_padding)
                    cy2 = min(H, y2 + crop_padding)

                    crop = np_img[cy1:cy2, cx1:cx2].copy()
                    predictor.set_image(crop)

                    local_box = np.array([x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1], dtype=np.float32)
                    masks_out, scores, _ = predictor.predict(box=local_box, multimask_output=False)

                    crop_alpha = masks_out[0].astype(np.float32)
                    obj_region = (labels == label_id)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    obj_expanded = cv2.dilate(obj_region.astype(np.uint8), kernel, iterations=3)
                    full_alpha = np.zeros((H, W), dtype=np.float32)
                    full_alpha[cy1:cy2, cx1:cx2] = crop_alpha
                    unified_alpha = np.maximum(unified_alpha, full_alpha * obj_expanded)

                logger.info("[ZIM-Refine] %d large: %.2fs", len(large_objects), time.time() - t_large)

            refined_mask = torch.from_numpy(unified_alpha)

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

        # Resolve mask to 2D
        if mask.ndim == 4:
            mask_2d = mask[0, :, :, 0]
        elif mask.ndim == 3:
            mask_2d = mask[0]
        else:
            mask_2d = mask

        # Resize mask if dimensions don't match
        if mask_2d.shape[0] != H or mask_2d.shape[1] != W:
            mask_2d = torch.nn.functional.interpolate(
                mask_2d.unsqueeze(0).unsqueeze(0).float(),
                size=(H, W), mode="bilinear", align_corners=False,
            ).squeeze(0).squeeze(0)

        if invert_mask:
            mask_2d = 1.0 - mask_2d

        # Find bounding box of non-zero mask region
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

            # Apply padding — clamped to image bounds (never errors)
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

        # Crop
        cropped_image = image[:, bbox_y:bbox_y + bbox_h, bbox_x:bbox_x + bbox_w, :]
        cropped_mask_2d = mask_2d[bbox_y:bbox_y + bbox_h, bbox_x:bbox_x + bbox_w]

        if B > 1:
            cropped_mask = cropped_mask_2d.unsqueeze(0).expand(B, -1, -1)
        else:
            cropped_mask = cropped_mask_2d.unsqueeze(0)

        # Masked composite (preview)
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

        # Resize crop if needed
        crop = processed_crop
        if crop.shape[1] != bh or crop.shape[2] != bw:
            crop = torch.nn.functional.interpolate(
                crop.permute(0, 3, 1, 2).float(),
                size=(bh, bw), mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1)

        # Match channels
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
            # Feathered blend
            if blend_mask is not None:
                if blend_mask.ndim == 3:
                    alpha = blend_mask[0]
                else:
                    alpha = blend_mask
                if alpha.shape[0] != bh or alpha.shape[1] != bw:
                    alpha = torch.nn.functional.interpolate(
                        alpha.unsqueeze(0).unsqueeze(0).float(),
                        size=(bh, bw), mode="bilinear", align_corners=False,
                    ).squeeze(0).squeeze(0)
            else:
                # Auto-generate feathered edge mask
                alpha = torch.ones(bh, bw, dtype=torch.float32)
                if feather_radius > 0:
                    alpha_np = (alpha.numpy() * 255).astype(np.uint8)
                    # Erode then blur for smooth falloff
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
