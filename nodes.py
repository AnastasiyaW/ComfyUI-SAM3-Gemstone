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
# Multi-prompt stone detection
# ---------------------------------------------------------------------------
STONE_PROMPTS = [
    "gemstone", "diamond", "precious stone",
    "emerald", "ruby", "sapphire",
    "amethyst", "topaz", "opal", "garnet", "tourmaline",
    "aquamarine", "tanzanite", "morganite", "peridot",
    "marquise cut stone", "pear cut stone", "oval gemstone",
    "round brilliant stone", "cushion cut stone",
    "crystal", "transparent stone", "faceted stone",
]
TILE_PROMPTS = ["diamond", "gemstone"]


# ============================================================================
#  Shared helpers — module-level functions
# ============================================================================
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


def _load_sam3_model() -> dict:
    """Load or return cached SAM3 model + processor wrapped in ModelPatcher.

    Returns dict with keys 'patcher' (ModelPatcher) and 'processor' (Sam3Processor).
    """
    t0 = time.time()
    device = mm.get_torch_device()
    offload_device = mm.unet_offload_device()

    cache_key = "sam3"
    if cache_key in _sam3_cache:
        logger.info("Returning cached SAM3 model (key=%s)", cache_key)
        return _sam3_cache[cache_key]

    logger.info("=" * 60)
    logger.info("SAM3 LOAD START  device=%s  offload=%s", device, offload_device)

    if device.type != "cuda":
        raise RuntimeError(
            f"SAM3-Gemstone requires CUDA. Got device={device}. No CPU fallback."
        )

    prop = torch.cuda.get_device_properties(device)
    vram_bytes = getattr(prop, "total_memory", None) or getattr(prop, "total_mem", 0)
    logger.info("GPU: %s  (compute %d.%d, %.1f GB VRAM)",
                prop.name, prop.major, prop.minor, vram_bytes / (1024 ** 3))

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

    # Build model on CPU for ModelPatcher flow
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

    # Monkey-patch Sam3Image.device — read-only @property but
    # ModelPatcher needs to set model.device during offload/load cycles.
    cls = type(model)
    device_attr = getattr(cls, "device", None)
    if isinstance(device_attr, property) and device_attr.fset is None:
        old_prop = cls.device
        cls.device = property(old_prop.fget, lambda self, val: setattr(self, "_device", val))
        logger.info("Patched Sam3Image.device: added setter for ModelPatcher compatibility")

    logger.info("Model built on CPU, wrapping in ModelPatcher...")

    # Wrap in ModelPatcher — ComfyUI manages GPU offload
    patcher = comfy.model_patcher.ModelPatcher(
        model,
        load_device=device,
        offload_device=offload_device,
    )
    logger.info("ModelPatcher created: size=%.1f MB, load=%s, offload=%s",
                patcher.model_size() / (1024 ** 2), device, offload_device)

    # Build processor + monkey-patch presence_logit_dec bypass
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

        state["boxes"] = boxes
        state["scores"] = out_probs
        state["masks_lowres"] = out_masks
        return state

    if hasattr(processor, '_forward_grounding'):
        processor._forward_grounding = types.MethodType(_forward_grounding_no_presence, processor)
        logger.info("Patched Sam3Processor: presence_logit_dec multiplier DISABLED")

    pipe = {"patcher": patcher, "processor": processor}
    _sam3_cache[cache_key] = pipe

    logger.info("SAM3 LOAD DONE in %.1fs", time.time() - t0)
    return pipe


def _normalize_mask_to_2d(mask: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Normalize ComfyUI mask tensor to 2D (H, W) float."""
    if mask.ndim == 4:
        m = mask[0, :, :, 0]
    elif mask.ndim == 3:
        m = mask[0]
    else:
        m = mask
    if m.shape[0] != H or m.shape[1] != W:
        m = torch.nn.functional.interpolate(
            m.unsqueeze(0).unsqueeze(0).float(),
            size=(H, W), mode="nearest",
        ).squeeze(0).squeeze(0)
    return m.float()


def _roi_from_mask(mask_2d: torch.Tensor, H: int, W: int, padding_ratio: float = 0.05):
    """Get ROI bounding box from binary mask. Returns (x1, y1, x2, y2) or None."""
    binary = mask_2d > 0.5
    nonzero = torch.nonzero(binary, as_tuple=False)
    if nonzero.shape[0] == 0:
        return None
    y_min = nonzero[:, 0].min().item()
    y_max = nonzero[:, 0].max().item() + 1
    x_min = nonzero[:, 1].min().item()
    x_max = nonzero[:, 1].max().item() + 1
    # Padding
    rw = x_max - x_min
    rh = y_max - y_min
    pad_x = int(rw * padding_ratio)
    pad_y = int(rh * padding_ratio)
    x1 = max(0, x_min - pad_x)
    y1 = max(0, y_min - pad_y)
    x2 = min(W, x_max + pad_x)
    y2 = min(H, y_max + pad_y)
    return (x1, y1, x2, y2)


def _mask_to_points(obj_mask_u8: np.ndarray, bx1: int, by1: int, bx2: int, by2: int,
                    H: int, W: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract rich point prompts from binary mask for ZIM.

    Foreground: centroid + 4 quadrant points (5 points).
    Background: 4 points outside bbox with adaptive margin.
    """
    ys, xs = np.where(obj_mask_u8 > 0)
    if len(ys) == 0:
        return None, None

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


def _split_large_blob(blob_mask: np.ndarray, min_area: int, max_crop: int) -> list:
    """Split a large connected component into smaller sub-objects via progressive erosion."""
    results = []
    H, W = blob_mask.shape

    for kern_size in [3, 5, 9, 15, 25]:
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kern_size, kern_size))
        eroded = cv2.erode(blob_mask, kern, iterations=1)
        n_labels, sub_labels = cv2.connectedComponents(eroded, connectivity=8)

        if n_labels - 1 < 2:
            continue

        sub_objects = []
        for sid in range(1, n_labels):
            ys, xs = np.where(sub_labels == sid)
            if len(ys) < min_area:
                continue
            sub_mask_dilated = np.zeros_like(blob_mask)
            sub_mask_dilated[sub_labels == sid] = 1
            sub_mask_dilated = cv2.dilate(sub_mask_dilated, kern, iterations=1)
            sub_mask_final = (sub_mask_dilated & blob_mask).astype(np.uint8)
            fys, fxs = np.where(sub_mask_final > 0)
            if len(fys) < min_area:
                continue
            fx1, fy1 = int(fxs.min()), int(fys.min())
            fx2, fy2 = int(fxs.max()) + 1, int(fys.max()) + 1
            sub_objects.append((sub_mask_final, fx1, fy1, fx2, fy2))

        if len(sub_objects) >= 2:
            return sub_objects

    # Could not split — return as single object
    ys, xs = np.where(blob_mask > 0)
    if len(ys) == 0:
        return []
    return [(blob_mask, int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)]


# ============================================================================
#  SAM3 Gemstone Segmentation (V2: search mask + ZIM + 3 size layers)
# ============================================================================
class SAM3GemstoneV2:
    """SAM3 gemstone segmentation with search mask, integrated ZIM refinement,
    and 3-layer size-stratified output (large / medium / small stones).

    search_mask: white = search region, black = skip.
    SAM3 detects gemstones in the white region, ZIM refines edges.
    Output: 3 MASK layers split by stone area as % of image."""

    DEFAULT_FULL_PROMPTS = "\n".join(STONE_PROMPTS)
    DEFAULT_TILE_PROMPTS = "\n".join(TILE_PROMPTS)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "search_mask": ("MASK", {
                    "tooltip": "White = search region, Black = skip. "
                               "SAM3 only detects gemstones within the white area.",
                }),
                "full_image_prompts": ("STRING", {
                    "default": cls.DEFAULT_FULL_PROMPTS,
                    "multiline": True,
                    "placeholder": "Prompts for full-image pass (one per line)",
                }),
                "tile_prompts": ("STRING", {
                    "default": cls.DEFAULT_TILE_PROMPTS,
                    "multiline": True,
                    "placeholder": "Prompts for SAHI tile pass (one per line)",
                }),
                "sahi_tile_size": ("INT", {
                    "default": 1024, "min": 256, "max": 2048, "step": 128,
                }),
                "mask_mode": (["sam3_mask", "bbox_fill", "bbox_tight"], {
                    "default": "sam3_mask",
                }),
                "mask_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                }),
                "score_threshold": ("FLOAT", {
                    "default": 0.10, "min": 0.01, "max": 1.0, "step": 0.01,
                }),
                "nms_iou_threshold": ("FLOAT", {
                    "default": 0.50, "min": 0.1, "max": 1.0, "step": 0.05,
                }),
                "small_pct_threshold": ("FLOAT", {
                    "default": 0.1, "min": 0.01, "max": 5.0, "step": 0.01,
                    "tooltip": "Stones below this % of image area → 'small' layer",
                }),
                "large_pct_threshold": ("FLOAT", {
                    "default": 2.0, "min": 0.1, "max": 20.0, "step": 0.1,
                    "tooltip": "Stones above this % of image area → 'large' layer",
                }),
            },
            "optional": {
                "max_detections": ("INT", {
                    "default": 256, "min": 16, "max": 1024, "step": 16,
                }),
                "enable_zim_refinement": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Run ZIM per-stone refinement for pixel-perfect alpha edges.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("overlay", "mask_large", "mask_medium", "mask_small",
                    "stats_text", "gem_count", "coverage_pct")
    FUNCTION = "run"
    CATEGORY = "🔮 HappyIn SAM3"
    OUTPUT_NODE = True

    # -----------------------------------------------------------------
    #  Helpers
    # -----------------------------------------------------------------
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
                     max_per_prompt: int = 50) -> tuple[list, list, list]:
        """Run multiple prompts on cached image state. Returns (boxes, scores, masks_lowres)."""
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
            masks_lr = result.get("masks_lowres")

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

    # -----------------------------------------------------------------
    #  ZIM refinement (integrated from former ZIMRefineMask)
    # -----------------------------------------------------------------
    def _zim_refine(self, np_img: np.ndarray, unified_binary: np.ndarray,
                    H: int, W: int, search_mask_np: np.ndarray | None = None) -> np.ndarray:
        """Run ZIM refinement on each connected component of the unified SAM3 mask.

        Returns unified_alpha (H, W) float32 with soft ZIM edges.
        """
        try:
            predictor = _load_zim_model()
        except Exception as e:
            logger.warning("[ZIM] Cannot load: %s — falling back to SAM3 masks", e)
            return unified_binary.astype(np.float32)

        min_object_area = 50
        crop_padding = 30
        MAX_ZIM_CROP = 1024

        num_labels, labels = cv2.connectedComponents(
            unified_binary.astype(np.uint8), connectivity=8)
        n_objects = num_labels - 1
        logger.info("[ZIM-Refine] %d objects from SAM3 mask", n_objects)

        if n_objects == 0:
            return unified_binary.astype(np.float32)

        # Collect objects, split large blobs
        objects = []
        skipped_tiny = 0
        for label_id in range(1, num_labels):
            ys, xs = np.where(labels == label_id)
            obj_area = len(ys)
            if obj_area < min_object_area:
                skipped_tiny += 1
                continue
            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
            bbox_w, bbox_h = x2 - x1, y2 - y1

            if max(bbox_w, bbox_h) > MAX_ZIM_CROP:
                obj_mask = (labels == label_id).astype(np.uint8)
                split_objects = _split_large_blob(obj_mask, min_object_area, MAX_ZIM_CROP)
                logger.info("[ZIM-Refine] Large blob %d (%dx%d) → %d sub-objects",
                            label_id, bbox_w, bbox_h, len(split_objects))
                objects.extend(split_objects)
            else:
                obj_mask = (labels == label_id).astype(np.uint8)
                objects.append((obj_mask, x1, y1, x2, y2))

        logger.info("[ZIM-Refine] %d objects (skipped %d tiny)", len(objects), skipped_tiny)

        unified_alpha = np.zeros((H, W), dtype=np.float32)
        zim_refined = 0

        if objects:
            t_refine = time.time()
            for idx, (obj_mask_u8, x1, y1, x2, y2) in enumerate(objects):
                mm.throw_exception_if_processing_interrupted()

                cx1 = max(0, x1 - crop_padding)
                cy1 = max(0, y1 - crop_padding)
                cx2 = min(W, x2 + crop_padding)
                cy2 = min(H, y2 + crop_padding)

                crop = np_img[cy1:cy2, cx1:cx2].copy()
                predictor.set_image(crop)

                local_box = np.array([x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1], dtype=np.float32)
                local_mask = obj_mask_u8[cy1:cy2, cx1:cx2]
                crop_h, crop_w = cy2 - cy1, cx2 - cx1
                pt_coords, pt_labels = _mask_to_points(
                    local_mask, x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1, crop_h, crop_w)

                try:
                    if pt_coords is not None:
                        masks_out, iou_scores, _ = predictor.predict(
                            point_coords=pt_coords, point_labels=pt_labels,
                            box=local_box, multimask_output=True)
                    else:
                        masks_out, iou_scores, _ = predictor.predict(
                            box=local_box, multimask_output=True)

                    best_idx = int(np.argmax(iou_scores))
                    crop_alpha = masks_out[best_idx].astype(np.float32)

                    unified_alpha[cy1:cy2, cx1:cx2] = np.maximum(
                        unified_alpha[cy1:cy2, cx1:cx2], crop_alpha)
                    zim_refined += 1
                except Exception as e:
                    logger.warning("[ZIM] Object %d failed: %s — keeping SAM3 mask", idx, e)
                    # Fallback: use original binary for this object
                    unified_alpha[cy1:cy2, cx1:cx2] = np.maximum(
                        unified_alpha[cy1:cy2, cx1:cx2],
                        obj_mask_u8[cy1:cy2, cx1:cx2].astype(np.float32))

                if (idx + 1) % 50 == 0:
                    logger.info("[ZIM-Refine] %d/%d done (%.1fs)",
                                idx + 1, len(objects), time.time() - t_refine)

            logger.info("[ZIM-Refine] %d/%d objects refined in %.2fs",
                        zim_refined, len(objects), time.time() - t_refine)

        # Add tiny objects that were skipped
        for label_id in range(1, num_labels):
            ys, xs = np.where(labels == label_id)
            if len(ys) < min_object_area:
                unified_alpha = np.maximum(
                    unified_alpha, (labels == label_id).astype(np.float32))

        # Intersect with search mask
        if search_mask_np is not None:
            unified_alpha *= search_mask_np

        return unified_alpha

    # -----------------------------------------------------------------
    #  Size splitting
    # -----------------------------------------------------------------
    @staticmethod
    def _split_by_size(unified_alpha: np.ndarray, H: int, W: int,
                       small_pct: float, large_pct: float):
        """Split unified mask into 3 layers by stone area as % of image."""
        binary = (unified_alpha > 0.5).astype(np.uint8)
        image_area = H * W
        num_labels, labels = cv2.connectedComponents(binary, connectivity=8)

        mask_large = np.zeros((H, W), dtype=np.float32)
        mask_medium = np.zeros((H, W), dtype=np.float32)
        mask_small = np.zeros((H, W), dtype=np.float32)

        n_large = n_medium = n_small = 0

        for label_id in range(1, num_labels):
            component = (labels == label_id)
            area_pct = component.sum() / image_area * 100
            component_alpha = unified_alpha * component.astype(np.float32)

            if area_pct >= large_pct:
                mask_large = np.maximum(mask_large, component_alpha)
                n_large += 1
            elif area_pct >= small_pct:
                mask_medium = np.maximum(mask_medium, component_alpha)
                n_medium += 1
            else:
                mask_small = np.maximum(mask_small, component_alpha)
                n_small += 1

        logger.info("[SizeSplit] large=%d, medium=%d, small=%d (thresholds: <%.2f%% small, >%.2f%% large)",
                    n_large, n_medium, n_small, small_pct, large_pct)

        return (
            torch.from_numpy(mask_large).float(),
            torch.from_numpy(mask_medium).float(),
            torch.from_numpy(mask_small).float(),
            n_large + n_medium + n_small,
        )

    # -----------------------------------------------------------------
    #  Process single image (SAM3 detection pipeline)
    # -----------------------------------------------------------------
    def _process_single_image(
        self, img_tensor_single: torch.Tensor, processor, device,
        prompts: list[str], tile_prompts_list: list[str], H: int, W: int,
        score_threshold: float, nms_iou_threshold: float,
        min_area_pct: float, max_area_pct: float, max_detections: int,
        enable_sahi: bool, sahi_tile_size: int, sahi_overlap: float,
        mask_threshold: float, mask_mode: str,
        roi_bbox: tuple[int, int, int, int] | None = None,
    ) -> torch.Tensor:
        """Run SAM3 detection pipeline on one image. Returns unified mask (H, W) float."""
        np_img = (img_tensor_single.cpu().numpy() * 255).astype(np.uint8)

        all_boxes = []
        all_scores = []
        all_masks = []

        # Crop to ROI if provided
        if roi_bbox is not None:
            rx1, ry1, rx2, ry2 = roi_bbox
            roi_img = np_img[ry1:ry2, rx1:rx2]
            roi_pil = Image.fromarray(roi_img)
            roi_h, roi_w = ry2 - ry1, rx2 - rx1
        else:
            roi_pil = Image.fromarray(np_img)
            rx1, ry1 = 0, 0
            roi_h, roi_w = H, W

        # Full-frame pass on ROI
        t_si = time.time()
        state = processor.set_image(roi_pil)
        logger.info("set_image (ROI %dx%d): %.2fs", roi_w, roi_h, time.time() - t_si)
        logger.info("[Full-frame] Running %d prompts...", len(prompts))
        ff_boxes, ff_scores, ff_masks = self._run_prompts(processor, state, prompts)

        # Remap ROI boxes to full-image coordinates
        for i in range(len(ff_scores)):
            box = ff_boxes[i].clone()
            box[0] += rx1; box[1] += ry1; box[2] += rx1; box[3] += ry1
            all_boxes.append(box)
            all_scores.append(ff_scores[i])
        for m in ff_masks:
            all_masks.append((m, rx1, ry1, rx1 + roi_w, ry1 + roi_h))
        logger.info("[Full-frame] Total detections: %d", len(ff_scores))
        del state

        # SAHI tiling within ROI
        if enable_sahi:
            tiles = self._get_tiles(roi_h, roi_w, sahi_tile_size, sahi_overlap)
            # Offset tiles to full-image coordinates
            tiles_full = [(t[0] + rx1, t[1] + ry1, t[2] + rx1, t[3] + ry1) for t in tiles]
            logger.info("[SAHI] %d tiles (%dx%d), tile_prompts=%d",
                        len(tiles_full), sahi_tile_size, sahi_tile_size, len(tile_prompts_list))

            for t_idx, (tx1, ty1, tx2, ty2) in enumerate(tiles_full):
                mm.throw_exception_if_processing_interrupted()
                tile_pil = Image.fromarray(np_img[ty1:ty2, tx1:tx2])
                tile_state = processor.set_image(tile_pil)
                t_boxes, t_scores, t_masks = self._run_prompts(processor, tile_state, tile_prompts_list)
                del tile_state

                for i in range(len(t_scores)):
                    box = t_boxes[i].clone()
                    box[0] += tx1; box[1] += ty1; box[2] += tx1; box[3] += ty1
                    all_boxes.append(box)
                    all_scores.append(t_scores[i])
                for m in t_masks:
                    all_masks.append((m, tx1, ty1, tx2, ty2))

                if (t_idx + 1) % 10 == 0:
                    logger.info("[SAHI] Tile %d/%d done", t_idx + 1, len(tiles_full))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("[SAHI] Total detections after tiling: %d", len(all_scores))

        # NMS
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
            return torch.zeros(H, W, dtype=torch.float32)

        boxes_filtered = boxes_t[indices_above]
        scores_filtered = scores_t[indices_above]

        keep_nms = nms(boxes_filtered.to(device), scores_filtered.to(device), nms_iou_threshold)
        kept_indices = indices_above[keep_nms.cpu()]
        logger.info("[NMS] iou=%.2f: %d -> %d", nms_iou_threshold, len(indices_above), len(keep_nms))

        # Area filter
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

        logger.info("[Area filter] %d -> %d", len(keep_nms), len(surviving_indices))

        if len(surviving_indices) > max_detections:
            surviving_indices.sort(key=lambda i: all_scores[i].item(), reverse=True)
            surviving_indices = surviving_indices[:max_detections]

        if not surviving_indices:
            return torch.zeros(H, W, dtype=torch.float32)

        logger.info("[Surviving] %d detections → mask merge (mode=%s)", len(surviving_indices), mask_mode)

        # Merge masks
        unified_mask = torch.zeros(H, W, dtype=torch.float32)
        has_masks = len(all_masks) == len(all_boxes) and mask_mode == "sam3_mask"

        t_merge = time.time()

        if mask_mode in ("bbox_fill", "bbox_tight"):
            shrink = 0.0 if mask_mode == "bbox_fill" else 0.1
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
            n_flooded = 0
            for idx_val in surviving_indices:
                mask_lr, tx1, ty1, tx2, ty2 = all_masks[idx_val]
                tile_h, tile_w = ty2 - ty1, tx2 - tx1

                mask_up = torch.nn.functional.interpolate(
                    mask_lr.unsqueeze(0).unsqueeze(0).float(),
                    size=(tile_h, tile_w),
                    mode="bilinear", align_corners=False,
                ).squeeze(0).squeeze(0).sigmoid()

                if mask_threshold > 0.0:
                    mask_up = (mask_up > mask_threshold).float()

                box = all_boxes[idx_val]
                bx1, by1 = int(box[0].item()), int(box[1].item())
                bx2, by2 = int(box[2].item()), int(box[3].item())
                bbox_w, bbox_h = bx2 - bx1, by2 - by1
                pad = max(3, int(max(bbox_w, bbox_h) * 0.05))

                region_x1 = max(0, bx1 - pad)
                region_y1 = max(0, by1 - pad)
                region_x2 = min(W, bx2 + pad)
                region_y2 = min(H, by2 + pad)

                lx1 = max(0, min(region_x1 - tx1, tile_w))
                ly1 = max(0, min(region_y1 - ty1, tile_h))
                lx2 = max(0, min(region_x2 - tx1, tile_w))
                ly2 = max(0, min(region_y2 - ty1, tile_h))

                if lx2 > lx1 and ly2 > ly1:
                    cropped = mask_up[ly1:ly2, lx1:lx2]
                    th = min(cropped.shape[0], region_y2 - region_y1)
                    tw = min(cropped.shape[1], region_x2 - region_x1)
                    cropped = cropped[:th, :tw]
                    ry2_safe = region_y1 + th
                    rx2_safe = region_x1 + tw

                    fill_ratio = cropped.mean().item()
                    if fill_ratio > 0.90:
                        n_flooded += 1
                        yy, xx = torch.meshgrid(
                            torch.linspace(-1, 1, th),
                            torch.linspace(-1, 1, tw),
                            indexing="ij",
                        )
                        ellipse = ((xx ** 2 + yy ** 2) <= 1.0).float()
                        unified_mask[region_y1:ry2_safe, region_x1:rx2_safe] = torch.max(
                            unified_mask[region_y1:ry2_safe, region_x1:rx2_safe], ellipse)
                    else:
                        unified_mask[region_y1:ry2_safe, region_x1:rx2_safe] = torch.max(
                            unified_mask[region_y1:ry2_safe, region_x1:rx2_safe], cropped)

            logger.info("[Merge] %d masks merged (%d flooded→ellipse) in %.2fs",
                        len(surviving_indices), n_flooded, time.time() - t_merge)
        else:
            logger.warning("[Merge] No masks — using box fill fallback")
            for idx_val in surviving_indices:
                box = all_boxes[idx_val]
                x1, y1 = int(box[0].item()), int(box[1].item())
                x2, y2 = int(box[2].item()), int(box[3].item())
                unified_mask[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = 1.0

        return unified_mask

    # -----------------------------------------------------------------
    #  Main execution
    # -----------------------------------------------------------------
    def run(
        self,
        image: torch.Tensor,
        search_mask: torch.Tensor,
        full_image_prompts: str,
        tile_prompts: str,
        sahi_tile_size: int,
        mask_mode: str = "sam3_mask",
        mask_threshold: float = 0.5,
        score_threshold: float = 0.10,
        nms_iou_threshold: float = 0.50,
        small_pct_threshold: float = 0.1,
        large_pct_threshold: float = 2.0,
        max_detections: int = 256,
        enable_zim_refinement: bool = True,
    ):
        t0 = time.time()

        min_area_pct = 0.001
        max_area_pct = 15.0
        enable_sahi = True
        sahi_overlap = 0.3

        pipe = _load_sam3_model()
        patcher = pipe["patcher"]
        processor = pipe["processor"]

        logger.info("Requesting ComfyUI to load SAM3 model to GPU...")
        mm.load_model_gpu(patcher)
        device = patcher.load_device
        processor.device = device

        # Parse prompts
        prompts = [p.strip() for p in full_image_prompts.strip().splitlines() if p.strip()]
        t_prompts = [p.strip() for p in tile_prompts.strip().splitlines() if p.strip()]
        if not prompts:
            prompts = STONE_PROMPTS
        if not t_prompts:
            t_prompts = TILE_PROMPTS

        B, H, W, C = image.shape
        logger.info("=" * 60)
        logger.info("GEMSTONE V2 START  B=%d  H=%d  W=%d  device=%s", B, H, W, device)

        all_masks_large = []
        all_masks_medium = []
        all_masks_small = []
        all_overlays = []
        total_gems = 0
        total_coverage = 0.0
        stats_lines = []

        try:
            for b_idx in range(B):
                t_batch = time.time()
                logger.info("--- Batch %d/%d ---", b_idx + 1, B)

                # Parse search mask
                search_2d = _normalize_mask_to_2d(search_mask, H, W)
                roi_bbox = _roi_from_mask(search_2d, H, W, padding_ratio=0.05)

                if roi_bbox is None:
                    logger.warning("Empty search mask — skipping batch %d", b_idx)
                    empty = torch.zeros(H, W, dtype=torch.float32)
                    all_masks_large.append(empty)
                    all_masks_medium.append(empty)
                    all_masks_small.append(empty)
                    all_overlays.append(image[b_idx].cpu().clone())
                    stats_lines.append(f"[Batch {b_idx}] Empty search mask")
                    continue

                # SAM3 detection
                with torch.inference_mode():
                    unified_mask = self._process_single_image(
                        image[b_idx], processor, device, prompts, t_prompts, H, W,
                        score_threshold, nms_iou_threshold,
                        min_area_pct, max_area_pct, max_detections,
                        enable_sahi, sahi_tile_size, sahi_overlap,
                        mask_threshold, mask_mode,
                        roi_bbox=roi_bbox,
                    )

                # Intersect with search mask
                search_np = (search_2d > 0.5).cpu().numpy().astype(np.float32)
                unified_np = unified_mask.cpu().numpy()
                unified_np *= search_np

                # ZIM refinement
                if enable_zim_refinement:
                    np_img = (image[b_idx].cpu().numpy() * 255).astype(np.uint8)
                    unified_binary = (unified_np > 0.5).astype(np.uint8)
                    unified_alpha = self._zim_refine(
                        np_img, unified_binary, H, W, search_mask_np=search_np)
                else:
                    unified_alpha = unified_np

                # Split by size
                m_large, m_medium, m_small, n_gems = self._split_by_size(
                    unified_alpha, H, W, small_pct_threshold, large_pct_threshold)

                total_gems += n_gems
                cov_pct = round(float((unified_alpha > 0.5).sum()) / (H * W) * 100, 2)
                total_coverage += cov_pct
                stats_lines.append(f"[Batch {b_idx}] Gems: {n_gems} | Coverage: {cov_pct}%")

                # Overlay: red=large, green=medium, blue=small
                img_t = image[b_idx].cpu().clone()
                alpha_vis = 0.35
                colors = {
                    "large": torch.tensor([1.0, 0.3, 0.3]),
                    "medium": torch.tensor([0.3, 1.0, 0.3]),
                    "small": torch.tensor([0.3, 0.3, 1.0]),
                }
                overlay = img_t.clone()
                for name, mask_layer in [("large", m_large), ("medium", m_medium), ("small", m_small)]:
                    m3 = mask_layer.unsqueeze(-1)
                    overlay = overlay * (1 - m3 * alpha_vis) + colors[name] * m3 * alpha_vis
                overlay = overlay.clamp(0, 1)

                all_masks_large.append(m_large)
                all_masks_medium.append(m_medium)
                all_masks_small.append(m_small)
                all_overlays.append(overlay)

                logger.info("Batch %d done: %.2fs", b_idx + 1, time.time() - t_batch)

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        overlay_batch = torch.stack(all_overlays, dim=0)
        mask_large_batch = torch.stack(all_masks_large, dim=0)
        mask_medium_batch = torch.stack(all_masks_medium, dim=0)
        mask_small_batch = torch.stack(all_masks_small, dim=0)

        avg_coverage = total_coverage / B if B > 0 else 0.0
        stats_lines.insert(0, f"=== SAM3 Gemstone V2 ({B} images) ===")
        stats_lines.append(f"Total gems: {total_gems} | Avg coverage: {avg_coverage:.1f}%")
        stats_text = "\n".join(stats_lines)

        logger.info("GEMSTONE V2 DONE in %.1fs", time.time() - t0)

        return (overlay_batch, mask_large_batch, mask_medium_batch, mask_small_batch,
                stats_text, total_gems, round(avg_coverage, 2))


# ============================================================================
#  SAM3 Boolean Switch — route by detection
# ============================================================================
class SAM3Boolean:
    """SAM3 boolean switch node.
    Runs SAM3 with a text prompt. If the object is found (score > threshold),
    image goes to found_image output; otherwise to not_found_image.
    Optional search_mask limits the search region."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "gemstone",
                    "placeholder": "What to search for",
                    "tooltip": "SAM3 text prompt. If detected with score > threshold → found=True.",
                }),
                "score_threshold": ("FLOAT", {
                    "default": 0.20, "min": 0.01, "max": 1.0, "step": 0.01,
                }),
            },
            "optional": {
                "search_mask": ("MASK", {
                    "tooltip": "Optional: white = search region, black = skip. "
                               "Limits SAM3 detection to the white area.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "BOOLEAN")
    RETURN_NAMES = ("found_image", "not_found_image", "detected_mask", "found")
    FUNCTION = "run"
    CATEGORY = "🔮 HappyIn SAM3"

    def run(self, image: torch.Tensor, prompt: str, score_threshold: float = 0.20,
            search_mask: torch.Tensor | None = None):
        t0 = time.time()

        pipe = _load_sam3_model()
        patcher = pipe["patcher"]
        processor = pipe["processor"]

        logger.info("Requesting ComfyUI to load SAM3 model to GPU...")
        mm.load_model_gpu(patcher)
        device = patcher.load_device
        processor.device = device

        B, H, W, C = image.shape
        logger.info("[Boolean] START  B=%d  H=%d  W=%d  prompt='%s'  threshold=%.2f",
                    B, H, W, prompt, score_threshold)

        found = False
        all_masks = []

        for b_idx in range(B):
            np_img = (image[b_idx].cpu().numpy() * 255).astype(np.uint8)

            # ROI from search mask if provided
            roi_bbox = None
            search_np = None
            if search_mask is not None:
                search_2d = _normalize_mask_to_2d(search_mask, H, W)
                roi_bbox = _roi_from_mask(search_2d, H, W, padding_ratio=0.05)
                search_np = (search_2d > 0.5).cpu().numpy().astype(np.float32)

            if roi_bbox is not None:
                rx1, ry1, rx2, ry2 = roi_bbox
                roi_img = np_img[ry1:ry2, rx1:rx2]
                pil_img = Image.fromarray(roi_img)
            else:
                pil_img = Image.fromarray(np_img)
                rx1, ry1 = 0, 0

            with torch.inference_mode():
                state = processor.set_image(pil_img)
                result = processor.set_text_prompt(prompt=prompt, state=state)

            boxes = result["boxes"]
            scores = result["scores"]
            masks_lr = result.get("masks_lowres")

            n = scores.shape[0] if scores.ndim > 0 else (1 if scores.numel() > 0 else 0)

            batch_mask = torch.zeros(H, W, dtype=torch.float32)

            if n > 0:
                scores_cpu = scores.cpu()
                if scores_cpu.ndim == 0:
                    scores_cpu = scores_cpu.unsqueeze(0)

                above = scores_cpu >= score_threshold
                if above.any():
                    found = True
                    # Build mask from detected regions
                    if masks_lr is not None:
                        masks_cpu = masks_lr.cpu()
                        if masks_cpu.ndim == 2:
                            masks_cpu = masks_cpu.unsqueeze(0)

                        roi_h = (ry1 if roi_bbox else 0)
                        tile_h = pil_img.size[1]
                        tile_w = pil_img.size[0]

                        for i in range(masks_cpu.shape[0]):
                            if scores_cpu[i] >= score_threshold:
                                m_up = torch.nn.functional.interpolate(
                                    masks_cpu[i].unsqueeze(0).unsqueeze(0).float(),
                                    size=(tile_h, tile_w),
                                    mode="bilinear", align_corners=False,
                                ).squeeze().sigmoid()
                                m_binary = (m_up > 0.5).float()

                                # Place mask in full image coordinates
                                if roi_bbox is not None:
                                    full_m = torch.zeros(H, W, dtype=torch.float32)
                                    rx1_, ry1_, rx2_, ry2_ = roi_bbox
                                    mh = min(m_binary.shape[0], ry2_ - ry1_)
                                    mw = min(m_binary.shape[1], rx2_ - rx1_)
                                    full_m[ry1_:ry1_ + mh, rx1_:rx1_ + mw] = m_binary[:mh, :mw]
                                    batch_mask = torch.max(batch_mask, full_m)
                                else:
                                    batch_mask = torch.max(batch_mask, m_binary)
                    else:
                        # Fallback: bbox fill
                        boxes_cpu = boxes.cpu()
                        if boxes_cpu.ndim == 1:
                            boxes_cpu = boxes_cpu.unsqueeze(0)
                        for i in range(boxes_cpu.shape[0]):
                            if scores_cpu[i] >= score_threshold:
                                bx1 = int(boxes_cpu[i][0].item()) + rx1
                                by1 = int(boxes_cpu[i][1].item()) + ry1
                                bx2 = int(boxes_cpu[i][2].item()) + rx1
                                by2 = int(boxes_cpu[i][3].item()) + ry1
                                batch_mask[max(0, by1):min(H, by2), max(0, bx1):min(W, bx2)] = 1.0

            # Intersect with search mask
            if search_np is not None:
                batch_mask *= torch.from_numpy(search_np)

            all_masks.append(batch_mask)
            processor.reset_all_prompts(state)
            del state

        mask_batch = torch.stack(all_masks, dim=0)

        if found:
            found_image = image
            not_found_image = torch.zeros_like(image)
        else:
            found_image = torch.zeros_like(image)
            not_found_image = image

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("[Boolean] DONE in %.1fs  found=%s", time.time() - t0, found)

        return (found_image, not_found_image, mask_batch, found)


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
    CATEGORY = "🔮 HappyIn SAM3"

    def crop(self, image, mask, padding, invert_mask):
        t0 = time.time()
        B, H, W, C = image.shape

        mask_2d = _normalize_mask_to_2d(mask, H, W)

        if invert_mask:
            mask_2d = 1.0 - mask_2d

        mask_binary = (mask_2d > 0.5).float()
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

            pad_top = min(padding, y_min)
            pad_bottom = min(padding, H - 1 - y_max)
            pad_left = min(padding, x_min)
            pad_right = min(padding, W - 1 - x_max)

            bbox_x = x_min - pad_left
            bbox_y = y_min - pad_top
            bbox_w = (x_max + 1 + pad_right) - bbox_x
            bbox_h = (y_max + 1 + pad_bottom) - bbox_y

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

    Two resize modes when the crop changed size:
      fit_to_bbox  — shrink/stretch crop to fit original bbox region.
      resize_canvas — scale entire original image so bbox matches crop size."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "processed_crop": ("IMAGE",),
                "bbox_data": ("GEMSTONE_BBOX",),
                "resize_mode": (["fit_to_bbox", "resize_canvas"], {
                    "default": "fit_to_bbox",
                }),
                "blend_mode": (["replace", "feather"], {"default": "replace"}),
                "feather_radius": ("INT", {
                    "default": 8, "min": 0, "max": 100, "step": 1,
                }),
            },
            "optional": {
                "blend_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result",)
    FUNCTION = "stitch"
    CATEGORY = "🔮 HappyIn SAM3"

    def stitch(self, original_image, processed_crop, bbox_data, resize_mode,
               blend_mode, feather_radius, blend_mask=None):
        t0 = time.time()
        B, H, W, C = original_image.shape
        bx = bbox_data["x"]
        by = bbox_data["y"]
        bw = bbox_data["width"]
        bh = bbox_data["height"]

        crop = processed_crop
        crop_h, crop_w = crop.shape[1], crop.shape[2]

        logger.info("[InpaintStitch] bbox=(%d,%d) %dx%d | crop=%dx%d | mode=%s/%s",
                    bx, by, bw, bh, crop_w, crop_h, resize_mode, blend_mode)

        if resize_mode == "resize_canvas" and (crop_h != bh or crop_w != bw):
            scale_y = crop_h / bh
            scale_x = crop_w / bw
            scale = max(scale_x, scale_y)
            new_H = int(round(H * scale))
            new_W = int(round(W * scale))

            logger.info("[InpaintStitch] resize_canvas: scale=%.3f, %dx%d → %dx%d",
                        scale, W, H, new_W, new_H)

            result = torch.nn.functional.interpolate(
                original_image.permute(0, 3, 1, 2).float(),
                size=(new_H, new_W), mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1)

            bx = int(round(bx * scale))
            by = int(round(by * scale))
            bw = int(round(bw * scale))
            bh = int(round(bh * scale))
            H, W = new_H, new_W

            if crop.shape[1] != bh or crop.shape[2] != bw:
                crop = torch.nn.functional.interpolate(
                    crop.permute(0, 3, 1, 2).float(),
                    size=(bh, bw), mode="bilinear", align_corners=False,
                ).permute(0, 2, 3, 1)
        else:
            if crop.shape[1] != bh or crop.shape[2] != bw:
                logger.info("[InpaintStitch] fit_to_bbox: resizing crop %dx%d → %dx%d",
                            crop_w, crop_h, bw, bh)
                crop = torch.nn.functional.interpolate(
                    crop.permute(0, 3, 1, 2).float(),
                    size=(bh, bw), mode="bilinear", align_corners=False,
                ).permute(0, 2, 3, 1)

            result = original_image.clone()

        if crop.shape[0] < B:
            crop = crop.expand(B, -1, -1, -1)

        if crop.shape[3] != C:
            if crop.shape[3] > C:
                crop = crop[:, :, :, :C]
            else:
                pad_ch = torch.ones(crop.shape[0], crop.shape[1], crop.shape[2],
                                    C - crop.shape[3],
                                    dtype=crop.dtype, device=crop.device)
                crop = torch.cat([crop, pad_ch], dim=3)

        paste_y2 = min(by + bh, H)
        paste_x2 = min(bx + bw, W)
        paste_h = paste_y2 - by
        paste_w = paste_x2 - bx

        if blend_mode == "replace":
            result[:, by:paste_y2, bx:paste_x2, :] = crop[:, :paste_h, :paste_w, :]
        else:
            if blend_mask is not None:
                if blend_mask.ndim == 3:
                    alpha = blend_mask[0]
                elif blend_mask.ndim == 4:
                    alpha = blend_mask[0, :, :, 0]
                else:
                    alpha = blend_mask
                if alpha.shape[0] != bh or alpha.shape[1] != bw:
                    alpha = torch.nn.functional.interpolate(
                        alpha.unsqueeze(0).unsqueeze(0).float(),
                        size=(bh, bw), mode="bilinear", align_corners=False,
                    ).squeeze(0).squeeze(0)
            else:
                alpha = torch.ones(bh, bw, dtype=torch.float32)
                if feather_radius > 0:
                    alpha_np = (alpha.numpy() * 255).astype(np.uint8)
                    k_size = feather_radius * 2 + 1
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                    eroded = cv2.erode(alpha_np, kernel, iterations=1)
                    blurred = cv2.GaussianBlur(eroded, (k_size, k_size), 0)
                    alpha = torch.from_numpy(blurred.astype(np.float32) / 255.0)

            alpha_3d = alpha[:paste_h, :paste_w].unsqueeze(-1)
            for b in range(B):
                orig_region = result[b, by:paste_y2, bx:paste_x2, :]
                result[b, by:paste_y2, bx:paste_x2, :] = (
                    orig_region * (1 - alpha_3d) + crop[b, :paste_h, :paste_w, :] * alpha_3d
                )

        logger.info("[InpaintStitch] Result: %dx%d  Done in %.2fs",
                    result.shape[2], result.shape[1], time.time() - t0)
        return (result,)


# ============================================================================
#  Simple Gemstone Crop — trim image to mask bounds with safe padding
# ============================================================================
class SimpleGemstoneCrop:
    """Crop image to the bounding box of the mask region.
    Trims empty edges so only the area with content remains.
    Padding is safe: if there's no room to pad, it just doesn't pad."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "padding": ("INT", {
                    "default": 16, "min": 0, "max": 500, "step": 1,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "info")
    FUNCTION = "crop"
    CATEGORY = "🔮 HappyIn SAM3"

    def crop(self, image, mask, padding):
        t0 = time.time()
        B, H, W, C = image.shape

        mask_2d = _normalize_mask_to_2d(mask, H, W)

        mask_binary = (mask_2d > 0.5).float()
        nonzero = torch.nonzero(mask_binary, as_tuple=False)

        if nonzero.shape[0] == 0:
            logger.info("[SimpleCrop] Empty mask — returning full image")
            info = f"Empty mask — no crop | {W}x{H}"
            mask_out = mask_2d.unsqueeze(0) if B == 1 else mask_2d.unsqueeze(0).expand(B, -1, -1)
            return (image, mask_out, info)

        y_min = nonzero[:, 0].min().item()
        y_max = nonzero[:, 0].max().item()
        x_min = nonzero[:, 1].min().item()
        x_max = nonzero[:, 1].max().item()

        pad_top = min(padding, y_min)
        pad_bottom = min(padding, H - 1 - y_max)
        pad_left = min(padding, x_min)
        pad_right = min(padding, W - 1 - x_max)

        cy1 = y_min - pad_top
        cy2 = y_max + 1 + pad_bottom
        cx1 = x_min - pad_left
        cx2 = x_max + 1 + pad_right

        logger.info("[SimpleCrop] Crop: %dx%d from %dx%d",
                    cx2 - cx1, cy2 - cy1, W, H)

        cropped_image = image[:, cy1:cy2, cx1:cx2, :]
        cropped_mask_2d = mask_2d[cy1:cy2, cx1:cx2]

        if B > 1:
            cropped_mask = cropped_mask_2d.unsqueeze(0).expand(B, -1, -1)
        else:
            cropped_mask = cropped_mask_2d.unsqueeze(0)

        crop_w, crop_h = cx2 - cx1, cy2 - cy1
        info = (f"Crop: {crop_w}x{crop_h} at ({cx1},{cy1}) | Orig: {W}x{H} | "
                f"Pad: T={pad_top} B={pad_bottom} L={pad_left} R={pad_right}")

        logger.info("[SimpleCrop] Done in %.2fs", time.time() - t0)
        return (cropped_image, cropped_mask, info)


# ============================================================================
#  Mask Donor — restore missing regions from donor mask
# ============================================================================
class MaskDonor:
    """Compare base mask with donor mask and restore missing regions.

    Finds connected components in the donor that are absent (or mostly absent)
    from the base mask, then adds those regions to produce a repaired mask.

    Use case: SAM3 occasionally misses a stone — the donor mask (from another
    pass or method) has that stone. This node transplants only the missing
    pieces while keeping the base mask as the primary source of truth.

    Logic per donor blob:
      overlap = (donor_blob & base_binary).sum / donor_blob.sum
      if overlap < overlap_threshold → blob is "missing" → add it from donor
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_mask": ("MASK", {
                    "tooltip": "Primary mask — everything from here is kept as-is.",
                }),
                "donor_mask": ("MASK", {
                    "tooltip": "Donor mask — missing regions are transplanted from here.",
                }),
                "overlap_threshold": ("FLOAT", {
                    "default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "If a donor blob overlaps base by LESS than this ratio, "
                               "it is considered 'missing' and added. "
                               "0.0 = add only completely absent blobs; "
                               "0.5 = add blobs that are less than 50% covered.",
                }),
                "min_blob_area": ("INT", {
                    "default": 100, "min": 1, "max": 100000, "step": 10,
                    "tooltip": "Ignore donor blobs smaller than this (in pixels). "
                               "Prevents noise from being transplanted.",
                }),
            },
            "optional": {
                "search_mask": ("MASK", {
                    "tooltip": "Optional: limit donor transplant to white region only.",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK", "INT", "STRING")
    RETURN_NAMES = ("repaired_mask", "added_regions", "base_only", "added_count", "info")
    FUNCTION = "run"
    CATEGORY = "🔮 HappyIn SAM3"

    def run(self, base_mask: torch.Tensor, donor_mask: torch.Tensor,
            overlap_threshold: float = 0.30, min_blob_area: int = 100,
            search_mask: torch.Tensor | None = None):
        t0 = time.time()

        # Handle batch dimension — use first frame
        if base_mask.ndim == 3:
            B = base_mask.shape[0]
        elif base_mask.ndim == 2:
            B = 1
        else:
            B = base_mask.shape[0]

        # We'll process per-batch
        all_repaired = []
        all_added = []
        all_base_only = []
        total_added = 0
        info_lines = []

        for b_idx in range(B):
            # Extract 2D masks
            if base_mask.ndim == 2:
                base_2d = base_mask.float()
            elif base_mask.ndim == 3:
                base_2d = base_mask[b_idx].float()
            else:
                base_2d = base_mask[b_idx, :, :, 0].float() if base_mask.ndim == 4 else base_mask[0].float()

            if donor_mask.ndim == 2:
                donor_2d = donor_mask.float()
            elif donor_mask.ndim == 3:
                donor_2d = donor_mask[min(b_idx, donor_mask.shape[0] - 1)].float()
            else:
                donor_2d = donor_mask[min(b_idx, donor_mask.shape[0] - 1), :, :, 0].float() if donor_mask.ndim == 4 else donor_mask[0].float()

            H, W = base_2d.shape

            # Resize donor if needed
            if donor_2d.shape[0] != H or donor_2d.shape[1] != W:
                donor_2d = torch.nn.functional.interpolate(
                    donor_2d.unsqueeze(0).unsqueeze(0),
                    size=(H, W), mode="nearest",
                ).squeeze(0).squeeze(0)

            # Optional search mask
            search_np = None
            if search_mask is not None:
                s2d = _normalize_mask_to_2d(search_mask, H, W)
                search_np = (s2d > 0.5).cpu().numpy().astype(np.uint8)

            base_np = base_2d.cpu().numpy()
            donor_np = donor_2d.cpu().numpy()

            base_binary = (base_np > 0.5).astype(np.uint8)
            donor_binary = (donor_np > 0.5).astype(np.uint8)

            # Apply search mask to donor
            if search_np is not None:
                donor_binary = donor_binary & search_np

            # Find connected components in donor
            num_labels, donor_labels = cv2.connectedComponents(donor_binary, connectivity=8)

            added_mask = np.zeros((H, W), dtype=np.float32)
            n_added = 0
            n_skipped_small = 0
            n_skipped_overlap = 0

            for label_id in range(1, num_labels):
                blob = (donor_labels == label_id)
                blob_area = blob.sum()

                if blob_area < min_blob_area:
                    n_skipped_small += 1
                    continue

                # Compute overlap with base
                overlap = (blob & (base_binary > 0)).sum()
                overlap_ratio = overlap / blob_area if blob_area > 0 else 0.0

                if overlap_ratio < overlap_threshold:
                    # This blob is missing from base → transplant it
                    # Use donor's actual values (not just binary) for soft edges
                    blob_float = blob.astype(np.float32)
                    donor_values = donor_np * blob_float
                    added_mask = np.maximum(added_mask, donor_values)
                    n_added += 1
                    logger.info("[MaskDonor] Blob %d: area=%d overlap=%.1f%% → ADDED",
                                label_id, blob_area, overlap_ratio * 100)
                else:
                    n_skipped_overlap += 1

            total_added += n_added

            # Build repaired mask: base + added donor regions
            repaired_np = np.maximum(base_np, added_mask)

            # Clamp to [0, 1]
            repaired_np = np.clip(repaired_np, 0.0, 1.0)
            added_mask = np.clip(added_mask, 0.0, 1.0)

            all_repaired.append(torch.from_numpy(repaired_np).float())
            all_added.append(torch.from_numpy(added_mask).float())
            all_base_only.append(base_2d.cpu())

            info_line = (f"[Batch {b_idx}] Donor blobs: {num_labels - 1} | "
                         f"Added: {n_added} | "
                         f"Skipped (small): {n_skipped_small} | "
                         f"Skipped (overlap): {n_skipped_overlap}")
            info_lines.append(info_line)
            logger.info("[MaskDonor] %s", info_line)

        repaired_batch = torch.stack(all_repaired, dim=0)
        added_batch = torch.stack(all_added, dim=0)
        base_only_batch = torch.stack(all_base_only, dim=0)

        info_lines.insert(0, f"=== MaskDonor ({B} images) ===")
        info_lines.append(f"Total added regions: {total_added}")
        info_text = "\n".join(info_lines)

        logger.info("[MaskDonor] DONE in %.2fs — added %d regions", time.time() - t0, total_added)

        return (repaired_batch, added_batch, base_only_batch, total_added, info_text)


# ============================================================================
#  Hole Donor — transplant holes from reference mask into base
# ============================================================================
class HoleDonor:
    """Compare base mask with reference mask and cut holes that exist in
    reference but are absent from base.

    Inverse of MaskDonor: instead of adding missing WHITE blobs, this node
    adds missing BLACK holes (gaps inside the mask).

    How it works:
      1. Invert both masks → holes become blobs.
      2. Find connected components of inverted reference (= holes in reference).
      3. For each hole-blob: check overlap with inverted base.
         If the hole is mostly ABSENT from base → cut it from base.

    Use case: reference mask has proper cutouts between stones (chain gaps,
    mounting holes) that the base mask filled in by mistake. This node
    transplants those holes into the base.

    Logic per reference hole:
      overlap = (ref_hole & base_hole).sum / ref_hole.sum
      if overlap < overlap_threshold → hole is missing in base → cut it out
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_mask": ("MASK", {
                    "tooltip": "Primary mask — holes will be cut INTO this mask.",
                }),
                "reference_mask": ("MASK", {
                    "tooltip": "Reference mask — holes are taken FROM this mask.",
                }),
                "overlap_threshold": ("FLOAT", {
                    "default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "If a reference hole overlaps base holes by LESS than "
                               "this ratio, it is considered 'missing' and cut out. "
                               "0.0 = cut only completely absent holes; "
                               "0.5 = cut holes that are less than 50% present.",
                }),
                "min_hole_area": ("INT", {
                    "default": 100, "min": 1, "max": 100000, "step": 10,
                    "tooltip": "Ignore reference holes smaller than this (pixels). "
                               "Prevents noise artifacts.",
                }),
            },
            "optional": {
                "search_mask": ("MASK", {
                    "tooltip": "Optional: only consider holes within the white region.",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK", "INT", "STRING")
    RETURN_NAMES = ("repaired_mask", "cut_holes", "base_only", "holes_count", "info")
    FUNCTION = "run"
    CATEGORY = "🔮 HappyIn SAM3"

    def run(self, base_mask: torch.Tensor, reference_mask: torch.Tensor,
            overlap_threshold: float = 0.30, min_hole_area: int = 100,
            search_mask: torch.Tensor | None = None):
        t0 = time.time()

        if base_mask.ndim == 3:
            B = base_mask.shape[0]
        elif base_mask.ndim == 2:
            B = 1
        else:
            B = base_mask.shape[0]

        all_repaired = []
        all_holes = []
        all_base_only = []
        total_holes = 0
        info_lines = []

        for b_idx in range(B):
            # Extract 2D masks
            if base_mask.ndim == 2:
                base_2d = base_mask.float()
            elif base_mask.ndim == 3:
                base_2d = base_mask[b_idx].float()
            else:
                base_2d = base_mask[b_idx, :, :, 0].float() if base_mask.ndim == 4 else base_mask[0].float()

            if reference_mask.ndim == 2:
                ref_2d = reference_mask.float()
            elif reference_mask.ndim == 3:
                ref_2d = reference_mask[min(b_idx, reference_mask.shape[0] - 1)].float()
            else:
                ref_2d = reference_mask[min(b_idx, reference_mask.shape[0] - 1), :, :, 0].float() if reference_mask.ndim == 4 else reference_mask[0].float()

            H, W = base_2d.shape

            # Resize reference if needed
            if ref_2d.shape[0] != H or ref_2d.shape[1] != W:
                ref_2d = torch.nn.functional.interpolate(
                    ref_2d.unsqueeze(0).unsqueeze(0),
                    size=(H, W), mode="nearest",
                ).squeeze(0).squeeze(0)

            # Optional search mask
            search_np = None
            if search_mask is not None:
                s2d = _normalize_mask_to_2d(search_mask, H, W)
                search_np = (s2d > 0.5).cpu().numpy().astype(np.uint8)

            base_np = base_2d.cpu().numpy()
            ref_np = ref_2d.cpu().numpy()

            base_binary = (base_np > 0.5).astype(np.uint8)
            ref_binary = (ref_np > 0.5).astype(np.uint8)

            # Invert: holes become blobs
            # We only care about holes INSIDE the mask area, so we look at
            # the union coverage to define "inside"
            union_binary = np.maximum(base_binary, ref_binary)

            # Holes in reference = white in union but black in reference
            ref_holes = union_binary & (1 - ref_binary)
            # Holes in base = white in union but black in base
            base_holes = union_binary & (1 - base_binary)

            # Apply search mask
            if search_np is not None:
                ref_holes = ref_holes & search_np

            # Find connected components of reference holes
            num_labels, hole_labels = cv2.connectedComponents(ref_holes, connectivity=8)

            cut_mask = np.zeros((H, W), dtype=np.float32)
            n_cut = 0
            n_skipped_small = 0
            n_skipped_overlap = 0

            for label_id in range(1, num_labels):
                hole_blob = (hole_labels == label_id)
                hole_area = hole_blob.sum()

                if hole_area < min_hole_area:
                    n_skipped_small += 1
                    continue

                # Check if this hole already exists in base
                overlap = (hole_blob & (base_holes > 0)).sum()
                overlap_ratio = overlap / hole_area if hole_area > 0 else 0.0

                if overlap_ratio < overlap_threshold:
                    # This hole is missing from base → cut it
                    cut_mask = np.maximum(cut_mask, hole_blob.astype(np.float32))
                    n_cut += 1
                    logger.info("[HoleDonor] Hole %d: area=%d overlap=%.1f%% → CUT",
                                label_id, hole_area, overlap_ratio * 100)
                else:
                    n_skipped_overlap += 1

            total_holes += n_cut

            # Apply: subtract holes from base
            repaired_np = base_np.copy()
            repaired_np[cut_mask > 0.5] = 0.0
            repaired_np = np.clip(repaired_np, 0.0, 1.0)
            cut_mask = np.clip(cut_mask, 0.0, 1.0)

            all_repaired.append(torch.from_numpy(repaired_np).float())
            all_holes.append(torch.from_numpy(cut_mask).float())
            all_base_only.append(base_2d.cpu())

            info_line = (f"[Batch {b_idx}] Ref holes: {num_labels - 1} | "
                         f"Cut: {n_cut} | "
                         f"Skipped (small): {n_skipped_small} | "
                         f"Skipped (overlap): {n_skipped_overlap}")
            info_lines.append(info_line)
            logger.info("[HoleDonor] %s", info_line)

        repaired_batch = torch.stack(all_repaired, dim=0)
        holes_batch = torch.stack(all_holes, dim=0)
        base_only_batch = torch.stack(all_base_only, dim=0)

        info_lines.insert(0, f"=== HoleDonor ({B} images) ===")
        info_lines.append(f"Total holes cut: {total_holes}")
        info_text = "\n".join(info_lines)

        logger.info("[HoleDonor] DONE in %.2fs — cut %d holes", time.time() - t0, total_holes)

        return (repaired_batch, holes_batch, base_only_batch, total_holes, info_text)


# ============================================================================
#  MAPPINGS
# ============================================================================
NODE_CLASS_MAPPINGS = {
    "SAM3Gemstone": SAM3GemstoneV2,
    "SAM3Boolean": SAM3Boolean,
    "MaskPositive": MaskDonor,
    "MaskNegative": HoleDonor,
    "GemstoneInpaintCrop": GemstoneInpaintCrop,
    "GemstoneInpaintStitch": GemstoneInpaintStitch,
    "SimpleGemstoneCrop": SimpleGemstoneCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3Gemstone": "🔮 HappyIn SAM3 Gemstone",
    "SAM3Boolean": "🔮 HappyIn SAM3 Boolean Switch",
    "MaskPositive": "🔮 HappyIn Mask Donor Positive (Add Missing)",
    "MaskNegative": "🔮 HappyIn Mask Donor Negative (Cut Holes)",
    "GemstoneInpaintCrop": "🔮 HappyIn SAM3 Inpaint Crop",
    "GemstoneInpaintStitch": "🔮 HappyIn SAM3 Inpaint Stitch",
    "SimpleGemstoneCrop": "🔮 HappyIn SAM3 Simple Crop",
}
