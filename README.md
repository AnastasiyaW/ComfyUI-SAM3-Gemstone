# ComfyUI-SAM3-Gemstone

Gemstone segmentation nodes for ComfyUI powered by **SAM3** (Segment Anything Model 3) and **ZIM** (Zero-Shot Image Matting).

Designed for jewelry photography — detects and segments individual gemstones with pixel-perfect alpha edges, even stones as small as 10px.

## How It Works

The pipeline combines two models in sequence:

```
Input image (1500–4000px)
       │
       ├──► [SAM3 full-frame] 22 text prompts (gem types + cut shapes)
       │         catches medium/large stones
       │
       ├──► [SAM3 SAHI tiles] 384px tiles, overlap 0.3, 2 prompts
       │         catches small stones (10–30px) that full-frame misses
       │
       ├──► NMS dedup (IoU 0.5) + quality filters (area, solidity)
       │         removes duplicates and false positives
       │
       └──► [ZIM per-detection crop] box + point prompts from SAM3 mask
                 pixel-perfect sigmoid alpha [0,1] edges
                 ▼
           Final mask (float, not binary — smooth semi-transparent edges)
```

### Why two passes?

SAM3 internally resizes all input to **1008x1008**. On a 4000px image, a 10px gemstone becomes ~2.5px — invisible. SAHI tiles (384px) zoom in so small stones occupy enough pixels for SAM3 to detect them.

### Why ZIM after SAM3?

SAM3 produces decent masks but with jagged binary edges. ZIM generates **sigmoid alpha mattes** [0, 1] with pixel-perfect smooth edges. Each detected stone is cropped and fed to ZIM individually (with centroid and background points from the SAM3 mask), so even on large images every stone gets maximum resolution.

## Nodes

### SAM3 Gemstone

Main detection + segmentation node.

| Input | Type | Description |
|---|---|---|
| `image` | IMAGE | Input jewelry photo |
| `full_image_prompts` | STRING (multiline) | Text prompts for full-frame pass, one per line. Default: 22 prompts covering gem types (emerald, ruby, sapphire...), cut shapes (marquise, pear, oval...), and generic fallbacks |
| `tile_prompts` | STRING (multiline) | Prompts for SAHI tile pass, one per line. Default: `diamond`, `gemstone` (smaller set for speed) |
| `sahi_tile_size` | INT | Tile size in pixels. Default: 384. Smaller = better for tiny stones, but slower |
| `use_zim_refinement` | BOOLEAN | Refine edges with ZIM. Default: true |

| Output | Type | Description |
|---|---|---|
| `overlay` | IMAGE | Original image with green overlay on detected gems |
| `mask` | MASK | Combined mask of all detected gems (float [0,1] with ZIM, binary without) |
| `cropped` | IMAGE | Gems on white background |
| `stats_text` | STRING | Detection statistics |
| `gem_count` | INT | Number of detected gems |
| `coverage_pct` | FLOAT | Percentage of image covered by gems |

**Detection pipeline internals:**

1. **Full-frame pass** — all prompts run on the full image (catches medium/large stones)
2. **SAHI tile pass** — tile prompts run on overlapping tiles (catches small stones)
3. **Score filter** — discard detections with confidence < 0.10
4. **NMS** — remove duplicate detections (IoU threshold 0.50)
5. **Area filter** — discard masks with area < 0.001% or > 15% of image
6. **Solidity filter** — discard non-convex shapes (solidity < 0.30)
7. **Top-K** — keep max 128 highest-scoring detections
8. **ZIM refinement** — each surviving detection is cropped, SAM3 mask provides centroid (foreground) + 4 background points outside bbox, ZIM produces pixel-perfect alpha
9. **Merge** — max-union of all refined masks preserving alpha values

### ZIM Refine Mask

Standalone edge refinement node. Takes any rough mask and refines edges with ZIM.

| Input | Type | Description |
|---|---|---|
| `image` | IMAGE | Original image |
| `mask` | MASK | Rough mask to refine |
| `chain_mode` | BOOLEAN | Enable for chains/small links (uses more permissive settings) |

| Output | Type | Description |
|---|---|---|
| `refined_mask` | MASK | Refined mask with pixel-perfect alpha edges |
| `overlay` | IMAGE | Green overlay visualization |

**Presets:**

| Parameter | Normal | Chain Mode |
|---|---|---|
| min_object_area | 50 px | 5 px |
| crop_padding | 30 px | 15 px |
| band_outer | 10 px | 6 px |
| zim_min_coverage | 10% | 5% |

Every object (small and large) is individually cropped before feeding to ZIM for maximum resolution.

### Gemstone Inpaint Crop

Crops image and mask around the mask bounding box with padding. Useful for inpainting workflows — crop the gem area, inpaint, then stitch back.

| Input | Type | Description |
|---|---|---|
| `image` | IMAGE | Full image |
| `mask` | MASK | Gem mask |
| `padding` | INT | Extra pixels around bbox. Default: 32 |
| `invert_mask` | BOOLEAN | Flip mask before cropping |

| Output | Type | Description |
|---|---|---|
| `cropped_image` | IMAGE | Cropped region |
| `cropped_mask` | MASK | Cropped mask |
| `masked_composite` | IMAGE | Masked region only |
| `bbox_data` | GEMSTONE_BBOX | Coordinates for stitching back |
| `info` | STRING | Crop dimensions info |

### Gemstone Inpaint Stitch

Pastes a processed crop back into the original image using bbox_data from Gemstone Inpaint Crop.

| Input | Type | Description |
|---|---|---|
| `original_image` | IMAGE | Full original image |
| `processed_crop` | IMAGE | Processed (inpainted) crop |
| `bbox_data` | GEMSTONE_BBOX | From Gemstone Inpaint Crop |
| `blend_mode` | replace / feather | How to paste back |
| `feather_radius` | INT | Edge feathering (feather mode only). Default: 8 |
| `blend_mask` | MASK (optional) | Custom blend mask |

## Installation

### Models Required

**SAM3 checkpoint** (~3.3 GB):
```
ComfyUI/models/sam3/sam3.pt
```
Download from [facebookresearch/sam3](https://github.com/facebookresearch/sam3).

**ZIM model** (ONNX, ~300 MB):
```
ComfyUI/models/zim/zim_vit_b_2043/encoder.onnx
ComfyUI/models/zim/zim_vit_b_2043/decoder.onnx
```
Download from [HuggingFace naver-iv/zim-anything-vitb](https://huggingface.co/naver-iv/zim-anything-vitb).

### Dependencies

Installed automatically by ComfyUI Manager, or manually:

```bash
pip install sam3>=0.1.2 --no-deps
pip install opencv-python-headless>=4.8.0
pip install torchvision
pip install zim_anything --no-deps
```

ZIM requires `segment-anything`:
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Requirements

- **CUDA GPU** required (no CPU fallback)
- ~4 GB VRAM for SAM3 + ZIM
- Optimized for NVIDIA H100 (TF32, cuDNN benchmark auto-enabled on Ampere+)

## Technical Details

### SAM3 Monkey-Patches

Two patches are applied at load time for ComfyUI compatibility:

1. **`Sam3Image.device` setter** — SAM3's device property is read-only, but ComfyUI's ModelPatcher needs to set it during GPU offload. A setter is patched that writes to `_device`.

2. **`presence_logit_dec` bypass** — SAM3 multiplies detection scores by a "presence" logit that aggressively suppresses detections for prompts it considers unlikely. This is disabled to allow all 22 gem prompts to work without score suppression.

### ZIM Alpha Matting

ZIM's `predict()` returns sigmoid float masks in [0, 1] — these are true alpha mattes, not binary masks. The pipeline preserves this by:

- Skipping morphological closing when ZIM is active (it binarizes alpha)
- Using `torch.max` merge instead of binary OR (preserves soft edges)
- Constraining ZIM output to a dilated SAM3 zone (prevents bleeding into neighboring objects)

### Point Prompts for ZIM

For each detection, the SAM3 binary mask provides guidance to ZIM:

- **1 foreground point**: centroid of the SAM3 mask pixels
- **4 background points**: midpoints of each bbox edge, offset 3px outward

This tells ZIM exactly what to segment within the crop, producing much better results than box-only prompts.

## License

MIT
