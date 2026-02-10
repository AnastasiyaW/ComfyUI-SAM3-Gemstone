"""ComfyUI-SAM3-Gemstone — SAM 3 gemstone segmentation nodes for ComfyUI.

Optimised for NVIDIA H100 (Hopper) with BF16, TF32, Flash Attention 2
through PyTorch SDPA, and torch.compile(max-autotune).
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
