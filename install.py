"""Auto-install script executed by ComfyUI Manager."""

import subprocess
import sys


def pip_install(*args):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", *args],
        stdout=subprocess.DEVNULL,
    )


def main():
    try:
        import sam3  # noqa: F401
    except ImportError:
        print("[SAM3-Gemstone] Installing sam3 package (--no-deps to protect existing torch/CUDA)...")
        pip_install("--no-deps", "sam3==0.1.2")

    try:
        import cv2  # noqa: F401
    except ImportError:
        print("[SAM3-Gemstone] Installing opencv-python-headless...")
        pip_install("opencv-python-headless>=4.8.0")

    try:
        from torchvision.ops import nms  # noqa: F401
    except ImportError:
        print("[SAM3-Gemstone] torchvision not found (needed for NMS). Installing...")
        pip_install("torchvision")

    # ZIM (optional — for edge refinement)
    try:
        from zim_anything import ZimPredictor  # noqa: F401
    except ImportError:
        print("[SAM3-Gemstone] Installing zim_anything (ZIM edge refinement)...")
        # ZIM requires segment-anything from git
        try:
            from segment_anything import SamPredictor  # noqa: F401
        except ImportError:
            print("[SAM3-Gemstone] Installing segment-anything (ZIM dependency)...")
            pip_install("git+https://github.com/facebookresearch/segment-anything.git@6fdee8f2727f4506cfbbe553e23b895e27956588")
        pip_install("--no-deps", "zim_anything")

    print("[SAM3-Gemstone] Dependencies OK.")


main()
