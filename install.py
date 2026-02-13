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
        pip_install("--no-deps", "sam3>=0.1.2")

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

    print("[SAM3-Gemstone] Dependencies OK.")


main()
