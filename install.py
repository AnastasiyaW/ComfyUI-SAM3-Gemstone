"""Auto-install script executed by ComfyUI Manager."""

import subprocess
import sys
import os


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
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("[SAM3-Gemstone] Installing huggingface_hub...")
        pip_install("huggingface_hub>=0.20.0")

    print("[SAM3-Gemstone] Dependencies OK.")


main()
