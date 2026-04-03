"""
Device utilities for TRM on Apple Silicon (MPS).
"""

import torch


def get_device() -> str:
    """Return 'mps' if available, else raise — MPS is required on macOS.

    Requires macOS 12.3+ and PyTorch >= 2.0.
    """
    if torch.backends.mps.is_available():
        return 'mps'
    raise RuntimeError(
        "MPS (Metal Performance Shaders) not available.\n"
        "Requirements:\n"
        "  - macOS 12.3 or later\n"
        "  - PyTorch >= 2.0  (pip3 install torch)\n"
        "  - Apple Silicon or AMD GPU\n"
        "Check: python3 -c \"import torch; print(torch.backends.mps.is_available())\""
    )


def print_device_info() -> None:
    """Print MPS device information."""
    device = get_device()
    print("=" * 50)
    print("MPS (Metal) GPU READY")
    print("=" * 50)
    print(f"PyTorch version : {torch.__version__}")
    print(f"Device          : {device}")
    print(f"MPS available   : {torch.backends.mps.is_available()}")
    print(f"MPS built       : {torch.backends.mps.is_built()}")
    print("=" * 50)
