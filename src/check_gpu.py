"""Check MPS (Metal GPU) availability on macOS."""
from device import print_device_info, get_device
import sys
import os

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


print_device_info()
device = get_device()
print(f"\nReady to train. Run:\n  python3 train.py --task sudoku\n  python3 train.py --task maze")
