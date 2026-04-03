"""Check MPS (Metal GPU) availability on macOS."""
from device import print_device_info, get_device

print_device_info()
device = get_device()
print(f"\nReady to train. Run:\n  python3 train.py --task sudoku\n  python3 train.py --task maze")
