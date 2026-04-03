"""
Data module for TRM — Sudoku and Maze datasets.
"""

from data.sudoku import SudokuDataset
from data.maze import MazeDataset


def get_dataset(task: str, split: str = 'train', seed: int = 42,
                num_samples: int = 10000, **kwargs):
    """Create a dataset for the given task.

    Args:
        task:        ``"sudoku"`` or ``"maze"``.
        split:       ``"train"`` or ``"val"`` (affects seed offset).
        seed:        Base random seed.
        num_samples: Number of samples to generate.
        **kwargs:    Forwarded to the dataset constructor
                     (e.g. ``min_givens``, ``max_givens``, ``grid_size``).

    Returns:
        A ``torch.utils.data.Dataset`` yielding ``(input, target)`` pairs.
    """
    seed_offset = 0 if split == 'train' else 10000
    effective_seed = seed + seed_offset

    if task == 'sudoku':
        return SudokuDataset(
            num_samples=num_samples,
            seed=effective_seed,
            augment=(split == 'train'),
            **kwargs,
        )
    elif task == 'maze':
        return MazeDataset(
            num_samples=num_samples,
            seed=effective_seed,
            augment=(split == 'train'),
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown task '{task}'. Choose 'sudoku' or 'maze'.")
