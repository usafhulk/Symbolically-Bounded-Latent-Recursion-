"""
Sudoku dataset generation for TRM.

Generates random valid Sudoku puzzles by constructing a full solution via
backtracking, then removing cells to create the puzzle.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Puzzle generation
# ---------------------------------------------------------------------------

def _fill_grid(grid: np.ndarray, rng: np.random.RandomState) -> bool:
    """Fill an empty 9x9 grid with a valid Sudoku solution via backtracking."""
    for r in range(9):
        for c in range(9):
            if grid[r, c] == 0:
                nums = list(range(1, 10))
                rng.shuffle(nums)
                for n in nums:
                    if _is_valid_placement(grid, r, c, n):
                        grid[r, c] = n
                        if _fill_grid(grid, rng):
                            return True
                        grid[r, c] = 0
                return False
    return True


def _is_valid_placement(grid: np.ndarray, row: int, col: int, num: int) -> bool:
    """Check if placing *num* at (row, col) violates Sudoku constraints."""
    if num in grid[row]:
        return False
    if num in grid[:, col]:
        return False
    br, bc = 3 * (row // 3), 3 * (col // 3)
    if num in grid[br:br + 3, bc:bc + 3]:
        return False
    return True


def generate_puzzle(
    num_givens: int,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a Sudoku puzzle/solution pair.

    Args:
        num_givens: Number of revealed cells (17-80).
        rng:        Random state for reproducibility.

    Returns:
        (puzzle, solution) — each is a 9×9 ``np.ndarray`` of dtype ``int64``
        with values 1-9 for the solution and 0 for blanks in the puzzle.
    """
    if rng is None:
        rng = np.random.RandomState()
    grid = np.zeros((9, 9), dtype=np.int64)
    _fill_grid(grid, rng)
    solution = grid.copy()

    # Remove cells
    num_remove = 81 - num_givens
    indices = list(range(81))
    rng.shuffle(indices)
    puzzle = solution.copy()
    for idx in indices[:num_remove]:
        puzzle[idx // 9, idx % 9] = 0

    return puzzle, solution


def difficulty_score(puzzle: np.ndarray) -> int:
    """Return the number of empty cells in a puzzle (flattened or 9×9)."""
    return int((puzzle == 0).sum())


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _augment_sudoku(
    puzzle: np.ndarray,
    solution: np.ndarray,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random digit-relabelling augmentation."""
    perm = np.arange(10)  # 0 stays 0
    perm[1:] = rng.permutation(np.arange(1, 10))
    aug_puzzle = perm[puzzle]
    aug_solution = perm[solution]
    return aug_puzzle, aug_solution


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SudokuDataset(Dataset):
    """PyTorch Dataset of random Sudoku puzzles.

    Each sample is ``(puzzle, solution)`` — both ``torch.long`` tensors of
    shape ``(81,)`` with values in ``0-9`` (puzzle) and ``1-9`` (solution).

    Args:
        num_samples: Number of puzzles to generate.
        min_givens:  Minimum number of revealed cells (default 17).
        max_givens:  Maximum number of revealed cells (default 35).
        seed:        Random seed for reproducibility.
        augment:     Apply digit-relabelling augmentation.
    """

    def __init__(
        self,
        num_samples: int,
        min_givens: int = 17,
        max_givens: int = 35,
        seed: int = 42,
        augment: bool = True,
    ):
        self.num_samples = num_samples
        self.min_givens = min_givens
        self.max_givens = max_givens
        self.augment = augment
        self.rng = np.random.RandomState(seed)

        self.puzzles = []
        self.solutions = []
        for _ in range(num_samples):
            givens = self.rng.randint(min_givens, max_givens + 1)
            puzzle, solution = generate_puzzle(givens, self.rng)
            self.puzzles.append(puzzle.flatten())
            self.solutions.append(solution.flatten())

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        puzzle = self.puzzles[idx].copy()
        solution = self.solutions[idx].copy()

        if self.augment:
            puzzle, solution = _augment_sudoku(
                puzzle.reshape(9, 9), solution.reshape(9, 9), np.random,
            )
            puzzle = puzzle.flatten()
            solution = solution.flatten()

        return (
            torch.tensor(puzzle, dtype=torch.long),
            torch.tensor(solution, dtype=torch.long),
        )
