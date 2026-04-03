"""
Maze dataset generation for TRM.

Generates random mazes using randomised DFS (recursive back-tracker),
then solves them with BFS to produce binary path targets.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from collections import deque
from typing import Tuple, Optional, List


# ---------------------------------------------------------------------------
# Maze generation (randomised DFS / recursive back-tracker)
# ---------------------------------------------------------------------------

def generate_maze(
    grid_size: int = 9,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """Generate a random maze via randomised DFS.

    The grid uses odd-sized dimensions.  Cell encoding:
        0 = wall, 1 = open, 2 = start, 3 = goal

    Args:
        grid_size: Side length (forced odd).
        rng:       Random state for reproducibility.

    Returns:
        (grid, start, goal) where grid is ``(gs, gs)`` ``int64`` array.
    """
    if rng is None:
        rng = np.random.RandomState()
    gs = grid_size if grid_size % 2 == 1 else grid_size + 1
    maze = np.zeros((gs, gs), dtype=np.int64)  # 0 = wall

    # Carve passages using DFS from (1, 1)
    start_cell = (1, 1)
    maze[start_cell] = 1
    stack = [start_cell]
    while stack:
        r, c = stack[-1]
        neighbours = []
        for dr, dc in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
            nr, nc = r + dr, c + dc
            if 0 < nr < gs and 0 < nc < gs and maze[nr, nc] == 0:
                neighbours.append((nr, nc, r + dr // 2, c + dc // 2))
        if neighbours:
            nr, nc, wr, wc = neighbours[rng.randint(len(neighbours))]
            maze[wr, wc] = 1
            maze[nr, nc] = 1
            stack.append((nr, nc))
        else:
            stack.pop()

    # Place start and goal at opposite corners
    start = (1, 1)
    goal = (gs - 2, gs - 2)
    maze[start] = 2
    maze[goal] = 3

    return maze, start, goal


# ---------------------------------------------------------------------------
# BFS solver
# ---------------------------------------------------------------------------

def solve_maze(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """Solve a maze with BFS, returning the shortest path.

    Returns:
        List of (row, col) from start to goal, inclusive.
    """
    H, W = grid.shape
    visited = set()
    parent = {}
    queue = deque([start])
    visited.add(start)

    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            # Reconstruct path
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = parent.get(cur)
            return list(reversed(path))
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited:
                if grid[nr, nc] != 0:  # not a wall
                    visited.add((nr, nc))
                    parent[(nr, nc)] = (r, c)
                    queue.append((nr, nc))

    return [start]  # fallback: no path found


def make_path_target(
    grid: np.ndarray,
    path: List[Tuple[int, int]],
) -> np.ndarray:
    """Create a binary mask of the solution path.

    Returns:
        ``(H, W)`` ``int64`` array with 1 on path cells, 0 elsewhere.
    """
    target = np.zeros_like(grid, dtype=np.int64)
    for r, c in path:
        target[r, c] = 1
    return target


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _augment_maze(
    grid: np.ndarray,
    target: np.ndarray,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random rotation/flip augmentation."""
    k = rng.randint(4)
    grid = np.rot90(grid, k).copy()
    target = np.rot90(target, k).copy()
    if rng.rand() > 0.5:
        grid = np.fliplr(grid).copy()
        target = np.fliplr(target).copy()
    return grid, target


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MazeDataset(Dataset):
    """PyTorch Dataset of random mazes with BFS-solved path targets.

    Each sample is ``(maze_flat, target_flat)`` — both ``torch.long`` tensors
    of shape ``(grid_size**2,)``.  Maze values are ``{0,1,2,3}``; target
    values are ``{0,1}``.

    Args:
        num_samples: Number of mazes to generate.
        grid_size:   Side length (forced odd).
        seed:        Random seed.
        augment:     Apply rotation/flip augmentation.
    """

    def __init__(
        self,
        num_samples: int,
        grid_size: int = 9,
        seed: int = 42,
        augment: bool = True,
    ):
        self.num_samples = num_samples
        self.grid_size = grid_size if grid_size % 2 == 1 else grid_size + 1
        self.augment = augment
        self.rng = np.random.RandomState(seed)

        self.grids = []
        self.targets = []
        for _ in range(num_samples):
            grid, start, goal = generate_maze(self.grid_size, self.rng)
            path = solve_maze(grid, start, goal)
            target = make_path_target(grid, path)
            self.grids.append(grid)
            self.targets.append(target)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        grid = self.grids[idx].copy()
        target = self.targets[idx].copy()

        if self.augment:
            grid, target = _augment_maze(grid, target, np.random)

        return (
            torch.tensor(grid.flatten(), dtype=torch.long),
            torch.tensor(target.flatten(), dtype=torch.long),
        )
