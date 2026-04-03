"""
Evaluation metrics for TRM — Sudoku and Maze tasks.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional
from collections import defaultdict


# ---------------------------------------------------------------------------
# Sudoku
# ---------------------------------------------------------------------------

def _is_valid_sudoku(grid: np.ndarray) -> bool:
    for i in range(9):
        row = grid[i][grid[i] > 0]
        if len(row) != len(set(row)):
            return False
        col = grid[:, i][grid[:, i] > 0]
        if len(col) != len(set(col)):
            return False
    for br in range(3):
        for bc in range(3):
            box = grid[br*3:br*3+3, bc*3:bc*3+3].flatten()
            box = box[box > 0]
            if len(box) != len(set(box)):
                return False
    return True


def evaluate_sudoku(model, loader: DataLoader, device: str = 'mps') -> Dict[str, float]:
    """Evaluate on Sudoku puzzles.

    Returns cell_accuracy, puzzle_accuracy, constraint_satisfaction_rate.
    """
    model.eval()
    cell_correct = cell_total = puzzle_correct = puzzle_total = valid_total = 0

    with torch.no_grad():
        for x_input, y_true in tqdm(loader, desc='Eval Sudoku', leave=False):
            x_input, y_true = x_input.to(device), y_true.to(device)
            y_hat = model(x_input, training=False)
            pred = y_hat.argmax(dim=-1) + 1  # 0-indexed -> 1-9
            mask = (x_input == 0)

            for b in range(x_input.shape[0]):
                m = mask[b]
                n_empty = m.sum().item()
                n_correct = (pred[b][m] == y_true[b][m]).sum().item()
                cell_correct += n_correct
                cell_total += n_empty
                puzzle_correct += int(n_correct == n_empty)
                puzzle_total += 1

                inp = x_input[b].cpu().numpy().reshape(9, 9)
                p = pred[b].cpu().numpy().reshape(9, 9)
                full = np.where(inp == 0, p, inp)
                valid_total += int(_is_valid_sudoku(full))

    return {
        'cell_accuracy': cell_correct / max(cell_total, 1),
        'puzzle_accuracy': puzzle_correct / max(puzzle_total, 1),
        'constraint_satisfaction_rate': valid_total / max(puzzle_total, 1),
    }


def evaluate_sudoku_by_difficulty(
    model, loader: DataLoader, device: str = 'mps',
    buckets: Optional[list] = None,
) -> Dict[str, dict]:
    """Puzzle accuracy grouped by number of empty cells."""
    if buckets is None:
        buckets = [0, 30, 45, 55, 65]
    model.eval()
    results = defaultdict(lambda: {'correct': 0, 'total': 0})

    with torch.no_grad():
        for x_input, y_true in loader:
            x_input, y_true = x_input.to(device), y_true.to(device)
            y_hat = model(x_input, training=False)
            pred = y_hat.argmax(dim=-1) + 1
            mask = (x_input == 0)

            for b in range(x_input.shape[0]):
                m = mask[b]
                n_empty = m.sum().item()
                label = f">{buckets[-1]}"
                for i in range(len(buckets) - 1):
                    if buckets[i] <= n_empty < buckets[i+1]:
                        label = f"{buckets[i]}-{buckets[i+1]}"
                        break
                correct = int((pred[b][m] == y_true[b][m]).all().item())
                results[label]['correct'] += correct
                results[label]['total'] += 1

    return {
        label: {'puzzle_accuracy': d['correct'] / max(d['total'], 1), 'count': d['total']}
        for label, d in sorted(results.items())
    }


# ---------------------------------------------------------------------------
# Maze
# ---------------------------------------------------------------------------

def _path_connects(pred_mask: np.ndarray, maze: np.ndarray) -> bool:
    from collections import deque
    H, W = maze.shape
    start = goal = None
    for r in range(H):
        for c in range(W):
            if maze[r, c] == 2:
                start = (r, c)
            elif maze[r, c] == 3:
                goal = (r, c)
    if start is None or goal is None:
        return False
    if not pred_mask[start[0], start[1]] or not pred_mask[goal[0], goal[1]]:
        return False
    visited, queue = set(), deque([start])
    visited.add(start)
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and (nr,nc) not in visited and pred_mask[nr,nc]:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return False


def evaluate_maze(model, loader: DataLoader, device: str = 'mps',
                  grid_size: int = 9) -> Dict[str, float]:
    """Evaluate on maze solving.

    Returns path_f1, solve_rate, path_optimality.
    """
    model.eval()
    gs = grid_size if grid_size % 2 == 1 else grid_size + 1
    tp = fp = fn = solved = total = 0
    opt_sum = 0.0

    with torch.no_grad():
        for x_input, y_true in tqdm(loader, desc='Eval Maze', leave=False):
            x_input, y_true = x_input.to(device), y_true.to(device)
            y_hat = model(x_input, training=False)
            pred = (y_hat.squeeze(-1) > 0).long()

            for b in range(x_input.shape[0]):
                p = pred[b].cpu().numpy()
                t = y_true[b].cpu().numpy()
                m = x_input[b].cpu().numpy().reshape(gs, gs)
                tp += ((p == 1) & (t == 1)).sum()
                fp += ((p == 1) & (t == 0)).sum()
                fn += ((p == 0) & (t == 1)).sum()
                if _path_connects(p.reshape(gs, gs), m):
                    solved += 1
                    true_len = t.sum()
                    if true_len > 0:
                        opt_sum += p.sum() / true_len
                total += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        'path_f1': f1,
        'path_precision': precision,
        'path_recall': recall,
        'solve_rate': solved / max(total, 1),
        'path_optimality': opt_sum / max(solved, 1),
    }


# ---------------------------------------------------------------------------
# Per-step accuracy (any task)
# ---------------------------------------------------------------------------

def evaluate_per_step(model, loader: DataLoader, device: str = 'mps',
                      max_batches: int = 50) -> Dict[int, float]:
    """Accuracy at each recursive supervision step.

    Uses return_all_steps=True.
    """
    model.eval()
    step_correct: Dict[int, int] = defaultdict(int)
    step_total: Dict[int, int] = defaultdict(int)

    with torch.no_grad():
        for i, (x_input, y_true) in enumerate(loader):
            if i >= max_batches:
                break
            x_input, y_true = x_input.to(device), y_true.to(device)
            _, all_preds = model(x_input, training=False, return_all_steps=True)
            for s, y_hat in enumerate(all_preds):
                step_correct[s] += model.task_head.check_correct(
                    y_hat, y_true, x_input).sum().item()
                step_total[s] += x_input.shape[0]

    return {s: step_correct[s] / max(step_total[s], 1) for s in sorted(step_correct)}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def evaluate(model, loader: DataLoader, task: str,
             device: str = 'mps', **kwargs) -> Dict[str, float]:
    """Dispatch to task-specific evaluator."""
    if task == 'sudoku':
        return evaluate_sudoku(model, loader, device)
    elif task == 'maze':
        return evaluate_maze(model, loader, device, grid_size=kwargs.get('grid_size', 9))
    else:
        raise ValueError(f"Unknown task '{task}'.")
