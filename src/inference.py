"""
Inference script for Tiny Recursive Model (TRM) — macOS / MPS

Usage:
    python3 inference.py sudoku
    python3 inference.py maze
    python3 inference.py per_step sudoku
"""

from evaluate import evaluate, evaluate_per_step
from device import get_device
from model import create_trm_model
import numpy as np
import json
import torch
import sys
import os

# Resolve paths: src/ siblings + repo root (for data/)
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC_DIR)
for _p in (_SRC_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class TRMInference:
    """Inference wrapper for TRM."""

    def __init__(self, checkpoint_path: str, config_path: str = None):
        self.device = get_device()

        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(checkpoint_path), 'config.json')
        with open(config_path) as f:
            self.config = json.load(f)

        self.task = self.config['task']

        kwargs = {k: self.config[k] for k in
                  ('dim', 'n_layers', 'n_heads', 'n_recursions', 'n_cycles', 'n_supervision')}
        if self.task == 'maze':
            kwargs['grid_size'] = self.config.get('grid_size', 9)

        self.model = create_trm_model(task=self.task, **kwargs).to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if 'ema_shadow' in ckpt:
            self.model.load_state_dict(
                {n: ckpt['ema_shadow'].get(n, p.data)
                 for n, p in self.model.named_parameters()}, strict=False)
            print("Loaded EMA weights")
        else:
            self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        print(f"Model loaded on {self.device} (task: {self.task})")

    @torch.no_grad()
    def predict(self, x_input: torch.Tensor, return_all_steps: bool = False):
        if x_input.dim() == 1:
            x_input = x_input.unsqueeze(0)
        x_input = x_input.to(self.device)
        return self.model(x_input, training=False, return_all_steps=return_all_steps)


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo_sudoku():
    print("=== Sudoku Demo ===\n")
    from data.sudoku import generate_puzzle, difficulty_score
    from viz.visualize import print_sudoku

    rng = np.random.RandomState(42)
    puzzle, solution = generate_puzzle(30, rng)
    print(f"Puzzle ({difficulty_score(puzzle)} empty cells):")

    flat = torch.tensor(puzzle.flatten(), dtype=torch.long)
    try:
        inf = TRMInference('checkpoints_sudoku/best_model.pt')
        y_hat = inf.predict(flat)
        pred = (y_hat.argmax(dim=-1)[0] + 1).cpu().numpy().reshape(9, 9)
        pred = np.where(puzzle == 0, pred, puzzle)
        print("\nPrediction (blue=given, green=correct, red=wrong):")
        print_sudoku(puzzle, pred, solution)
        mask = (puzzle == 0)
        n = mask.sum()
        c = (pred[mask] == solution[mask]).sum()
        print(f"\nCell accuracy: {c}/{n} ({c/n*100:.1f}%)")
    except FileNotFoundError:
        print("No checkpoint. Train first: python3 train.py --task sudoku")
        print_sudoku(puzzle, solution, solution)


def demo_maze():
    print("=== Maze Demo ===\n")
    from data.maze import generate_maze, solve_maze, make_path_target
    from viz.visualize import print_maze

    rng = np.random.RandomState(42)
    grid, start, goal = generate_maze(9, rng)
    path = solve_maze(grid, start, goal)
    true_target = make_path_target(grid, path)

    print(f"Maze (path length={len(path)}):")
    print_maze(grid, true_path=true_target)

    flat = torch.tensor(grid.flatten(), dtype=torch.long)
    try:
        inf = TRMInference('checkpoints_maze/best_model.pt')
        y_hat = inf.predict(flat)
        pred_2d = (y_hat.squeeze(-1) >
                   0).long()[0].cpu().numpy().reshape(grid.shape)
        print("\nPredicted path (green=correct, red=false pos, yellow=missed):")
        print_maze(grid, pred_path=pred_2d, true_path=true_target)
    except FileNotFoundError:
        print("No checkpoint. Train first: python3 train.py --task maze")


def demo_per_step(task: str = 'sudoku'):
    print(f"=== Per-Step Demo ({task}) ===\n")
    ckpt = f'checkpoints_{task}/best_model.pt'
    try:
        inf = TRMInference(ckpt)
        if task == 'sudoku':
            from data.sudoku import generate_puzzle
            rng = np.random.RandomState(42)
            puzzle, solution = generate_puzzle(30, rng)
            x = torch.tensor(puzzle.flatten(), dtype=torch.long)
            sol = torch.tensor(solution.flatten())
        else:
            from data.maze import generate_maze, solve_maze
            rng = np.random.RandomState(42)
            grid, start, goal = generate_maze(9, rng)
            x = torch.tensor(grid.flatten(), dtype=torch.long)

        _, all_preds = inf.predict(x, return_all_steps=True)
        print(f"Supervision steps: {len(all_preds)}")

        for i, pred in enumerate(all_preds):
            if task == 'sudoku':
                p = pred.argmax(dim=-1)[0] + 1
                mask = torch.tensor(puzzle.flatten() == 0)
                acc = (p.cpu()[mask] == sol[mask]).float().mean().item()
                print(f"  Step {i:2d}: cell accuracy = {acc:.4f}")
            else:
                p = (pred.squeeze(-1) > 0).long()[0]
                print(
                    f"  Step {i:2d}: predicted path cells = {p.sum().item()}")
    except FileNotFoundError:
        print(f"No checkpoint for {task}. Train first.")


def demo_eval(task: str = 'sudoku'):
    """Run full evaluation with per-step accuracy (failure valley diagnostic)."""
    ckpt = f'checkpoints_{task}/best_model.pt'
    try:
        inf = TRMInference(ckpt)
    except FileNotFoundError:
        print(f"No checkpoint for {task}. Train first.")
        return

    from data import get_dataset
    from torch.utils.data import DataLoader
    from viz import plot_per_step_accuracy

    ds_kwargs = {}
    if task == 'sudoku':
        ds_kwargs = {'min_givens': 17, 'max_givens': 35}
    elif task == 'maze':
        ds_kwargs = {'grid_size': inf.config.get('grid_size', 9)}

    ds = get_dataset(task, 'val', seed=42, num_samples=5000, **ds_kwargs)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    # Per-step accuracy — the failure valley curve
    print("\nPer-step accuracy (failure valley diagnostic):")
    step_acc = evaluate_per_step(inf.model, loader, inf.device)
    for s, acc in step_acc.items():
        bar = '#' * int(acc * 40)
        print(f"  Step {s:2d}: {acc:.4f} |{bar}")

    plot_per_step_accuracy(step_acc,
                           title=f'{task.title()} — Accuracy vs Recursive Step',
                           save_path=f'checkpoints_{task}/per_step_accuracy.png')

    # Detailed metrics
    print("\nDetailed evaluation:")
    results = evaluate(inf.model, loader, task, inf.device,
                       grid_size=inf.config.get('grid_size', 9))
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    import sys

    demos = {'sudoku': demo_sudoku, 'maze': demo_maze}
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == 'per_step':
            demo_per_step(sys.argv[2] if len(sys.argv) > 2 else 'sudoku')
        elif mode == 'eval':
            demo_eval(sys.argv[2] if len(sys.argv) > 2 else 'sudoku')
        elif mode in demos:
            demos[mode]()
        else:
            print(
                f"Unknown mode '{mode}'. Options: sudoku, maze, per_step [task], eval [task]")
    else:
        print("Usage: python3 inference.py [sudoku|maze|per_step|eval] [task]")
        demo_sudoku()
