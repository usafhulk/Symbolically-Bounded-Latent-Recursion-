"""
Visualization utilities for TRM experiments.

Provides formatted console output and matplotlib plots for
Sudoku, Maze, and per-step accuracy analysis.
"""

import numpy as np
from typing import Dict, List, Optional

try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Sudoku visualization
# ---------------------------------------------------------------------------

def print_sudoku(
    puzzle: np.ndarray,
    prediction: np.ndarray,
    solution: Optional[np.ndarray] = None,
) -> str:
    """Format a Sudoku grid showing given cells, correct/wrong predictions.

    Args:
        puzzle: 9x9 array (0 = empty).
        prediction: 9x9 array of predicted digits (1-9).
        solution: 9x9 array of true digits (1-9). If None, no color marking.

    Returns:
        Formatted string representation.
    """
    # ANSI colors
    GREEN = '\033[92m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

    puzzle = puzzle.reshape(9, 9)
    prediction = prediction.reshape(9, 9)
    if solution is not None:
        solution = solution.reshape(9, 9)

    lines = []
    lines.append('+-------+-------+-------+')
    for i in range(9):
        if i > 0 and i % 3 == 0:
            lines.append('+-------+-------+-------+')
        row_parts = []
        for j in range(9):
            if j % 3 == 0:
                row_parts.append('|')

            if puzzle[i, j] != 0:
                # Given cell (blue)
                row_parts.append(f' {BLUE}{puzzle[i, j]}{RESET}')
            else:
                val = prediction[i, j]
                if solution is not None:
                    if val == solution[i, j]:
                        row_parts.append(f' {GREEN}{val}{RESET}')
                    else:
                        row_parts.append(f' {RED}{val}{RESET}')
                else:
                    row_parts.append(f' {val}')

        row_parts.append('|')
        lines.append(' '.join(row_parts))
    lines.append('+-------+-------+-------+')

    result = '\n'.join(lines)
    print(result)
    return result


# ---------------------------------------------------------------------------
# Maze visualization
# ---------------------------------------------------------------------------

_MAZE_CHARS = {
    0: '\u2588\u2588',  # wall (full block)
    1: '  ',            # open
    2: 'SS',            # start
    3: 'GG',            # goal
}


def print_maze(
    maze: np.ndarray,
    pred_path: Optional[np.ndarray] = None,
    true_path: Optional[np.ndarray] = None,
) -> str:
    """Render a maze as colored ASCII art.

    Args:
        maze: H x W array (0=wall, 1=open, 2=start, 3=goal).
        pred_path: H x W binary array of predicted path.
        true_path: H x W binary array of true path.

    Returns:
        Formatted string.
    """
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    RESET = '\033[0m'

    H, W = maze.shape
    lines = []

    for r in range(H):
        row = []
        for c in range(W):
            cell = maze[r, c]
            is_pred = pred_path is not None and pred_path[r, c] == 1
            is_true = true_path is not None and true_path[r, c] == 1

            if cell == 0:
                row.append(_MAZE_CHARS[0])
            elif cell == 2:
                row.append(f'{CYAN}SS{RESET}')
            elif cell == 3:
                row.append(f'{CYAN}GG{RESET}')
            elif is_pred and is_true:
                row.append(f'{GREEN}**{RESET}')  # correct path
            elif is_pred and not is_true:
                row.append(f'{RED}xx{RESET}')     # false positive
            elif not is_pred and is_true:
                row.append(f'{YELLOW}..{RESET}')  # missed path
            else:
                row.append(_MAZE_CHARS[1])

        lines.append(''.join(row))

    result = '\n'.join(lines)
    print(result)
    return result


# ---------------------------------------------------------------------------
# Per-step accuracy plot
# ---------------------------------------------------------------------------

def plot_per_step_accuracy(
    step_accuracies: Dict[int, float],
    title: str = 'Accuracy vs Recursive Step',
    save_path: Optional[str] = None,
) -> None:
    """Plot accuracy at each recursive supervision step.

    Args:
        step_accuracies: Dict mapping step index to accuracy.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    if not HAS_MPL:
        print("matplotlib not installed; printing table instead.")
        for step, acc in sorted(step_accuracies.items()):
            bar = '#' * int(acc * 50)
            print(f"  Step {step:2d}: {acc:.4f} |{bar}")
        return

    steps = sorted(step_accuracies.keys())
    accs = [step_accuracies[s] for s in steps]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, accs, 'o-', color='#2196F3', linewidth=2, markersize=6)
    ax.set_xlabel('Recursive Step')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_multi_task_per_step(
    task_results: Dict[str, Dict[int, float]],
    save_path: Optional[str] = None,
) -> None:
    """Plot per-step accuracy for multiple tasks on one chart.

    Args:
        task_results: Dict mapping task name to step_accuracies dict.
        save_path: If provided, save figure.
    """
    if not HAS_MPL:
        for task, accs in task_results.items():
            print(f"\n--- {task} ---")
            for step, acc in sorted(accs.items()):
                bar = '#' * int(acc * 50)
                print(f"  Step {step:2d}: {acc:.4f} |{bar}")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {'sudoku': '#2196F3', 'maze': '#FF9800'}

    for task, accs in task_results.items():
        steps = sorted(accs.keys())
        vals = [accs[s] for s in steps]
        ax.plot(steps, vals, 'o-', label=task,
                color=colors.get(task, None), linewidth=2, markersize=5)

    ax.set_xlabel('Recursive Step')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Recursive Step (All Tasks)')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)
