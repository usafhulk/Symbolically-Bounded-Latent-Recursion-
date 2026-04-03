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


# ---------------------------------------------------------------------------
# Z-state evolution plot
# ---------------------------------------------------------------------------

def plot_z_state_evolution(
    diagnostics: dict,
    title: str = 'Z-State Evolution Over Recursive Steps',
    save_path: Optional[str] = None,
) -> None:
    """Plot z-state and y-state L2 norms across supervision steps.

    Shows whether latent representations are stable (bounded norms) or
    collapsing/exploding (diverging norms). This is the key representation
    collapse diagnostic.

    Args:
        diagnostics: Dict from model.diagnostic_forward() with keys
                     'z_norms', 'y_norms', 'z_cosines'.
        save_path: If provided, save figure to this path.
    """
    if not HAS_MPL:
        print("matplotlib not installed; skipping z-state plot.")
        return

    z_norms = diagnostics['z_norms']
    y_norms = diagnostics['y_norms']
    z_cosines = diagnostics.get('z_cosines', [])
    steps = list(range(len(z_norms)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Panel 1: Z-norm evolution (linear scale)
    axes[0].plot(steps, z_norms, 'o-', color='steelblue',
                 linewidth=2, markersize=5)
    axes[0].set_xlabel('Supervision Step')
    axes[0].set_ylabel('Mean ||z||')
    axes[0].set_title('Z-State Norm (linear)')
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Y-norm evolution
    axes[1].plot(steps, y_norms, 's-', color='coral',
                 linewidth=2, markersize=5)
    axes[1].set_xlabel('Supervision Step')
    axes[1].set_ylabel('Mean ||y||')
    axes[1].set_title('Y-State Norm (scratchpad)')
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Z cosine similarity between consecutive steps
    if z_cosines:
        cos_steps = list(range(1, len(z_cosines) + 1))
        axes[2].plot(cos_steps, z_cosines, 'D-', color='mediumseagreen',
                     linewidth=2, markersize=5)
        axes[2].axhline(y=1.0, color='gray', linestyle='--',
                        alpha=0.5, label='identical')
        axes[2].set_ylim(-0.1, 1.1)
        axes[2].legend(fontsize=8)
    axes[2].set_xlabel('Supervision Step')
    axes[2].set_ylabel('Cosine Similarity')
    axes[2].set_title('Z-State Stability (consecutive)')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Scratchpad (y-state) heatmap
# ---------------------------------------------------------------------------

def plot_scratchpad_heatmap(
    diagnostics: dict,
    title: str = 'Scratchpad (y) Activation Heatmap',
    save_path: Optional[str] = None,
) -> None:
    """Plot heatmap of mean |y| activations: supervision steps vs sequence position.

    This is the failure valley heatmap -- if activations wash out or become
    uniform across positions at later steps, the model has lost information.

    Args:
        diagnostics: Dict from model.diagnostic_forward() with 'y_heatmaps'.
        save_path: If provided, save figure to this path.
    """
    if not HAS_MPL:
        print("matplotlib not installed; skipping scratchpad heatmap.")
        return

    y_heatmaps = diagnostics['y_heatmaps']
    heatmap = np.stack(y_heatmaps)  # [n_supervision, seq_len]

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(heatmap.T, aspect='auto',
                   cmap='RdBu_r', interpolation='nearest')
    ax.set_xlabel('Supervision Step')
    ax.set_ylabel('Sequence Position (cell index)')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='Mean |y| activation')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    epoch_losses: List[float],
    epoch_accs: List[float],
    val_accs: Optional[List[float]] = None,
    title: str = 'Training Curves',
    save_path: Optional[str] = None,
) -> None:
    """Plot loss and accuracy over epochs.

    Args:
        epoch_losses: Average loss per epoch.
        epoch_accs: Average training accuracy per epoch.
        val_accs: Validation accuracy per epoch (optional).
        save_path: If provided, save figure to this path.
    """
    if not HAS_MPL:
        print("matplotlib not installed; skipping training curves.")
        return

    epochs = list(range(1, len(epoch_losses) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Panel 1: Loss
    axes[0].plot(epochs, epoch_losses, '-', color='steelblue', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Accuracy
    axes[1].plot(epochs, epoch_accs, '-', color='steelblue', linewidth=2,
                 label='Train')
    if val_accs:
        axes[1].plot(epochs[:len(val_accs)], val_accs, '-', color='coral',
                     linewidth=2, label='Val')
        axes[1].legend()
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Halt confidence curve
# ---------------------------------------------------------------------------

def plot_halt_confidence(
    diagnostics: dict,
    title: str = 'Halt Confidence Over Recursive Steps',
    save_path: Optional[str] = None,
) -> None:
    """Plot halt probability (q_hat) across supervision steps.

    Shows whether the model develops confidence over time. In healthy
    training, halt probability should increase as reasoning progresses.

    Args:
        diagnostics: Dict from model.diagnostic_forward() with 'halt_probs'.
        save_path: If provided, save figure to this path.
    """
    if not HAS_MPL:
        print("matplotlib not installed; skipping halt confidence plot.")
        return

    halt_probs = diagnostics['halt_probs']
    steps = list(range(len(halt_probs)))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, halt_probs, 'o-', color='darkorange',
            linewidth=2, markersize=6)
    ax.axhline(y=0.5, color='gray', linestyle='--',
               alpha=0.5, label='threshold')
    ax.set_xlabel('Supervision Step')
    ax.set_ylabel('Mean Halt Probability (q_hat)')
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)
