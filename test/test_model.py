"""
Test suite for TRM — Sudoku and Maze tasks.
"""

import torch
import numpy as np
from model import create_trm_model, SudokuHead, MazeHead


def get_test_device() -> str:
    """Use MPS if available, otherwise CPU (for CI/test environments)."""
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def test_model_creation():
    print("Testing model creation...")
    for task, extra in [('sudoku', {}), ('maze', {'grid_size': 9})]:
        m = create_trm_model(task=task, dim=64, n_layers=2, n_heads=4,
                             n_recursions=3, n_cycles=2, n_supervision=4, **extra)
        print(f"  {task}: {sum(p.numel() for p in m.parameters()):,} params")
    print("  OK\n")


def test_sudoku_forward():
    print("Testing Sudoku forward pass...")
    model = create_trm_model(task='sudoku', dim=64, n_layers=2, n_heads=4,
                             n_recursions=3, n_cycles=2, n_supervision=4)
    B = 4
    x = torch.randint(0, 10, (B, 81))
    y = torch.randint(1, 10, (B, 81))

    model.eval()
    with torch.no_grad():
        y_hat = model(x, training=False)
    assert y_hat.shape == (B, 81, 9), f"Bad shape: {y_hat.shape}"
    print(f"  Inference: {x.shape} -> {y_hat.shape}")

    model.train()
    losses, preds, halts = model(x, y, training=True)
    assert len(losses) > 0
    print(f"  Training: {len(losses)} steps, loss={losses[0].item():.4f}")
    print("  OK\n")


def test_maze_forward():
    print("Testing Maze forward pass...")
    model = create_trm_model(task='maze', grid_size=9, dim=64, n_layers=2, n_heads=4,
                             n_recursions=3, n_cycles=2, n_supervision=4)
    B, gs = 4, 9
    x = torch.randint(0, 4, (B, gs * gs))
    y = torch.randint(0, 2, (B, gs * gs))

    model.eval()
    with torch.no_grad():
        y_hat = model(x, training=False)
    assert y_hat.shape == (B, gs * gs, 1), f"Bad shape: {y_hat.shape}"
    print(f"  Inference: {x.shape} -> {y_hat.shape}")

    model.train()
    losses, preds, halts = model(x, y, training=True)
    assert len(losses) > 0
    print(f"  Training: {len(losses)} steps, loss={losses[0].item():.4f}")
    print("  OK\n")


def test_per_step_predictions():
    print("Testing per-step predictions...")
    model = create_trm_model(task='sudoku', dim=64, n_layers=2, n_heads=4,
                             n_recursions=3, n_cycles=2, n_supervision=6)
    x = torch.randint(0, 10, (4, 81))
    model.eval()
    with torch.no_grad():
        y_hat, all_preds = model(x, training=False, return_all_steps=True)
    assert len(all_preds) == 6
    for i, p in enumerate(all_preds):
        assert p.shape == (4, 81, 9), f"Step {i}: {p.shape}"
    print(f"  {len(all_preds)} steps, each {all_preds[0].shape}")
    print("  OK\n")


def test_dynamic_seq_len():
    print("Testing dynamic sequence lengths...")
    model = create_trm_model(task='sudoku', dim=64, n_layers=2, n_heads=4,
                             n_recursions=2, n_cycles=2, n_supervision=4)
    # Sudoku always 81, but test that the model can handle variable-length inputs
    # by temporarily monkey-patching (just tests the architecture)
    for seq_len in [64, 81, 100]:
        x = torch.randint(0, 10, (2, seq_len))
        # Bypass task head encoding to test core model directly
        emb = model.task_head.embedding(x)
        y = model.y_init.expand(2, seq_len, -1)
        z = model.z_init.expand(2, seq_len, -1)
        out = model.forward_network(emb, y, z)
        assert out.shape == (2, seq_len, 64), f"seq_len={seq_len}: {out.shape}"
        print(f"  seq_len={seq_len}: core output {out.shape}")
    print("  OK\n")


def test_backward_passes():
    print("Testing backward passes...")
    for task, x_fn, y_fn, extra in [
        ('sudoku', lambda: torch.randint(0, 10, (4, 81)),
                   lambda: torch.randint(1, 10, (4, 81)), {}),
        ('maze',   lambda: torch.randint(0, 4, (4, 81)),
                   lambda: torch.randint(0, 2, (4, 81)), {'grid_size': 9}),
    ]:
        model = create_trm_model(task=task, dim=64, n_layers=2, n_heads=4,
                                 n_recursions=2, n_cycles=2, n_supervision=4, **extra)
        model.train()
        losses, _, _ = model(x_fn(), y_fn(), training=True)
        loss = torch.stack(losses).mean()
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        opt.zero_grad()
        loss.backward()
        opt.step()
        has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grads
        print(f"  {task}: loss={loss.item():.4f}  gradients OK")
    print("  OK\n")


def test_mps_forward():
    print("Testing MPS forward pass...")
    device = get_test_device()
    model = create_trm_model(task='sudoku', dim=64, n_layers=2, n_heads=4,
                             n_recursions=3, n_cycles=2, n_supervision=4).to(device)
    x = torch.randint(0, 10, (4, 81)).to(device)
    model.eval()
    with torch.no_grad():
        y_hat = model(x, training=False)
    assert y_hat.shape == (4, 81, 9)
    print(f"  Device: {device}  output: {y_hat.shape}")
    print("  OK\n")


def test_data_generators():
    print("Testing data generators...")
    from data.sudoku import SudokuDataset
    from data.maze import MazeDataset
    from data import get_dataset

    ds = SudokuDataset(8, min_givens=25, max_givens=35, seed=42, augment=False)
    p, s = ds[0]
    assert p.shape == (81,) and s.shape == (81,)
    assert (s >= 1).all() and (s <= 9).all()
    print(f"  Sudoku: puzzle {p.shape}, solution range [{s.min()},{s.max()}]")

    ds = MazeDataset(8, grid_size=9, seed=42, augment=False)
    m, t = ds[0]
    assert m.shape == (81,) and set(t.unique().tolist()).issubset({0, 1})
    print(f"  Maze: maze {m.shape}, target values {t.unique().tolist()}")

    for task in ['sudoku', 'maze']:
        ds = get_dataset(task, 'train', seed=42, num_samples=8)
        assert len(ds) == 8
        print(f"  get_dataset('{task}'): {len(ds)} samples")
    print("  OK\n")


def run_all_tests():
    print("=" * 55)
    print("TRM Test Suite")
    print("=" * 55 + "\n")
    try:
        test_model_creation()
        test_sudoku_forward()
        test_maze_forward()
        test_per_step_predictions()
        test_dynamic_seq_len()
        test_backward_passes()
        test_mps_forward()
        test_data_generators()
        print("=" * 55)
        print("All tests passed!")
        print("=" * 55)
        return True
    except Exception as e:
        import traceback
        print(f"\nFAILED: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    torch.manual_seed(42)
    sys.exit(0 if run_all_tests() else 1)
