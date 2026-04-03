"""
Training script for Tiny Recursive Model (TRM) — macOS / MPS

Usage:
    python3 train.py --task sudoku
    python3 train.py --task sudoku --n_recursions 12 --dim 256
    python3 train.py --task maze --grid_size 11
"""

from evaluate import evaluate, evaluate_per_step
from device import get_device, print_device_info
from model import create_trm_model
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import sys
import os
import json
import argparse
import time

# Resolve paths: src/ siblings + repo root (for data/)
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC_DIR)
for _p in (_SRC_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class EMA:
    """Exponential Moving Average of model parameters"""

    def __init__(self, model, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {n: p.data.clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = self.decay * \
                    self.shadow[n] + (1 - self.decay) * p.data

    def apply_shadow(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data = self.shadow[n]

    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.data = self.backup[n]
        self.backup = {}


class TRMTrainer:
    """Trainer for TRM models"""

    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        device: str = 'mps',
        save_dir: str = 'checkpoints',
        task: str = 'sudoku',
    ):
        if not torch.backends.mps.is_available() and device == 'mps':
            raise RuntimeError(
                "MPS not available. See device.py for requirements.")

        self.device = device
        self.task = task
        print(f"\nMoving model to {device.upper()}...")
        self.model = model.to(device)
        print(
            f"Model on {device.upper()} — {sum(p.numel() for p in model.parameters()):,} params\n")

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = save_dir
        self.max_steps = max_steps

        self.optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                               weight_decay=weight_decay)
        self.warmup_steps = warmup_steps
        self.base_lr = lr

        self.use_ema = use_ema
        if use_ema:
            self.ema = EMA(model, decay=ema_decay)

        self.step = 0
        self.best_val_acc = 0.0
        os.makedirs(save_dir, exist_ok=True)

    def get_lr(self) -> float:
        if self.step < self.warmup_steps:
            return self.base_lr * self.step / max(self.warmup_steps, 1)
        return self.base_lr

    def _set_lr(self):
        for pg in self.optimizer.param_groups:
            pg['lr'] = self.get_lr()

    def train_step(self, batch) -> dict:
        self.model.train()
        x_input, y_true = [t.to(self.device) for t in batch]

        losses, predictions, halts = self.model(x_input, y_true, training=True)
        loss = torch.stack(losses).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self._set_lr()

        if self.use_ema:
            self.ema.update()

        # Performance optimization: Only check accuracy every 10 steps
        acc = 0.0
        if self.step % 10 == 0:
            with torch.no_grad():
                acc = self.model.task_head.check_correct(
                    predictions[-1], y_true, x_input).mean().item()

        self.step += 1
        return {
            'loss': loss.item(),
            'accuracy': acc,
            'num_steps': len(predictions),
            'avg_halt': torch.stack(halts).mean().item(),
        }

    @torch.no_grad()
    def evaluate(self, loader) -> float:
        self.model.eval()
        if self.use_ema:
            self.ema.apply_shadow()

        total_correct = total = 0
        for batch in tqdm(loader, desc='Evaluating', leave=False):
            x_input, y_true = [t.to(self.device) for t in batch]
            y_hat = self.model(x_input, training=False)
            total_correct += self.model.task_head.check_correct(
                y_hat, y_true, x_input).sum().item()
            total += x_input.size(0)

        if self.use_ema:
            self.ema.restore()
        return total_correct / max(total, 1)

    def train(self, num_epochs: int):
        print(f"Task: {self.task}  |  Device: {self.device}")

        for epoch in range(num_epochs):
            metrics = {'loss': [], 'accuracy': [],
                       'num_steps': [], 'avg_halt': []}
            pbar = tqdm(self.train_loader,
                        desc=f'Epoch {epoch+1}/{num_epochs}')

            for batch in pbar:
                m = self.train_step(batch)
                for k, v in m.items():
                    metrics[k].append(v)
                pbar.set_postfix(loss=f"{m['loss']:.4f}", acc=f"{m['accuracy']:.3f}",
                                 steps=m['num_steps'], lr=f"{self.get_lr():.2e}")
                if self.step >= self.max_steps:
                    break

            avg = {k: sum(v) / len(v) for k, v in metrics.items()}
            print(f"\nEpoch {epoch+1} — loss: {avg['loss']:.4f}  acc: {avg['accuracy']:.3f}"
                  f"  steps: {avg['num_steps']:.1f}  halt: {avg['avg_halt']:.3f}")

            if self.val_loader is not None:
                val_acc = self.evaluate(self.val_loader)
                print(f"  Val accuracy: {val_acc:.3f}")
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.save_checkpoint('best_model.pt')
                    print(f"  New best: {val_acc:.3f}")

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')

            if self.step >= self.max_steps:
                print(f"Reached max steps ({self.max_steps})")
                break

    def save_checkpoint(self, filename: str):
        path = os.path.join(self.save_dir, filename)
        ckpt = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
            'best_val_acc': self.best_val_acc,
        }
        if self.use_ema:
            ckpt['ema_shadow'] = self.ema.shadow
        torch.save(ckpt, path)
        print(f"  Saved: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.step = ckpt['step']
        self.best_val_acc = ckpt.get('best_val_acc', 0.0)
        if self.use_ema and 'ema_shadow' in ckpt:
            self.ema.shadow = ckpt['ema_shadow']
        print(
            f"Loaded {path} — step {self.step}, best val acc {self.best_val_acc:.3f}")


# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------

TASK_DEFAULTS = {
    'sudoku': {
        'dim': 128,
        'n_layers': 2,
        'n_heads': 8,
        'n_recursions': 8,
        'n_cycles': 3,
        'n_supervision': 14,
        'batch_size': 64,
        'lr': 3e-4,
        'weight_decay': 0.1,
        'num_epochs': 100,
        'max_steps': 100000,
        'warmup_steps': 1000,
        'num_train': 10000,  # lowering for the purpose of initial build phase
        'num_val': 5000,
        'min_givens': 17,
        'max_givens': 35,
    },
    'maze': {
        'dim': 64,
        'n_layers': 2,
        'n_heads': 8,
        'n_recursions': 6,
        'n_cycles': 2,
        'n_supervision': 10,
        'batch_size': 128,  # saturating GPU Cores
        'lr': 3e-4,
        'weight_decay': 0.1,
        'num_epochs': 100,
        'max_steps': 80000,
        'warmup_steps': 1000,
        'num_train': 10000,  # lowering for the purpose of initial build phase
        'num_val': 5000,
        'grid_size': 9,
    },
}


def parse_args():
    p = argparse.ArgumentParser(description='Train TRM (macOS MPS)')
    p.add_argument('--task', required=True, choices=['sudoku', 'maze'])

    # Architecture
    p.add_argument('--dim', type=int)
    p.add_argument('--n_layers', type=int)
    p.add_argument('--n_heads', type=int)
    p.add_argument('--n_recursions', '--num_steps', type=int)
    p.add_argument('--n_cycles', type=int)
    p.add_argument('--n_supervision', type=int)

    # Training
    p.add_argument('--batch_size', type=int)
    p.add_argument('--lr', type=float)
    p.add_argument('--weight_decay', type=float)
    p.add_argument('--num_epochs', '--epochs', type=int)
    p.add_argument('--max_steps', type=int)
    p.add_argument('--warmup_steps', type=int)
    p.add_argument('--num_train', type=int)
    p.add_argument('--num_val', type=int)
    p.add_argument('--seed', type=int, default=42)

    # Task-specific
    p.add_argument('--grid_size', type=int)
    p.add_argument('--min_givens', type=int)
    p.add_argument('--max_givens', type=int)

    p.add_argument('--save_dir', type=str)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ---- MPS check (hard fail) ----
    print_device_info()
    device = get_device()

    # ---- Merge defaults + CLI overrides ----
    cfg = TASK_DEFAULTS[args.task].copy()
    for k, v in vars(args).items():
        if v is not None and k in cfg:
            cfg[k] = v
    cfg['task'] = args.task
    cfg['device'] = device
    save_dir = args.save_dir or f'checkpoints_{args.task}'
    cfg['save_dir'] = save_dir

    print(f"\nTask: {args.task}")
    for k, v in sorted(cfg.items()):
        print(f"  {k}: {v}")

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)

    # ---- Datasets ----
    from data import get_dataset
    print("\nGenerating datasets...")
    train_ds = get_dataset(args.task, 'train', seed=args.seed,
                           num_samples=cfg['num_train'], **{
                               k: cfg[k] for k in
                               (['min_givens', 'max_givens']
                                if args.task == 'sudoku' else ['grid_size'])
                               if k in cfg
                           })
    val_ds = get_dataset(args.task, 'val', seed=args.seed,
                         num_samples=cfg['num_val'], **{
                             k: cfg[k] for k in
                             (['min_givens', 'max_givens']
                              if args.task == 'sudoku' else ['grid_size'])
                             if k in cfg
                         })
    print(f"  train: {len(train_ds)}   val: {len(val_ds)}")

    # ---- Optimized Data Loaders for M2 Max ----
    # num_workers=4 uses the P-Cores to prep data while GPU trains.
    # pin_memory=True speeds up the transfer to Unified Memory/MPS.
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # ---- Model ----
    model_kwargs = {k: cfg[k] for k in ('dim', 'n_layers', 'n_heads', 'n_recursions',
                                        'n_cycles', 'n_supervision')}
    if args.task == 'maze':
        model_kwargs['grid_size'] = cfg['grid_size']
    model = create_trm_model(task=args.task, **model_kwargs)

    # ---- Train ----
    trainer = TRMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay'],
        warmup_steps=cfg['warmup_steps'],
        max_steps=cfg['max_steps'],
        device=device,
        save_dir=save_dir,
        task=args.task,
    )
    t_start = time.time()
    trainer.train(num_epochs=cfg['num_epochs'])
    train_time = time.time() - t_start

    # ---- Post-training evaluation (Phase 1 baseline evidence) ----
    print("\n" + "=" * 60)
    print("POST-TRAINING EVALUATION")
    print("=" * 60)

    trainer.save_checkpoint('final_model.pt')

    if trainer.use_ema:
        trainer.ema.apply_shadow()

    # Per-step accuracy — this IS the failure valley curve
    print("\nPer-step accuracy (failure valley diagnostic):")
    step_acc = evaluate_per_step(trainer.model, val_loader, device)
    for s, acc in step_acc.items():
        bar = '#' * int(acc * 40)
        print(f"  Step {s:2d}: {acc:.4f} |{bar}")

    # Save per-step accuracy plot
    from viz import plot_per_step_accuracy
    results_dir = os.path.join(_REPO_ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    run_name = f'{args.task}_{time.strftime("%Y%m%d_%H%M%S")}'
    plot_path = os.path.join(results_dir, f'{run_name}_per_step.png')
    plot_per_step_accuracy(step_acc,
                           title=f'{args.task.title()} — Accuracy vs Recursive Step',
                           save_path=plot_path)

    # Detailed task metrics
    print("\nDetailed evaluation:")
    detailed = evaluate(trainer.model, val_loader, args.task, device,
                        grid_size=cfg.get('grid_size', 9))
    for k, v in detailed.items():
        print(f"  {k}: {v:.4f}")

    if trainer.use_ema:
        trainer.ema.restore()

    # ---- Save experiment record to results/ (git-tracked) ----
    experiment = {
        'config': cfg,
        'training': {
            'total_steps': trainer.step,
            'best_val_acc': trainer.best_val_acc,
            'training_time_sec': round(train_time, 1),
        },
        'per_step_accuracy': {str(k): round(v, 6) for k, v in step_acc.items()},
        'evaluation': {k: round(v, 6) for k, v in detailed.items()},
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    results_path = os.path.join(results_dir, f'{run_name}_results.json')
    with open(results_path, 'w') as f:
        json.dump(experiment, f, indent=2)
    print(f"\nExperiment results saved to {results_path}")
    print(f"Plot saved to {plot_path}")
