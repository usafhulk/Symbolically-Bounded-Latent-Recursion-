# TRM Tag Guide

## Convention

```
exp/<task>/<run-id>
```

Examples: `exp/sudoku/r001`, `exp/maze/r003`, `exp/trm/baseline`

---

## Create a Tag

```bash
git tag -a exp/sudoku/r001 -m "
Task: sudoku
  batch_size: 64
  device: mps
  dim: 128
  lr: 0.0003
  max_givens: 35
  max_steps: 100000
  min_givens: 17
  n_cycles: 3
  n_heads: 8
  n_layers: 2
  n_recursions: 4
  n_supervision: 7
  num_epochs: 10
  num_train: 10000
  num_val: 5000
  save_dir: checkpoints_sudoku
  task: sudoku
  warmup_steps: 1000
  weight_decay: 0.1
Generating datasets...
  train: 10000   val: 5000
Moving model to MPS...
Model on MPS — 530,048 params
val_accuracy: 0.000
notes: baseline architecture verification
"
```

## Push a Tag

```bash
git push origin exp/sudoku/r001
```

## List Tags

```bash
git tag -l "exp/sudoku/*"   # filter by task
git tag -l "exp/*"          # all experiments
```

## Inspect a Tag

```bash
git show exp/sudoku/r001
```

## Where You Are Relative to Last Tag

```bash
git describe --tags
# → exp/sudoku/r001-12-gabcdef  (12 commits past tag, on commit abcdef)
```

---

## Run ID Convention

| ID | Meaning |
|----|---------|
| `baseline` | pre-training reference point |
| `r001`, `r002` … | sequential experiment runs |
| `r001-ablation-halting` | named variant of a run |

---

## Experiment Log

Keep `EXPERIMENT_LOG.md` updated alongside tags.

| Tag | Date | Task | Key Change | Val Acc | Notes |
|-----|------|------|------------|---------|-------|
| exp/trm/baseline | 2026-03-14 | sudoku+maze | initial architecture | — | pre-training startblock |
