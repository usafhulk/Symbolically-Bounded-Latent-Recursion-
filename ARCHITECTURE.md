# Architecture: Symbolically Bounded Latent Recursion

A living document tracking the architecture of this neuro-symbolic Sudoku solver — what exists, why it exists, what failed, and where it's going.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Current Architecture (as of R025)](#current-architecture-as-of-r025)
   - [Core Components](#core-components)
   - [Data Flow](#data-flow)
   - [Key Architectural Invariants](#key-architectural-invariants)
3. [Architectural Decision Log](#architectural-decision-log)
4. [Symbolic Integration: Z3](#symbolic-integration-z3)
5. [Constraint Enforcement History](#constraint-enforcement-history)
6. [Performance Ceiling Map](#performance-ceiling-map)
7. [Open Questions & Next Moves](#open-questions--next-moves)
8. [File Map](#file-map)

---

## System Overview

This is a **Tiny Recursive Model (TRM)** — a small transformer that iterates a latent state `z` over many recursion steps before emitting a prediction. The hypothesis: a small model that thinks for longer beats a big model that thinks once.

Applied to Sudoku: the TRM is integrated with a **Z3 SMT solver** that prunes logically impossible digit assignments from the model's logits, providing hard symbolic constraints at inference time (and during training via gradient-transparent masking).

**Task:** 9×9 Sudoku (primary), 4×4 Sudoku (planned return)
**Primary metric:** Constraint Satisfaction Rate (CSR) — % of predictions that satisfy all Sudoku row/col/box constraints

---

## Current Architecture (as of R025)

### Core Components

#### 1. `SudokuHead` — Input Encoder / Output Decoder
*`notebooks/trm_experiment.ipynb`, Cell 7 | `src/model.py` SudokuHead*

```
Input:  [B, 81] integers  (0 = empty, 1–9 = given clue)
Output: [B, 81, 9] logits over digits 1–9
```

- **Embedding:** `nn.Embedding(10, dim)` — maps each cell value to a `dim`-dimensional vector
- **Positional encoding:** learned 2D: `row_embed(row_idx) + col_embed(col_idx)` — separate row and column embeddings broadcast over the 81-cell sequence. **No explicit box embedding** — the model must discover box structure from attention.
- **Decoder:** single `nn.Linear(dim, 9, bias=False)`
- **Loss:** Cross-entropy on empty cells only (`x_input == 0`). Given clues are masked with `ignore_index=-100`.
- **Accuracy metric:** per-cell accuracy over empty cells (changed from whole-board accuracy during R013)

#### 2. `TinyRecursiveModel` — Core Recursion Engine
*`notebooks/trm_experiment.ipynb`, Cell 7*

**Learned state vectors:**
```python
y_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)  # scratchpad init
z_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)  # recurrence init
```
Both broadcast to `[B, seq_len, dim]` at runtime.

**Shared transformer layers:**
```python
self.layers = nn.ModuleList([TransformerLayer(dim, n_heads) for _ in range(n_layers)])
```
Weight-shared across all recursion steps. The same 2-layer transformer is applied repeatedly — compactness is the point.

**Recurrence stabilizer (added R008):**
```python
self.z_norm_layer = RMSNorm(dim)
```
Applied to `z` inside the micro-recursion loop. Prevents representation collapse (z-norms were hitting 1e17+ without this).

**Halt gate:**
```python
self.q_head = nn.Linear(dim, 1, bias=False)
```
Predicts when the model is confident enough to stop. Trained with BCE against per-step correctness. Currently: halt confidence plateaus early (0.64–0.65), not meaningfully used for early exit at inference.

#### 3. `latent_recursion` — The Inner Loop (16 micro-steps)
```python
def latent_recursion(self, x, y, z, x_input=None):
    xy = x + y
    for _ in range(self.n_recursions):           # 16 iterations
        z = self.forward_network(xy + z)          # transformer pass
        z = self.z_norm_layer(z)                  # collapse prevention
        if self.use_z3_pruning and x_input is not None:
            # Z3 latent projection (see below)
            ...
    y = self.forward_network(y + z)               # y update from final z
    return y, z
```

After 16 micro-steps, `z` holds a deeply iterated latent state. `y` is updated once from `z` at the end.

#### 4. Z3 Integration — Two Injection Points

**Point A: Inside `latent_recursion` (every micro-step)**
```python
step_logits = self.task_head.decode(z)                           # peek at z's current belief
pruned_logits = prune_illegal_logits_z3(step_logits, hard_preds) # mask illegal digits
step_probs = F.softmax(pruned_logits, dim=-1)                    # differentiable
z_symbolic = step_probs @ self.task_head.embedding.weight[1:]   # project back to latent space
z = self.z_norm_layer(z + z_symbolic)                           # inject + re-normalize
```
This makes Z3 a **latent-space constraint projector** — the symbolic truth is injected back into `z` at every micro-step, not just at the output.

**Point B: Output readout in `deep_recursion`**
```python
y_hat = prune_illegal_logits_z3(y_hat, hard_preds)  # hard mask on final logits
```
A safety net — prevents any logically impossible digit from appearing in the final prediction regardless of what the network learned.

#### 5. `deep_recursion` — One Macro-Step
```python
def deep_recursion(self, x, y, z, with_gradients=False, x_input=None):
    y, z = self.latent_recursion(x, y, z, x_input=x_input)
    y_norm = self.output_norm(y)
    y_hat = self.task_head.decode(y_norm)
    if self.use_z3_pruning:
        y_hat = prune_illegal_logits_z3(y_hat, hard_preds)   # Point B
    q_logit = self.q_head(y_norm.mean(dim=1))
    return (y, z), y_hat, q_logit
```

#### 6. `forward` — Supervision Loop (4 macro-steps)
```python
for _ in range(self.n_supervision):             # 4 supervision steps
    (y, z), y_hat, q_logit = self.deep_recursion(...)
    # NO detach() — full BPTT through all 4 steps
    predictions.append(y_hat)
    losses.append(pred_loss + 0.1 * halt_loss)
```
**No `.detach()` between supervision steps** — gradients flow through the entire unrolled computation (BPTT over 4 × 16 = 64 effective transformer passes).

**Constraint loss** (added R022–R025, currently reverted to R024 style — final step only):
```python
c_loss = sudoku_constraint_loss(predictions[-1])
loss = loss + get_constraint_weight() * c_loss
```

### Data Flow

```
x_input [B, 81]
    │
    ▼ SudokuHead.encode()
x   [B, 81, dim]   ← token embeddings + row/col positional encodings
    │
    ▼ (loop: n_supervision=4 macro-steps)
    │
    ├─ latent_recursion (16 micro-steps each):
    │       xy = x + y
    │       for 16 steps:
    │           z = transformer(xy + z)
    │           z = RMSNorm(z)                ← collapse prevention
    │           z += Z3_projection(z)         ← symbolic injection
    │       y = transformer(y + z)
    │
    ├─ constraint_prop(y)                     ← R026: group attention (PENDING)
    │
    ├─ y_norm = output_norm(y)
    ├─ y_hat  = Linear(y_norm)               ← [B, 81, 9] logits
    └─ y_hat  = Z3_prune(y_hat)             ← hard output mask
    
    ▼
y_hat [B, 81, 9]   → argmax + 1 → predicted digits 1–9
```

### Key Architectural Invariants

| Property | Value | Why |
|---|---|---|
| Weight sharing | All n_recursions + n_supervision steps use the same `layers` | Parameter efficiency; forces iterative refinement |
| No detach between macro-steps | Full BPTT unroll | Gradients reach the earliest supervision steps |
| z_norm_layer inside micro-loop | RMSNorm on z every step | Prevents z-norm explosion (R008 discovery) |
| Z3 at latent level | Injected into z every micro-step | Z3 shapes the latent manifold, not just the output |
| Z3 at output level | Hard mask on y_hat | Safety net against impossible predictions |
| No box positional embedding | Row + col embeddings only | Box topology must be discovered from attention |
| Loss on empty cells only | `ignore_index=-100` on given clues | Don't penalize the model for cells it didn't predict |

---

## Architectural Decision Log

### R001–R007 | Baseline: 0% Puzzle Accuracy
**Problem:** Representation collapse — `z` norms hitting `1e17+` after a few recursion steps. The recurrent transformer without stabilization has no bound on norm growth.

**Lesson:** Weight-shared transformers applied recursively amplify norms exponentially. Some form of normalization inside the loop is mandatory.

---

### R008 | `z_norm_layer` — Internal Recurrence Stabilizer ✅
**Change:** Added `RMSNorm` applied to `z` inside the micro-recursion loop.

**Effect:** z-norms dropped from `1e17+` to `~1.0×` per step. 4×4 accuracy jumped significantly. The fundamental training instability was solved.

**Code:**
```python
z = self.z_norm_layer(z)  # inside latent_recursion for-loop
```

---

### R010 | Z3 Latent Projection ✅
**Change:** Z3 pruning moved from output-only to also inside the micro-recursion loop. Model decodes `z` at every micro-step, Z3 identifies impossible digits, the constraint-consistent probability distribution is projected back into latent space via the embedding matrix.

**Effect:** Symbolic constraints now shape `z` during the latent computation, not just at readout. The embedding matrix serves double duty as a constraint→latent projection matrix.

**Code:**
```python
step_probs = F.softmax(pruned_logits, dim=-1)
z_symbolic = step_probs @ self.task_head.embedding.weight[1:]  # [B, 81, dim]
z = self.z_norm_layer(z + z_symbolic)
```

---

### R014 | Full BPTT Unroll — No `.detach()` Between Macro-Steps ✅
**Change:** Removed `y, z = y.detach(), z.detach()` between supervision steps.

**Effect:** 9×9 puzzle accuracy jumped from ~0% to 4.9%, CSR to 18.7%. With detach, gradients only see one macro-step; without it, gradients flow through the full recursion depth, enabling the model to learn iterative refinement strategies that span multiple macro-steps.

**Note:** The multi-cycle inference (`n_cycles > 1` without gradients) was also commented out — eval depth now matches train depth exactly.

---

### R017/R018 | Z3 Ablation: Z3 Does Real Work
**Finding:** Z3-off vs Z3-on at 5k training data: no measurable difference. Z3 at clue-only pruning (hard_preds = given clues only) prunes very few cells when the model is mostly random. Z3 only proves its value when the model is good enough that its predictions start violating constraints meaningfully.

---

### R021 | Generalization Gap Identified
**Finding:** 15k data (no dropout) closed the train/val generalization gap but CSR collapsed to 6.1%. Revealed **objective misalignment**: cross-entropy loss rewards per-cell accuracy but does not reward constraint coherence across cells. A model can maximize CE while producing constraint-incoherent outputs.

---

### R022 | Z3 at Scale — Confirmed Real Work ✅
**Change:** Z3=True + 15k training data.

**Effect:** CSR recovered from 6.1% (R021, no Z3) to 12.6% (+6.5 pts). Confirmed: Z3 does more than filter — it actively shapes the latent trajectory through the symbolic injection in `latent_recursion`.

---

### R023 | Constraint Auxiliary Loss Alone — Weaker Than Z3
**Change:** Auxiliary loss penalizing constraint violations on softmax probabilities (no Z3).

**Effect:** CSR 7.2% — weaker than Z3 alone (12.6%). Epoch-1 train accuracy collapsed to 17.3% due to constraint penalty hitting full strength at random initialization before the model could learn basic cell predictions.

**Code:**
```python
def sudoku_constraint_loss(logits):
    probs = torch.softmax(logits, dim=-1).view(B, 9, 9, 9)
    penalty  = (probs.sum(dim=2) - 1.0).pow(2).mean()   # rows
    penalty += (probs.sum(dim=1) - 1.0).pow(2).mean()   # cols
    p = probs.view(B, 3, 3, 3, 3, 9)
    penalty += (p.sum(dim=(2, 4)) - 1.0).pow(2).mean()  # boxes
    return penalty / 3.0
```

---

### R024 | Z3 + Constraint Loss (Final Step) + Warmup — Best So Far ✅
**Change:** Z3=True + constraint loss on `predictions[-1]` only + LR/constraint warmup.

**Effect:** CSR 13.2% — new record. Warmup ramp prevented epoch-1 collapse (40.0% vs R023's 17.3%).

**Warmup:**
```python
def get_constraint_weight():
    return constraint_weight * min(1.0, step / max(warmup_steps, 1))
```

---

### R025 | Constraint Loss on ALL Supervision Steps — Regression ❌
**Change:** Applied constraint loss to `predictions[0]` through `predictions[3]` (all steps).

**Effect:** CSR regressed to 10.5% (−2.7 pts vs R024). Constraint gradient at intermediate macro-steps competes with CE before representations are mature. More constraint gradient = worse CSR.

**Lesson:** The constraint-gradient design space is exhausted. No training-objective manipulation within this architecture will break through the ~13% CSR ceiling. The bottleneck is structural.

---

## Symbolic Integration: Z3

The Z3 integration in this project operates at two levels and uses two completely different codebases:

### 1. Tensorized Z3 Pruning (active — notebook Cell 2)
`notebooks/trm_experiment.ipynb`, Cell 2

Fully vectorized, GPU-resident, no Python loops over cells. Uses PyTorch tensor operations to compute which digits are already present in each cell's row/col/box, then masks those logits to `-20.0`.

**Works on any batch size, runs entirely on CUDA.** This is what runs during training and inference.

```python
def prune_illegal_logits_z3(y_hat, hard_preds, grid_size=9):
    # Vectorized constraint masking
    # row_mask | col_mask | box_mask → apply -20.0 to illegal positions
```

### 2. True Z3 Solver (legacy, 4×4 only)
`symbolic/z3_sudoku.py`

Uses the actual Z3 SMT solver. Runs on CPU, has a Python loop over cells and digits, calls Z3's `s.check()` for each (cell, digit) pair. Too slow for 9×9 training but provides true logical inference — not just "digit already exists in row/col/box" but "digit is impossible given *any* constraint propagation."

**This is the symbolic foundation the project was originally built on.** The tensorized version is an approximation of Z3's constraint-only pruning (equivalent for clue-based pruning, weaker for full inference-time solving).

---

## Constraint Enforcement History

| Run | Z3 | Constraint Loss | Applied To | CSR | Notes |
|---|---|---|---|---|---|
| R021 | ❌ | ❌ | — | 6.1% | Baseline collapse |
| R022 | ✅ | ❌ | — | 12.6% | Z3 alone +6.5 pts |
| R023 | ❌ | ✅ | Final step | 7.2% | Cold-start collapse |
| R024 | ✅ | ✅ | Final step only | **13.2%** | **Current best** |
| R025 | ✅ | ✅ | All 4 steps | 10.5% | −2.7 pts regression |

**The constraint-gradient axis is closed.** R024 is the optimal configuration within the current architecture.

---

## Performance Ceiling Map

```
CSR (Constraint Satisfaction Rate)

0%   ─── R001–R007  (collapse, z-norms 1e17+)
         │
         ▼ R008: z_norm_layer added
~4%  ─── R008–R013  (4×4 working, 9×9 partial)
         │
         ▼ R014: full BPTT unroll
~18% ─── R014–R017  (9×9 initial breakthrough)
         │
         ▼ R018: no dropout, 5k data ceiling
15%  ─── R018       (5k data neural ceiling)
         │
         ▼ R021: 15k data, objective misalignment
6%   ─── R021       (CSR collapse despite data scaling)
         │
         ▼ R022: Z3 + 15k data
12.6%─── R022
         │
         ▼ R024: Z3 + constraint loss (final step) + warmup
13.2%─── R024  ← CURRENT CEILING
         │
         ▼ R025: constraint loss on all steps (regression)
10.5%─── R025
         │
         ▼ R026: ConstraintPropagationLayer (PENDING)
  ?% ─── R026  Does structural inductive bias break 13%?
```

**Why the ceiling exists:** The current transformer layers have no inductive bias toward Sudoku's constraint-group topology (row/col/box). Every cell attends to every other cell with equal structural priority. Gradient-based constraint enforcement cannot overcome this — it can nudge representations but cannot inject the topological prior.

---

## Open Questions & Next Moves

### R026 — `ConstraintPropagationLayer` (Immediate Next)

**Theory:** The CSR ceiling is architectural. Adding a dedicated attention layer where each cell attends **only within its row, column, and 3×3 box** gives the model native Sudoku topology as an inductive bias.

**New component to add to Cell 7:**
```python
class ConstraintAttention(nn.Module):
    """Masked attention: each cell attends only to its row, col, and box peers."""
    def __init__(self, dim, n_heads, grid_size=9):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        N, sub = grid_size, int(grid_size ** 0.5)
        cells = N * N
        attn_bias = torch.full((cells, cells), float('-inf'))
        for i in range(cells):
            ri, ci = i // N, i % N
            for j in range(cells):
                rj, cj = j // N, j % N
                if ri == rj or ci == cj or (ri//sub == rj//sub and ci//sub == cj//sub):
                    attn_bias[i, j] = 0.0
        self.register_buffer('attn_bias', attn_bias)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=self.attn_bias[None, None])
        return self.proj(out.transpose(1, 2).contiguous().reshape(B, L, D))


class ConstraintPropagationLayer(nn.Module):
    """Residual layer: propagates constraint-group information after latent recursion."""
    def __init__(self, dim, n_heads, grid_size=9):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.attn = ConstraintAttention(dim, n_heads, grid_size)

    def forward(self, y):
        return y + self.attn(self.norm(y))
```

**Where it sits in `TinyRecursiveModel`:**
```python
# __init__:
self.constraint_prop = ConstraintPropagationLayer(dim, n_heads, grid_size=task_head.N)

# deep_recursion, after latent_recursion:
y, z = self.latent_recursion(x, y, z, x_input=x_input)
y = self.constraint_prop(y)   # ← constraint-group propagation
y_norm = self.output_norm(y)
```

**Diagnostic signals to watch:**
- If CSR jumps meaningfully (>16%): architectural bottleneck confirmed, port to 4×4
- If CSR stays flat (~13%): latent space itself is the limit, move to 4×4 regardless
- Per-step accuracy arc: if it starts climbing step-over-step (not flat), the recursion is now iteratively propagating constraints

---

### R027 — Recursion Depth Ablation

**Theory:** The flat per-step accuracy curve (≤0.15 pt spread across 4 steps in every run) might be a depth issue rather than a representation issue. Doubling `n_recursions` to 32 tests whether the model just needs more inner iterations.

**Change:** `n_recursions = 32` (or `n_cycles = 4`), everything else identical to R024.

**Signal:** If per-step accuracy starts climbing, depth was the bottleneck. If still flat, it's representational.

---

### Return to 4×4 (After R026/R027)

**Why 4×4:**
- 4×4 has a dramatically simpler constraint space — only 4 rows, 4 cols, 4 boxes of 4 cells each
- Fewer cells (16 vs 81) means the model's attention can fully resolve constraint groups without topology priors
- At 4×4, changes to architectural components (depth, constraint loss, propagation layers) produce cleaner signal because the problem is closer to the model's representational capacity
- The full Z3 solver (`symbolic/z3_sudoku.py`) works at 4×4 — can use true SMT solving, not just the tensorized approximation

**What to bring back:**
- `ConstraintPropagationLayer` if R026 shows signal — will be even more decisive at 4×4
- `warmup` — confirmed critical, carry forward always
- Z3 latent injection — confirmed working, carry forward
- Full BPTT (no detach) — foundational, never remove

---

## File Map

```
repo/
├── notebooks/
│   └── trm_experiment.ipynb       # Primary experiment notebook (all active runs)
│       ├── Cell 2: Z3 tensorized logit pruning (prune_illegal_logits_z3)
│       ├── Cell 3: Experiment config (dim, n_layers, n_heads, ...)
│       ├── Cell 4: Dataset generation
│       ├── Cell 7: Model architecture (TinyRecursiveModel, SudokuHead, ...)
│       ├── Cell 8: Training loop (EMA, optimizer, constraint loss, warmup)
│       ├── Cell 9: Sudoku validity checker (_is_valid_sudoku)
│       ├── Cell 10: Evaluation + diagnostics
│       └── Cell 13: Save results JSON
│
├── src/                            # Reference implementations (not used for active runs)
│   ├── model.py                    # TRM + task heads (older, no Z3 latent injection)
│   └── train.py                    # Training script (CLI-driven, not used for GPU runs)
│
├── symbolic/
│   └── z3_sudoku.py               # True Z3 solver (4×4, CPU, legacy)
│
├── results/
│   └── exp-sudoku-r0XX/           # Per-run results: JSON + PNG plots
│
├── RESULTS.md                     # Full experiment log (R001–R025)
└── ARCHITECTURE.md                # This file
```

**Important:** The canonical model architecture for all runs from R010 onward lives in the **notebook** (Cell 7), not `src/model.py`. The notebook version includes Z3 latent injection, the `z_norm_layer` inside the micro-loop, and the updated `check_correct` (per-cell, not whole-board). `src/model.py` is older and does not reflect the current experiment state.

---

*Last updated: R025 complete, R026 pending*
