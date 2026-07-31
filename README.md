# Symbolically Bounded Latent Recursion

A from-scratch PyTorch investigation into whether a **Tiny Recursive Model (TRM)** — a small, weight-shared transformer that iterates a latent state `z` over many recursion steps — performs genuine *sequential* computation, or whether its apparent "reasoning depth" is an artifact of measurement.

The headline result is a **negative one, established through a controlled diagnostic chain**: on the tasks tested, the TRM's recursion does not iterate. It solves problems with a shallow, single-pass parallel circuit, and when a task is constructed to *require* step-by-step composition, the architecture fails to learn it. This repo documents how that conclusion was reached, run by run, with full instrumentation.

---

## What this project actually found

The work ran in two phases across 25 Sudoku runs and 12 pointer-chasing runs. Each run logs a hypothesis, a single changed variable, results, latent-state diagnostics, and an interpretation.

**Phase 1 — Sudoku (runs 001–025): stabilizing the recursion, then hitting a ceiling.**

- **Representation collapse (r001–r007).** The unregularized weight-shared recurrence blows up: the latent state `z` grows ~198× per step, its norm reaching 1e17+ within a few steps, freezing the direction while the magnitude explodes. No puzzle is ever solved.
- **A single-layer fix (r008).** Adding `RMSNorm` to `z` inside the recurrence loop drops the per-step growth from ~198× to ~1.0× and is the most impactful change in the series — 4×4 puzzle accuracy jumps from 0% to 54%, and for the first time accuracy *increases* across recursion steps.
- **Objective misalignment (r021).** Scaling to 15k training puzzles solves the generalization gap but collapses constraint satisfaction — revealing that cell-level cross-entropy rewards per-cell accuracy without rewarding cross-cell constraint coherence.
- **Symbolic integration (r022–r025).** Z3 SMT pruning injected into the latent loop recovers constraint satisfaction to ~13%, but a full sweep of the constraint-enforcement design space (Z3 vs. auxiliary loss, final-step vs. all-step) never breaks that ceiling. The bottleneck is shown to be structural, not a training-signal problem.

A persistent signal runs through all of Phase 1: **per-step accuracy is flat** — the model front-loads its answer into the first step and the later recursion steps do no additional work. But Sudoku can't disambiguate "not iterating" from "correctly front-loading an easy cell," so Phase 2 was built to remove that ambiguity.

**Phase 2 — Pointer-chasing (runs pc-r001–pc-r012): a shortcut-free test of iteration.**

The task is K composed random permutations, with a mandatory leakage gate proving no shallow shortcut can solve it. This makes "does the recursion compose K sequential operations?" directly testable.

- **A false ceiling, then a real one (pc-r006 → pc-r007).** An apparent "3-hop composition ceiling" turned out to be an **fp16/GradScaler numerical artifact** (silent optimizer step-skipping); switching to bf16 reproduced full composition and reversed a chance-level result to 100%. This is a clean example of an infrastructure bug masquerading as a scientific finding — and of catching it.
- **The compute-starve sweep (pc-r008–r011).** Holding everything fixed and sweeping the inner recursion budget `n_recursions` ∈ {8, 4, 2, 1} isolates the mechanism. The answer appears in the output at **outer step 0 for every budget**, down to a single transformer pass. All latent diagnostics are invariant across an 8× compute range; only training wall-clock changes. Verdict: the solution is a **shallow parallel-composition circuit**, and the outer recursion loop performs no run-time computation.
- **The forcing test (pc-r012).** Revealing only one permutation per step makes single-pass composition impossible — genuine iteration becomes the only route. The model **fails at chance**, and the failure localizes precisely at the first step that requires composing a new input into a carried state. The recursion can *hold* a state but never learns to *update* it.

The result that survives: probe-verified composition in the latent state and perfect downward depth-transfer. The result that does not: that the recursion's outer steps function as reasoning steps.

---

## Why this matters

Recursive and "latent reasoning" architectures are often credited with iterative computation on the basis of accuracy metrics alone. This project shows — with a reproducible diagnostic protocol — that **accuracy can completely conceal the absence of recursive computation**, and provides the instruments (per-step latent decoding, a stop-gradient probe, compute-starve ablation, and shortcut-free task construction) needed to tell the difference.

---

## Diagnostic instruments

Every run captures the same latent-state diagnostics, which are what make the negative result airtight rather than speculative:

- **z-norm / y-norm trajectories** — detect representation collapse and norm explosion
- **z-cosine similarity across steps** — measures whether the latent state actually moves or is frozen
- **halt probability** — whether the model's stopping signal tracks real completion
- **stop-gradient probe (`z_probe`)** — honest readout of what the latent state encodes, without interfering with training
- **per-step / eval-by-depth accuracy** — separates front-loading from iteration
- **compute-starve sweep** — isolates whether recursion depth is used or merely present
- **leakage gate** — proves the task admits no shallow shortcut before any GPU time is spent

---

## Architecture

A weight-shared 2-layer transformer applied recursively over a latent state `z`, with:

- an inner recurrence loop (`n_recursions`) with `RMSNorm` stabilization
- an outer deep-supervision loop (`n_supervision`) trained with full backpropagation-through-time (no detachment between steps)
- a learned halt gate
- optional Z3 SMT constraint pruning injected at both the latent and output levels (Sudoku)
- EMA weight averaging

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full component breakdown and decision log, and [`RESULTS.md`](RESULTS.md) for the complete run-by-run experiment log.

## Project structure

```
├── data/          # Dataset generation (Sudoku, Maze, pointer-chasing)
├── symbolic/      # Z3 neuro-symbolic integration
├── src/           # Reference model + training implementations
├── notebooks/     # Primary experiment notebook (active runs)
├── results/       # Per-run diagnostics: JSON + plots
├── ARCHITECTURE.md
└── RESULTS.md     # Full experiment log (37+ runs)
```

## Setup

```bash
git clone https://github.com/usafhulk/Symbolically-Bounded-Latent-Recursion-.git
cd Symbolically-Bounded-Latent-Recursion-
pip install -r requirements.txt
```

## Usage

```bash
python check_gpu.py                      # verify GPU
python train.py --task sudoku            # train on Sudoku
python train.py --task maze --grid_size 11
```

---

*Research conducted as part of a PhD in Artificial Intelligence. The experiment log is a living document; interpretations are revised as runs accumulate, and errata are recorded in-line rather than edited away.*
