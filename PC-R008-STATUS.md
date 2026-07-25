# PC-R008 Experiment Status

## Overall Status

PC-R008 (compute-starve sweep, n_recursions ∈ {1,2,4,8}) has one arm completed and recorded (R=8, absolute, seed=42) that fully replicates pc-r007's 100% K=4 accuracy, but with a regression toward complete halt-at-step-0 front-loading; the remaining three arms (R=1, R=2, R=4) — which are the experiment's primary scientific payload — have no recorded results in either the JSON directory or RESULTS.md.

---

## Executive Summary (30-second read)

| Area | Status |
|---|---|
| Experiment Goal | ⚠️ Partially Executed |
| Training (R=8 arm) | ✅ Complete |
| Validation (R=8 arm) | ✅ 100% K=4 accuracy |
| Performance (K=4 final) | ✅ TRM=1.0, Baseline=1.0 |
| Stability | ✅ No norm explosion, z stable |
| Reproducibility | ⚠️ Halt behavior regressed vs pc-r007 anchor |
| Ready for Next Phase? | No — 3 of 4 arms unexecuted |

PC-R008 was designed to test whether K=4 composition degrades when the inner recurrence depth (n_recursions) is reduced from 8 toward 1 — isolating whether pc-r007's success required deep inner compute or genuine outer-step iteration. The only completed arm (R=8) matches the pc-r007 anchor configuration and again achieves 100% K=4 accuracy for both TRM and baseline. However, R=8 is the control arm — it cannot answer whether shrinking compute breaks the result. The remaining arms (R=1, R=2, R=4) have no evidence of completion. A secondary concern is that the R=8 TRM now shows halt_prob=1.0 at step 0 (complete front-loading), a regression from pc-r007's 0.094 at step 0 (one genuine recursion step), suggesting the 25-epoch training duration caused more aggressive front-loading than the 15-epoch pc-r007 run. The baseline per-step accuracy profile cleanly validates the leakage check: it reads chance at steps below K and 1.0 exactly at step K−1, as expected for a per-hop multi-task head.

---

## Experiment Timeline

1. **Previous state (pc-r007, absolute, seed=42):** TRM solved K=4 at 100% with 4-hop z-supervision withheld. Phase transition at epoch 4. halt_prob=0.094 at step 0, 1.0 from step 1 — one genuine outer recursion step before the answer appeared. z_probe_acc ≥ 0.992 at all steps (including unsupervised). z_taskdecode_acc flipped to 1.0 at step 3, coinciding with composition saturation.

2. **What changed (pc-r008 design):** Single variable swept: n_recursions ∈ {1, 2, 4, 8}. Everything else held fixed (arm=absolute, seed=42, bf16, 100k train, K=4, dim=256, n_supervision=8). Epoch budget reduced from 50 (pc-r006) / 15 (pc-r007) to **25 epochs**. GPU: NVIDIA L4.

3. **What was tested:** Only R=8 arm is present. This is the anchor — identical inner depth to pc-r007 — providing a reproducibility check but no new signal.

4. **What happened:** R=8 achieved 100% K=4 val accuracy for both TRM (5.8 h) and baseline (9.5 min). Phase transition at epoch 4 (TRM: 62→99% train acc). Val checkpoints reported at epochs 10, 20, 25 — all 100%. halt_prob collapsed to 1.0 at step 0 for the TRM (regression from pc-r007's 0.094). z_probe_acc saturated at every step.

5. **Current outcome:** One of four planned arms completed. The scientific question (does K=4 degrade at R<8?) is unanswered. The R=8 arm replicates accuracy but shows increased front-loading relative to pc-r007.

---

## Metric Comparison

| Metric | pc-r007 (anchor, n\_rec=8) | pc-r008 (R=8 arm) | Change | Interpretation |
|---|---|---|---|---|
| TRM K=4 val acc (final step) | 1.000 | 1.000 | = | Full replication |
| TRM K=2 val acc | 1.000 | 1.000 | = | Downward transfer intact |
| TRM K=3 val acc | 1.000 | 1.000 | = | Downward transfer intact |
| Baseline K=4 val acc | 1.000 | 1.000 | = | Unchanged |
| TRM halt\_prob step 0 | 0.094 | 1.000 | ↓ (regression) | More extreme front-loading |
| TRM halt\_prob step 1+ | 1.000 | 1.000 | = | Converges after one step (pc-r007) or immediately (pc-r008) |
| TRM y-decode step 0 | 0.081 | 1.000 | ↑ | Correct answer at step 0 in pc-r008 |
| TRM z\_taskdecode steps 0–2 | ~chance | ~chance (0.107, 0.105, 0.090) | ≈ | z not y-decodable at partial steps |
| TRM z\_taskdecode step 3+ | 1.000 | 1.000 | = | Composition saturated at step 3 |
| TRM z\_probe\_acc steps 0–7 | ≥0.992 | ≥0.999 | ≈ | Full saturation in both |
| TRM z\_aux\_acc steps 0–2 | 1.000 | 1.000 | = | Supervised steps clean |
| z norm (final step) | 12.986 | 10.264 | ↓ slightly | Slightly more compact, no instability |
| z cosine step 0→1 | 0.659 | 0.316 | ↓ | Larger state update on first step |
| z cosine step 1→2 | — | 0.345 | N/A | Second real update before convergence |
| TRM training time | 12,734 s (15 ep) | 20,915 s (25 ep) | ↑ | L4 vs L4 — comparable per-epoch pace |
| Baseline training time | 349 s (15 ep) | 571 s (25 ep) | ↑ proportional | Expected |
| TRM train accuracy final | Not Available | 1.000 (epoch 19+) | — | Fully converged |
| Phase transition epoch | 4 | 4 | = | Consistent |

---

## Major Findings

### Finding 1

**Observation:** K=4 accuracy replicates at 100% for both TRM and baseline in the R=8 arm.

**Evidence:**
- JSON `eval_by_K.trm["4"].final_acc = 1.0`; `eval_by_K.baseline["4"].final_acc = 1.0`
- TRM val_accs: [1.0, 1.0, 1.0] at all three checkpoint epochs

**Interpretation:** The pc-r007 result is reproducible under the pc-r008 configuration. The reduction from 50 to 25 epochs was sufficient — convergence occurred around epoch 19.

**Confidence: High**

---

### Finding 2

**Observation:** The TRM regressed from one genuine outer recursion step (pc-r007) to complete halt-at-step-0 (pc-r008).

**Evidence:**
- pc-r007 halt_probs: `[0.094, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000]`
- pc-r008 halt_probs: `[1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000]`
- pc-r007 y_decode step 0: 0.081 (chance); pc-r008 y_decode step 0: 1.000

**Interpretation:** The TRM in pc-r008 answers correctly before any outer recursion step occurs. With n_recursions=8 inner updates per outer step, the first outer step involves 8 inner transformer passes (n_cycles is confirmed inert per repository memory). The model may have learned to compute all 4 hops within those 8 inner passes and embed the result before the first outer readout. The 25-epoch training (10 more than pc-r007) likely allowed the model to further optimize the front-loading strategy.

**Confidence: High** (halt_prob=1.0 is unambiguous; causal interpretation of why is Medium confidence)

---

### Finding 3

**Observation:** The baseline per-step accuracy profile validates the evaluation methodology cleanly.

**Evidence:**
- `baseline.eval_by_K["4"].step_accs = {0: 0.062, 1: 0.063, 2: 0.064, 3: 1.0}`
- `baseline.eval_by_K["3"].step_accs = {0: 0.0635, 1: 0.063, 2: 1.0, 3: 1.0}`
- `baseline.eval_by_K["2"].step_accs = {0: 0.062, 1: 1.0, 2: 1.0, 3: 1.0}`

**Interpretation:** The multi-task baseline activates exactly at step K−1 for depth K (chance below, 1.0 at and above). This validates that the evaluation correctly measures per-step accuracy for each K level. Chance = 1/16 ≈ 0.0625, matching observed pre-activation values.

**Confidence: High**

---

### Finding 4

**Observation:** z_probe_acc is saturated (≥0.999) at all 8 steps, including unsupervised steps 3–7.

**Evidence:**
- `z_probe_acc = [1.0, 1.0, 0.999, 1.0, 1.0, 1.0, 1.0, 1.0]`

**Interpretation:** The stop-gradient probe reads the composition correctly from z throughout. Combined with halt_prob=1.0 at step 0, z already holds the final answer before any outer supervision step occurs. The dip to 0.999 at step 2 is trivial, not meaningful.

**Confidence: High**

---

### Finding 5

**Observation:** z_taskdecode_acc flips from chance to 1.0 at step 3, identical to pc-r007.

**Evidence:**
- `z_taskdecode_acc = [0.107, 0.105, 0.090, 1.0, 1.0, 1.0, 1.0, 1.0]`

**Interpretation:** The task output decoder (y-space) cannot read the answer from z at steps 0–2, but can from step 3 onward — despite z_probe reading it correctly from step 0. z represents the composition in a latent geometry that aligns with output space only after sufficient outer steps. Exact replication of pc-r007's transition point.

**Confidence: High**

---

### Finding 6

**Observation:** The primary experimental payload (R=1, R=2, R=4 arms) is absent. No results exist for these configurations.

**Evidence:**
- `results/exp-pointer_chase-r008/` contains only one JSON: `pc-r008_absolute_s42_nrec8_20260724_143708_results.json`
- No files with `nrec1`, `nrec2`, `nrec4` patterns exist anywhere in the results directory
- RESULTS.md contains no pc-r008 entry

**Interpretation:** The compute-starve sweep's central question is unanswerable from available evidence. These arms need to be run.

**Confidence: High** (absence of files is directly observable)

---

## Problems Identified

### Problem 1 — Halt Behavior Regression

**Severity: High**

**Evidence:** pc-r007 halt_prob at step 0 = 0.094; pc-r008 halt_prob at step 0 = 1.000. pc-r008 y_decode at step 0 = 1.000 (correct answer before first outer recursion step).

**Impact:** The R=8 anchor now behaves differently from pc-r007, complicating intra-sweep comparisons for the upcoming R=1/2/4 arms.

**Recommended next action:** Run R=8 again at 15 epochs to determine if the halt regression is epoch-driven. Standardize epoch budget across all arms.

---

### Problem 2 — Three Arms Missing (Critical)

**Severity: Critical**

**Evidence:** Only one of four planned arms (R=8) is in `results/exp-pointer_chase-r008/`. No R=1, R=2, or R=4 results exist anywhere. RESULTS.md has no pc-r008 entry.

**Impact:** The experiment's hypothesis ("does K=4 degrade when inner recursion depth shrinks?") is completely unanswered. The R=8 result is a control arm — it carries no new information over pc-r007.

**Recommended next action:** Run R=4, R=2, and R=1 arms. R=4 is the most informative starting point.

---

### Problem 3 — Notebook Outputs Are Stale

**Severity: Medium**

**Evidence:** `trm_experiment.ipynb` is titled "pc-r007" but its outputs are from a Sudoku run (`sudoku_20260526_010415`, exp/sudoku/r020, 2026-05-26). `trm_experiment_fixed.ipynb` is similarly stale. Neither contains any pc-r008 output.

**Impact:** The notebooks cannot serve as a cross-check for the JSON results file.

**Recommended next action:** Clear stale outputs and re-run the notebook for pc-r008, or update cell outputs to reflect the actual run.

---

### Problem 4 — n_cycles Errata Not in RESULTS.md

**Severity: Low**

**Evidence:** pc-r008 JSON notes field states: "This corrects pc-r007 finding 7, which mis-attributed the per-step inner compute to n_cycles=3." RESULTS.md pc-r007 finding 7 says "≥6 inner recurrent applications by step 1" — based on n_cycles=3 being active, which it is not (n_cycles is confirmed inert in TinyRecursiveModel).

**Impact:** A reader of RESULTS.md alone would not see this correction.

**Recommended next action:** Add errata to RESULTS.md under pc-r007 finding 7.

---

## Successes

**Success 1: K=4 accuracy at 100% replicates.**
Evidence: `eval_by_K.trm["4"].final_acc = 1.0`; val_accs = [1.0, 1.0, 1.0].

**Success 2: Downward depth transfer (K=2, K=3) replicates.**
Evidence: `eval_by_K.trm["2"].final_acc = 1.0`, `eval_by_K.trm["3"].final_acc = 1.0`, all step_accs = 1.0.

**Success 3: z_probe_acc saturates at every step including unsupervised steps 3–7.**
Evidence: `z_probe_acc = [1.0, 1.0, 0.999, 1.0, 1.0, 1.0, 1.0, 1.0]`. 4-hop composition present in z without explicit 4-hop z-supervision.

**Success 4: Latent state numerically stable.**
Evidence: `z_norms = [10.283, 10.317, 10.320, 10.269, 10.264, 10.263, 10.263, 10.264]`. No norm explosion or collapse.

**Success 5: Baseline per-step accuracy validates the evaluation instrument.**
Evidence: Baseline K=4 step_accs: [chance, chance, chance, 1.0]. Correct per-hop head behavior.

**Success 6: Phase transition at epoch 4 replicates.**
Evidence: TRM epoch_accs: epochs 1–3 near chance (0.062, 0.063, 0.099); epoch 4 = 0.990.

---

## Remaining Risks

**Risk 1: Compute-starve hypothesis completely untested.**
All three informative arms (R=4, R=2, R=1) are unrun. Either outcome — K=4 holds at R=1 (outer iteration doing the work) or collapses (result is inner-compute-dependent) — is important and currently unknown.

**Risk 2: Training-duration confound in halt behavior.**
pc-r007 (15 epochs): halt_prob=0.094 at step 0. pc-r008 (25 epochs): halt_prob=1.0 at step 0. These are not equivalent conditions. Intra-sweep comparisons need a fixed epoch budget.

**Risk 3: One-pass shortcut risk at R=1.**
With n_recursions=1, if the model solves K=4 within the first outer step's single inner update, the "outer-step as hop clock" interpretation collapses. The halt behavior at R=1 will be the decisive diagnostic.

**Risk 4: Single seed only.**
All pc-r008 results are seed=42. pc-r007 was designed as a 2-arm × 3-seed matrix. Robustness across seeds 43 and 44 is unknown.

**Risk 5: RESULTS.md documentation lag.**
RESULTS.md is at least one entry behind the results directory. The gap between authoritative JSON evidence and the narrative record is growing.

---

## Consistency Check

**Partially consistent** — with important caveats.

| Source | Status | Notes |
|---|---|---|
| JSON results file (pc-r008 R=8) | ✅ Internally consistent | All metrics self-consistent; no contradictions |
| RESULTS.md | ❌ No pc-r008 entry | Cannot check consistency — entry absent |
| trm_experiment.ipynb | ❌ Stale outputs | May 2026 Sudoku run; unrelated to pc-r008 |
| trm_experiment_fixed.ipynb | ❌ Stale outputs | Sudoku notebook; unrelated to pc-r008 |

**Discrepancies:**
1. Notebook outputs are from a Sudoku run weeks before the pointer-chasing series.
2. RESULTS.md has no pc-r008 entry despite the JSON being committed on 2026-07-24.
3. pc-r008 halt_prob=1.0 at step 0 vs pc-r007's 0.094 is undocumented and unexplained in available records.

---

## Overall Assessment

**Current maturity level: Proof of Concept**

The pointer-chasing series has demonstrated that the TRM can solve K=4 at 100% with intermediate z-supervision, that the result transfers to shallower depths (K=2, K=3) without retraining, and that z encodes the correct composition as verified by a stop-gradient probe. These are genuine scientific results. However, the thesis question — whether the TRM's outer recursion performs genuine hop-by-hop sequential iteration — remains open. Every run that achieves 100% does so via early halting (front-loading), not via the outer supervision steps acting as a hop clock. The architecture is stable and the experimental methodology is well-designed, but the primary open question (does recursion depth matter?) is untested. This places the project at Proof of Concept: the task can be solved, but the mechanism it was designed to test has not been demonstrated.

---

## Recommended Next Experiments

| Priority | Recommendation | Why |
|---|---|---|
| High | Run pc-r008 R=4, R=2, R=1 arms (arm=absolute, seed=42, 25-epoch budget) | The experiment's entire scientific payload. R=8 is the control; R<8 is the test. |
| High | Run pc-r008 R=8 at 15 epochs to check if halt regression is epoch-driven | If halt_prob reverts to 0.094, the epoch count caused the regression. If it stays 1.0, pc-r007's 0.094 was transient. |
| Medium | Add RESULTS.md entry for pc-r008 (R=8 arm) from the JSON | Notebooks are stale. The JSON is authoritative. RESULTS.md should document what was run. |
| Medium | Complete pc-r007 seed matrix (seeds 43, 44; both arms) | Needed to confirm pc-r007's 100% result is not seed-specific before building further on it. |
| Low | Add errata to RESULTS.md pc-r007 finding 7 (n_cycles vs n_recursions) | The correction exists in the pc-r008 JSON notes but has not propagated to the narrative record. |

---

## Confidence Assessment

**Overall Confidence: Medium**

**What limits confidence:**
1. **Notebook outputs are completely stale.** The only evidence is the JSON results file — no cross-check from a notebook run.
2. **RESULTS.md is not updated.** No narrative entry for pc-r008 to compare against.
3. **Three of four arms are missing.** The scientific question is 75% unanswered.
4. **halt_prob discrepancy vs pc-r007 is unexplained** in available documentation.

---

## Bottom Line

### What We Know
- PC-R008 is a compute-starve sweep over n_recursions ∈ {1, 2, 4, 8}, arm=absolute, seed=42, testing whether K=4 composition degrades with less inner compute per outer step.
- The R=8 arm completed on 2026-07-24, achieving 100% K=4 val accuracy for both TRM and baseline, with downward transfer to K=2 and K=3 at 100%.
- z_probe_acc is saturated at every step (≥0.999); 4-hop composition is encoded in z without explicit 4-hop z-supervision.
- The baseline per-step accuracy profile reads chance below step K−1 and 1.0 at step K−1, validating the evaluation methodology.
- The TRM shows halt_prob=1.0 at step 0 — a regression from pc-r007's 0.094 at step 0.
- n_cycles is confirmed inert in TinyRecursiveModel. True inner compute per outer step = n_recursions transformer applications.
- The notebook files contain stale outputs from a May 2026 Sudoku run — they are not evidence for pc-r008.
- RESULTS.md has no pc-r008 entry.

### What We Don't Know
- Whether K=4 accuracy degrades at R=4, R=2, or R=1 — the entire scientific payload of pc-r008.
- Whether the halt_prob regression (1.0 at step 0 in pc-r008 vs 0.094 in pc-r007) is caused by the 25 vs 15 epoch training duration, seed-level noise, or something else.
- Whether the TRM's outer recursion steps are doing genuine hop-by-hop compositional work, or whether all computation is packed into the first outer step's n_recursions inner passes.
- Whether the pc-r007 result is stable across seeds 43 and 44.

### Next Best Action
**Run the three missing pc-r008 arms (R=4, R=2, R=1) with the same configuration.** This is the single experiment that converts pc-r008 from a control arm that adds nothing over pc-r007 into the decisive test of whether inner recursion depth drives the compositional result. If K=4 holds at R=1 (one inner update per outer step), the outer supervision loop is the hop clock and the recursion mechanism is doing genuine iterative work. If K=4 collapses as R drops, the result is compute-dependent and the outer iteration is incidental — a finding that would force a significant reinterpretation of everything since pc-r005.
