# Boundary-condition & endpoint sensitivity — experiments

**Date:** 2026-10-18 · **Follow-up to:** `prime_tone_eemd_results.md` (first EEMD run)
**Run script:** `../exp_boundary_sensitivity.py` · **Data:** `../boundary_sensitivity_out/results.json`
**Signal:** same 27-tone FLAC (N=48 000, 48 kHz, endpoints x[0]=+0.090370, x[-1]=+0.199143 — non-zero).

**Why this experiment.** The first run's G3 showed CPU-E1 vs GPU-E1 disagree at
O(1) in every mode on the 27-tone signal, while the suite's A2 gate (per-IMF
≲ 1e-8) holds on the sparse reference sweep. EMD/EEMD sensitivity to boundary
conditions is well-known in general, but how it behaves in *this exact context*
(GPU DSBT kernel vs CPU Thomas, natural-BC both sides, dense multi-tone input,
1 s window) was unknown. These experiments measure it.

---

## 1. Three sensitivity scales, all measured (CPU E=1 unless noted)

| Perturbation | Scale | Measured effect |
|---|---|---|
| **BC choice**: `end='natural'` vs `end='mirror'` (same backend, same input) | macroscopic | 10 modes both; per-mode relmax **0.56–1.86** (max 1.86, mode 4). Band peaks **identical** in both (same 6 tone bands + 4 low-freq modes) — the BC change redistributes energy *within* bands, it does not re-partition them |
| **Random input perturbation** x + s·std(x)·noise, s = 1e-16…1e-9 | 1e-16…1e-9 · std | **Linear, no cliff**: max relmax 4.3e-15 → 1.6e-8, amplification factor ≈ 14–43 (≈17) at every scale. n_modes = 10 throughout |
| **Backend difference** (CPU exact Thomas vs GPU DSBT; same BC=natural, same input, E=1) | ~1e-16 internal state (DSBT seam ≤ 6.9e-18·C) | **O(1) in every mode**: relmax 0.43–2.59, 11–90 % of samples > 0.1·max; band boundary moves (mode 1: CPU 6827–10909 Hz vs GPU 10909–16381 Hz); difference spread over the whole time axis (not edge-localized) |
| **DC offset** x + 3.6e-13 (control) | 1e-12 · std | IMFs identical to **2.8e-15**, n_modes equal, residual shifts by exactly +3.614e-13 — DC-invariant, machinery validated |

**Reading.** The decomposition is locally well-conditioned as a map on inputs
(condition number ≈ 17, linear to s = 1e-9·std). Yet a ~1e-16-scale *internal*
difference (backend) produces O(1) partition differences — far beyond what the
input condition number would predict (~1e-15). The backend divergence is
therefore not a small-input-perturbation effect: the two backends land on
opposite sides of sift *decision* boundaries (extrema strictness, SD stop)
during the ~20 siftings — a path bifurcation, not a continuous deviation
(bifurcation mechanism inferred from rows 1–3; the O(1) differences themselves
are measured). BC choice is the macroscopic version of the same phenomenon:
it changes the edge envelopes, flips decisions, and redistributes within bands
at O(1).

## 2. Tukey windowing sweep (r = taper fraction of N; r=0 = raw)

Measured per window (E=1 CPU/GPU for the G3 comparison; E=64 GPU for stability):

| r | xw[0], xw[-1] | 0.43 Hz AM in input | G3 max relmax (CPU-E1 vs GPU-E1) | E=64 n_modes | E=64 band modes 1–7 |
|---|---|---|---|---|---|
| 0 (raw) | 0.090 / 0.199 | 7.5e-3 | **2.59** (mode 2) | 11 | (baseline) |
| 0.005 | 0 / 0 | 5.2e-3 | **10.45** (mode 1) | 10 | unchanged |
| 0.01 | 0 / 0 | 2.7e-3 | **11.13** (mode 3) | 11 | unchanged |
| 0.02 | 0 / 0 | 1.3e-3 | **6.80** (mode 5) | 11 | unchanged |
| 0.05 | 0 / 0 | 4.9e-4 | **18.45** (mode 4) | 11 | unchanged |

Per-mode G3 relmax (CPU-E1 vs GPU-E1):

| mode | r=0 | r=0.005 | r=0.01 | r=0.02 | r=0.05 |
|---|---|---|---|---|---|
| 1 | 1.05 | **10.45** | 1.05 | 1.05 | 1.06 |
| 2 | **2.59** | 8.72 | 1.11 | 1.01 | 1.01 |
| 3 | 0.86 | 9.18 | **11.13** | 2.80 | 1.17 |
| 4 | 0.43 | 2.39 | 0.41 | 4.70 | **18.45** |
| 5 | 1.08 | 0.90 | 2.45 | **6.80** | 10.72 |
| 6 | 1.20 | 1.64 | 4.38 | **5.46** | 0.68 |
| 7 | 0.77 | 1.00 | 1.02 | 1.32 | 1.26 |
| 8 | 0.94 | 0.85 | 1.03 | 0.74 | 1.32 |
| 9 | 1.05 | 0.80 | 1.05 | 1.49 | 0.89 |
| 10 | 1.28 | 1.76 | 0.54 | 0.77 | 2.13 |

Additional measurements:

* **Windowing changes the E=1 CPU band partition itself**: at r=0.005 the
  CPU's mode 2 shifts from 2731/3413/4093 Hz to 2039/2731/3413 Hz, and mode 3
  from 853/1021/1361 to 683/853/1021 (band boundaries move with the window).
* **The E=64 ensemble band structure is invariant to windowing**: modes 1–7
  peaks are identical for all r (10909/13649/16381 | 3413/4093/5449 |
  1361/1699/2039 | 421/509/683 | 167/211/251 | 53/61/83 | 43±0.43); only the
  low-frequency modes 8–11 reshuffle (e.g. mode 8: 18.6/20.5/21.1 Hz at r=0 →
  15.4/17.6/18.6 at r=0.005 → 11.7/17.6/18.6 at r=0.01).
* **Windowing reduces the input's low-frequency content**: 0.43 Hz AM
  7.5e-3 → 4.9e-4 (r=0→0.05); 2–21 Hz components mostly decrease (4.65 Hz:
  6.9e-3 → 6.3e-4; 18.59 Hz: 8.4e-3 → 2.2e-3), with a non-monotonic rise at
  2–3 Hz for small r (2.06 Hz: 1.4e-3 → 4.1e-3 at r=0.02 → 2.6e-4 at r=0.05).
* **Windowing increases BC sensitivity**: natural vs mirror (CPU E=1) max
  relmax 1.86 (raw) → **4.38** (r=0.01), concentrated in mid-bands (modes
  3–6: 4.38, 4.23, 1.95, 3.62).

**Reading.** Forcing zero endpoints with a Tukey taper does **not** smooth the
decomposition: single-trial cross-backend divergence *increases* by up to 7×
(max 18.5×), BC sensitivity increases, and the E=1 band partition shifts. The
stable object — the E=64 ensemble band structure — was already stable without
windowing, so windowing buys nothing for band stability and costs single-trial
reproducibility. (Hypothesis for the increase, not confirmed: near the taper
the spline envelope is set by interior knots while the signal → 0, so the
envelope/signal ratio is large there and the ~1e-16 internal differences are
relatively amplified at the edges.)

## 3. Practical conclusions (measured basis)

1. **On dense multi-tone signals, do not expect A2-level (1e-8) cross-backend
   agreement.** Expect O(1) partition differences at E=1 (band boundaries
   included). The ensemble (E≥64) is what makes the result usable: band
   structure, mode count, and partition of unity are stable across E=64→128,
   across window choices, and bit-identical for same seed.
2. **Document the BC used.** Natural-BC (GPU-only option) and mirror-BC
   (CPU-only) give different within-band compositions at O(1); they agree on
   band structure for this signal.
3. **Do not window this class of signal to force zero endpoints** expecting
   smoother EMD behavior — measured worse on every single-trial metric.
4. **DC offsets are harmless** (IMFs invariant to 3e-15, DC lands in the
   residual) — no need to detrend before EMD.
5. **Gates appropriate for this signal class**: partition of unity,
   same-seed determinism, residual content, max_imf_reached, E-stability of
   n_modes + band peaks. Per-tone IF gates (first run's G5/G6) are invalid
   for band modes.

## 4. What remains unknown (honestly)

* Whether the DSBT seam error (structured, at block rims, repeated per sift)
  amplifies coherently more than random 1-ulp noise — the perturbation curve
  used random perturbations; a structured-perturbation probe was not run.
* Whether the E=64 ensemble band structure is also invariant to BC choice
  (BC was only compared at E=1).
* The mechanism by which the 27-tone envelope locks onto log-adjacent clusters
  (first report, §7.1).
* E→∞ convergence rate (first report, §7.3).

## 5. Artifacts

| Artifact | Location |
|---|---|
| Experiment script | `exp_boundary_sensitivity.py` |
| Machine-readable data | `boundary_sensitivity_out/results.json` |
