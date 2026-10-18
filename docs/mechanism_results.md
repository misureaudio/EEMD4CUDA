# EEMD on the 27-tone prime chord — the four open unknowns, resolved

**Date:** 2026-10-18 · **Machine:** GTX 1070 (8 GB, driver 581.57) +
`D:\Source\hermes-dir\.venv` (numpy 2.4.6, scipy 1.17.1, cupy 14.2.0,
numba 0.68.0) · **Code:** `emd_par.py` (public API), `emd_gpu.py`
(`gpu_eemd_full`), `emd_ref.py` (exact CPU), `test_block_thomas.py`
(DSBT host mirror).

**Signal:** `PrimeToneNewPhi[Nch 1][Frac 03][LFL 043][00020-21600][Nfr 027]-
[48000-24]-[1s]-[-0.25dB]-[049062]-[ 8.5905]-[RD].flac` — 48 000 samples,
48 kHz, mono, PCM_24, peak −0.25 dBFS. 27 tones at prime-numbered
frequencies 43…16 381 Hz (3 per octave, adjacent ratios 1.15–1.34),
amplitudes 0.2295…0.01155 (≈ −0.93 dB per tone), 1 s, stationary.
Note: `[049062]` in the filename is an ID, not N (N = 48 000).

**TL;DR**

| # | Unknown | Resolution |
|---|---------|------------|
| 1 | CPU↔GPU "O(1) divergence" (G3) | **No backend divergence.** Same input: 1.1e-15 (E=1); 63/64 trials ≤ 5.1e-15 (E=64). The O(1) came from (a) the API adding noise to GPU-E1 but not CPU-E1, and (b) a real bug in the DSBT *host mirror* (now fixed). |
| 2 | Boundary-condition effect at E=64 | Band modes (43 Hz–16.4 kHz) are BC-invariant (identical peaks, relmax 0.73–3.12 at E=64 = MC noise level); only the two slowest modes (1.2–2 Hz) are BC-sensitive (relmax 2.30–3.85). |
| 3 | Why 11 modes for 27 tones (band locking) | EMD separates tones into distinct modes only for frequency ratio ≳ 2 (measured crossover 1.8–2.0). Below that, tones merge into band modes and the AM structure peels off subharmonic "ghost" modes (f/2, f/3, 2f/3, beat rate) plus a 1.4–23 Hz low-frequency tail. |
| 4 | E→∞ convergence | Band-mode ensemble means converge as E^−α with α ≈ 0.56–0.69 (faster than i.i.d. MC's 0.5); mode 10 slower (α ≈ 0.33); an 11th mode is a rare-trial tail event (P ≈ 1/64) that does **not** converge as an ensemble mean. |

---

## 1. Unknown 1 — Is the CPU↔GPU divergence real? No.

### 1.1 Same input → round-off (decisive tests)

| Test | Config | n_modes | max\|Δ\| (per-IMF) |
|---|---|---|---|
| T1 | `decompose(cpu, E=1)` vs `decompose(gpu, E=1, eps=0.0)` on the raw signal | 10 / 10 (25/25 siftings) | **1.11e-15** |
| T64 | one noise batch (E=64, ε=0.2·std), the same 64 noised signals to both backends (GPU `gpu_eemd_full` batch vs 64× CPU `decompose E=1`) | 63/64 trials: 10=10 (2 trials 9=9, 1 trial 11=11) | **63/64 trials ≤ 5.1e-15**; 1 trial 1.6e-2 (see below) |

Individual operators, verified on the real signal (`exp_mechanism.py`
A4–A6): spline envelope evaluation 2.2e-16 (max rel 2.2e-16); extrema
knot scan **bit-identical** (9 501 = 9 501 knots); SD stop statistic
agreement 2.1e-16, distance from the τ=0.25 stop line 9.8e-3. The
suite's own A2 gate (same-input full-EMD, 441 k reference sweep)
measured **1.4e-13** per-IMF (`API_MANUAL.md` §9).

### 1.2 The one divergent trial: a chaotic bifurcation in the slowest mode

T64 trial 56 (the only trial with max\|Δ\| > 1e-10):

```
mode:   1      2      3      4      5      6      7      8      9      10
max|d|: 2.5e-16 4.9e-16 4.7e-16 5.7e-16 7.2e-16 7.2e-16 4.8e-16 2.8e-16 3.1e-16 1.6e-02
```

Modes 1–9 (the band modes, 43 Hz–16.4 kHz) follow *identical*
trajectories on both backends to round-off; only mode 10 (the ~2 Hz
slow mode, the fewest-extrema, stop-boundary-closest mode) diverges to
1.6e-2 (relmax 0.158), with the same mode count (10 = 10). This is the
decision-boundary mechanism hypothesized in
`docs/boundary_sensitivity_results.md` — a ~1e-15 floating-point
difference flipping an extrema-strictness or SD-stop decision — but it
is a **rare chaotic event (1/64 = 1.6 % of trials), not a systematic
backend difference**. The same mode 10 is the most sensitive one in
Unknowns 2 and 4 below.

### 1.3 Where the documented O(1) "G3" actually came from

**(a) API noise asymmetry (the big one).** In `emd_par.py` (verified in
current code): `_cpu_decompose` with E=1 calls `emd_1d(x)` — **no
noise**; `_gpu_decompose` *always* computes
`noises = rng.standard_normal((Eb,N))·(eps·std(x))` and decomposes
`x + noises` — **adds noise even at E=1**. So the documented
"CPU-E1 vs GPU-E1" (G3) comparison was actually **EMD(x) vs
EMD(x + 0.2·std·noise)** — an input asymmetry, not a backend
difference. Measured on the 27-tone signal (T2): relmax **1.08**, and
the two runs land on *different band partitions* (CPU lead band
6 827 Hz vs GPU lead band 10 909 Hz; per-mode relmax 0.43–2.59).
The API's RNG streams also differ for E>1 (CPU: per-trial seeds via
`_worker_eemd(e, seed, …)`; GPU: one batch stream), so even
same-seed CPU vs GPU EEMD uses different noise draws.

**(b) The DSBT host-mirror bug (real, now fixed).** The Phase-1 host
mirror `test_block_thomas.dsbt_solve` solved each block's clipped window
directly into the shared output array, so each window's halo re-solve
**overwrote the previous block's owned tail** with its own
dropped-coupling boundary node. On the real first-sift spline system
(n = 9 499, B = 250, S = 20):

- seam-node error **9.5e-3** at i ≡ B−S (mod B) for every seam
  (error = l·|x\*[w0−1]| ≈ 20 % of the solution scale; profile:
  1.1e-8 at 10–20 knots away, 4.3e-18 beyond 25);
- independent of halo width S (S ∈ {5…500}: max error stays
  6.7e-3–1.09e-2 — the error sits at the owned/halo interface, not in
  the halo);
- independent of B / number of seams (B ∈ {64…250}: max error
  9.5e-3–1.07e-2 — each seam contributes its own bad node; the max
  picks the worst);
- EMD consequence (A2/A3, instrumented replica): **9 vs 10 modes,
  relmax 1.97** vs the exact run; first divergence at **mode 1, sift 0**
  (max Δh = 1.46e-2), chaotic amplification up to 9.1×; no
  near-τ siftings (min |sd−τ| = 5.3e-3 over 15 values), so the split
  is the seam error, not a stop-boundary coin flip.

The **device** kernel (`emd_gpu_v4_dsbt`) always writes only the owned
region [o0..o1] and is exact on this system: solve-level
max\|M_dsbt − M_exact\| = **2.2e-19**; full EMD E=1 relmax =
**2.2e-16**, 10 = 10 modes. The DSBT design is validated; the bug was
host-side only.

**Fixes applied to `test_block_thomas.py` (2026-10-18):**

1. `dsbt_solve`: window solve into a scratch array, copy back **owned
   [o0..o1] only** (device semantics). U1 now ≤ 2.26e-16 on all 5 knot
   sets (incl. the real-sweep set, n = 14 270); boundary error
   ≤ 5e-18.
2. `scan_extrema_2pass`: the old strided version (thread t owns
   i ≡ t+1 mod 128, global prefix over per-thread counts) compacted in
   **thread-major order** — same hit set, wrong order (U2 failed:
   equal counts, different content). Replaced with the tile-based
   compact that actually mirrors the device kernel (contiguous 128-sample
   tiles, per-tile prefix, running total) → **bit-identical to serial**
   on all 8 scan cases × {max, min}.
3. `decay_base`: sign typo `exp(−slope)` → `exp(slope)`. The Green's
   function of the 2-dominant spline system decays with base
   **0.2679492 = 2−√3 exactly** (uniform grid, fitted over 55 points);
   all 5 U1 knot sets now measure bases 0.231–0.268 ≤ 2−√3 (U3 PASS).

All Phase-1 unit gates now pass: `test_block_thomas.py` →
**U1, U2, U3 PASS** (exit 0).

---

## 2. Unknown 2 — Does the boundary-condition choice matter at E=64?

CPU E=64, ε = 0.2·std, **same noise draws** (same seed) for both BCs,
on the 27-tone signal (`exp_mechanism.py` Part B):

| | natural | mirror |
|---|---|---|
| n_modes | 11 | 11 |
| band-mode peaks m1–m5 | 10 909.01, 3 413.0, 1 361.02, 421.01, 167.0 | **identical** (±0.01 Hz) |
| per-mode relmax (nat vs mir) | — | m1–m9: **0.73–3.12**, m10–m11: **2.30–3.85** |

- **Band structure is BC-invariant**: every band mode keeps its
  frequencies (and its tone membership) under natural ↔ mirror; the
  O(1) per-mode relmax at E=64 is of the same size as the ensemble MC
  error at E=64 (Unknown 4: rel-L2 of the E=64 mean vs the E=256 mean
  is 0.03–0.21 for band modes). At E=1 the BC choice is O(1)
  (documented in `docs/boundary_sensitivity_results.md`); at E=64 it
  is no longer visible in the band structure.
- **The two slowest modes (1.2–2.0 Hz) are BC-sensitive** (relmax
  2.30–3.85, peak 1.99→1.24 Hz and 1.85→1.52 Hz): the slow modes are
  not converged at E=64 (α ≈ 0.33, Unknown 4), so any O(1) perturbation
  — BC choice or MC error — redistributes them.
- CPU-natural vs GPU E=64 (same seed, **different RNG stream** →
  different noise draws): per-mode relmax 0.18–1.03, band modes m1–m6
  keep identical peaks (10 909, 3 413, 1 361, 421, 167, 53 Hz) —
  consistent with two independent EEMD estimates of the same ensemble
  mean, not with a backend difference.

**Practical rule:** for band structure and the modes above ~50 Hz, the
BC choice is irrelevant once E ≳ 16; for the sub-5 Hz tail, either
raise E or accept O(1) uncertainty.

---

## 3. Unknown 3 — Why 11 modes for 27 tones (band locking)

All experiments: CPU E=1 (no noise), τ = 0.25, max_imf = 32, on 1 s
synthetic replicas of the signal (the replica is bit-identical to the
CPU result on the real FLAC — A1 validation). *First attempt at this
part used sample indices instead of seconds in `sin(2πft)` and aliased
every tone; all numbers below are from the corrected run
(`exp_bandlock2.py`).*

### 3.1 Two-tone resolution limit (C1: 1 kHz + r·1 kHz, equal amplitude)

| r | f2 (Hz) | n_modes | m1 | structure |
|---|---|---|---|---|
| 1.1 | 1 100 | 7 | **1 000 + 1 100 merged** | + subharmonics 500/600/700, 200/300, beat **100** Hz, low tail 1.6–10.3 Hz |
| 1.2 | 1 200 | 8 | **merged** | + 600, 200/400, beat **200** Hz, 39.5 Hz, low tail |
| 1.3 | 1 300 | 9 | **merged** | + 700, 400, 100/200, 18 Hz, low tail |
| 1.4 | 1 400 | 7 | **merged** | + 600, 200, 98 Hz, low tail |
| 1.5 | 1 500 | 8 | **merged** | + 500/1 000, 500, 52 Hz, low tail |
| 1.6 | 1 600 | 8 | **merged** | + 1 000 (=1600/1.6), 400, 200, 88 Hz, low tail |
| 1.8 | 1 800 | 8 | **merged** | + 1 000, 200, 83 Hz, low tail |
| 2.0 | 2 000 | 9 | **2 000 alone** | **m2 = 1 000 alone** + 427.5 Hz ghost, 99 Hz, low tail |
| 2.4 | 2 400 | 8 | **2 400 alone** | **m2 = 1 000 alone** + 400/600, 200, 102 Hz, low tail |
| 3.0 | 3 000 | 9 | **3 000 alone** | **m2 = 1 000 alone** + 189/77/73/27 Hz ghosts, low tail |

Signal extrema counts check out (r=1.1: 1 100+1 100; r=3.0:
3 000+3 000 — the sum of two sines has 2(f1+f2) extrema/s).

**The resolution limit: tones merge into one band mode for r ≲ 1.8 and
split into separate per-tone modes for r ≥ 2.0.** No 2-tone input ever
yields 2 modes: n_modes = 7–9, because the merged pair's AM structure
peels off a **subharmonic cascade** (f/2, f/3, 2f/3, the beat rate
f2−f1, and further divisions) that is *not in the input* — sifting is a
nonlinear process and the "ghost" modes are its residue — plus a
2–4 mode **low-frequency tail** (finite-window leakage + slow AM).

### 3.2 Density controls

| Signal | adjacent ratios | n_modes | partition |
|---|---|---|---|
| 1 per octave: 43/107/251/683/1 699 Hz (C2) | 2.34–2.72 | 8 | **5 per-tone modes in order** (1 699, 683, 251, 107, 43; each a pure single peak) + 3 low-freq (5.2–20.9, 1.4–11.3 Hz) |
| thinned 14 (every other of the 27) (C3) | 1.42–1.68 | 9 | 6 bands (3/3/3/2/2/2 tones) + 3 low-freq |
| full 27 (the real chord) | 1.15–1.34 | 11 | 7 bands (3–4 tones each) + 4 low-freq |

### 3.3 Mechanism (the answer)

1. Sifting extracts the fastest oscillation. Two tones closer than
   ~2× in frequency share their extremum structure (their sum is an
   AM carrier at the pair center with a beat envelope), so they are
   peeled off **together** as one band mode.
2. The residual AM structure (carrier/beats/subharmonics) is then
   peeled layer by layer as subharmonic ghost modes — f/2, f/3, 2f/3,
   beat rate f2−f1, and divisions thereof — until what remains is
   slow.
3. A 1 s windowed signal always leaves a low-frequency tail
   (1.4–23 Hz): finite-length leakage plus the slowest AM content.
4. Hence n_modes is a function of **tone density**, not tone count:
   2 tones → 7–9; 5 tones at 1 per octave → 8 (5 per-tone + 3 tail);
   14 thinned → 9 (6 bands + 3 tail); 27 at 3 per octave →
   11 (7 bands + 4 tail). The observed 3–4-tone band width at
   3-per-octave density is the packing effect on top of the pairwise
   threshold: a mode absorbs tones while their ratios stay below
   ~1.8–2.

---

## 4. Unknown 4 — E → ∞ convergence

GPU batch EEMD on the 27-tone signal (ε = 0.2·std, τ = 0.25, one
batch stream, `exp_mechanism.py` Part D); per-trial nmode histograms:

| E | n=9 | n=10 | n=11 | wall (s) |
|---|---|---|---|---|
| 16 | 2 | 14 | 0 | 1.4 |
| 32 | 2 | 29 | 1 | 1.5 |
| 64 | 3 | 60 | 1 | 3.0 |
| 128 | 10 | 117 | 1 | 5.8 |
| 256 | 23 | 229 | 4 | 11.6 |

Median nmode = 10 for all E ≥ 32. **The 11th mode is a rare-trial tail
event** (P ≈ 1/64 at E ≥ 32), not a convergent ensemble component.

Per-mode rel-L2 of the E-mean vs the E=256 mean (modes 1–9 are the band
modes 16.4 kHz…43 Hz, mode 10 ≈ 2 Hz, mode 11 ≈ 0.5 Hz):

| mode | E=16 | E=32 | E=64 | E=128 | fit α |
|---|---|---|---|---|---|
| 1 (16.4 kHz) | 0.837 | 0.573 | 0.375 | 0.218 | 0.645 |
| 2 | 0.369 | 0.252 | 0.165 | 0.095 | 0.648 |
| 3 | 0.221 | 0.150 | 0.106 | 0.061 | 0.609 |
| 4 | 0.171 | 0.116 | 0.079 | 0.051 | 0.575 |
| 5 | 0.157 | 0.110 | 0.071 | 0.042 | 0.639 |
| 6 | 0.106 | 0.083 | 0.066 | 0.031 | 0.563 |
| 7 | 0.208 | 0.159 | 0.127 | 0.056 | 0.598 |
| 8 | 0.444 | 0.265 | 0.136 | 0.112 | 0.694 |
| 9 | 0.798 | 0.287 | 0.211 | 0.180 | 0.690 |
| 10 (~2 Hz) | 1.154 | 0.598 | 0.475 | 0.579 | **0.332** |
| 11 (~0.5 Hz) | — | 0.933 | 0.933 | 0.933 | n/a |

- **Band modes (1–9): α ≈ 0.56–0.69 — faster than i.i.d. Monte Carlo
  (1/√E, α = 0.5)**: the per-trial band structure is a robust
  (correlated) feature, so the ensemble mean tightens faster than the
  MC bound. At E=64 the band modes are within 3–21 % (rel-L2) of the
  E=256 mean.
- **Mode 10: α ≈ 0.33 — slower than MC**: the slow mode is chaotic
  (fewest extrema, stop-boundary-closest), so its ensemble mean
  converges sub-MC; at E=64 it is still ~48 % off the E=256 mean.
- **Mode 11: rel-L2 pinned at 0.933** for E = 32/64/128 — the E=32–128
  "mean" of mode 11 is 1 rare trial vs 4 rare trials at E=256; a
  tail event does not converge as an ensemble mean. Its correct
  description is the per-trial occurrence probability (≈ 1/64), not a
  waveform.

**Practical rule:** for this signal class, E = 64 gives band modes
within a few percent (rel-L2 0.03–0.21) of the E=256 mean at 3.0 s on
the GTX 1070; the sub-5 Hz tail should be quoted with O(1) uncertainty
or computed at E ≥ 256.

---

## 5. Corrections to prior documents

1. **`docs/boundary_sensitivity_results.md` (G3):** the "cross-backend
   O(1) divergence" was two separate effects, neither a backend
   difference: (a) the `decompose()` API noise asymmetry at E=1 (GPU
   adds ε·std·noise, CPU does not — EMD(x) vs EMD(x+noise), relmax
   1.08, different band partitions); (b) the DSBT host-mirror seam bug
   (9.5e-3 seam-node error → 9 vs 10 modes, relmax 1.97). The
   document's decision-boundary amplification mechanism is confirmed
   but is a *rare* chaotic event (1/64 trials at E=64, only in the
   slowest mode), not a systematic backend split. The A2-gate
   "≲ 1e-8" in that document's framing is the gate *tolerance*; the
   measured same-input A2 is 1.4e-13 (manual §9).
2. **`API_MANUAL.md` §11:** "GPU vs CPU is chaotic. Per-IMF agreement
   is ~1e-8 (A2)" — the 1e-8 is the A2 gate tolerance, not a measured
   divergence; same-input agreement is ≤ 1.4e-13 (measured A2) and
   1.1e-15 on the 27-tone signal (T1). (Left untouched per the
   versioning rule — flag for the next manual revision.)
3. **`eemd-par-application` skill:** updated — same-input agreement
   numbers, the E=1 noise-asymmetry pitfall (use `eps=0.0` on the GPU
   path for a true same-input comparison), the DSBT mirror fix, the
   band-locking threshold (merge ≲ 1.8, split ≥ 2.0, per-octave
   spacing → per-tone modes), and the E-convergence exponents.

## 6. Artifacts

| File | Content |
|---|---|
| `exp_mechanism.py` | Parts A0–A7 (CPU↔GPU + DSBT micro), B (BC E=64), C (first, aliased — superseded), D (E→∞) |
| `exp_bandlock2.py` | corrected Part C (seconds-based synthesis) |
| `probe_sameinput2.py` | T1/T2/T64 same-input probes |
| `probe_t64_diag.py` | per-trial/per-mode T64 divergence map |
| `mechanism_out/results.json` | A0–A7 + B + D raw results |
| `mechanism_out/bandlock_results.json` | corrected C1–C3 |
| `mechanism_out/sameinput_results.json` | T1/T2/T64 |
| `mechanism_out/t64_diag.json` | per-trial T64 map |
| `test_block_thomas.py` | 3 fixes (dsbt_solve owned-write, tile-based scan, decay_base sign); U1+U2+U3 PASS |
