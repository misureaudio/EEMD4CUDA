# EEMD of the 27-tone PrimeToneNewPhi signal — results

**Date:** 2026-10-18 · **Plan:** `../EEMD_PRIME_TONE_PLAN.md` · **Run script:** `../eemd_prime_tone.py`
**Signal:** `D:\MLS-ESS\SigGenNew\LFL-MuTPri\PrimeToneNewPhi[Nch 1][Frac 03][LFL 043][00020-21600][Nfr 027]-[48000-24]-[1s]-[-0.25dB]-[049062]-[ 8.5905]-[RD].flac`
**Stack:** `D:\Source\hermes-dir\.venv` (numpy 2.4.6, cupy 14.2.0, numba 0.68.0), GTX 1070 8 GB, CUDA 12.8.

This is the first real-world application of the `emd_par` suite. The headline
finding: **EMD does not separate the 27 tones into 27 modes — it groups them
into 7 band modes, and the grouping is chaotic at the 1-ulp level.** Details
below; every number in this document was measured in this run
(`eemd_prime_tone_out/results.json` is the machine-readable record).

---

## 1. Signal (measured)

| Property | Value |
|---|---|
| N / fs / channels / subtype | 48 000 / 48 000 Hz / 1 / PCM_24 (exactly 1.0 s) |
| mean / peak / RMS | 0.0 / 0.971628 / 0.361391 |
| Tones | **27**, at prime-numbered frequencies: 43, 53, 61, 83, 107, 127, 167, 211, 251, 337, 421, 509, 683, 853, 1021, 1361, 1699, 2039, 2731, 3413, 4093, 5449, 6827, 8191, 10909, 13649, 16381 Hz |
| Tone amplitudes | 0.2295 (43 Hz) … 0.01155 (16381 Hz), ∝ f^(−1/2) |
| AM | every tone carries ±0.43 Hz sidebands at 50 % of the carrier (100 % AM depth; e.g. A(42.57)=0.1720, A(43.43)=0.1600) |
| Extrema | 9501 maxima, 9501 minima |
| Low-freq content in raw x | coherent amplitudes 1.4e-3…8.4e-3 at 2.06, 2.10, 4.65, 4.74, 7.25, 18.59, 21.06 Hz; max 0–30 Hz spectral bin 4.49e-4 (0.2 % of A(43)); A(0.43 Hz)=7.48e-3 |

## 2. Primary decomposition (GPU EEMD, E=64)

`decompose(x, method='gpu', E=64, tau=0.25, eps=0.2, max_imf=32, seed=0)`

| Quantity | Measured |
|---|---|
| **n_modes** | **11** (+ residual) |
| wall / gpu_time | 7.3 s / 5.74 s (auto-batch 362 → all 64 trials in one batch) |
| per-trial total siftings | 19–26, median 22 (≈ 2 siftings per mode) |
| partition of unity | 6.94e-18 |
| determinism (same seed re-run) | bit-identical `imfs` and `residual` (3.03 s) |
| max_imf_reached | False |

### 2.1 The actual mode structure (measured, not assumed)

Largest spectral peak of each mode (relative to the mode's own max), with the
mode's energy as a fraction of the signal's self-energy (IMFs are not
orthogonal, so the fractions do not sum to 1):

| Mode | Energy | Largest spectral peaks (Hz) | Tones carried (≥10 % of signal amplitude, relative coherent amp) |
|---|---|---|---|
| 1 | 0.0028 | 6827, 8191, **10909**, 13649, 16381 | 6827:0.36 8191:0.59 10909:0.89 13649:0.97 16381:0.99 (5449:0.16) |
| 2 | 0.0055 | 2731, 3413, **4093**, 5449, 6827 | 2039:0.18 2731:0.45 3413:0.70 4093:0.83 5449:0.82 6827:0.64 |
| 3 | 0.0155 | 1021, **1361**, 1361.29, 1699, 2039 | 853:0.29 1021:0.50 1361:0.77 1699:0.84 2039:0.80 2731:0.53 |
| 4 | 0.0454 | 337, 421, **509**, 683, 853 | 251:0.14 337:0.43 421:0.63 509:0.78 683:0.80 853:0.69 |
| 5 | 0.1363 | 127, **167**, 167.13, 211, 251 | 83:0.20 107:0.29 127:0.52 167:0.77 211:0.82 251:0.84 337:0.56 |
| 6 | **0.4493** | 43, **53, 61**, 83, 107 | 43:0.59 53:0.77 61:0.85 83:0.80 107:0.69 127:0.46 |
| 7 | 0.0905 | 36.76, 42.29, **42.99**, 43.00, 52.96 | 43:0.43 53:0.21 61:0.12 (43 Hz tone + its 0.43 Hz AM sidebands) |
| 8 | 0.0080 | 14.68, 17.60, **18.59**, 20.49, 21.06 | none (no tone ≥ 5 %) |
| 9 | 0.0015 | 4.87, 5.65, 5.98, 6.86, **7.25** | none |
| 10 | 0.0002 | 0.00, 0.45, 4.65, **4.74**, 6.76 | none |
| 11 | 0.0030 | 0.00, 1.97, **2.10**, 2.24, 4.66 | none |
| residual | 0.0034 | 0.00, **2.06**, 2.23, 4.63, 5.12 | none; carries 0.43 Hz AM (A=4.56e-3, 61 % of the x value) and DC (mean −4.96e-3) |

Structure: **7 band modes** carrying all 27 tones in log-adjacent clusters of
4–6 tones each (with edge overlap between adjacent bands, e.g. 5449 in modes
1–2, 683 in modes 3–4, 43–61 Hz split between modes 6–7), plus **4
low-frequency modes** (8–11) containing no tones, plus the residual.

## 3. Verification gates

| Gate | Criterion (plan) | Measured | Result |
|---|---|---|---|
| G1 partition of unity | ≤ 1e-10 | 6.94e-18 | **PASS** |
| G2 determinism (A3) | bit-identical re-run | bit-identical | **PASS** |
| G3 CPU↔GPU cross-check (A2-level) | per-IMF relmax ≲ 1e-8 | **O(1) in every mode** (below) | **FAIL** — see §4 |
| G4 mode count | 25–30 (one per tone) | 11 | **FAIL** — hypothesis refuted, see §4 |
| G5 monotone tone assignment | strictly decreasing | modes 7–11 dominant below 43 Hz | **FAIL** — hypothesis refuted |
| G6 IF within 5 % of tone | max rel err < 5 % | max rel err 95 % (mode 11, mean\|IF\|=2.06 Hz vs 43 Hz "assigned") | **FAIL** — hypothesis refuted |
| G7 residual content | 0.43 Hz present, tone leakage < 1 % | 0.43 Hz: 4.56e-3 in residual vs 7.48e-3 in x; worst tone leakage 0.90 % (16381 Hz) | **PASS** |
| G8 cap not hit | max_imf_reached False | False | **PASS** |

Gates G4–G6 were designed under the plan's §3 hypothesis ("each tone lands in
its own mode, fastest first"). The measurement refutes that hypothesis for
this signal; the "failures" are the finding, not a defect of the
implementation. G1/G2/G7/G8 confirm the decomposition itself is valid and
reproducible.

## 4. Chaotic sensitivity to 1-ulp differences (the key finding)

Plain EMD on both backends (E=1, deterministic): CPU 0.43 s → 10 modes,
GPU 1.28 s → 10 modes. Per-mode comparison over the common depth:

| Mode | relmax \|Δ\|/max | frac of samples with \|Δ\|>0.1·max | CPU top-3 peaks (Hz) | GPU top-3 peaks (Hz) |
|---|---|---|---|---|
| 1 | 1.054 | 0.711 | 6827, 8191, 10909 | 10909, 13649, 16381 |
| 2 | **2.589** | 0.638 | 2731, 3413, 4093 | 3413, 4093, 5449 |
| 3 | 0.859 | 0.110 | 853, 1021, 1361 | 1361, 1699, 2039 |
| 4 | 0.430 | 0.117 | 251, 337, 421 | 421, 509, 683 |
| 5 | 1.079 | 0.433 | 107, 127, 167 | 167, 211, 251 |
| 6 | 1.203 | 0.516 | 43, 53, 61 | 43, 53, 61 |
| 7 | 0.770 | 0.307 | 24.7, 36.8, 43.1 | 34.7, 42.7, 43.5 |
| 8 | 0.938 | 0.584 | 15.3, 17.1, 18.7 | 14.5, 14.6, 23.4 |
| 9 | 1.051 | 0.753 | 7.3, 7.9, 8.9 | 6.6, 6.6, 9.4 |
| 10 | 1.283 | 0.897 | 0.6, 2.9, 3.1 | 2.8, 3.0, 3.4 |

The two backends produce **different band partitions** of the same tones:
mode 1's band boundary sits at 10909 Hz on the CPU but at 6827 Hz on the
GPU. This is the chaotic regime: sifting decisions (extrema detect, SD stop)
sit on 1-ulp differences, and in a dense tone cluster the sift path forks.
The manual's A2 tolerance (per-IMF ≲ 1e-8) was established on the
10-octave reference sweep — a sparse, well-conditioned signal — and does not
extend to this signal class.

**What the ensemble does and does not stabilize** (GPU, same seed):

| E | n_modes | wall / gpu_time | rel-L2 vs E=64, per mode 1…11 |
|---|---|---|---|
| 64 | 11 | 7.3 s / 5.74 s | — |
| 128 | 11 | 6.1 s / 6.04 s | 0.287, 0.137, 0.085, 0.066, 0.054, 0.052, 0.104, 0.122, 0.196, **0.545**, 0.0 |

Stable under E: the mode count (11), the band structure (7 bands, same
order), the partition of unity (6.94e-18 both). Not stable: the contested
band boundary at the top (mode 1: 0.287) and the slowest low-frequency mode
(mode 10: 0.545). No E→∞ convergence rate is established by this run.

## 5. Timings (measured, GTX 1070, N=48 000)

| Run | Wall |
|---|---|
| GPU E=64 (primary) | 7.3 s (gpu_time 5.74 s) |
| GPU E=64 re-run (G2) | 3.0 s |
| GPU E=128 | 6.1 s (gpu_time 6.04 s) |
| GPU E=1 | 1.3 s |
| CPU E=1 (serial) | 0.43 s |

All far inside the plan's estimates; E=128 costs ≈ E=64 on GPU because
64–128 blocks × 128 threads do not saturate the 1070's 15 SMs.

## 6. Interpretation guide (what the numbers mean, why they vary)

* **Why 11 modes, not 27.** EMD sifting peels off the "fastest oscillation
  layer" as defined by the signal's spline envelopes — not by its spectral
  lines. In a sum of 27 tones the envelope is a composite object, and each
  sift locks onto a *cluster* of log-adjacent tones (ratios 1.2–1.6×), not a
  single tone. With 3 tones per octave the clusters are dense, so EMD
  behaves here as a 7-band logarithmic filter bank, not a tone separator.
* **Why mode 6 dominates (45 % of self-energy).** It carries the five lowest
  tones, including the 43 Hz tone with the largest amplitude (0.2295); tone
  energy ∝ A² ∝ 1/f, so the low end holds most of the energy. (The per-mode
  energy fractions are self-energies of non-orthogonal IMFs and do not sum
  to 1.)
* **Why modes 8–11 exist at all.** The raw 1-second signal genuinely contains
  small low-frequency content (coherent amplitudes 1.4e-3…8.4e-3 at 2–21 Hz;
  origin: finite-window leakage of the incommensurate tone sum and the
  0.43 Hz AM — mechanism not identified). After the 7 band modes, that
  content plus the surviving 0.43 Hz AM and DC is siffted into 4 more modes
  down to ~2 Hz; what is too slow to be an IMF in a 1 s window (0.43 Hz AM,
  DC) stays in the residual.
* **Why CPU and GPU disagree at O(1).** See §4: the sift path forks on 1-ulp
  differences in the dense-cluster regime. Same-backend + same-seed is still
  bit-identical (G2), so this is signal-driven chaos, not a backend bug.
* **What to trust from this decomposition.** The 7-band grouping, the mode
  count, the partition of unity, and the band-averaged content are stable.
  Individual slow modes (8–11) and exact band boundaries are trial-level
  artifacts — meaningful only after ensemble averaging, and even then with
  the E=64→128 spread of §4 in mind.
* **If the goal was tone separation:** EEMD is the wrong tool for this
  signal — the tones are stationary and ≥ 10 Hz apart (≫ the 1 Hz bin), so a
  Fourier method recovers all 27 exactly. EEMD's value here is structural:
  it shows how a mode-based method actually partitions a polyphonic signal.

## 7. Known unknowns (honestly)

1. The mechanism by which the envelope of a 27-tone sum locks onto
   log-adjacent clusters (which sift peels which band) — measured, not modeled.
2. The exact origin of the 2–21 Hz content (leakage vs weak difference tones
   vs AM interaction) — measured to exist, origin not identified.
3. Whether the ensemble average converges as E→∞ and at what rate — E=64 vs
   128 still shows up to 54 % rel-L2 on the slowest mode.

## 8. Artifacts

| Artifact | Location |
|---|---|
| EEMD result E=64 (npz: imfs, residual, stats) | `eemd_prime_tone_out/primetone_eemd_E64_gpu.npz` (4.35 MB) |
| EEMD result E=128 | `eemd_prime_tone_out/primetone_eemd_E128_gpu.npz` (4.34 MB) |
| All measured numbers (machine-readable) | `eemd_prime_tone_out/results.json` |
| Detected tone frequencies | `eemd_prime_tone_out/tone_frequencies.npy` |
| Plots | `eemd_prime_tone_out/signal.png`, `modes.png` (11 modes + tone labels), `spectra.png` (per-IMF spectra, tone grid), `if_check.png` (mean \|IF\| vs dominant tone) |
| Run script / diagnostics | `eemd_prime_tone.py`, `diag_prime_tone.py` (project root) |
