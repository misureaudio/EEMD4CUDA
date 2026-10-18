# Plan — EEMD of the PrimeToneNewPhi FLAC (LFL-MuTPri)

**Target signal.**
`D:\MLS-ESS\SigGenNew\LFL-MuTPri\PrimeToneNewPhi[Nch 1][Frac 03][LFL 043][00020-21600][Nfr 027]-[48000-24]-[1s]-[-0.25dB]-[049062]-[ 8.5905]-[RD].flac`

**Decomposition API.** `emd_par.decompose()` from this project (`API_MANUAL.md` §3).
**Status.** Executed 2026-10-18 — see `docs/prime_tone_eemd_results.md`. The §3
hypothesis (one mode per tone, ~27 modes) was **refuted by measurement**: the
decomposition yields **11 modes** — 7 band modes grouping the 27 tones into
log-adjacent clusters, plus 4 low-frequency modes with no tone content — and
the band partition is chaotic at the 1-ulp level (CPU-E1 vs GPU-E1 differ at
O(1) in every mode). Gates G4–G6 failed against the hypothesis, not against
the implementation; G1/G2/G7/G8 passed. The results document is authoritative.

---

## 1. Goal

Apply Ensemble Empirical Mode Decomposition (EEMD) to the FLAC, obtain the
ensemble-averaged IMFs + residual, verify the result against the manual's
correctness gates (§8/§11 of `API_MANUAL.md`), and save reproducible artifacts
(npz + plots + a results report citing only measured numbers).

---

## 2. Verified environment (measured, 2026-10-18)

| Item | Measured value |
|---|---|
| FLAC frames / sr / ch / subtype | **48 000** / 48 000 Hz / 1 / PCM_24 → `N = 48 000`, exactly 1.0 s |
| Peak / RMS | 0.971628 (−0.25 dBFS ✓) / 0.361391 |
| GPU | NVIDIA **GTX 1070**, 8192 MiB, driver 581.57 |
| venv | `D:\Source\hermes-dir\.venv` (standing rule: only this interpreter) |
| numpy / scipy / cupy / numba / soundfile / matplotlib | 2.4.6 / 1.17.1 / 14.2.0 / 0.68.0 / 0.14.0 / 3.11.2 |
| cupy on this Pascal card | **works** — `getDeviceCount()=1`, alloc+compute verified |
| numba.cuda on this Pascal card | **works** — `is_available()=True`, toy kernel verified |
| `CUDA_HOME` | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8` (satisfies NVVM bootstrap, manual pitfall 7; no local `cuda_libs/` dir) |
| CPU | `os.cpu_count() = 12` |
| Smoke: `decompose(x, method='cpu', E=1)`, N=4000, 3-tone | 0.01 s, 3 modes, `recon_error = 4.4e-16` |
| Smoke: `decompose(x, method='gpu', E=2, max_imf=4)`, N=4000 | 4.7 s incl. one-time JIT compile, 4 modes, `recon_error = 2.2e-16`, `stats['max_imf_reached']=False` |

**Conclusion:** both backends are usable; GPU is the production path and is the
primary method. The manual's benchmark numbers were taken on RTX 4070 / GB10 —
expect the 1070 (1:32 FP64, lower bandwidth) to be a few times slower per
trial, which is immaterial at this signal length.

---

## 3. Signal characterization (pre-run, measured)

- **27 tones** (matches `Nfr 027`), at **prime-numbered frequencies**:
  43, 53, 61, 83, 107, 127, 167, 211, 251, 337, 421, 509, 683, 853, 1021,
  1361, 1699, 2039, 2731, 3413, 4093, 5449, 6827, 8191 Hz, plus 3 more in
  8.2–21.6 kHz (below the 5 % detection floor of the quick FFT; Phase 1 of
  the run script confirms the full list).
- **Amplitude law** ≈ `A(f) ∝ f^(−1/2)` (power ∝ 1/f): 1.000 @ 43 Hz → 0.071 @ 8191 Hz.
- Each tone carries **constant 0.43 Hz AM sidebands** (±0.43 Hz around every
  tone, visible at the 43 Hz tone as 42.57 / 43.43 Hz).
- **Stationary**: first-half vs second-half spectrum correlation = 1.0000.
- Consequences for EMD:
  - Tones are well separated (ratios 1.2–1.33×) → each tone should land in its
    own mode, fastest first (mode 1 ≈ highest tone … last mode ≈ 43 Hz).
    Expected `n_modes` ≈ 27 (± a few; the 0.43 Hz AM completes < 1 cycle in
    1 s, is below `min_extrema = 4`, and should end up in the **residual**).
  - Stationarity + pure tones → ensemble averaging converges quickly; the
    default `E = 64` is ample (Monte-Carlo rate O(E^(−1/2)), manual §2).

---

## 4. API mapping (why these calls)

Public entry point only — manual §1 "rule of thumb": import **`emd_par`**,
call `decompose()`. Do **not** import versioned files (`emd_gpu_v1..v5`,
`emd_par_v0/v1/v2`) — manual pitfall 8.

| Need | API object | Manual ref |
|---|---|---|
| Load FLAC → float64 mono | `soundfile.read(path, dtype='float64')` | — (external) |
| EEMD, best backend | `M.decompose(x, method='gpu', E=64, tau=0.25, eps=0.2, max_imf=32, seed=0)` | §3.1 |
| Result container | `EMDResult`: `.imfs [n_modes, N]`, `.residual [N]`, `.n_modes`, `.stats`, `.recon_error()`, `.save(path)` | §3.2 |
| CPU cross-check | `M.decompose(x, method='cpu', E=8, tau=0.25, eps=0.2, max_imf=32, seed=0)` | §3.1, §4.3 |
| Determinism re-run | same GPU call again, same `seed` → must be bit-identical (gate A3) | §8 |

`emd_par` imports `emd_ref` at module level (numpy only) and `emd_gpu`
**lazily inside `_gpu_decompose`** — so the CPU path works even if the GPU
stack breaks, and the GPU import cost is paid only when used.

---

## 5. Parameter choices (with rationale)

| Param | Value | Rationale |
|---|---|---|
| `method` | `'gpu'` | Production path; N = 48 000 is small → one batch, seconds of compute. CPU kept as cross-check only. |
| `E` | **64** (primary), 128 (convergence check) | API default; O(E^(−1/2)) MC rate makes 64 stable for a stationary multi-tone; 128 compares against 64 to confirm convergence (Δ per IMF ≲ Monte-Carlo floor). |
| `eps` | 0.2 (default) | Standard EEMD noise amplitude (fraction of std(x)). |
| `tau` | 0.25 (default) | Rilling–Flandrin–Gonçalves SD tolerance; pure tones converge in few sifts. |
| `max_sifts` | 50 (default) | Safety cap; won't bind for tones. |
| `max_imf` | **32** | Expect ~27 modes; explicit cap avoids the GPU `-1` → hard-cap-64 semantics (pitfall 1) and trims per-trial buffers (`out` slab is `max_imf·N`). |
| `min_extrema` / `end` | leave defaults | **CPU-only** parameters — silently ignored on GPU (pitfall 2). Natural-BC default is fine here. |
| `seed` | 0 | Reproducibility; same-backend same-seed is bit-identical (gate A3). |
| `batch` / `mem_limit_gb` | `None` (auto) | Auto-batch reads cupy pool accounting (pitfall 6). Footprint: ≈ `N·8·(4+32) + 10·(N/2)·8` ≈ **15.7 MB/trial** (§5.3 model) → E=64 ≈ 1.0 GB ≪ 8 GB → single batch. |
| `n_workers` (CPU check) | `None` (auto → min((8−1)//2, 12) = 3) | Manual formula; small E, short wall time. |
| Input scaling | as read (peak 0.9716) | EMD is scale-invariant in mode structure; keeping native scale makes IMFs sum exactly to x and keeps `recon_error()` meaningful. |

---

## 6. Execution steps

All commands from `D:\Source\hermes-dir\EEMD4CUDA-main` with the venv
interpreter (imports `emd_par` / `emd_ref` / `emd_gpu` from this directory).

### Phase 0 — script scaffold

Create `eemd_prime_tone.py` in this directory. Layout:

```python
import sys, time, pathlib
import numpy as np, soundfile as sf
sys.path.insert(0, str(pathlib.Path(__file__).parent))   # emd_par lives here
import emd_par as M

FLAC = r"D:\MLS-ESS\SigGenNew\LFL-MuTPri\PrimeToneNewPhi[Nch 1][Frac 03][LFL 043]" \
       r"[00020-21600][Nfr 027]-[48000-24]-[1s]-[-0.25dB]-[049062]-[ 8.5905]-[RD].flac"
OUT  = pathlib.Path(__file__).parent / "eemd_prime_tone_out"
OUT.mkdir(exist_ok=True)

x, sr = sf.read(FLAC, dtype="float64")
assert x.ndim == 1 and sr == 48000 and len(x) == 48000
N = len(x)
```

### Phase 1 — characterization (for the record)

- `mean(x)` (DC → expected ≈ 0; any DC lands in the residual), peak, RMS.
- rFFT peak list (same method as §3) → confirm all **27** tone frequencies
  and the ±0.43 Hz sideband structure; save `tone_frequencies.npy`.
- Extrema count of `x` (sanity vs the GPU knot capacity `K = max(16, N//2)` —
  can never bind, manual §7).

### Phase 2 — primary EEMD (GPU)

```python
t0 = time.time()
r = M.decompose(x, method="gpu", E=64, tau=0.25, eps=0.2,
                max_imf=32, seed=0)
wall = time.time() - t0
r.save(OUT / "primetone_eemd_E64_gpu.npz")
```

Record: `wall`, `r.n_modes`, `r.stats` (`batch`, `gpu_time_s`,
`per_trial_total_sifts`, `max_imf_reached` — must be `False`),
`r.recon_error()`.

### Phase 3 — verification gates (pass/fail, from manual §8/§11)

| # | Gate | Pass criterion | Source |
|---|---|---|---|
| G1 | Partition of unity | `r.recon_error() ≤ 1e-10` (expect ~1e-16) | A2 |
| G2 | Determinism | re-run Phase 2 with same `seed` → `np.array_equal` on `imfs` **and** `residual` | A3 |
| G3 | CPU cross-check | `decompose(x, method='cpu', E=8, seed=0, max_imf=32)`; compare over common depth `min(n_modes_gpu, n_modes_cpu)`: per-IMF `max|Δ| / max|imf_gpu|` **≲ 1e-8** — *not* bit-equality (chaotic amplification, pitfall 3) | A2, pitfall 3 |
| G4 | Mode count | `25 ≤ n_modes ≤ 30` (27 tones; ± for AM/DC leakage) | §3 |
| G5 | Mode–tone assignment | per-IMF rFFT argmax ∈ detected tone set; order monotone decreasing in frequency (fastest first) | §3 |
| G6 | Instantaneous frequency | for each mode, Hilbert (scipy.signal.hilbert) → mean |IF| within 5 % of its assigned tone frequency (Bedrosian sanity) | — |
| G7 | Residual content | residual spectrum: 0.43 Hz AM peak (±0.5 Hz) + DC present; no tone energy > 1 % of that tone's amplitude | §3 |
| G8 | Cap not hit | `stats['max_imf_reached'] is False` | pitfall 1 |

### Phase 4 — convergence check (GPU)

```python
r128 = M.decompose(x, method="gpu", E=128, tau=0.25, eps=0.2,
                   max_imf=32, seed=0)
```

Report per-IMF relative L2 between `r` (E=64) and `r128` over the common
depth; expect a drop consistent with O(E^(−1/2)) (≈ 2^(−1/2) × the E=32-vs-64
gap if that is also run). Save as `primetone_eemd_E128_gpu.npz`.

### Phase 5 — artifacts & report

- Plots (matplotlib, PNG into `eemd_prime_tone_out/`):
  1. `signal.png` — x(t) + residual(t) (zoomed).
  2. `modes.png` — small-multiples of all IMFs (time).
  3. `spectra.png` — per-IMF amplitude spectra, marked with the tone grid.
  4. `if_check.png` — instantaneous frequency per mode vs assigned tone.
- `docs/prime_tone_eemd_results.md` (create `docs/` in this project) —
  **measured numbers only**: runtime, n_modes, per-gate results, per-mode
  tone table (mode → tone Hz → mean IF → max|Δ| vs CPU), residual findings.
  No anecdotes, no unmeasured claims.

---

## 7. Runtime & memory estimates (order of magnitude)

| Run | Estimate | Basis |
|---|---|---|
| GPU E=64, N=48 000 | **≈ 1–3 s compute** + one-time kernel JIT ≈ 30–60 s (cached per N) | 4070 `T_block` = 0.50 s @ N=441k (manual §9) → ≈ 0.05 s @ 48k; 1070 slower (1:32 FP64) → ×3–5 margin; 64 blocks × 128 threads fit 15 SMs in one batch |
| GPU E=128 | ≈ 2–6 s compute | same, doubled |
| CPU E=8 (G3) | ≈ 10–60 s | ~100 sifts/trial × ~50 ms @ N=48k, 3 workers (auto) |
| GPU E=64 device memory | ≈ 1.0–1.2 GB | §5.3 footprint model, `max_imf=32` |

All estimates are well inside the 8 GB VRAM and a few-minute wall budget.

---

## 8. Pitfalls to respect (manual §11, filtered to this run)

1. **`max_imf` semantics differ by backend** — set it explicitly (32); never
   rely on `-1` on the GPU (hard cap 64 + possible `RuntimeWarning`).
2. **`end` / `min_extrema` are silently ignored on GPU** (natural-BC,
   min_extrema=4 hardcoded) — do not pass non-default values with
   `method='gpu'`; if mirror-BC is ever wanted, that is a CPU run.
3. **GPU vs CPU is not bit-equal** — chaotic amplification ×10²–10³ per full
   decomposition; gate G3 uses a 1e-8-level tolerance, never `allclose`
   defaults.
4. **`gpu_eemd_full` trims to max(nmode)** — irrelevant here (we go through
   `decompose`, which averages per mode over the trials that produced it),
   but G3 must compare over the *common* depth.
5. **cupy pool accounting** — leave `batch=None`; `emd_par` reads the pool's
   cached bytes, not just `memGetInfo()` (auto-batch would silently degrade
   to 1 otherwise).
6. **NVVM bootstrap on Windows** — `CUDA_HOME` is set (verified); if a
   `numba.cuda` link error ever appears, check that
   `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin` still
   contains `nvvm.dll` / `cudart*.dll`.
7. **Do not import versioned files** (`emd_gpu_v2_dsbt..py` … `_v5`,
   `emd_par_v*`) — review-trail snapshots only.
8. **The filename `[049062]` is not the frame count** — actual N = 48 000
   (verified). Any script that trusts the name will be off by 1 062 samples.
9. **The 0.43 Hz AM is sub-cycle in a 1 s window** — do not expect it as an
   IMF; gate G7 checks it lands in the residual.

---

## 9. Deliverables

| Artifact | Location |
|---|---|
| Run script | `EEMD4CUDA-main/eemd_prime_tone.py` |
| EEMD result (E=64, primary) | `EEMD4CUDA-main/eemd_prime_tone_out/primetone_eemd_E64_gpu.npz` |
| EEMD result (E=128, convergence) | `…/primetone_eemd_E128_gpu.npz` |
| Plots (4) | `…/signal.png`, `modes.png`, `spectra.png`, `if_check.png` |
| Results report (measured numbers only) | `EEMD4CUDA-main/docs/prime_tone_eemd_results.md` |

**Out of scope** (optional follow-ups if requested): CEEMDAN comparison
(CPU-only, `emd_ref.ceemdan`, §4.3), deeper Bedrosian/HT analysis
(`ht_bedrosian/`), the sibling 44.1 kHz FLAC in the same folder.
