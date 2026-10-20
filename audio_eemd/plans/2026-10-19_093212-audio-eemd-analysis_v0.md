# Audio EEMD Analysis Tool — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** A CLI Python program that reads a user-assigned FLAC/WAV file, extracts its metadata, decomposes the signal with EEMD (reusing the `eemd4cuda` suite), and emits a reproducible bundle of numerical data and plots — including standard time-frequency frameworks (STFT, per-IMF Hilbert spectrum, Morlet CWT, and the `tftb` toolbox suite) for cross-validation.

**Architecture:** A thin pipeline package `audio_eemd/` at the workspace root (`C:\Users\MATTIA\source\hermes-dir\audio_eemd\`) with five modules — audio I/O, EEMD stage, spectra/TFR, report (numerical outputs), plots — driven by one argparse CLI. The EEMD core is **not reimplemented**: the stage module wraps `eemd4cuda.emd_par.decompose()` (auto CPU/GPU, DSBT kernel) and only adds I/O, segmentation, and analysis layers.

**Tech Stack:** Python 3.14 (workspace `.venv`), soundfile 0.14, numpy 2.4.6, scipy 1.18, matplotlib 3.11.1, pyloudnorm, tqdm; GPU path via cupy 14.1.1 + numba 0.66.0 on an RTX A2000 12 GB (1 device confirmed). `tftb 0.3.0.dev0` (dev build) is used **only** behind a validation gate and a `--tftb` flag.

---

## 1. Context & verified facts

| Fact | Value | Source |
|---|---|---|
| EEMD entry point | `emd_par.decompose(x, method='auto', E=64, eps=0.2, tau=0.25, max_sifts=50, max_imf=-1, min_extrema=4, end='natural', n_workers=None, seed=0, parallel=True, batch=None, nthreads=128) -> EMDResult` | `eemd4cuda/API_MANUAL.md` §3, verified by `inspect.signature` |
| `EMDResult` | `imfs [n_modes, N]`, `residual [N]`, `n_modes`, `stats`, `recon_error()`, `save/load(.npz)` | API manual §3.2 |
| GPU caveats | natural-BC only, `min_extrema=4` hardcoded, `max_imf=-1` = cap 64 (RuntimeWarning + `stats['max_imf_reached']`); `end`/`min_extrema` silently ignored on GPU | API manual §11 |
| GPU footprint model | ≈ `N·8·(4+max_imf) + 10·(N/2)·8` bytes/trial → ≈ 84 MB/trial at N=441 000, max_imf=16 | API manual §5.3 |
| Chaos | same seed → bit-identical (A3); different seed → different mode depth (A4); cross-backend agreement ~1e-8, not bit-equal | API manual §11.3 |
| Test audio | `PrimeToneNewPhi[Nch 1][Frac 003][00043-16381][Nfr 027]-[44100-16]-[1800s]-[-0.25dB]-[ 8.4809]-[RD].flac` — FLAC, PCM_16, 44 100 Hz, mono, 1 800 s, 79 380 000 frames, peak ≈ 0.972, 1-s RMS ≈ 0.366, low-frequency content (42–167 Hz region) | `sf.info` probe |
| venv (verified importable) | numpy, scipy, matplotlib, soundfile, pandas, cupy (1 device), numba, pyloudnorm (`Meter` OK), tqdm, click, rich, jupyterlab, pillow, `tftb 0.3.0.dev0` | `pip list` + import probes |
| tftb layout (dev0) | `tftb.processing.analysis` (`tfristft`, `momttfr`, `holder`, `tfrideal`, …), `tftb.processing.reassigned` (`spectrogram`, `morlet_scalogram`, `smoothed_pseudo_wigner_ville`, `reassigned_gabor`, `pseudo_wigner_ville`, `pseudo_page`), `tftb.processing.wavelet`, `tftb.processing.ambiguity`, `tftb.processing.plotifl` | `pkgutil` probe |
| Missing from scipy | `scipy.signal.cwt` does **not** exist in scipy 1.18 → Morlet CWT must be implemented manually (rfft) or via tftb | probe |
| GPU | NVIDIA RTX A2000 12 GB, 1 device, cupy 14.1.1 | `nvidia-smi`, `cp.cuda.runtime.getDeviceCount()` |
| Repo state | **No git repository** in the workspace → no commit steps; version by file content only | `ls .git` |
| Test-segment policy | **Per user instruction: test with 1–5 s clips** from the PrimeTone FLAC (e.g. `--start 10 --duration 5`), never the full 30-min file | user |

**Hard constraint — file length.** A full 30-min file is 79.4 M samples: one float64 copy is 635 MB, and per-trial GPU footprint would be ~1.4 GB × E — infeasible for any meaningful E. The tool therefore operates on a **user-selected segment** (`--start/--stop` or `--duration`). v1 refuses files > 30 s without an explicit segment; chunked full-file decomposition is a documented v1.1 extension (see §9).

**EEMD scale invariance.** EEMD noise is `eps·std(x)`-scaled, so the decomposition is invariant to input scaling — no normalization is applied; the original scale is preserved in all outputs and reported in metadata.

---

## 2. CLI design

Entry point: `audio_eemd/audio_eemd.py` (argparse, stdlib only — no new dependencies).

```
.venv/Scripts/python.exe audio_eemd/audio_eemd.py FILE [options]

Positional:
  FILE                  .flac or .wav path (any extension accepted by soundfile is
                        also accepted, but .flac/.wav are the documented inputs)

Segment (default: whole file if ≤ 30 s, else error):
  --start SEC           start time in seconds (default 0)
  --stop SEC            stop time in seconds (default = start + duration or EOF)
  --duration SEC        alternative: window length from --start
  --max-file-sec SEC    raise if segment longer than this (default 30)

EEMD:
  --method {auto,cpu,gpu}   default auto
  --E N                     ensemble size, default 64 (E=1 → plain EMD)
  --eps F                   noise amplitude, fraction of std(x), default 0.2
  --tau F                   SD stop criterion, default 0.25
  --max-imf N               default -1 (full; GPU caps at 64)
  --max-sifts N             default 50
  --seed N                  default 0
  --mem-limit-gb F          GPU device-memory ceiling for auto-batching

Spectra:
  --nperseg N               STFT window, default 4096
  --noverlap N              default nperseg//2
  --wavelet                 enable Morlet CWT figure (Tier 3)
  --tftb                    enable tftb module (Tier 3, requires gate G1 to pass)

Numerical extras:
  --csv                     also write per-IMF CSV dumps (Tier 4)
  --entropy                 per-IMF sample entropy (Tier 4)
  --fractal                 per-IMF Higuchi fractal dimension (Tier 4)

Output:
  --out DIR                 output dir, default ./eemd_out/<stem>_<start>-<stop>s/
  --fmt {png,pdf,svg}       figure format, default png
  --dpi N                   default 150
```

Exit codes: `0` success · `2` bad usage / bad segment · `3` file unreadable · `4` EEMD failure (recon error or backend error) · `5` tftb gate failed (only with `--tftb`).

---

## 3. Proposed outputs (the framework menu)

Each item: definition, output artifact, cost, rationale. **Core = always produced; flag = behind a CLI flag.**

### Tier 0 — Metadata (always) → `report.json` block `metadata` + `metadata.csv`

* Container: format, subtype (PCM_16…), sample rate, channels, frames, duration, bit depth, encoder info (`sf.info`).
* Audio stats of the **segment**: peak, RMS, crest factor, integrated LUFS (pyloudnorm `Meter(sr).integrated_loudness`, default blocking — valid for ≥ 400 ms; label as approximate for < 2 s), spectral centroid and flatness of the segment (periodogram).
* *Why:* every analysis downstream is meaningless without the exact signal description; LUFS gives a perceptual loudness anchor the user asked metadata to cover.

### Tier 1 — EEMD core (always) → `imfs.npz`, `report.json` block `decomposition`

* `imfs.npz`: `x` (segment, original scale), `t`, `imfs [n_modes, N]`, `residual`, `n_modes`, scalar `stats` (method, E, tau, seed, per-trial sifting counts, gpu_time_s, max_imf_reached).
* **Partition-of-unity check** `recon_error()` = max|x − (Σ imfs + residual)| — written to report; a hard gate (fail exit 4 if > 1e-6·max|x|).
* Per-IMF table (one row per mode k): `std`, **energy fraction** E_k = Σ imf_k² / Σ x², **dominant frequency** (periodogram peak of imf_k), zero-crossing count, sifting counts (from `stats`).
* Residual stats (same row set) — for a 16-bit file the residual/last IMFs sit near the quantization floor (≈ −96 dBFS); the report annotates which IMFs are above/below the quantization-noise estimate 2⁻¹⁵·√3·RMS so the user doesn't read quantization noise as content.
* *Why:* the decomposition itself plus its self-verification and a per-mode "what is this mode" table.

### Tier 2 — Standard time-frequency frameworks (always) → figures + `stft.npz`, `hilbert.npz`

The cross-validation layer: EEMD is a time-varying, adaptive, nonlinear decomposition; fixed-basis TFRs are the reference the user can reason about.

1. **STFT (scipy.signal.stft, Hann, 50 % overlap, nperseg default 4096)** — of the original segment **and per IMF**.
   * `stft.npz`: `freqs`, `times`, `Zxx_orig [F, T]`, `Zxx_imf [k, F, T]`.
   * Figures: `fig_03_stft_original.png` (log-magnitude spectrogram), `fig_04_stft_imfs.png` (grid, one spectrogram per mode, shared color scale).
   * *Why:* the user's named example framework; per-IMF STFTs are the standard way to show what each mode carries and expose mode mixing (an IMF with multiple STFT ridges = mode mixing).
2. **Hilbert spectrum / instantaneous frequency (per IMF)** — analytic signal `z_k = imf_k + i·H(imf_k)` (scipy.signal.hilbert), instantaneous frequency `ω_k(t) = d/dt arg z_k`, instantaneous amplitude `A_k(t) = |z_k|`; Hilbert spectrum = `A_k(t)·ω_k(t)` ridges.
   * `hilbert.npz`: per-k `inst_freq [N]`, `inst_amp [N]`; stats: median/IQR of `ω_k` (positive part only).
   * Figure: `fig_05_hilbert.png` (per-IMF IF ridges over time + amplitude envelope).
   * *Why:* the canonical EEMD companion (Wu–Yeh–Lan 2001); the IF statistics in the per-mode table come from here. Caveat (stated in the report): IF of a noisy/complex IMF is ill-conditioned — this is expected and is precisely what the ensemble averaging in EEMD is for.
3. **Energy distribution** — `fig_06_energy.png`: bar of E_k per mode + cumulative line, log scale.
   * *Why:* one-glance answer to "where is the signal's energy across scales".

### Tier 3 — Extended frameworks (flag-gated, **recommended**)

1. **`--wavelet`: Morlet CWT scalogram** of the original segment. Manual FFT-based Morlet (no dependency; `scipy.signal.cwt` does not exist in scipy 1.18): scales from 100 Hz to Nyquist, ~40 log-spaced, `f_c = 1` (cycles), `ψ_a(t) = π^{-1/4} e^{i 2 f_c t/a} e^{-t²/(2a²)}`; `C(a, t)` via rfft of `x·ψ_a` with zero-padding for linear convolution. Figure `fig_07_wavelet.png` (log |C| scalogram, time × log-frequency).
   * *Why:* second reference framework with adaptive time-frequency resolution; contrasts directly with EEMD's data-driven basis.
2. **`--tftb`: tftb toolbox suite** (dev0 — behind gate G1, §7). One figure `fig_08_tftb.png` with up to 3 panels: `tftb.processing.reassigned.spectrogram` (tftb STFT), `smoothed_pseudo_wigner_ville` (SPWVD — Cohen-class, negative values clipped, note in caption), `reassigned_gabor` (reassignment — the "known framework" with the sharpest ridges). Plus optional numerical: `tftb.processing.analysis.holder` (Hölder-exponent estimate) per IMF when `--fractal` is also set.
   * *Why:* the user asked for "known analysis frameworks"; tftb is the standard Python TFR toolbox and is already in the venv. Dev0 → never a hard dependency: if gate G1 fails the flag errors out (exit 5) and core outputs are untouched.

### Tier 4 — Numerical extras (flag-gated, optional)

1. **`--csv`**: per-IMF CSV `imf_<k>.csv` (`time_s, value`) + `residual.csv`. ~2 MB per mode at 5 s/44.1 kHz — hence off by default.
2. **`--entropy`**: per-IMF sample entropy (m=2, r=0.2·std(imf_k)) on a decimated series (≤ 4096 points, stride = N//4096) so cost is bounded; one column in the per-mode table.
3. **`--fractal`**: per-IMF Higuchi fractal dimension (m = 1…10, on decimated series) — one column in the per-mode table. (tftb `holder` used instead when `--tftb` is active.)
4. **Coarse per-IMF STFT CSV** (`--csv`): binned `freq_bin × time_bin` log-magnitude tables (≤ 256 × 256) for spreadsheet use without 2 MB files.
* *Why:* machine-readable analysis beyond the figures, kept opt-in because each adds nontrivial compute or file bulk.

### Figure inventory (core, in production order)

| File | Content |
|---|---|
| `fig_01_signal.png` | segment waveform + residual + running reconstruction (Σ imfs₁..k), 3 rows |
| `fig_02_imfs.png` | stacked IMFs, one row per mode, shared x-axis, auto y-limits per row |
| `fig_03_stft_original.png` | log-magnitude STFT of the segment |
| `fig_04_stft_imfs.png` | per-IMF STFT grid, shared color scale |
| `fig_05_hilbert.png` | per-IMF instantaneous-frequency ridges + amplitude envelope |
| `fig_06_energy.png` | energy-fraction bars + cumulative |
| `fig_07_wavelet.png` | (flag) Morlet scalogram |
| `fig_08_tftb.png` | (flag) tftb spectrogram / SPWVD / reassignment |

All figures: matplotlib Agg backend, consistent style module (one `style.py`-like dict in `plots.py`), title embeds the identifying metadata line (file stem, segment, sr, E, tau, seed), 150 dpi PNG by default.

### Output directory layout

```
eemd_out/<stem>_<start>-<stop>s/
  report.json            # metadata + parameters + per-mode table + recon error + timings + caveats
  metadata.csv           # flat metadata (also in report.json)
  imfs.npz               # x, t, imfs, residual, n_modes, stats
  stft.npz               # freqs, times, Zxx_orig, Zxx_imf
  hilbert.npz            # per-k inst_freq, inst_amp
  imf_<k>.csv            # (--csv)
  fig_01..06.<fmt>       # core figures
  fig_07/08.<fmt>        # flagged
```

`report.json` top-level keys: `metadata`, `segment`, `parameters`, `environment` (backend resolved, E, n_modes, per-trial sifting, gpu_time_s, max_imf_reached), `decomposition` (per-mode table + recon_error), `timings` (per-stage wall clock), `caveats` (static list: GPU natural-BC/min_extrema, seed, quantization-floor annotation, LUFS approximation).

---

## 4. Module design

```
audio_eemd/
  __init__.py            # version string
  audio_io.py            # loading, metadata, segmentation, mono
  eemd_stage.py          # emd_par bootstrap + decompose wrapper
  spectra.py             # STFT, Hilbert IF, Morlet CWT, entropy, Higuchi
  report.py              # report.json / metadata.csv / npz assembly
  plots.py               # all figures
  audio_eemd.py          # CLI entry (argparse + orchestration, tqdm stages)
  tests/
    test_audio_io.py
    test_eemd_stage.py
    test_spectra.py
    test_report.py
    test_plots.py
    test_cli.py
    test_tftb_gate.py
```

Key signatures:

```python
# audio_io.py
@dataclass
class AudioFile:
    path: Path; sr: int; channels: int; frames: int; duration: float
    fmt: str; subtype: str; x: np.ndarray          # float64, shape (C, N), original scale
    meta: dict                                      # raw sf.info fields
    def to_mono(self) -> np.ndarray                 # mean over channels
def load_audio(path: str) -> AudioFile              # ValueError → exit 3
def select_segment(a: AudioFile, start: float, stop: float) -> np.ndarray  # float64 [N]
def audio_stats(x: np.ndarray, sr: int) -> dict     # peak, rms, crest, lufs, centroid, flatness
```

```python
# eemd_stage.py
@dataclass
class Decomp:
    x: np.ndarray; imfs: np.ndarray; residual: np.ndarray; n_modes: int
    stats: dict; recon: float; wall_s: float
def run_eemd(x, method='auto', E=64, eps=0.2, tau=0.25, max_imf=-1,
             max_sifts=50, seed=0, mem_limit_gb=None) -> Decomp
    # sys.path bootstrap for eemd4cuda (idempotent, relative to this file:
    # Path(__file__).resolve().parent.parent / "eemd4cuda");
    # timing via time.perf_counter; recon gate: recon > 1e-6*max|x| → RuntimeError
```

```python
# spectra.py
def stft_block(x, sr, nperseg=4096, noverlap=None) -> (freqs, times, Zxx)
def hilbert_if(imf, sr) -> (inst_freq, inst_amp)      # scipy.signal.hilbert
def if_stats(inst_freq) -> dict                        # median, iqr over inst_freq > 0
def morlet_cwt(x, sr, f_lo=100.0, n_scales=40, f_c=1.0) -> (freqs, C)   # rfft-based
def sample_entropy(x, m=2, r_frac=0.2, max_len=4096) -> float
def higuchi_dim(x, kmax=10, max_len=4096) -> float
```

`report.py`: `build_report(audio, seg_info, params, env, decomp, mode_table, timings, caveats) -> dict`, `write_report(outdir, ...)`, `write_imfs_npz(outdir, ...)`, `write_stft_npz`, `write_hilbert_npz`, optional CSV writers. `plots.py`: one function per figure, `def fig_0N_*(ctx) -> Path`, where `ctx` is a dataclass bundling everything; `render_all(ctx, enabled) -> list[Path]`.

---

## 5. Step-by-step tasks

No git repo in the workspace → **no commit steps**; each task ends with a passing test run. All commands run from `C:\Users\MATTIA\source\hermes-dir` with `.venv/Scripts/python.exe` (alias `PY` below). The PrimeTone FLAC path must always be single-quoted in bash (brackets are glob characters).

### Task 0: Scaffold + dependency gate

**Objective:** package skeleton and a hard import gate so later tasks never discover a missing dependency at runtime.

**Files:** create `audio_eemd/__init__.py`, `audio_eemd/tests/__init__.py`, `audio_eemd/tests/conftest.py`.

**Step 1:** `conftest.py` defines fixtures:

```python
import sys, numpy as np, pytest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FLAC = str(ROOT / "PrimeToneNewPhi[Nch 1][Frac 003][00043-16381][Nfr 027]-[44100-16]-[1800s]-[-0.25dB]-[ 8.4809]-[RD].flac")

@pytest.fixture(scope="session")
def clip1():          # 1 s at t=10, mono float64
    import soundfile as sf
    x, sr = sf.read(FLAC, start=10*sr, stop=11*sr, dtype="float64", always_2d=True)
    return x[:, 0], sr

@pytest.fixture(scope="session")
def clip5():          # 5 s at t=10
    import soundfile as sf
    x, sr = sf.read(FLAC, start=10*sr, stop=15*sr, dtype="float64", always_2d=True)
    return x[:, 0], sr

@pytest.fixture
def two_tones():      # mode-mixing reference: 100 Hz + 105 Hz, 4 s @ 8 kHz
    sr = 8000; t = np.arange(4*sr)/sr
    return (np.sin(2*np.pi*100*t) + np.sin(2*np.pi*105*t))/2, sr
```

**Step 2:** `audio_eemd/tests/test_deps.py` — one test per hard dependency (`soundfile, numpy, scipy, matplotlib, pyloudnorm, tqdm`, and `import emd_par` after the eemd4cuda bootstrap); tftb import is **not** asserted here.

**Step 3:** Run `PY -m pytest audio_eemd/tests/test_deps.py -v` → 7 passed.

### Task 1: `audio_io.py`

**Objective:** load FLAC/WAV, full metadata, segment selection, mono, audio stats.

**Files:** create `audio_eemd/audio_io.py`, `audio_eemd/tests/test_audio_io.py`.

**Step 1 (failing tests):**

```python
def test_load_flac_metadata(clip1):
    from audio_eemd.audio_io import load_audio
    from pathlib import Path
    a = load_audio(FLAC)
    assert a.sr == 44100 and a.channels == 1 and a.frames == 79_380_000
    assert a.subtype == "PCM_16" and a.fmt.startswith("FLAC")

def test_segment_shape(clip1):
    from audio_eemd.audio_io import load_audio, select_segment
    a = load_audio(FLAC)
    x = select_segment(a, 10.0, 15.0)
    assert x.shape == (5 * 44100,) and x.dtype == np.float64

def test_segment_bounds(clip1):
    from audio_eemd.audio_io import load_audio, select_segment
    a = load_audio(FLAC)
    with pytest.raises(ValueError):
        select_segment(a, 10.0, 1e9)      # stop beyond EOF
    with pytest.raises(ValueError):
        select_segment(a, -1.0, 5.0)      # negative start

def test_stats_keys(clip1):
    from audio_eemd.audio_io import load_audio, select_segment, audio_stats
    a = load_audio(FLAC)
    s = audio_stats(select_segment(a, 10.0, 15.0), a.sr)
    for k in ("peak", "rms", "crest", "lufs", "centroid", "flatness"):
        assert k in s and np.isfinite(s[k])
    assert s["peak"] < 1.0 and s["rms"] < s["peak"]
```

**Step 2:** run → FAIL (module missing).

**Step 3 (implementation):** per §4 signatures. Notes: `load_audio` uses `sf.info` + `sf.read(..., dtype="float64", always_2d=True)` (full file read is fine: 79.4 M samples = 635 MB float64 — acceptable once; segment is then sliced; if memory is a concern later, add streaming read). `audio_stats`: LUFS via `Meter(sr).integrated_loudness(x)` guarded by `try/except` → `None` on failure (still recorded, caveat added); centroid = `Σ f|X(f)| / Σ |X(f)|` over `rfft(x·hann)`.

**Step 4:** run → 4 passed.

### Task 2: `eemd_stage.py`

**Objective:** wrap `emd_par.decompose` with bootstrap, timing, recon gate.

**Files:** create `audio_eemd/eemd_stage.py`, `audio_eemd/tests/test_eemd_stage.py`.

**Step 1 (failing tests):**

```python
def test_emd_single_cpu(two_tones):
    from audio_eemd.eemd_stage import run_eemd
    x, sr = two_tones
    d = run_eemd(x, method="cpu", E=1, tau=0.25)
    assert d.n_modes >= 2
    assert d.recon < 1e-10 * x.max()
    assert d.imfs.shape[1] == len(x)

def test_eemd_ensemble_cpu(two_tones):
    from audio_eemd.eemd_stage import run_eemd
    x, sr = two_tones
    d = run_eemd(x, method="cpu", E=8, seed=0)
    assert d.n_modes >= 1 and d.recon < 1e-8 * x.max()
    assert d.stats.get("E") == 8

def test_seed_determinism(two_tones):
    from audio_eemd.eemd_stage import run_eemd
    x, sr = two_tones
    a = run_eemd(x, method="cpu", E=4, seed=7)
    b = run_eemd(x, method="cpu", E=4, seed=7)
    assert np.array_equal(a.imfs, b.imfs)
```

**Step 2:** run → FAIL.

**Step 3 (implementation):** bootstrap inserts `<root>/eemd4cuda` into `sys.path` once (guard flag on the module); `run_eemd` calls `M.decompose(x, method=..., E=E, eps=eps, tau=tau, max_imf=max_imf, max_sifts=max_sifts, seed=seed, mem_limit_gb=mem_limit_gb)`, wraps into `Decomp`, computes `recon = res.recon_error()`, raises `RuntimeError` if `recon > 1e-6 * max|x|`. `stats` is copied verbatim (it already carries method/E/sifting/gpu_time_s/max_imf_reached).

**Step 4:** run → 3 passed. (These are CPU-only; GPU is exercised in Task 7.)

### Task 3: `spectra.py`

**Objective:** STFT, Hilbert IF, Morlet CWT, entropy, Higuchi.

**Files:** create `audio_eemd/spectra.py`, `audio_eemd/tests/test_spectra.py`.

**Step 1 (failing tests):**

```python
def test_stft_tone_peak():
    import numpy as np
    from audio_eemd.spectra import stft_block
    sr = 8000; x = np.sin(2*np.pi*1000*np.arange(sr)/sr)
    f, tt, Z = stft_block(x, sr, nperseg=1024)
    assert np.argmax(np.abs(Z).mean(axis=1)) == np.argmin(np.abs(f - 1000))

def test_hilbert_if_sine():
    import numpy as np
    from audio_eemd.spectra import hilbert_if, if_stats
    sr = 8000; x = np.sin(2*np.pi*1000*np.arange(sr)/sr)
    w, A = hilbert_if(x, sr)
    s = if_stats(w)
    assert abs(s["median"] - 1000) < 10 and np.isclose(A.std(), 0, atol=0.05)

def test_morlet_chirp_ridge():
    import numpy as np
    from audio_eemd.spectra import morlet_cwt
    sr = 8000; t = np.arange(sr)/sr
    f = 100 + 900*t/t[-1]                      # 100 → 1000 Hz linear chirp
    x = np.sin(2*np.pi*f*t)
    freqs, C = morlet_cwt(x, sr, f_lo=50, n_scales=32)
    ridge = freqs[np.argmax(np.abs(C), axis=0)]
    assert ridge[0] < 300 and ridge[-1] > 700  # ridge follows the chirp

def test_entropy_ordering():
    import numpy as np
    from audio_eemd.spectra import sample_entropy, higuchi_dim
    rng = np.random.default_rng(0)
    tone = np.sin(2*np.pi*100*np.arange(4096)/1000)
    noise = rng.standard_normal(4096)
    assert sample_entropy(tone) < sample_entropy(noise)
    assert higuchi_dim(noise) > higuchi_dim(tone)
```

**Step 2:** run → FAIL.

**Step 3 (implementation):**
* `stft_block`: thin wrapper `scipy.signal.stft(x, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap or nperseg//2, boundary="pad", padded=False)`.
* `hilbert_if`: `z = scipy.signal.hilbert(imf)`; `phase = np.unwrap(np.angle(z))`; `inst_freq = np.gradient(phase, 1/sr)/(2π)`; `inst_amp = np.abs(z)`.
* `morlet_cwt`: scales `a_j = f_c·sr/f_j` for `f_j` log-spaced from `f_lo` to `sr/2`; kernel `ψ(t) = π^{-1/4} e^{i2f_c t/a} e^{-t²/2a²}`; per scale `C_j = ifft(rfft(x, 2N)·rfft(ψ_j, 2N))[:N]` (zero-padded circular = linear convolution), 40 scales × 220 500-sample rfft ≈ < 1 s.
* `sample_entropy`: standard m=2, r=0.2·std, on series decimated to ≤ 4096 (stride = max(1, N//4096)).
* `higuchi_dim`: K = 1…10, `L_k = (N-k)/k·Σ |x[i+ki]−x[i]|/(k·mean_d)`, `D = slope(log L vs log k)`.

**Step 4:** run → 4 passed.

### Task 4: `report.py`

**Objective:** assemble all numerical outputs.

**Files:** create `audio_eemd/report.py`, `audio_eemd/tests/test_report.py`.

**Step 1 (failing test):**

```python
def test_report_roundtrip(tmp_path, two_tones):
    from audio_eemd.eemd_stage import run_eemd
    from audio_eemd.report import build_report, write_all
    from audio_eemd.spectra import stft_block, hilbert_if, if_stats
    x, sr = two_tones
    d = run_eemd(x, method="cpu", E=4, seed=0)
    rep = build_report(audio_meta={"sr": sr, "fmt": "test"}, segment={"start": 0.0, "stop": 4.0},
                       params={"E": 4}, decomp=d, sr=sr)
    write_all(tmp_path, rep, x, d, extra={"stft": stft_block(x, sr, 1024),
                                          "hilbert": [hilbert_if(m, sr) for m in d.imfs]})
    import json
    r = json.loads((tmp_path / "report.json").read_text())
    for k in ("metadata", "segment", "parameters", "environment", "decomposition", "timings", "caveats"):
        assert k in r
    rows = r["decomposition"]["imf"]
    assert len(rows) == d.n_modes
    assert abs(sum(row["energy_frac"] for row in rows) + r["decomposition"]["residual"]["energy_frac"] - 1.0) < 1e-9
    assert (tmp_path / "imfs.npz").exists() and (tmp_path / "stft.npz").exists() and (tmp_path / "hilbert.npz").exists()
    assert (tmp_path / "metadata.csv").exists()
```

**Step 2:** run → FAIL.

**Step 3 (implementation):** per §3 Tier 1/2 artifacts. Per-mode row: `k, std, energy_frac, dominant_freq_hz, n_zerocross, if_median_hz, if_iqr_hz, sifting` (sifting from `d.stats` if per-mode available, else `null`). Quantization-floor annotation: `qfloor = 2**-15 * 1.732 * rms` (16-bit) — flag rows with `std < qfloor` as `"quantization_level": true` (only when `metadata.subtype == "PCM_16"`). Static `caveats` list per §3.

**Step 4:** run → 1 passed.

### Task 5: `plots.py`

**Objective:** the six core figures (+ two flagged).

**Files:** create `audio_eemd/plots.py`, `audio_eemd/tests/test_plots.py`.

**Step 1 (failing test):**

```python
def test_core_figures(tmp_path, two_tones):
    from audio_eemd.eemd_stage import run_eemd
    from audio_eemd.spectra import stft_block, hilbert_if
    from audio_eemd.plots import render_all
    import numpy as np
    x, sr = two_tones
    d = run_eemd(x, method="cpu", E=4, seed=0)
    ctx = make_ctx(x=x, sr=sr, decomp=d, stft=stft_block(x, sr, 1024),
                   hilb=[hilbert_if(m, sr) for m in d.imfs],
                   meta={"file": "test", "segment": "0-4s"}, outdir=tmp_path, fmt="png")
    paths = render_all(ctx, wavelet=False, tftb=False)
    names = {p.name for p in paths}
    assert {"fig_01_signal.png", "fig_02_imfs.png", "fig_03_stft_original.png",
            "fig_04_stft_imfs.png", "fig_05_hilbert.png", "fig_06_energy.png"} <= names
    for p in paths:
        assert p.stat().st_size > 10_000     # non-trivial content
```

**Step 2:** run → FAIL.

**Step 3 (implementation):** one function per figure per §3 inventory; shared style dict (figure dpi, font scale, colormaps: `magma` for spectrograms, `viridis` for scalograms); long signals downsampled for line plots (decimate to ≤ 20 000 points via `scipy.signal.decimate` or stride); per-IMF spectrogram grid uses `max(n_modes, 4)` rows with shared `vmin/vmax` (99th percentile of |Z| across all modes to avoid one outlier mode killing the scale). `fig_07` (Morlet) and `fig_08` (tftb) implemented but only called when flags set.

**Step 4:** run → 1 passed.

### Task 6: CLI `audio_eemd.py`

**Objective:** argparse entry, orchestration with staged progress, exit codes.

**Files:** create `audio_eemd/audio_eemd.py`, `audio_eemd/tests/test_cli.py`.

**Step 1 (failing tests):**

```python
import subprocess, sys
PY = sys.executable

def run_cli(*args):
    return subprocess.run([PY, str(Path(__file__).parents[1] / "audio_eemd.py"), *args],
                          capture_output=True, text=True, cwd=str(Path(__file__).parents[2]))

def test_help():
    r = run_cli("--help")
    assert r.returncode == 0 and "--tftb" in r.stdout

def test_missing_file(tmp_path):
    r = run_cli(str(tmp_path / "nope.flac"))
    assert r.returncode == 3

def test_bad_segment():
    r = run_cli(FLAC, "--start", "10", "--stop", "9")
    assert r.returncode == 2

def test_smoke_1s(tmp_path):
    r = run_cli(FLAC, "--start", "10", "--duration", "1", "--method", "cpu",
                "--E", "8", "--out", str(tmp_path / "smoke"))
    assert r.returncode == 0, r.stderr
    out = tmp_path / "smoke"
    for f in ("report.json", "imfs.npz", "stft.npz", "hilbert.npz",
              "fig_01_signal.png", "fig_06_energy.png"):
        assert (out / f).exists()
```

**Step 2:** run → FAIL.

**Step 3 (implementation):** per §2. Orchestration order with `tqdm`/plain stage prints: (0) load+metadata → (1) segment → (2) EEMD (longest stage; print backend + E + wall time) → (3) spectra → (4) report → (5) figures. Segment policy: no `--start/--stop/--duration` → whole file if `duration ≤ --max-file-sec` (30) else exit 2 with a message. Exit-code mapping per §2. `--tftb` runs gate G1 first (Task 8) → exit 5 on failure.

**Step 4:** run → 4 passed.

### Task 7: End-to-end validation on the PrimeTone FLAC (the user-specified test)

**Objective:** prove the pipeline on real audio: 1 s smoke (CPU) and 5 s standard run (auto → GPU), per the user's "1 to 5 seconds" instruction.

**Steps:**
1. `PY audio_eemd/audio_eemd.py 'PrimeToneNewPhi[...]RD].flac' --start 10 --duration 1 --method cpu --E 8 --out eemd_out/smoke_1s` → exit 0; inspect `report.json` (n_modes, per-mode table, recon error, LUFS ≈ sane for a −0.25 dBFS file, i.e. integrated loudness in the −25…−15 LUFS range for this material — sanity, not a gate).
2. `PY audio_eemd/audio_eemd.py 'PrimeToneNewPhi[...]RD].flac' --start 10 --duration 5 --E 64 --out eemd_out/std_5s` → exit 0; verify: `environment.backend == "gpu"` (A2000 present), `recon_error < 1e-6·max|x|`, all 6 core figures + 3 npz + report exist, `fig_04` has one row per mode, per-mode `if_median_hz` monotonically decreasing in k (expected for EEMD of music-like content; if violated, note in report rather than fail — mode mixing is legitimate), timings recorded.
3. Repeat step 2 with `--seed 1` → `imfs` differ (chaos check, mirrors gate A4) but report structure identical.
4. `--csv --entropy --fractal` run on the 1 s clip → CSVs exist, entropy/fractal columns populated.
5. Record wall times in `eemd_out/std_5s/report.json` and compare against the §8 budget.

### Task 8: tftb gate + `--tftb` module (optional tier)

**Objective:** validate the dev0 tftb build before any tftb output is trusted.

**Files:** create `audio_eemd/tests/test_tftb_gate.py`, extend `spectra.py` with `tftb_panel(x, sr)`.

**Step 1 (gate tests):**

```python
def test_tftb_stft_matches_scipy(two_tones):
    import numpy as np
    from tftb.processing.analysis import tfristft
    from audio_eemd.spectra import stft_block
    x, sr = two_tones
    f, tt, Z = stft_block(x, sr, 1024)
    # tftb tfristft signature: (x, nperseg, noverlap, window) — verify at runtime;
    # assert the two spectrograms correlate > 0.99 on |Z| after resampling tftb's
    # frequency grid onto scipy's (np.interp), same grid alignment.
    ...

def test_tftb_spwvd_runs(two_tones):
    from tftb.processing.reassigned import smoothed_pseudo_wigner_ville
    x, sr = two_tones
    T = smoothed_pseudo_wigner_ville(x, fs=sr)   # arg names verified at runtime
    assert np.isfinite(T).all() and T.shape[0] >= 128
```

**Step 2:** run. **If either fails → tftb module is dropped from the plan** (flag stays, returns exit 5 with a clear message); do not patch the dev0 package. If they pass:
**Step 3:** `tftb_panel` produces the 3-panel `fig_08_tftb.png` (tftb spectrogram, SPWVD clipped at 0 with caption note, reassignment); `render_all(..., tftb=True)` wires it in; `--fractal` uses `tftb.processing.analysis.holder` per IMF when `--tftb` is active.
**Step 4:** re-run Task 7 step 2 with `--tftb` → `fig_08_tftb.png` exists, exit 0.

### Task 9: README + caveats finalization

**Files:** create `audio_eemd/README.md`.

Content: usage examples (the exact Task 7 commands), full output map (§3/§4), parameter reference, caveats section (EEMD chaos & seed; GPU natural-BC / min_extrema=4 / max_imf cap 64; cross-backend non-bit-equality ~1e-8; 16-bit quantization floor in residual; LUFS approximate on short clips; tftb dev0 status), v1.1 roadmap (chunked full-file decomposition: fixed 10 s windows, 50 % overlap, per-chunk EEMD, concatenated per-mode statistics — results are window-dependent and must be labeled as such).

---

## 6. Tests / validation summary

| Level | What | Gate |
|---|---|---|
| Unit | Tasks 1–5 test files, run with `PY -m pytest audio_eemd/tests -x -q` | all pass |
| Integration | Task 6 CLI tests incl. 1 s end-to-end | exit codes + file existence |
| Real audio | Task 7: 1 s CPU smoke + 5 s E=64 GPU standard + seed-1 chaos check + extras run | recon < 1e-6·max, GPU backend used, figures non-trivial, monotone IF check noted |
| Optional | Task 8: tftb gate (scipy cross-check ρ>0.99) then 5 s + `--tftb` | exit 0, fig_08 present |
| Regression | re-run `eemd4cuda/test_block_thomas.py` (DSBT host gates, no GPU) once at the start to confirm the EMD library itself is green on this host | exit 0 |

---

## 7. Performance budget (RTX A2000 12 GB, 5 s @ 44.1 kHz = N = 220 500)

| Stage | Estimate | Basis |
|---|---|---|
| FLAC read + metadata | ~3–5 s (full-file read, 635 MB float64) | file 119 MB |
| EEMD E=64, max_imf=16, GPU | per-trial footprint ≈ N·8·20 + 10·(N/2)·8 ≈ 44 MB → 64 trials ≈ 2.8 GB (fits 12 GB); wall ≈ 10–30 s | API manual §5.3/§9, A2000-class results |
| STFT orig + per-IMF (≤ ~20 modes) | < 2 s | rfft |
| Hilbert IF per IMF | < 1 s | scipy.hilbert |
| Morlet CWT (40 scales) | < 2 s | rfft, zero-padded |
| Entropy + Higuchi (decimated ≤ 4096) | < 1 s | bounded |
| Figures | ~5–15 s | 6–8 figures, 150 dpi |
| **Total, standard run** | **~30–60 s** | |

CPU-only fallback (E=64, 8 workers): expect ~5–15× the EEMD stage (ProcessPool, dynamic scheduling) — still minutes, acceptable for the 1 s smoke with E=8.

---

## 8. Risks, tradeoffs, open questions

**Risks / mitigations**
1. **tftb is a dev0 build** — API names/args verified only by probing; the gate in Task 8 is a hard decision point (drop, don't patch). Core pipeline has zero tftb dependency.
2. **EEMD chaos** — different seeds change mode depth (documented A4); seed is always written to `report.json`; the seed-1 re-run in Task 7 makes this explicit instead of surprising.
3. **GPU caveats** — natural-BC only, `min_extrema=4` hardcoded, `max_imf=-1` caps at 64: all surfaced in `caveats` and `environment` blocks; users needing `end='mirror'` must pass `--method cpu` (documented).
4. **16-bit quantization** — last IMFs/residual sit at the −96 dBFS floor; the `quantization_level` flag in the per-mode table prevents misreading.
5. **Memory** — full-file float64 load (635 MB) + 5 s segment + GPU 2.8 GB: fine on this host; `--max-file-sec` guard prevents accidental 30-min GPU runs.
6. **Bracketed filename** — all shell examples single-quote it; code uses `pathlib` only.
7. **IF ill-conditioning** — instantaneous frequency of complex IMFs is noisy; reported as stats (median/IQR), not as point estimates; caveat text included.

**Open questions (recommendations in bold)**
1. Default segment when the user gives no `--start/--duration`? **First 10 s**, with the whole file allowed up to `--max-file-sec 30`.
2. Default E? **64** (suite default; 2.8 GB on this GPU).
3. Figure format default? **PNG 150 dpi** (PDF/SVG available via `--fmt`).
4. Should `--csv` be on by default? **No** (2 MB/mode).
5. v1.1 chunked full-file decomposition: include in v1 or defer? **Defer** (documented in README); chunking changes EEMD results (window edge effects, per-chunk mode count) and needs its own validation story.
6. Stereo/multi-channel inputs: v1 **downmixes to mono** (documented); per-channel decomposition is a v1.1 option.

---

## 9. Files likely to change (final inventory)

Create: `audio_eemd/__init__.py`, `audio_eemd/audio_io.py`, `audio_eemd/eemd_stage.py`, `audio_eemd/spectra.py`, `audio_eemd/report.py`, `audio_eemd/plots.py`, `audio_eemd/audio_eemd.py`, `audio_eemd/README.md`, `audio_eemd/tests/{__init__,conftest,test_deps,test_audio_io,test_eemd_stage,test_spectra,test_report,test_plots,test_cli,test_tftb_gate}.py`.
Output (generated): `eemd_out/<stem>_<start>-<stop>s/**`.
Untouched: everything in `eemd4cuda/` (consumed via import only).
