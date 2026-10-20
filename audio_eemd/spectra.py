"""Time-frequency building blocks (plan Task 3).

Tier 2 wires `stft_block` + `hilbert_if`/`if_stats` into analyze().
Tier 3: `morlet_cwt` + the gated tftb suite (`check_tftb_gate`,
`tftb_panel`, `tftb_holder`). `sample_entropy`/`higuchi_dim` (Tier 4)
live here but are not wired into analyze() until their tier.

tftb (0.3.0.dev0, dev build) is imported lazily inside functions and is
only ever trusted after gate G1 (`check_tftb_gate`) passes — it is never
a hard dependency of the package.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as ss


def stft_block(x: np.ndarray, sr: int, nperseg: int = 4096,
               noverlap: int | None = None):
    """STFT of a 1-D signal. Returns (f, t, Zxx).

    scipy.signal.stft, Hann window, 50 % overlap by default,
    boundary='pad', padded=False.
    """
    if noverlap is None:
        noverlap = nperseg // 2
    f, t, Zxx = ss.stft(x, fs=sr, window="hann", nperseg=nperseg,
                        noverlap=noverlap, boundary="zeros", padded=False)
    return f, t, Zxx


def hilbert_if(imf: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Instantaneous frequency / amplitude of one IMF
    (Wu–Yeh–Lan 2001 Hilbert spectrum companion).

    z = imf + i·H(imf); inst_freq = d/dt arg(z) / 2π [Hz]; inst_amp = |z|.
    Returns (inst_freq [N], inst_amp [N]).

    Note: the IF of a noisy/complex IMF is ill-conditioned — that is expected
    and is precisely what the ensemble averaging in EEMD is for.
    """
    z = ss.hilbert(imf)
    phase = np.unwrap(np.angle(z))
    inst_freq = np.gradient(phase, 1.0 / sr) / (2.0 * np.pi)
    return inst_freq, np.abs(z)


def if_stats(w: np.ndarray) -> dict:
    """Median / IQR of an IF series, positive part only.

    Returns {"median", "iqr", "n_pos_frac"}; None values if no positive part.
    """
    wp = w[w > 0]
    if wp.size == 0:
        return {"median": None, "iqr": None, "n_pos_frac": 0.0}
    lo, hi = np.percentile(wp, [25, 75])
    return {"median": float(np.median(wp)), "iqr": float(hi - lo),
            "n_pos_frac": float(wp.size / w.size)}


def morlet_cwt(x: np.ndarray, sr: int, f_lo: float = 100.0, n_scales: int = 40,
               f_c: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Manual FFT-based Morlet CWT (scipy.signal.cwt is absent in scipy 1.18).

    Kernel ψ_a(n) = π^(-1/4)·e^{i 2π f_c n/a}·e^{-n²/(2a²)} with a = f_c·sr/f
    (sample units; the 2π makes the spectral peak land at f — without it the
    peak is at f/π); scales log-spaced from f_lo to ~Nyquist; per scale
    C_j = ifft(fft(x, 2N)·fft(ψ_j, 2N))[:N] (zero-padded = linear conv;
    full fft because the Morlet kernel is complex).
    Returns (freqs [n_scales], C [n_scales, N] complex).
    """
    N = len(x)
    nyq = sr / 2.0
    freqs = np.logspace(np.log10(f_lo), np.log10(nyq * 0.99), n_scales)
    X = np.fft.fft(x, 2 * N)
    C = np.empty((n_scales, N), dtype=np.complex128)
    for j, f in enumerate(freqs):
        a = f_c * sr / f
        half = int(np.ceil(5.0 * a / np.sqrt(2.0)))  # 5 sigmas of the envelope
        tn = np.arange(-half, half + 1)
        psi = (np.pi ** (-0.25) * np.exp(1j * 2.0 * np.pi * f_c * tn / a)
               * np.exp(-(tn ** 2) / (2.0 * a ** 2)))
        C[j] = np.fft.ifft(X * np.fft.fft(psi, 2 * N), 2 * N)[:N]
    return freqs, C


_MORLET_GAIN_CACHE: dict = {}


def morlet_cwt_gain(sr: int, freqs: np.ndarray,
                    f_c: float = 1.0) -> np.ndarray:
    """Per-scale CWT gain: |C| produced by a full-scale (amplitude 1.0)
    sine at the scale's own frequency.

    Measured through the exact kernel code path (same truncation and FFT
    as morlet_cwt), because the analytic Gaussian-sum approximation breaks
    for a < ~3 samples (the Morlet double-frequency term then adds up to a
    factor of 2). With the gains, |C|/gain = sine amplitude, so the
    scalogram can be read in dBFS (0 dBFS = full-scale sine).
    Cached per (sr, freqs, f_c).
    """
    key = (sr, tuple(np.asarray(freqs).tolist()), f_c)
    cached = _MORLET_GAIN_CACHE.get(key)
    if cached is not None:
        return cached
    gains = np.empty(len(freqs))
    for j, f in enumerate(freqs):
        a = f_c * sr / f
        half = int(np.ceil(5.0 * a / np.sqrt(2.0)))
        tn = np.arange(-half, half + 1)
        psi = (np.pi ** (-0.25) * np.exp(1j * 2.0 * np.pi * f_c * tn / a)
               * np.exp(-(tn ** 2) / (2.0 * a ** 2)))
        L = 2 * (2 * half + 1) + 8  # interior peak well inside the signal
        x = np.sin(2.0 * np.pi * f * np.arange(L) / sr)
        Cj = np.fft.ifft(np.fft.fft(x, 2 * L) * np.fft.fft(psi, 2 * L),
                         2 * L)[:L]
        gains[j] = np.abs(Cj[L // 3:2 * L // 3]).max()
    _MORLET_GAIN_CACHE[key] = gains
    return gains


def _decimate(x: np.ndarray, max_pts: int = 4096) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    stride = max(1, len(x) // max_pts)
    return x[::stride]


def sample_entropy(x: np.ndarray, m: int = 2, r_frac: float = 0.2) -> float:
    """Sample entropy (m=2, r=0.2·std) on a series decimated to ≤ 4096 pts.

    Lower = more regular. NaN for degenerate inputs.
    """
    x = _decimate(x)
    x = x - x.mean()
    std = float(x.std())
    n = len(x)
    if std <= 0 or n < 2 * m + 2:
        return float("nan")
    r = r_frac * std
    B = np.lib.stride_tricks.sliding_window_view(x, m + 1)[:-1]  # len m+1
    A = B[:, :m]                                                 # len m
    Bm = np.lib.stride_tricks.sliding_window_view(x, m)[:-1]     # len m

    def _count(X, Y, skip_self):
        cnt = 0
        nA, nY = X.shape[0], Y.shape[0]
        chunk = max(1, 250_000 // max(1, nY))
        for i0 in range(0, nA, chunk):
            d = np.abs(X[i0:i0 + chunk, None, :] - Y[None, :, :]).max(axis=2)
            ok = d <= r
            if skip_self:  # exclude trivial self-matches (same start index)
                rows = np.arange(i0, min(i0 + chunk, nA))
                ok &= ~(rows[:, None] == np.arange(nY)[None, :])
            cnt += int(ok.sum())
        return cnt

    # Richman–Moorman: B counts full (m+1)-coordinate matches within B;
    # A counts m-coordinate matches of A against the larger set Bm.
    Bcnt = _count(B, B, skip_self=True)
    Acnt = _count(A, Bm, skip_self=True)
    if Bcnt == 0 or Acnt == 0:
        return float("nan")
    return float(np.log(Acnt / Bcnt))


def higuchi_dim(x: np.ndarray, k_max: int = 10) -> float:
    """Higuchi fractal dimension (k = 1..10) on a series decimated to ≤ 4096.

    L_m(k) = (N-1)/(Q·k²)·Σ|Δ| with Q = ⌊(N-m)/k⌋ (no mean-d normalization —
    the k-scaling IS the estimator); L(k) = (1/k)Σ_m L_m(k); D = slope of
    log L(k) vs log(1/k). Reference values: smooth curve ≈ 1, random walk
    ≈ 1.5, 1-D white noise ≈ 2. NaN for degenerate inputs.
    """
    x = _decimate(x)
    n = len(x)
    if n < 2 * k_max:
        return float("nan")
    ks, Ls = [], []
    for k in range(1, k_max + 1):
        Lk = 0.0
        for m in range(1, k + 1):
            seg = x[m - 1::k]
            Q = len(seg) - 1
            if Q < 1:
                continue
            Lk += (n - 1) / (Q * k ** 2) * float(np.sum(np.abs(np.diff(seg))))
        Lk /= k
        if Lk > 0:
            ks.append(k)
            Ls.append(Lk)
    if len(ks) < 2:
        return float("nan")
    # D is the slope of log L vs log(1/k) = -log k
    return float(-np.polyfit(np.log(np.array(ks, dtype=float)),
                             np.log(np.array(Ls, dtype=float)), 1)[0])


# ------------------------------------------------ Tier 3: tftb (gated) --

class TFTBError(Exception):
    """tftb gate failed or import failed (Tier 3, plan Task 9)."""

    def __init__(self, message: str, code: str = "tftb.gate_failed"):
        super().__init__(message)
        self.code = code


_TFTB_GATE: dict | None = None


def check_tftb_gate() -> dict:
    """Gate G1 (plan Task 9): validate the dev0 tftb build before any tftb
    output is trusted. Runs once per process (cached result).

    Checks on a deterministic 100/105 Hz two-tone reference (4096 samples
    @ 8 kHz), tftb window passed to BOTH sides:
      1. tftb `spectrogram` vs scipy STFT — amplitude correlation > 0.99 on
         the common (resampled) frequency grid. Note: tftb returns |DFT|²,
         so the comparison is |Z| vs sqrt(|S|);
      2. `smoothed_pseudo_wigner_ville` runs and returns a finite matrix.
    Raises TFTBError (stable code) on any failure.
    """
    global _TFTB_GATE
    if _TFTB_GATE is not None:
        return _TFTB_GATE
    try:
        from importlib.metadata import version as _pkg_version
        from tftb.processing.reassigned import (
            smoothed_pseudo_wigner_ville,
            spectrogram,
        )
        from tftb.processing.utils import tftb_window
    except Exception as e:
        raise TFTBError(f"tftb import failed: {e!r}",
                        code="tftb.import_failed") from e
    try:
        ver = _pkg_version("tftb")
    except Exception:
        ver = "unknown"

    sr = 8000
    n = 4096
    t = np.arange(n) / sr
    xref = (np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 105 * t)) / 2
    win = tftb_window(255, "Hanning")
    f, tt, Z = ss.stft(xref, fs=sr, window=win, nperseg=255, noverlap=127,
                       boundary="zeros", padded=False)
    # tftb on the same time grid (scipy's frame centers, in sample indices)
    ts = (tt * sr).round().astype(int)
    ts = ts[(ts >= 0) & (ts < n)]
    St, _, _ = spectrogram(xref, time_samples=ts, n_fbins=256, window=win)
    # tftb's spectrogram is |DFT|² (energy) — compare amplitudes
    Spos = np.sqrt(np.abs(St[:256 // 2 + 1, :]))
    ft = np.arange(256 // 2 + 1) * sr / 256
    a = np.abs(Z[:, :len(ts)])
    if len(f) == len(ft):
        b = Spos
    else:  # resample tftb's frequency grid onto scipy's → (F, T)
        b = np.array([np.interp(f, ft, Spos[:, j])
                      for j in range(Spos.shape[1])]).T
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    if not (corr > 0.99):
        raise TFTBError(
            f"tftb gate G1 failed: tftb-vs-scipy STFT amplitude correlation "
            f"{corr:.4f} <= 0.99 (dev0 build mismatch)")

    T = smoothed_pseudo_wigner_ville(xref, timestamps=ts, n_fbins=256)[0]
    if not np.isfinite(T).all() or T.shape != (256, len(ts)):
        raise TFTBError(
            f"tftb gate G1 failed: SPWVD output not finite / wrong shape "
            f"{T.shape}")

    _TFTB_GATE = {"version": ver, "stft_corr": round(corr, 5),
                  "spwvd_shape": list(T.shape)}
    return _TFTB_GATE


def tftb_panel(x: np.ndarray, sr: int, n_fbins: int = 256,
               time_hop: int = 64) -> dict:
    """tftb reference TFR panel (Tier 3, `--tftb`). Gate G1 must pass.

    tftb's per-column cores are pure Python, so the time axis is
    subsampled by `time_hop` (noted in the report); at 44.1 kHz the
    defaults cost ~2 s per second of audio. Positive frequencies only
    (0 .. sr/2, `n_fbins` DFT bins). Arrays are stored [T, F] —
    `fig_08_tftb` transposes for pcolormesh.

    Returns a dict with keys: version, n_fbins, time_hop,
    spectrogram / spwvd / reassigned (+ t_*/f_* axes per panel).
    `spwvd` is a Cohen-class TFR and contains negative values
    (fig_08 clips them and says so).
    """
    gate = check_tftb_gate()
    from tftb.processing.reassigned import (
        reassigned_gabor,
        smoothed_pseudo_wigner_ville,
        spectrogram,
    )
    n = len(x)
    ts = np.arange(0, n, time_hop)
    S, _, _ = spectrogram(x, time_samples=ts, n_fbins=n_fbins)
    T, _, _ = smoothed_pseudo_wigner_ville(x, timestamps=ts, n_fbins=n_fbins)
    R, _, _ = reassigned_gabor(x, time_samples=ts, n_fbins=n_fbins)
    f = np.arange(n_fbins // 2 + 1) * sr / n_fbins
    t = ts.astype(float) / sr
    nf = n_fbins // 2 + 1
    return {
        "version": gate["version"],
        "n_fbins": n_fbins,
        "time_hop": time_hop,
        "spectrogram": np.abs(S[:nf, :]).T,
        "spwvd": np.real(T[:nf, :]).T,
        "reassigned": np.real(R[:nf, :]).T,
        "t_stft": t, "f_stft": f,
        "t_spwvd": t, "f_spwvd": f,
        "t_reass": t, "f_reass": f,
    }


def tftb_holder(x: np.ndarray, sr: int, n_fbins: int = 128,
                time_hop: int = 64) -> float | None:
    """Global Hölder exponent via `tftb.processing.analysis.holder` on the
    tftb spectrogram — the per-IMF fractal estimate used when `--tftb` is
    active (plan Tier 3/4). NaN → None. DC bin excluded (log 0).
    """
    check_tftb_gate()
    from tftb.processing.analysis import holder
    from tftb.processing.reassigned import spectrogram
    n = len(x)
    ts = np.arange(0, n, time_hop)
    S, _, _ = spectrogram(x, time_samples=ts, n_fbins=n_fbins)
    f = np.arange(n_fbins // 2 + 1) * sr / n_fbins
    Sp = np.abs(S[1:n_fbins // 2 + 1, :])  # drop DC
    h = holder(Sp, f[1:])
    h = float(h)
    return h if np.isfinite(h) else None
