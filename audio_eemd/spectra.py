"""Time-frequency building blocks (plan Task 3).

Tier 2 wires `stft_block` + `hilbert_if`/`if_stats` into analyze().
`morlet_cwt` (Tier 3) and `sample_entropy`/`higuchi_dim` (Tier 4) live here
but are not wired into analyze() until their tier.
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

    Kernel ψ_a(t) = π^(-1/4)·e^{i 2 f_c t/a}·e^{-t²/(2a²)} with a = f_c·sr/f;
    scales log-spaced from f_lo to ~Nyquist; per scale
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
        psi = (np.pi ** (-0.25) * np.exp(1j * 2.0 * f_c * tn / a)
               * np.exp(-(tn ** 2) / (2.0 * a ** 2)))
        C[j] = np.fft.ifft(X * np.fft.fft(psi, 2 * N), 2 * N)[:N]
    return freqs, C


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
