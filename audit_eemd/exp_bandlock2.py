"""exp_bandlock2.py — corrected Part C (band-locking mechanism).

FIX vs exp_mechanism.py C1-C3: the tone synthesis must use t in SECONDS
(t = arange(N)/FS), not sample indices — the first version aliased every
tone (f cycles/sample instead of f Hz) and the "peaks" were artifacts.

Run:  D:\\Source\\hermes-dir\\.venv\\Scripts\\python.exe exp_bandlock2.py
"""
import json
import pathlib
import sys
import time

import numpy as np
import soundfile as sf

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import emd_par as M  # noqa: E402

FLAC = (r"D:\MLS-ESS\SigGenNew\LFL-MuTPri\PrimeToneNewPhi[Nch 1][Frac 03][LFL 043]"
        r"[00020-21600][Nfr 027]-[48000-24]-[1s]-[-0.25dB]-[049062]-[ 8.5905]-[RD].flac")
FS = 48000
OUT = HERE / "mechanism_out"
N = 48000
T = np.arange(N, dtype=float) / FS
RES = {}


def top_peaks(y, fs, ntop=3, floor=0.02):
    n = len(y)
    S = np.abs(np.fft.rfft(y * np.hanning(n)))
    f = np.fft.rfftfreq(n, 1 / fs)
    S = S / S.max()
    out = []
    for i in np.argsort(S)[::-1][:40]:
        if S[i] < floor:
            break
        if 0 < i < len(S) - 1:
            a, b, c = (np.log(S[i - 1] + 1e-30), np.log(S[i] + 1e-30),
                       np.log(S[i + 1] + 1e-30))
            d = 0.5 * (a - c) / (a - 2 * b + c)
            out.append((f[i] + d * (f[1] - f[0]), float(S[i])))
        else:
            out.append((float(f[i]), float(S[i])))
    out.sort(key=lambda p: -p[1])
    out = out[:ntop]
    out.sort()
    return [[round(max(f, 0.0), 2), round(a, 3)] for f, a in out]


def run_cpu(x2):
    t0 = time.time()
    r = M.decompose(x2, method="cpu", E=1, tau=0.25, max_imf=32, seed=0)
    return r, time.time() - t0


def main():
    # ---------------- C1: 2-tone ratio sweep (proper seconds) ----------
    c1 = []
    for r_ratio in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.4, 3.0]:
        f1, f2 = 1000.0, 1000.0 * r_ratio
        x2 = 0.5 * np.sin(2 * np.pi * f1 * T) + 0.5 * np.sin(2 * np.pi * f2 * T)
        r, w = run_cpu(x2)
        nmax = int(np.sum((x2[1:-1] > x2[:-2]) & (x2[1:-1] > x2[2:])))
        nmin = int(np.sum((x2[1:-1] < x2[:-2]) & (x2[1:-1] < x2[2:])))
        peaks = [top_peaks(imf, FS) for imf in r.imfs]
        c1.append(dict(ratio=r_ratio, n_modes=int(r.n_modes),
                       nmax_signal=nmax, nmin_signal=nmin,
                       per_mode_sifts=[int(v) for v in
                                       r.stats.get("per_mode_sifts", [])],
                       peaks=peaks, wall_s=w))
        print(f"[C1] ratio={r_ratio:<4}: n_modes={r.n_modes} "
              f"extrema=({nmax},{nmin}) "
              f"peaks={[[p[0] for p in pk] for pk in peaks]}")
    RES["c1_two_tone_sweep"] = c1
    dump_path = OUT / "bandlock_results.json"
    with open(dump_path, "w") as f:
        json.dump(RES, f, indent=2, default=float)

    # ---------------- C2: 1-per-octave sparse control ------------------
    tone_f = [43.0, 53.0, 61.0, 83.0, 107.0, 127.0, 167.0, 211.0, 251.0,
              337.0, 421.0, 509.0, 683.0, 853.0, 1021.0, 1361.0, 1699.0,
              2039.0, 2731.0, 3413.0, 4093.0, 5449.0, 6827.0, 8191.0,
              10909.0, 13649.0, 16381.0]
    tone_amp = [0.2295, 0.20602, 0.18898, 0.16513, 0.14507, 0.13022,
                0.11659, 0.10353, 0.09294, 0.0822, 0.0734, 0.06505, 0.05784,
                0.05158, 0.046, 0.04102, 0.0366, 0.03263, 0.029, 0.02584,
                0.02304, 0.02055, 0.0183, 0.01631, 0.01454, 0.01295, 0.01155]
    sparse_idx = [0, 4, 8, 12, 16]   # 43, 107, 251, 683, 1699 Hz
    x_sparse = sum(tone_amp[i] * np.sin(2 * np.pi * tone_f[i] * T)
                   for i in sparse_idx)
    r, w = run_cpu(x_sparse)
    RES["c2_sparse_control"] = dict(
        tones=[tone_f[i] for i in sparse_idx], n_modes=int(r.n_modes),
        per_mode_sifts=[int(v) for v in r.stats.get("per_mode_sifts", [])],
        peaks=[top_peaks(imf, FS) for imf in r.imfs], wall_s=w)
    print(f"[C2] sparse 1-per-octave (5 tones): n_modes={r.n_modes} "
          f"peaks={[[p[0] for p in pk] for pk in RES['c2_sparse_control']['peaks']]}")
    with open(dump_path, "w") as f:
        json.dump(RES, f, indent=2, default=float)

    # ---------------- C3: thinned 14-tone control ----------------------
    thinned = list(range(0, 27, 2))
    x_thin = sum(tone_amp[i] * np.sin(2 * np.pi * tone_f[i] * T)
                 for i in thinned)
    r, w = run_cpu(x_thin)
    RES["c3_thinned27"] = dict(
        n_tones=len(thinned), n_modes=int(r.n_modes),
        per_mode_sifts=[int(v) for v in r.stats.get("per_mode_sifts", [])],
        peaks=[top_peaks(imf, FS) for imf in r.imfs], wall_s=w)
    print(f"[C3] thinned (14 tones): n_modes={r.n_modes} "
          f"peaks={[[p[0] for p in pk] for pk in RES['c3_thinned27']['peaks']]}")
    with open(dump_path, "w") as f:
        json.dump(RES, f, indent=2, default=float)
    print(f"\ndone -> {dump_path}")


if __name__ == "__main__":
    main()
