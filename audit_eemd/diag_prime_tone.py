"""Diagnostics for the 11-mode EEMD result: what are modes 8-11, and is the
CPU-E1 vs GPU-E1 divergence structural (different groupings) or localized?"""
import pathlib
import sys

import numpy as np
import soundfile as sf

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import emd_par as M  # noqa: E402

FLAC = (r"D:\MLS-ESS\SigGenNew\LFL-MuTPri\PrimeToneNewPhi[Nch 1][Frac 03][LFL 043]"
        r"[00020-21600][Nfr 027]-[48000-24]-[1s]-[-0.25dB]-[049062]-[ 8.5905]-[RD].flac")
FS = 48000


def top_peaks(y, fs, ntop=5, floor=0.02):
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
    # keep the ntop LARGEST-amplitude peaks, then sort by frequency for display
    out.sort(key=lambda p: -p[1])
    out = out[:ntop]
    out.sort()
    return out


def main():
    x, sr = sf.read(FLAC, dtype="float64")
    r = M.EMDResult.load(HERE / "eemd_prime_tone_out" / "primetone_eemd_E64_gpu.npz")
    imfs = r.imfs
    tot = (x ** 2).sum()

    print("=== per-mode energy + spectral peaks (E=64 GPU) ===")
    for k in range(r.n_modes):
        e = (imfs[k] ** 2).sum() / tot
        pp = top_peaks(imfs[k], FS)
        print(f"mode {k + 1:2d} | E={e:7.4f} | "
              + "  ".join(f"{f:.2f}:{a:.2f}" for f, a in pp))
    er = (r.residual ** 2).sum() / tot
    print(f"RESIDUAL | E={er:7.4f} | "
          + "  ".join(f"{f:.2f}:{a:.2f}" for f, a in top_peaks(r.residual, FS)))

    print("\n=== G3 diagnostic: CPU E=1 vs GPU E=1, per mode ===")
    rc = M.decompose(x, method="cpu", E=1, tau=0.25, max_imf=32, seed=0)
    rg = M.decompose(x, method="gpu", E=1, tau=0.25, max_imf=32, seed=0)
    for k in range(min(rc.n_modes, rg.n_modes)):
        d = np.abs(rc.imfs[k] - rg.imfs[k])
        m = np.max(np.abs(rg.imfs[k]))
        frac = float((d > 0.1 * m).mean())
        tmax = float(np.argmax(d)) / FS
        cpk = [f"{f:.1f}" for f, _ in top_peaks(rc.imfs[k], FS, 3)]
        gpk = [f"{f:.1f}" for f, _ in top_peaks(rg.imfs[k], FS, 3)]
        print(f"mode {k + 1:2d}: relmax={d.max() / m:7.3f} frac>0.1max={frac:5.3f} "
              f"t_argmax={tmax:6.3f}s cpu_pk={cpk} gpu_pk={gpk}")


if __name__ == "__main__":
    main()
