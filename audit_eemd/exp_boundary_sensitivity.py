"""Boundary-condition & endpoint sensitivity experiments for the 27-tone signal.

Context: first EEMD run (eemd_prime_tone.py) showed CPU-E1 vs GPU-E1 disagree
at O(1) in every mode (G3). Both backends are natural-BC, so this isolates
float-level chaos from BC/endpoint effects. Experiments:

  P1  BC comparison, CPU E=1: end='natural' vs end='mirror'
  P2  chaotic amplification curve, CPU E=1: input perturbation 1e-16..1e-9*std
  P3  Tukey window sweep r = 0/0.005/0.01/0.02/0.05:
      endpoints, low-freq input content, CPU-E1, GPU-E1 (G3), GPU-E64
  P4  Tukey x mirror: natural vs mirror on raw and r=0.01 (CPU E=1)

Run:  D:\\Source\\hermes-dir\\.venv\\Scripts\\python.exe exp_boundary_sensitivity.py
"""
import json
import pathlib
import sys
import time

import numpy as np
import soundfile as sf
from scipy.signal.windows import tukey

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import emd_par as M  # noqa: E402

FLAC = (r"D:\MLS-ESS\SigGenNew\LFL-MuTPri\PrimeToneNewPhi[Nch 1][Frac 03][LFL 043]"
        r"[00020-21600][Nfr 027]-[48000-24]-[1s]-[-0.25dB]-[049062]-[ 8.5905]-[RD].flac")
FS = 48000
OUT = HERE / "boundary_sensitivity_out"
OUT.mkdir(exist_ok=True)
LOWFREQ = [0.43, 2.06, 2.10, 4.65, 4.74, 7.25, 18.59, 21.06]
RES = {}


def dump():
    with open(OUT / "results.json", "w") as f:
        json.dump(RES, f, indent=2, default=float)


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
    return [[round(f, 2), round(a, 3)] for f, a in out]


def coherent_amp(y, f, fs):
    t = np.arange(len(y)) / fs
    return float(2.0 * abs(np.sum(y * np.exp(-2j * np.pi * f * t))) / len(y))


def mode_summary(r):
    return {"n_modes": int(r.n_modes),
            "mode_peaks": [top_peaks(imf, FS) for imf in r.imfs]}


def relmax_list(a, b):
    """per-mode max|a-b| / max|b| over the common depth"""
    common = min(a.n_modes, b.n_modes)
    return [float(np.max(np.abs(a.imfs[k] - b.imfs[k])) /
                  np.max(np.abs(b.imfs[k]))) for k in range(common)]


def cpu_e1(x, end="natural", seed=0):
    t0 = time.time()
    r = M.decompose(x, method="cpu", E=1, tau=0.25, max_imf=32, seed=seed,
                    end=end)
    return r, time.time() - t0


def gpu_e1(x):
    t0 = time.time()
    r = M.decompose(x, method="gpu", E=1, tau=0.25, max_imf=32, seed=0)
    return r, time.time() - t0


def gpu_e64(x):
    t0 = time.time()
    r = M.decompose(x, method="gpu", E=64, tau=0.25, eps=0.2, max_imf=32,
                    seed=0)
    s = r.stats
    sifs = s.get("per_trial_total_sifts")
    return r, time.time() - t0, {
        "n_modes": int(r.n_modes),
        "gpu_time_s": float(s.get("gpu_time_s", float("nan"))),
        "sifts_min": int(sifs.min()) if sifs is not None else None,
        "sifts_max": int(sifs.max()) if sifs is not None else None,
        "recon_error": float(r.recon_error()),
        "mode_peaks": [top_peaks(imf, FS) for imf in r.imfs],
    }


def main():
    x, sr = sf.read(FLAC, dtype="float64")
    assert sr == FS and len(x) == 48000
    N = len(x)
    std = float(x.std())

    RES["input"] = {
        "x0": float(x[0]), "xN1": float(x[-1]),
        "first8": [float(v) for v in x[:8]],
        "last8": [float(v) for v in x[-8:]],
        "max_abs_first100": float(np.abs(x[:100]).max()),
        "max_abs_last100": float(np.abs(x[-100:]).max()),
    }
    print(f"[IN] x[0]={x[0]:+.6f} x[-1]={x[-1]:+.6f}")
    dump()

    # ---------------- P1: BC comparison (CPU E=1) ----------------
    rc_nat, w_nat = cpu_e1(x, end="natural")
    rc_mir, w_mir = cpu_e1(x, end="mirror")
    rm = relmax_list(rc_nat, rc_mir)
    RES["p1_bc"] = {
        "natural": mode_summary(rc_nat), "wall_natural_s": w_nat,
        "mirror": mode_summary(rc_mir), "wall_mirror_s": w_mir,
        "per_mode_relmax_natural_vs_mirror": [round(v, 6) for v in rm],
        "max_relmax": float(max(rm)),
    }
    print(f"[P1] natural n={rc_nat.n_modes} ({w_nat:.2f}s) | "
          f"mirror n={rc_mir.n_modes} ({w_mir:.2f}s) | "
          f"max relmax = {max(rm):.3e}")
    dump()

    # ---------------- P2: chaotic amplification curve (CPU E=1) ----------------
    base = rc_nat
    curve = []
    for i, scale in enumerate([1e-16, 1e-15, 1e-14, 1e-13, 1e-12,
                               1e-11, 1e-10, 1e-9]):
        rng = np.random.default_rng(1000 + i)
        xp = x + scale * std * rng.standard_normal(N)
        r, w = cpu_e1(xp)
        rm = relmax_list(base, r)
        first_big = next((k + 1 for k, v in enumerate(rm) if v > 0.1), None)
        curve.append({"scale": scale, "n_modes": int(r.n_modes),
                      "wall_s": w, "max_relmax": float(max(rm)),
                      "first_mode_gt_0p1": first_big,
                      "per_mode_relmax": [round(v, 6) for v in rm]})
        print(f"[P2] scale={scale:.0e}: n={r.n_modes} max_relmax={max(rm):.3e} "
              f"first>0.1@mode {first_big}")
    RES["p2_perturbation_curve"] = curve
    dump()

    # ---------------- P3: Tukey window sweep ----------------
    sweep = []
    for r_tk in [0.0, 0.005, 0.01, 0.02, 0.05]:
        w = tukey(N, alpha=r_tk) if r_tk > 0 else np.ones(N)
        xw = x * w
        rec = {
            "r": r_tk,
            "xw0": float(xw[0]), "xwN1": float(xw[-1]),
            "lowfreq_in": {f"{f:g}": coherent_amp(xw, f, FS) for f in LOWFREQ},
        }
        rc, wc = cpu_e1(xw)
        rg, wg = gpu_e1(xw)
        rm = relmax_list(rc, rg)
        r64, w64, s64 = gpu_e64(xw)
        rec.update({
            "cpu_e1": mode_summary(rc) | {"wall_s": wc},
            "gpu_e1": {"n_modes": int(rg.n_modes), "wall_s": wg},
            "g3_per_mode_relmax": [round(v, 6) for v in rm],
            "g3_max_relmax": float(max(rm)),
            "gpu_e64": s64 | {"wall_s": w64},
        })
        sweep.append(rec)
        print(f"[P3] r={r_tk:<6}: xw[0]={xw[0]:+.2e} cpu_n={rc.n_modes} "
              f"gpu_n={rg.n_modes} E64_n={r64.n_modes} g3_max={max(rm):.3e} "
              f"({wc:.1f}/{wg:.1f}/{w64:.1f}s)")
    RES["p3_tukey_sweep"] = sweep
    dump()

    # ---------------- P4: Tukey x mirror (CPU E=1) ----------------
    p4 = {}
    for label, r_tk in [("raw", 0.0), ("tukey_r0.01", 0.01)]:
        xw = x if r_tk == 0 else x * tukey(N, alpha=r_tk)
        rn, wn = cpu_e1(xw, end="natural")
        rm_, wm = cpu_e1(xw, end="mirror")
        rl = relmax_list(rn, rm_)
        p4[label] = {
            "natural_n_modes": int(rn.n_modes), "mirror_n_modes": int(rm_.n_modes),
            "per_mode_relmax": [round(v, 6) for v in rl],
            "max_relmax": float(max(rl)),
            "natural_peaks": [top_peaks(imf, FS) for imf in rn.imfs],
            "mirror_peaks": [top_peaks(imf, FS) for imf in rm_.imfs],
        }
        print(f"[P4] {label}: natural n={rn.n_modes} mirror n={rm_.n_modes} "
              f"max_relmax={max(rl):.3e}")
    RES["p4_tukey_x_mirror"] = p4
    dump()
    print(f"\ndone -> {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
