"""EEMD of the PrimeToneNewPhi FLAC — executes EEMD_PRIME_TONE_PLAN.md.

Phases: 1 characterize, 2 GPU EEMD E=64 (primary), 3 gates G1-G8,
        4 GPU EEMD E=128 (convergence), 5 plots + results.json.

Run from this directory with the venv interpreter:
    D:\\Source\\hermes-dir\\.venv\\Scripts\\python.exe eemd_prime_tone.py
"""
import json
import pathlib
import sys
import time

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks, hilbert

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import emd_par as M  # noqa: E402

FLAC = (r"D:\MLS-ESS\SigGenNew\LFL-MuTPri\PrimeToneNewPhi[Nch 1][Frac 03][LFL 043]"
        r"[00020-21600][Nfr 027]-[48000-24]-[1s]-[-0.25dB]-[049062]-[ 8.5905]-[RD].flac")
OUT = HERE / "eemd_prime_tone_out"
OUT.mkdir(exist_ok=True)
FS = 48000
N_EXP = 48000
RES = {}  # accumulated measurements -> results.json


def dump():
    with open(OUT / "results.json", "w") as f:
        json.dump(RES, f, indent=2, default=float)


def parabolic(spec, freqs, i):
    """Parabolic peak refinement around bin i (log spectrum)."""
    if 0 < i < len(spec) - 1:
        a_, b_, c_ = (np.log(spec[i - 1] + 1e-30), np.log(spec[i] + 1e-30),
                      np.log(spec[i + 1] + 1e-30))
        d = 0.5 * (a_ - c_) / (a_ - 2 * b_ + c_)
    else:
        d = 0.0
    return freqs[i] + d * (freqs[1] - freqs[0])


def peak_freq(y, fs):
    n = len(y)
    spec = np.abs(np.fft.rfft(y * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1 / fs)
    return float(parabolic(spec, freqs, int(np.argmax(spec))))


def coherent_amp(y, f, fs):
    """One-sided amplitude of the complex exponential at f in real signal y."""
    t = np.arange(len(y)) / fs
    return float(2.0 * abs(np.sum(y * np.exp(-2j * np.pi * f * t))) / len(y))


def mean_abs_if(imf, fs, trim=0.10):
    """Mean |instantaneous frequency| via Hilbert, edges trimmed."""
    z = hilbert(imf)
    ifr = np.diff(np.unwrap(np.angle(z))) * fs / (2 * np.pi)
    m = max(int(trim * len(ifr)), 1)
    return float(np.mean(np.abs(ifr[m:-m])))


def stats_json(stats):
    out = {}
    for k, v in stats.items():
        if k == "x":
            continue
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def main():
    # ---------------- Phase 1: load + characterize ----------------
    t0 = time.time()
    x, sr = sf.read(FLAC, dtype="float64")
    assert x.ndim == 1, f"expected mono, got {x.shape}"
    assert sr == FS, f"unexpected sr {sr}"
    N = len(x)
    assert N == N_EXP, f"unexpected N {N}"
    t = np.arange(N) / FS

    spec = np.abs(np.fft.rfft(x * np.hanning(N)))
    freqs = np.fft.rfftfreq(N, 1 / FS)
    pk, _ = find_peaks(spec, height=0.02 * spec.max())
    groups, cur = [], [pk[0]]
    for i in pk[1:]:
        if freqs[i] - freqs[cur[-1]] < 2.0:      # merge sideband clusters
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    tone_f = []
    for g in groups:
        i = int(g[np.argmax(spec[g])])
        tone_f.append(float(parabolic(spec, freqs, i)))
    tone_amp = {f: coherent_amp(x, f, FS) for f in tone_f}
    np.save(OUT / "tone_frequencies.npy", np.array(tone_f))

    ext_max = int(np.sum((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:])))
    ext_min = int(np.sum((x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:])))

    RES["phase1"] = dict(
        load_s=time.time() - t0, N=N, sr=sr,
        mean=float(x.mean()), peak=float(np.abs(x).max()),
        rms=float(np.sqrt((x ** 2).mean())),
        n_tones=len(tone_f),
        tone_f=[round(f, 3) for f in tone_f],
        tone_amp=[round(a, 5) for a in (tone_amp[f] for f in tone_f)],
        n_extrema_max=ext_max, n_extrema_min=ext_min,
    )
    print(f"[P1] N={N} tones={len(tone_f)} ext(max,min)=({ext_max},{ext_min}) "
          f"{time.time()-t0:.2f}s")
    print("     tone_f =", [round(f, 1) for f in tone_f])
    dump()

    # ---------------- Phase 2: primary GPU EEMD, E=64 ----------------
    t0 = time.time()
    r = M.decompose(x, method="gpu", E=64, tau=0.25, eps=0.2, max_imf=32, seed=0)
    wall64 = time.time() - t0
    r.save(OUT / "primetone_eemd_E64_gpu.npz")
    RES["phase2"] = dict(wall_s=wall64, n_modes=r.n_modes,
                         recon_error=float(r.recon_error()),
                         stats=stats_json(r.stats))
    print(f"[P2] GPU E=64: {wall64:.1f}s n_modes={r.n_modes} "
          f"recon={r.recon_error():.2e} stats={ {k: v for k, v in r.stats.items() if k not in ('x', 'per_trial_total_sifts')} }")
    dump()

    # ---------------- Phase 3: gates G1-G8 ----------------
    gates = {}

    # G1 partition of unity
    g1_val = float(r.recon_error())
    gates["G1_partition_of_unity"] = dict(value=g1_val, pass_=g1_val <= 1e-10)

    # G2 determinism (same backend, same seed -> bit identical, manual A3)
    t0 = time.time()
    r2 = M.decompose(x, method="gpu", E=64, tau=0.25, eps=0.2, max_imf=32, seed=0)
    g2_wall = time.time() - t0
    g2_val = bool(np.array_equal(r.imfs, r2.imfs) and np.array_equal(r.residual, r2.residual))
    gates["G2_determinism"] = dict(bit_identical=g2_val, pass_=g2_val, rerun_wall_s=g2_wall)

    # G3 CPU-vs-GPU cross-check, plain EMD E=1 both sides (matches gate A2;
    #    E=1 avoids the E=64-vs-E=8 Monte-Carlo difference)
    t0 = time.time()
    rc = M.decompose(x, method="cpu", E=1, tau=0.25, max_imf=32, seed=0)
    wall_cpu1 = time.time() - t0
    t0 = time.time()
    rg1 = M.decompose(x, method="gpu", E=1, tau=0.25, max_imf=32, seed=0)
    wall_gpu1 = time.time() - t0
    common = min(rc.n_modes, rg1.n_modes)
    per_imf = [float(np.max(np.abs(rc.imfs[k] - rg1.imfs[k])) /
                     np.max(np.abs(rg1.imfs[k]))) for k in range(common)]
    g3_max = float(max(per_imf)) if per_imf else float("nan")
    g3_val = (g3_max <= 1e-8) and (abs(rc.n_modes - rg1.n_modes) <= 1)
    gates["G3_cpu_gpu_crosscheck"] = dict(
        cpu_E1_wall_s=wall_cpu1, gpu_E1_wall_s=wall_gpu1,
        n_modes_cpu=int(rc.n_modes), n_modes_gpu=int(rg1.n_modes),
        common_depth=common, per_imf_relmax=per_imf, max_relmax=g3_max,
        pass_=g3_val)

    # G4 mode count
    g4_val = 25 <= r.n_modes <= 30
    gates["G4_mode_count"] = dict(n_modes=r.n_modes, pass_=g4_val)

    # G5 mode->tone assignment + monotone order
    assigned, deltas = [], []
    for k in range(r.n_modes):
        f_pk = peak_freq(r.imfs[k], FS)
        j = int(np.argmin([abs(f_pk - f) for f in tone_f]))
        assigned.append(tone_f[j])
        deltas.append(f_pk - tone_f[j])
    mono = all(assigned[k] > assigned[k + 1] for k in range(len(assigned) - 1))
    g5_val = mono and all(abs(d) < 2.0 for d in deltas)
    gates["G5_mode_tone_assignment"] = dict(
        assigned_tone=[round(f, 2) for f in assigned],
        peak_offset_hz=[round(d, 3) for d in deltas],
        monotone_decreasing=mono, pass_=g5_val)

    # G6 instantaneous frequency vs assigned tone (5% tolerance)
    ifs, if_err = [], []
    for k in range(r.n_modes):
        mif = mean_abs_if(r.imfs[k], FS)
        ifs.append(mif)
        if_err.append(abs(mif - assigned[k]) / assigned[k])
    g6_val = all(e < 0.05 for e in if_err)
    gates["G6_instantaneous_frequency"] = dict(
        mean_abs_if=[round(v, 3) for v in ifs],
        rel_err=[round(e, 5) for e in if_err],
        max_rel_err=float(max(if_err)), pass_=g6_val)

    # G7 residual content: 0.43 Hz AM present, no tone leakage > 1%
    a043_x = coherent_amp(x, 0.43, FS)
    a043_res = coherent_amp(r.residual, 0.43, FS)
    leak = {round(f, 2): coherent_amp(r.residual, f, FS) / tone_amp[f]
            for f in tone_f}
    max_leak = max(leak.values())
    g7_val = (a043_res > 0.5 * a043_x) and (max_leak < 0.01)
    gates["G7_residual_content"] = dict(
        amp_043hz_in_x=a043_x, amp_043hz_in_residual=a043_res,
        residual_mean=float(r.residual.mean()),
        max_tone_leak_rel=float(max_leak),
        leak_worst_tone_hz=round(tone_f[int(np.argmax(list(leak.values())))], 2),
        pass_=g7_val)

    # G8 max_imf cap not hit
    g8_val = bool(r.stats.get("max_imf_reached", False) is False)
    gates["G8_max_imf_not_reached"] = dict(
        max_imf_reached=bool(r.stats.get("max_imf_reached", False)), pass_=g8_val)

    RES["phase3_gates"] = gates
    for name, g in gates.items():
        detail = {k: v for k, v in g.items()
                  if k not in ("pass_", "per_imf_relmax", "assigned_tone",
                               "peak_offset_hz", "mean_abs_if", "rel_err")}
        print(f"[P3] {name}: {'PASS' if g['pass_'] else 'FAIL'} {detail}")
    dump()

    # ---------------- Phase 4: convergence, GPU E=128 ----------------
    t0 = time.time()
    r128 = M.decompose(x, method="gpu", E=128, tau=0.25, eps=0.2, max_imf=32, seed=0)
    wall128 = time.time() - t0
    r128.save(OUT / "primetone_eemd_E128_gpu.npz")
    common = min(r.n_modes, r128.n_modes)
    rel_l2 = [float(np.linalg.norm(r128.imfs[k] - r.imfs[k]) /
                    np.linalg.norm(r.imfs[k])) for k in range(common)]
    RES["phase4"] = dict(wall_s=wall128, n_modes=r128.n_modes,
                         recon_error=float(r128.recon_error()),
                         rel_l2_vs_E64=[round(v, 12) for v in rel_l2],
                         stats=stats_json(r128.stats))
    print(f"[P4] GPU E=128: {wall128:.1f}s n_modes={r128.n_modes} "
          f"recon={r128.recon_error():.2e} max relL2 vs E64 = {max(rel_l2):.3e}")
    dump()

    # ---------------- Phase 5: plots ----------------
    nm = r.n_modes
    imfs = r.imfs

    # Mode x tone presence: coherent amplitude of each tone in each mode,
    # relative to that tone's amplitude in x. This is the actual grouping
    # structure EMD produced (the 27-modes hypothesis is not assumed).
    mt = np.zeros((nm, len(tone_f)))
    for k in range(nm):
        for j, f in enumerate(tone_f):
            mt[k, j] = coherent_amp(imfs[k], f, FS) / tone_amp[f]
    dom_tone = [tone_f[int(np.argmax(mt[k]))] for k in range(nm)]
    present = [[f for j, f in enumerate(tone_f) if mt[k, j] >= 0.10]
               for k in range(nm)]
    RES["phase5_mode_tone"] = dict(
        dominant_tone=[round(f, 2) for f in dom_tone],
        present_tone_ge_10pct=present,
        matrix=[[round(v, 4) for v in row] for row in mt])
    dump()

    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax[0].plot(t, x, lw=0.3)
    ax[0].set_ylabel("x(t)")
    ax[0].set_title("PrimeToneNewPhi (27 prime tones, 43 Hz - 21.6 kHz) + EEMD residual")
    ax[1].plot(t, r.residual, lw=0.3, color="tab:red")
    ax[1].set_ylabel("residual")
    ax[1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(OUT / "signal.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(nm, 1, figsize=(10, 0.85 * nm), sharex=True)
    for k in range(nm):
        axes[k].plot(t, imfs[k], lw=0.3)
        tones_lab = (", ".join(f"{f:.0f}" for f in present[k][:4])
                     + (f" +{len(present[k]) - 4} more" if len(present[k]) > 4 else ""))
        axes[k].set_ylabel(f"IMF {k + 1}\n{tones_lab}", rotation=0,
                           fontsize=7, va="center")
        axes[k].set_yticks([])
        for a in axes[k].xaxis.get_major_ticks():
            a.set_visible(False)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(OUT / "modes.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(nm, 1, figsize=(10, 0.85 * nm), sharex=True)
    for k in range(nm):
        S = np.abs(np.fft.rfft(imfs[k] * np.hanning(N)))
        axes[k].semilogy(freqs, S, lw=0.4)
        for f in tone_f:
            axes[k].axvline(f, color="gray", lw=0.3, alpha=0.4)
        axes[k].set_yticks([])
        for a in axes[k].yaxis.get_major_ticks():
            a.set_visible(False)
    for a in axes:
        a.set_xlim(0, FS / 2)
    axes[-1].set_xlabel("frequency (Hz)")
    fig.suptitle("Per-IMF amplitude spectra (gray lines: detected tones)", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "spectra.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * nm)))
    idx = np.arange(1, nm + 1)
    ax.bar(idx, ifs, width=0.6, alpha=0.6, label="mean |IF| (Hilbert)")
    ax.plot(idx, dom_tone, "kx", ms=8, mew=2, label="dominant tone (coherent)")
    ax.set_yscale("log")
    ax.set_xlabel("IMF index (1 = fastest)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title(f"EEMD E=64: instantaneous frequency vs tone (n_modes={nm})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "if_check.png", dpi=150)
    plt.close(fig)

    RES["phase5"] = dict(plots=["signal.png", "modes.png", "spectra.png", "if_check.png"])
    dump()

    # ---------------- summary ----------------
    print("\n=== SUMMARY ===")
    print(f"n_modes (E=64 GPU): {r.n_modes}")
    for name, g in gates.items():
        print(f"  {name}: {'PASS' if g['pass_'] else 'FAIL'}")
    npz = OUT / "primetone_eemd_E64_gpu.npz"
    print(f"saved: {npz} ({npz.stat().st_size/1e6:.2f} MB)")
    print(f"all results -> {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
