"""probe_sameinput2.py — decisive same-input CPU vs GPU comparisons.

T1  E=1:  decompose(cpu)  vs  decompose(gpu, eps=0.0)   [identical input]
T2  E=1:  decompose(cpu)  vs  decompose(gpu, eps=0.2)   [G3 artifact: the
      GPU path adds eps*std noise even at E=1, the CPU path does not]
T64 E=64: one noise batch, same 64 noised signals to BOTH backends
      (GPU: emd_gpu.gpu_eemd_full batch; CPU: 64 x decompose E=1)

Run:  D:\\Source\\hermes-dir\\.venv\\Scripts\\python.exe probe_sameinput2.py
"""
import json
import pathlib
import sys
import time

import numpy as np
import soundfile as sf

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import emd_par as M      # noqa: E402
import emd_gpu as EG     # noqa: E402

FLAC = (r"D:\MLS-ESS\SigGenNew\LFL-MuTPri\PrimeToneNewPhi[Nch 1][Frac 03][LFL 043]"
        r"[00020-21600][Nfr 027]-[48000-24]-[1s]-[-0.25dB]-[049062]-[ 8.5905]-[RD].flac")
FS = 48000
RES = {}

x, sr = sf.read(FLAC, dtype="float64")
N = len(x)
assert sr == FS and N == 48000


def relmax(a, b):
    d = max(float(np.max(np.abs(pa - pb))) for pa, pb in zip(a, b))
    s = max(float(np.max(np.abs(pa))) for pa in a)
    return d, d / s


print(f"signal: N={N} std={np.std(x):.6f}")

# ---------------- T1: E=1, SAME input (gpu eps=0) ----------------
t0 = time.time()
rc = M.decompose(x, method="cpu", E=1, tau=0.25, max_imf=32, seed=0)
w_cpu = time.time() - t0
t0 = time.time()
rg = M.decompose(x, method="gpu", E=1, tau=0.25, eps=0.0, max_imf=32, seed=0)
w_gpu = time.time() - t0
d1, rel1 = relmax(rc.imfs, rg.imfs)
RES["t1_e1_same_input"] = dict(
    cpu_n=int(rc.n_modes), gpu_n=int(rg.n_modes),
    max_abs=float(d1), relmax=float(rel1),
    cpu_wall_s=w_cpu, gpu_wall_s=w_gpu,
    cpu_sifts=int(rc.stats.get("total_sifts", -1)),
    gpu_sifts=int(rg.stats.get("per_trial_total_sifts", [-1])[0]))
print(f"[T1] E=1 same input: n={rc.n_modes}/{rg.n_modes} "
      f"max|d|={d1:.3e} relmax={rel1:.3e} "
      f"({w_cpu:.2f}s / {w_gpu:.2f}s, sifts {RES['t1_e1_same_input']['cpu_sifts']}/"
      f"{RES['t1_e1_same_input']['gpu_sifts']})")

# ---------------- T2: E=1, G3 artifact (gpu eps=0.2) ----------------
rg2 = M.decompose(x, method="gpu", E=1, tau=0.25, eps=0.2, max_imf=32, seed=0)
d2, rel2 = relmax(rc.imfs, rg2.imfs)
RES["t2_e1_g3_artifact"] = dict(
    cpu_n=int(rc.n_modes), gpu_n=int(rg2.n_modes),
    max_abs=float(d2), relmax=float(rel2))
print(f"[T2] E=1 G3 artifact (gpu eps=0.2): n={rc.n_modes}/{rg2.n_modes} "
      f"relmax={rel2:.3f}")

# ---------------- T64: E=64, SAME 64 noised signals ----------------
stdx = float(np.std(x))
rng = np.random.default_rng(0)
noises = rng.standard_normal((64, N)) * (0.2 * stdx)
batch = x + noises

t0 = time.time()
imfs_g, nsift_g, nmode_g = EG.gpu_eemd_full(
    batch, E=64, tau=0.25, max_sifts=50, max_imf=32, nthreads=128)
w_gpu64 = time.time() - t0

w_cpu64 = 0.0
max_trial_abs = 0.0
max_trial_rel = 0.0
nmode_c = []
for e in range(64):
    t0 = time.time()
    re = M.decompose(batch[e], method="cpu", E=1, tau=0.25, max_imf=32, seed=0)
    w_cpu64 += time.time() - t0
    nmode_c.append(int(re.n_modes))
    m = min(re.n_modes, int(nmode_g[e]))
    for i in range(m):
        da = float(np.max(np.abs(re.imfs[i] - imfs_g[e, i, :])))
        sa = float(np.max(np.abs(re.imfs[i])))
        max_trial_abs = max(max_trial_abs, da)
        max_trial_rel = max(max_trial_rel, da / sa)
    print(f"  trial {e:2d}: n_cpu={re.n_modes} n_gpu={int(nmode_g[e])}")

RES["t64_e64_same_batch"] = dict(
    cpu_nmode_hist={str(k): int(v) for k, v in zip(*np.unique(nmode_c,
                                                             return_counts=True))},
    gpu_nmode_hist={str(k): int(v) for k, v in zip(*np.unique(nmode_g,
                                                             return_counts=True))},
    per_trial_max_abs=float(max_trial_abs),
    per_trial_relmax=float(max_trial_rel),
    cpu_wall_s=w_cpu64, gpu_wall_s=w_gpu64)
print(f"[T64] E=64 same batch: per-trial max|d|={max_trial_abs:.3e} "
      f"relmax={max_trial_rel:.3e} "
      f"({w_cpu64:.1f}s / {w_gpu64:.1f}s)")

out = HERE / "mechanism_out" / "sameinput_results.json"
with open(out, "w") as f:
    json.dump(RES, f, indent=2)
print(f"\ndone -> {out}")
