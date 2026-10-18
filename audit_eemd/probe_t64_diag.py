"""probe_t64_diag.py — per-trial/per-mode divergence map for the E=64
same-batch CPU vs GPU comparison (T64). Locates the first mode where each
diverging trial splits."""
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, r"D:\Source\hermes-dir\EEMD4CUDA-main")
import emd_par as M    # noqa: E402
import emd_gpu as EG   # noqa: E402

FLAC = (r"D:\MLS-ESS\SigGenNew\LFL-MuTPri\PrimeToneNewPhi[Nch 1][Frac 03][LFL 043]"
        r"[00020-21600][Nfr 027]-[48000-24]-[1s]-[-0.25dB]-[049062]-[ 8.5905]-[RD].flac")
FS = 48000

x, sr = sf.read(FLAC, dtype="float64")
N = len(x)
stdx = float(np.std(x))
rng = np.random.default_rng(0)
noises = rng.standard_normal((64, N)) * (0.2 * stdx)
batch = x + noises

imfs_g, nsift_g, nmode_g = EG.gpu_eemd_full(
    batch, E=64, tau=0.25, max_sifts=50, max_imf=32, nthreads=128)

worst = (0.0, -1)
divergent = []
rows = []
for e in range(64):
    re = M.decompose(batch[e], method="cpu", E=1, tau=0.25, max_imf=32, seed=0)
    nc, ng = int(re.n_modes), int(nmode_g[e])
    m = min(nc, ng)
    per_mode = []
    for i in range(m):
        da = float(np.max(np.abs(re.imfs[i] - imfs_g[e, i, :])))
        per_mode.append(da)
    mx = max(per_mode) if per_mode else 0.0
    tag = ""
    if mx > 1e-10:
        first = int(np.argmax(np.array(per_mode) > 1e-10))
        divergent.append((e, nc, ng, mx, first, per_mode))
        tag = f"  <-- SPLIT at mode {first + 1} (n {nc}/{ng})"
    if mx > worst[0]:
        worst = (mx, e)
    rows.append(dict(trial=e, n_cpu=nc, n_gpu=ng, max_abs=mx,
                     first_split_mode=(int(np.argmax(np.array(per_mode) > 1e-10)) + 1)
                     if mx > 1e-10 else None,
                     per_mode=[float(v) for v in per_mode]))
    print(f"trial {e:2d}: n={nc}/{ng} max|d|={mx:.3e}{tag}")

import json
import pathlib
out = pathlib.Path(__file__).resolve().parent / "mechanism_out" / "t64_diag.json"
with open(out, "w") as f:
    json.dump(dict(worst_trial=int(worst[1]), worst_max_abs=float(worst[0]),
                   n_divergent=len(divergent), per_trial=rows), f, indent=2)
print(f"\nsaved -> {out}")

print(f"\nworst trial: {worst[1]} max|d|={worst[0]:.3e}")
print(f"divergent trials (>1e-10): {len(divergent)}/64")
for e, nc, ng, mx, first, pm in divergent:
    print(f"  trial {e}: n={nc}/{ng} first-split mode={first + 1} "
          f"per-mode: {[f'{v:.1e}' for v in pm]}")
