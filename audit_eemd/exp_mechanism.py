"""exp_mechanism.py — resolves the four open unknowns from
docs/boundary_sensitivity_results.md.

Context discovery (verified by reading the code): the INSTALLED production
emd_gpu.py (imported by emd_par) is the PRE-DSBT kernel — exact serial
Thomas + single-thread (tid==0) extrema scans, no EXACT/B/S params. The DSBT
+ parallel-scan kernel exists separately as emd_gpu_v4_dsbt.py / emd_gpu_dsbt/.
API_MANUAL.md describes the DSBT kernel as production -> manual/code desync.
Therefore the G3 O(1) CPU<->GPU divergence cannot be the DSBT seam error.

Part A: bisection of the CPU<->GPU divergence
  A0  real CPU E=1 + real GPU E=1 (reference)
  A1  instrumented CPU replica (exact Thomas) — validated bit-identical to A0
  A2  instrumented CPU replica (DSBT solve injected) — "what DSBT would change"
  A3  trajectory diff A1 vs A2 (per-sift amplification, first divergence)
  A4  M1: envelope eval CPU vs GPU on real first-sift knots
  A5  M2: knot collection CPU vs GPU on the same array (control)
  A6  M3: SD summation CPU vs GPU on real sift-1 arrays
  A7  SD-threshold proximity + extremum-boundary events (nm>=2 vs nmax>=1)
Part B: E=64 BC invariance (CPU natural vs CPU mirror vs GPU natural)
Part C: band-locking mechanism (2-tone ratio sweep, 1-per-octave control,
        thinned-27 control)
Part D: E->infinity convergence rate (E = 16..256 via gpu_eemd_full,
        per-trial nmode, per-mode rel-L2 vs E=256, exponent fit)

Run:  D:\\Source\\hermes-dir\\.venv\\Scripts\\python.exe exp_mechanism.py
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
import emd_par as M            # noqa: E402
import emd_ref as R            # noqa: E402
import emd_gpu as EG           # noqa: E402
from test_block_thomas import dsbt_solve  # noqa: E402
from numba import cuda         # noqa: E402

FLAC = (r"D:\MLS-ESS\SigGenNew\LFL-MuTPri\PrimeToneNewPhi[Nch 1][Frac 03][LFL 043]"
        r"[00020-21600][Nfr 027]-[48000-24]-[1s]-[-0.25dB]-[049062]-[ 8.5905]-[RD].flac")
FS = 48000
OUT = HERE / "mechanism_out"
OUT.mkdir(exist_ok=True)
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
    return [[round(max(f, 0.0), 2), round(a, 3)] for f, a in out]


def relmax_list(a, b):
    common = min(len(a), len(b))
    return [float(np.max(np.abs(a[k] - b[k])) / np.max(np.abs(b[k])))
            for k in range(common)]


# =====================================================================
# Instrumented CPU EMD replica (verbatim logic from emd_ref.py, solver
# switchable, per-sift trajectory recorded).
# =====================================================================
def _build_interior_system(tm, ym):
    p = len(tm)
    H = np.diff(tm)
    n = p - 2
    d = 2.0 * (H[0:n] + H[1:n + 1])
    l = np.zeros(n); u = np.zeros(n)
    l[1:] = H[1:n]
    u[0:n - 1] = H[1:n]
    return d, l, u, H, n


def _rhs(tm, ym, H, n):
    hp = H[0:n]; hn = H[1:n + 1]
    yp = ym[0:n]; yk = ym[1:n + 1]; yn = ym[2:n + 2]
    return 6.0 * ((yn - yk) / hn - (yk - yp) / hp)


def env_rep(t, x, want, solver, end='natural', rec=None):
    if want == 'max':
        k = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1
    else:
        k = np.where((x[1:-1] < x[:-2]) & (x[1:-1] < x[2:]))[0] + 1
    p = len(k)
    if p < 2:
        return np.full_like(t, x[k[0]] if p == 1 else 0.0), p, None
    tm, ym = t[k], x[k]
    if end == 'mirror' and p >= 3:
        dt0 = tm[1] - tm[0]; dt1 = tm[-1] - tm[-2]
        tm2 = np.concatenate(([tm[0] - dt0], tm, [tm[-1] + dt1]))
        ym2 = np.concatenate(([ym[1]], ym, [ym[-2]]))
    else:
        tm2, ym2 = tm, ym
    d, l, u, H, n = _build_interior_system(tm2, ym2)
    r = _rhs(tm2, ym2, H, n)
    if solver == 'exact':
        M_int = R.thomas(d, l, u, r)
    else:
        M_int = dsbt_solve(d, l, u, r, B=250, S=30)
    M_full = np.zeros(len(tm2)); M_full[1:-1] = M_int
    env = R._spline_eval(tm2, ym2, M_full, t)
    if rec is not None:
        rec.append(dict(knots=k.copy(), tm=tm2, ym=ym2, M=M_full, n=p, want=want))
    return env, p, M_full


def emd_1d_rep(x, t, solver, tau=0.25, max_sifts=50, max_imf=-1,
               min_extrema=4, end='natural'):
    x = np.asarray(x, dtype=float)
    imfs = []
    resid = x.copy()
    n_modes = 0
    total_sifts = 0
    per_mode_sifts = []
    traj = []
    while True:
        nmax = np.sum((resid[1:-1] > resid[:-2]) & (resid[1:-1] > resid[2:]))
        nmin = np.sum((resid[1:-1] < resid[:-2]) & (resid[1:-1] < resid[2:]))
        if nmax + nmin < min_extrema:
            break
        if max_imf != -1 and n_modes >= max_imf:
            break
        h = resid.copy()
        h_prev = None
        n_sift = 0
        mode_traj = []
        for _ in range(max_sifts):
            nmax = np.sum((h[1:-1] > h[:-2]) & (h[1:-1] > h[2:]))
            nmin = np.sum((h[1:-1] < h[:-2]) & (h[1:-1] < h[2:]))
            if nmax < 1 or nmin < 1:
                break
            recs = []
            eu, _, _ = env_rep(t, h, 'max', solver, end, recs)
            el, _, _ = env_rep(t, h, 'min', solver, end, recs)
            m = 0.5 * (eu + el)
            h_new = h - m
            n_sift += 1
            sd = None
            if h_prev is not None:
                num = np.sum((h_prev - h_new) ** 2)
                den = np.sum(h_prev ** 2)
                sd = num / den if den > 0 else 0.0
                if sd < tau:
                    h = h_new
                    mode_traj.append(dict(nmax=int(nmax), nmin=int(nmin), sd=sd,
                                          h=h_new.copy(), m=m.copy(),
                                          solves=recs, stop=True))
                    break
            h_prev = h_new
            h = h_new
            mode_traj.append(dict(nmax=int(nmax), nmin=int(nmin), sd=sd,
                                  h=h.copy(), m=m.copy(), solves=recs, stop=False))
        imfs.append(h)
        n_modes += 1
        total_sifts += n_sift
        per_mode_sifts.append(n_sift)
        traj.append(mode_traj)
        resid = resid - h
    stats = dict(n_modes=n_modes, total_sifts=total_sifts,
                 per_mode_sifts=per_mode_sifts)
    return imfs, resid, stats, traj


# =====================================================================
# GPU microkernels (replicate production fragments exactly)
# =====================================================================
@cuda.jit
def _k_eval(kt, ky, Mf, env, p, N):
    t = cuda.threadIdx.x
    for i in range(t, N, 128):
        env[i] = EG._spline_eval_bin(kt, ky, Mf, float(i), p)


@cuda.jit
def _k_scan(h, out, N, K):
    if cuda.threadIdx.x == 0:
        n = 0
        for i in range(1, N - 1):
            if (h[i] > h[i - 1]) and (h[i] > h[i + 1]):
                if n < K:
                    out[n] = i
                n += 1
        out[K] = n


@cuda.jit
def _k_sd(h, m, num, den, N):
    t = cuda.threadIdx.x
    nb = 0.0
    nd = 0.0
    for i in range(t, N, 128):
        nb += m[i] * m[i]
        nd += h[i] * h[i]
    red = cuda.shared.array(128, dtype=np.float64)
    red[t] = nb
    cuda.syncthreads()
    step = 64
    while step >= 1:
        if t < step:
            red[t] += red[t + step]
        cuda.syncthreads()
        step //= 2
    if t == 0:
        num[0] = red[0]
    red[t] = nd
    cuda.syncthreads()
    step = 64
    while step >= 1:
        if t < step:
            red[t] += red[t + step]
        cuda.syncthreads()
        step //= 2
    if t == 0:
        den[0] = red[0]


def gpu_eval(kt, ky, Mf, N):
    import cupy as cp
    env = cp.empty(N, dtype=np.float64)
    _k_eval[1, 128](cp.asarray(kt), cp.asarray(ky), cp.asarray(Mf), env,
                    len(kt), N)
    return cp.asnumpy(env)


def gpu_scan(h, K):
    import cupy as cp
    out = cp.zeros(K + 1, dtype=np.int64)
    _k_scan[1, 128](cp.asarray(h), out, len(h), K)
    o = cp.asnumpy(out)
    return o[:int(o[K])], int(o[K])


def gpu_sd(h, m):
    import cupy as cp
    num = cp.zeros(1, dtype=np.float64)
    den = cp.zeros(1, dtype=np.float64)
    _k_sd[1, 128](cp.asarray(h), cp.asarray(m), num, den, len(h))
    return float(cp.asnumpy(num)[0]), float(cp.asnumpy(den)[0])


def main():
    x, sr = sf.read(FLAC, dtype="float64")
    assert sr == FS and len(x) == 48000
    N = len(x)
    t = np.arange(N, dtype=float)

    # ---------------- A0: real CPU E=1 + real GPU E=1 ----------------
    t0 = time.time()
    rc = M.decompose(x, method="cpu", E=1, tau=0.25, max_imf=32, seed=0)
    w_cpu = time.time() - t0
    t0 = time.time()
    rg = M.decompose(x, method="gpu", E=1, tau=0.25, max_imf=32, seed=0)
    w_gpu = time.time() - t0
    RES["a0_reference"] = dict(
        cpu_n_modes=int(rc.n_modes), gpu_n_modes=int(rg.n_modes),
        cpu_wall_s=w_cpu, gpu_wall_s=w_gpu,
        cpu_total_sifts=int(rc.stats.get("total_sifts", -1)),
        gpu_total_sifts=int(rg.stats.get("per_trial_total_sifts", [-1])[0]),
        cpu_per_mode_sifts=[int(v) for v in rc.stats.get("per_mode_sifts", [])],
        per_mode_relmax=[round(v, 6) for v in relmax_list(rc.imfs, rg.imfs)],
        cpu_peaks=[top_peaks(imf, FS) for imf in rc.imfs],
        gpu_peaks=[top_peaks(imf, FS) for imf in rg.imfs])
    print(f"[A0] CPU n={rc.n_modes} ({w_cpu:.2f}s) GPU n={rg.n_modes} "
          f"({w_gpu:.2f}s) sifts {rc.stats.get('total_sifts')} vs "
          f"{rg.stats.get('per_trial_total_sifts')}")
    dump()

    # ---------------- A1: instrumented CPU replica (exact) ----------------
    t0 = time.time()
    imf_e, res_e, st_e, traj_e = emd_1d_rep(x, t, "exact")
    w_rep = time.time() - t0
    bit_ok = all(np.array_equal(a, b) for a, b in zip(imf_e, rc.imfs)) and \
        np.array_equal(res_e, rc.residual)
    RES["a1_replica_validation"] = dict(bit_identical_to_real_cpu=bool(bit_ok),
                                        wall_s=w_rep)
    print(f"[A1] replica exact: bit-identical to real CPU = {bit_ok} "
          f"({w_rep:.2f}s)")
    dump()

    # ---------------- A2: instrumented CPU replica (DSBT) ----------------
    t0 = time.time()
    imf_d, res_d, st_d, traj_d = emd_1d_rep(x, t, "dsbt")
    w_dsbt = time.time() - t0
    rd = relmax_list(imf_d, imf_e)
    RES["a2_cpu_dsbt"] = dict(
        n_modes=int(st_d["n_modes"]), wall_s=w_dsbt,
        total_sifts=int(st_d["total_sifts"]),
        per_mode_sifts=[int(v) for v in st_d["per_mode_sifts"]],
        per_mode_relmax_vs_exact=[round(v, 6) for v in rd],
        max_relmax=float(max(rd)),
        peaks=[top_peaks(imf, FS) for imf in imf_d])
    print(f"[A2] CPU+DSBT: n={st_d['n_modes']} max_relmax_vs_exact="
          f"{max(rd):.3e} ({w_dsbt:.2f}s)")
    dump()

    # ---------------- A3: trajectory diff exact vs dsbt ----------------
    diffs = []
    for m in range(min(len(traj_e), len(traj_d))):
        mode_diffs = []
        for s in range(min(len(traj_e[m]), len(traj_d[m]))):
            de = traj_e[m][s]; dd = traj_d[m][s]
            dh = float(np.max(np.abs(de["h"] - dd["h"])))
            knots_same = (de["solves"][0]["knots"].tolist() ==
                          dd["solves"][0]["knots"].tolist()) and \
                        (de["solves"][1]["knots"].tolist() ==
                         dd["solves"][1]["knots"].tolist())
            Me, Md = de["solves"][0]["M"], dd["solves"][0]["M"]
            dM = (float(np.max(np.abs(Me - Md)))
                  if Me.shape == Md.shape else None)
            mode_diffs.append(dict(sift=s, max_dh=dh, knots_same=knots_same,
                                   max_dM_solve0=dM,
                                   sd_exact=de["sd"], sd_dsbt=dd["sd"]))
        diffs.append(mode_diffs)
    first_div = None
    amplifs = []
    prev_dh = None
    for m, mode_diffs in enumerate(diffs):
        for d in mode_diffs:
            if d["max_dh"] > 0 and first_div is None:
                first_div = dict(mode=m + 1, sift=d["sift"], max_dh=d["max_dh"])
            if prev_dh is not None and prev_dh > 0 and d["max_dh"] > 0:
                amplifs.append(d["max_dh"] / prev_dh)
            prev_dh = d["max_dh"]
    RES["a3_trajectory_diff"] = dict(
        first_divergence=first_div,
        per_sift_max_dh=[[d["max_dh"] for d in md] for md in diffs],
        knot_divergence_sifts=[[d["sift"] for d in md if not d["knots_same"]]
                               for md in diffs],
        max_dM_solve0_per_mode=[max((d["max_dM_solve0"] for d in md
                                     if d["max_dM_solve0"] is not None),
                                    default=None)
                                for md in diffs],
        amplification_factors=[round(a, 4) for a in amplifs[:60]],
        amplification_max=float(max(amplifs)) if amplifs else None)
    print(f"[A3] first divergence: {first_div}")
    if amplifs:
        print(f"     per-sift amplification: max={max(amplifs):.3f} "
              f"mean={np.mean(amplifs):.3f}")
    dump()

    # ---------------- A4/A5/A6: micro-experiments M1/M2/M3 ----------------
    import cupy as cp
    h0 = traj_e[0][0]
    kt = h0["solves"][0]["tm"]; ky = h0["solves"][0]["ym"]; Mf = h0["solves"][0]["M"]
    p = len(kt)
    env_cpu = R._spline_eval(kt, ky, Mf, t)
    env_gpu = gpu_eval(kt, ky, Mf, N)
    m1 = dict(max_abs=float(np.max(np.abs(env_cpu - env_gpu))),
              max_rel=float(np.max(np.abs(env_cpu - env_gpu)) /
                            np.max(np.abs(env_cpu))))
    h_arr = x  # h at sift 0 == input
    k_cpu = np.where((h_arr[1:-1] > h_arr[:-2]) & (h_arr[1:-1] > h_arr[2:]))[0] + 1
    k_gpu, n_gpu = gpu_scan(h_arr, max(16, N // 2))
    m2 = dict(n_cpu=int(len(k_cpu)), n_gpu=int(n_gpu),
              identical=bool(np.array_equal(k_cpu, k_gpu)))
    s1 = traj_e[0][1]
    h_prev = traj_e[0][0]["h"]
    m_arr = s1["m"]
    h_new = s1["h"]
    num_c = float(np.sum((h_prev - h_new) ** 2))
    den_c = float(np.sum(h_prev ** 2))
    num_g, den_g = gpu_sd(h_prev, m_arr)
    sd_c = num_c / den_c
    sd_g = num_g / den_g
    m3 = dict(sd_cpu=sd_c, sd_gpu=sd_g, abs_diff=abs(sd_c - sd_g),
              rel_diff=abs(sd_c - sd_g) / sd_c,
              dist_to_tau_cpu=abs(sd_c - 0.25))
    RES["a456_micro"] = dict(M1_envelope_eval=m1, M2_knot_scan=m2,
                             M3_sd_summation=m3)
    print(f"[A4] M1 env eval CPU vs GPU: max_abs={m1['max_abs']:.3e} "
          f"max_rel={m1['max_rel']:.3e}")
    print(f"[A5] M2 knot scan: n_cpu={m2['n_cpu']} n_gpu={m2['n_gpu']} "
          f"identical={m2['identical']}")
    print(f"[A6] M3 SD: cpu={sd_c:.17g} gpu={sd_g:.17g} "
          f"rel_diff={m3['rel_diff']:.3e} dist_to_tau={m3['dist_to_tau_cpu']:.3e}")
    dump()

    # ---------------- A7: threshold proximity + boundary events ----------------
    sds = [d["sd"] for md in traj_e for d in md if d["sd"] is not None]
    sds = np.array(sds)
    prox = np.abs(sds - 0.25)
    nmax_sift = [d["nmax"] for md in traj_e for d in md]
    nmin_sift = [d["nmin"] for md in traj_e for d in md]
    a7 = dict(
        n_sd_values=len(sds),
        min_dist_to_tau=float(prox.min()),
        n_within_1e_6=int((prox < 1e-6).sum()),
        n_within_1e_9=int((prox < 1e-9).sum()),
        n_within_1e_12=int((prox < 1e-12).sum()),
        min_nmax_sift=int(min(nmax_sift)), min_nmin_sift=int(min(nmin_sift)),
        n_siftings_with_nmax_eq_1=int((np.array(nmax_sift) == 1).sum()),
        n_siftings_with_nmin_eq_1=int((np.array(nmin_sift) == 1).sum()))
    RES["a7_boundaries"] = a7
    print(f"[A7] min |sd-tau| = {prox.min():.3e}; "
          f"nmax/nmin==1 events: {a7['n_siftings_with_nmax_eq_1']}/"
          f"{a7['n_siftings_with_nmin_eq_1']}")
    dump()

    # ---------------- B: E=64 BC invariance ----------------
    t0 = time.time()
    bn = M.decompose(x, method="cpu", E=64, tau=0.25, eps=0.2, max_imf=32,
                     seed=0, end="natural")
    wn = time.time() - t0
    t0 = time.time()
    bm = M.decompose(x, method="cpu", E=64, tau=0.25, eps=0.2, max_imf=32,
                     seed=0, end="mirror")
    wm = time.time() - t0
    bg = M.decompose(x, method="gpu", E=64, tau=0.25, eps=0.2, max_imf=32,
                     seed=0)
    rb = relmax_list(bn.imfs, bm.imfs)
    RES["b_bc_e64"] = dict(
        cpu_natural_n_modes=int(bn.n_modes), cpu_mirror_n_modes=int(bm.n_modes),
        gpu_natural_n_modes=int(bg.n_modes),
        walls=dict(natural=wn, mirror=wm, gpu=0.0),
        natural_peaks=[top_peaks(imf, FS) for imf in bn.imfs],
        mirror_peaks=[top_peaks(imf, FS) for imf in bm.imfs],
        gpu_peaks=[top_peaks(imf, FS) for imf in bg.imfs],
        per_mode_relmax_natural_vs_mirror=[round(v, 6) for v in rb],
        max_relmax=float(max(rb)),
        per_mode_relmax_cpu_nat_vs_gpu=[round(v, 6) for v in
                                        relmax_list(bn.imfs, bg.imfs)])
    print(f"[B] E=64: natural n={bn.n_modes} ({wn:.1f}s) mirror n={bm.n_modes} "
          f"({wm:.1f}s) gpu n={bg.n_modes} | nat-vs-mir max relmax="
          f"{max(rb):.3e}")
    dump()

    # ---------------- C: band-locking mechanism ----------------
    tone_f = [43.0, 53.0, 61.0, 83.0, 107.0, 127.0, 167.0, 211.0, 251.0,
              337.0, 421.0, 509.0, 683.0, 853.0, 1021.0, 1361.0, 1699.0,
              2039.0, 2731.0, 3413.0, 4093.0, 5449.0, 6827.0, 8191.0,
              10909.0, 13649.0, 16381.0]
    tone_amp = [0.2295, 0.20602, 0.18898, 0.16513, 0.14507, 0.13022,
                0.11659, 0.10353, 0.09294, 0.0822, 0.0734, 0.06505, 0.05784,
                0.05158, 0.046, 0.04102, 0.0366, 0.03263, 0.029, 0.02584,
                0.02304, 0.02055, 0.0183, 0.01631, 0.01454, 0.01295, 0.01155]

    def run_cpu(x2):
        t0 = time.time()
        r = M.decompose(x2, method="cpu", E=1, tau=0.25, max_imf=32, seed=0)
        return r, time.time() - t0

    c1 = []
    for r_ratio in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.4, 3.0]:
        f1, f2 = 1000.0, 1000.0 * r_ratio
        x2 = 0.5 * np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
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
              f"peaks={[p[0][0] for p in peaks]}")
    RES["c1_two_tone_sweep"] = c1
    dump()

    sparse_idx = [0, 4, 8, 12, 16]   # 43, 107, 251, 683, 1699 Hz
    x_sparse = sum(tone_amp[i] * np.sin(2 * np.pi * tone_f[i] * t)
                   for i in sparse_idx)
    r, w = run_cpu(x_sparse)
    RES["c2_sparse_control"] = dict(
        tones=[tone_f[i] for i in sparse_idx], n_modes=int(r.n_modes),
        per_mode_sifts=[int(v) for v in r.stats.get("per_mode_sifts", [])],
        peaks=[top_peaks(imf, FS) for imf in r.imfs], wall_s=w)
    print(f"[C2] sparse 1-per-octave (5 tones): n_modes={r.n_modes}")
    dump()

    thinned = list(range(0, 27, 2))
    x_thin = sum(tone_amp[i] * np.sin(2 * np.pi * tone_f[i] * t)
                 for i in thinned)
    r, w = run_cpu(x_thin)
    RES["c3_thinned27"] = dict(
        n_tones=len(thinned), n_modes=int(r.n_modes),
        per_mode_sifts=[int(v) for v in r.stats.get("per_mode_sifts", [])],
        peaks=[top_peaks(imf, FS) for imf in r.imfs], wall_s=w)
    print(f"[C3] thinned (14 tones): n_modes={r.n_modes}")
    dump()

    # ---------------- D: E -> infinity convergence ----------------
    stdx = float(np.std(x))
    conv = {}
    avgs = {}
    for E in [16, 32, 64, 128, 256]:
        t0 = time.time()
        rng = np.random.default_rng(0)
        noises = rng.standard_normal((E, N)) * (0.2 * stdx)
        imfs_b, nsift, nmode = EG.gpu_eemd_full(x + noises, E=E, tau=0.25,
                                                max_sifts=50, max_imf=32,
                                                nthreads=128)
        w = time.time() - t0
        maxm = imfs_b.shape[1]
        avg = np.zeros((32, N)); cnt = np.zeros(32)
        for m in range(maxm):
            mask = (nmode > m)
            avg[m] += imfs_b[mask, m, :].sum(axis=0)
            cnt[m] += int(mask.sum())
        n_modes = int((cnt > 0).sum())
        avg = avg[:n_modes] / np.where(cnt[:n_modes] > 0, cnt[:n_modes], 1)[:, None]
        avgs[E] = avg
        hist = {int(k): int(v) for k, v in
                zip(*np.unique(nmode, return_counts=True))}
        conv[E] = dict(n_modes=n_modes, wall_s=w,
                       per_trial_nmode_hist=hist,
                       cnt_per_mode=[int(v) for v in cnt[:n_modes]])
        print(f"[D] E={E}: n_modes={n_modes} ({w:.1f}s) nmode_hist={hist}")
    ref = avgs[256]
    rel_l2 = {}
    for E in [16, 32, 64, 128]:
        common = min(len(avgs[E]), len(ref))
        rel_l2[E] = [float(np.linalg.norm(avgs[E][k] - ref[k]) /
                           np.linalg.norm(ref[k])) for k in range(common)]
    exponents = {}
    for k in range(min(len(v) for v in rel_l2.values())):
        Es = [16, 32, 64, 128]
        ys = [rel_l2[E][k] for E in Es]
        mask = np.array([y > 0 for y in ys])
        if mask.sum() >= 3:
            slope = np.polyfit(np.log2(np.array(Es)[mask]),
                               np.log2(np.array(ys)[mask]), 1)[0]
            exponents[f"mode_{k+1}"] = float(slope)
    RES["d_convergence"] = dict(
        per_E=conv,
        rel_l2_vs_E256={str(k): [round(v, 12) for v in vv]
                        for k, vv in rel_l2.items()},
        fit_exponents_neg_alpha=exponents)
    # sanity: my E=64 average vs the saved decompose npz
    saved = M.EMDResult.load(HERE / "eemd_prime_tone_out" /
                             "primetone_eemd_E64_gpu.npz")
    common = min(len(avgs[64]), saved.n_modes)
    chk = [float(np.linalg.norm(avgs[64][k] - saved.imfs[k]) /
                 np.linalg.norm(saved.imfs[k])) for k in range(common)]
    RES["d_convergence"]["e64_vs_saved_npz_rel_l2"] = [round(v, 15) for v in chk]
    print(f"[D] E=64 vs saved npz: max rel-L2 = {max(chk):.3e}")
    dump()
    print(f"\ndone -> {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
