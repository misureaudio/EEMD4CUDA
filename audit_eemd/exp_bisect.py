"""exp_bisect.py — pin down both divergence mechanisms.

Part 1: DSBT anatomy on the REAL first-sift system (n=9499 interior
unknowns from the 27-tone signal).
  P1a  max|M_dsbt - M_exact|, scale, error profile vs block boundaries
  P1b  S sweep (B=250): err vs S -> effective decay per step
  P1c  B sweep (S=30): err vs number of seams
  P1d  single interior window: standalone clipped solve vs global exact,
       error vs distance from window edges (measured decay base)
Part 2: CPU<->GPU first divergence (installed pre-DSBT kernel).
  P2a  debug GPU kernel (emd_gpu_dbg) vs instrumented CPU replica,
       sift-by-sift lockstep: first sift where knot sets / counts /
       sd / hv differ; margin analysis at the flipped comparison.

Run:  D:\\Source\\hermes-dir\\.venv\\Scripts\\python.exe exp_bisect.py
"""
import json
import math
import pathlib
import sys

import numpy as np
import soundfile as sf

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import emd_par as M            # noqa: E402
import emd_ref as R            # noqa: E402
from test_block_thomas import (dsbt_solve, build_system, rhs_of,  # noqa: E402
                               _dense, _window_thomas_np)
from exp_mechanism import emd_1d_rep  # noqa: E402

FLAC = (r"D:\MLS-ESS\SigGenNew\LFL-MuTPri\PrimeToneNewPhi[Nch 1][Frac 03][LFL 043]"
        r"[00020-21600][Nfr 027]-[48000-24]-[1s]-[-0.25dB]-[049062]-[ 8.5905]-[RD].flac")
OUT = HERE / "mechanism_out"
OUT.mkdir(exist_ok=True)
RES = {}


def dump():
    with open(OUT / "bisect_results.json", "w") as f:
        json.dump(RES, f, indent=2, default=float)


def main():
    x, sr = sf.read(FLAC, dtype="float64")
    N = len(x)
    t = np.arange(N, dtype=float)

    # CPU replica (exact) — provides the real first-sift knot system
    imf_e, res_e, st_e, traj_e = emd_1d_rep(x, t, "exact")
    s0 = traj_e[0][0]["solves"][0]      # mode 1, sift 0, MAX envelope
    tm, ym, M_full = s0["tm"], s0["ym"], s0["M"]
    d, l, u, H, n = build_system(tm)
    r = rhs_of(tm, ym, H, n)
    M_ex = R.thomas(d, l, u, r)
    print(f"[P1] system: p={len(tm)} knots, n={n} interior, "
          f"H min/mean/max = {H.min():.0f}/{H.mean():.2f}/{H.max():.0f}, "
          f"max|M_ex|={np.max(np.abs(M_ex)):.4e}")

    # ---------------- P1a: base case + error profile ----------------
    B, S = 250, 30
    M_ds = dsbt_solve(d, l, u, r, B=B, S=S)
    err = np.abs(M_ds - M_ex)
    scale = max(float(np.max(np.abs(M_ex))), 1e-300)
    nb = (n + B - 1) // B
    # distance of each interior index to nearest dropped-coupling boundary
    # boundaries at w0_k = k*B-S and w1_k = k*B+B-1+S (k=0..nb-1), clipped
    bounds = []
    for k in range(nb):
        w0 = max(0, k * B - S)
        w1 = min(n - 1, k * B + B - 1 + S)
        bounds += [w0, w1]
    bounds = np.array(sorted(set(bounds)))
    dist = np.min(np.abs(np.arange(n)[:, None] - bounds[None, :]), axis=1)
    p1a = dict(n=int(n), max_abs_err=float(err.max()),
               max_rel_err=float(err.max() / scale),
               mean_abs_err=float(err.mean()),
               max_M_scale=scale,
               err_at_dist_lt_5=float(err[dist < 5].max()),
               err_at_dist_10_20=float(err[(dist >= 10) & (dist <= 20)].max()),
               err_at_dist_gt_25=float(err[dist > 25].max()),
               top10_err_positions=[
                   dict(i=int(i), err=float(err[i]), dist_to_boundary=int(dist[i]),
                        H=float(H[i]), l=float(l[i]), u=float(u[i]),
                        d=float(d[i]))
                   for i in np.argsort(err)[::-1][:10]])
    RES["p1a_base"] = p1a
    np.savez(OUT / "dsbt_err_profile.npz", err=err, dist=dist, i=np.arange(n),
             M_ex=M_ex, M_ds=M_ds, H=H)
    print(f"[P1a] max_abs_err={err.max():.4e} (scale={scale:.4e}) "
          f"err(dist<5)={p1a['err_at_dist_lt_5']:.3e} "
          f"err(10-20)={p1a['err_at_dist_10_20']:.3e} "
          f"err(>25)={p1a['err_at_dist_gt_25']:.3e}")
    print("     top positions:",
          [(p['i'], f"{p['err']:.1e}", f"d={p['dist_to_boundary']}")
           for p in p1a["top10_err_positions"][:6]])
    dump()

    # ---------------- P1b: S sweep ----------------
    s_sweep = []
    for Ss in [5, 10, 20, 30, 60, 100, 200, 500]:
        e = np.abs(dsbt_solve(d, l, u, r, B=B, S=Ss) - M_ex).max()
        s_sweep.append(dict(S=int(Ss), max_abs_err=float(e)))
        print(f"[P1b] S={Ss:<4} max_err={e:.4e}")
    RES["p1b_s_sweep"] = s_sweep
    dump()

    # ---------------- P1c: B sweep ----------------
    b_sweep = []
    for Bb in [64, 125, 250, 500, 1000, 2500, 9499]:
        Bb = min(Bb, n)
        e = np.abs(dsbt_solve(d, l, u, r, B=Bb, S=S) - M_ex).max()
        b_sweep.append(dict(B=int(Bb), n_seams=int((n + Bb - 1) // Bb - 1),
                            max_abs_err=float(e)))
        print(f"[P1c] B={Bb:<5} seams={(n+Bb-1)//Bb-1:<4} max_err={e:.4e}")
    RES["p1c_b_sweep"] = b_sweep
    dump()

    # ---------------- P1d: single-window decay ----------------
    k = 10
    w0 = max(0, k * B - S)
    w1 = min(n - 1, k * B + B - 1 + S)
    # standalone clipped solve of the window (dense, independent)
    dw = d[w0:w1 + 1].copy()
    lw = l[w0:w1 + 1].copy()
    uw = u[w0:w1 + 1].copy()
    rw = r[w0:w1 + 1].copy()
    lw[0] = 0.0
    uw[-1] = 0.0
    xw = np.linalg.solve(_dense(dw, lw, uw, w1 - w0 + 1), rw)
    gex = M_ex[w0:w1 + 1]
    ew = np.abs(xw - gex)
    pos = np.arange(w0, w1 + 1)
    dleft = pos - w0
    dright = w1 - pos
    dmin = np.minimum(dleft, dright)
    # fit decay in the left half: log err vs dleft (excluding first 2)
    msk = (dleft >= 3) & (dleft <= S - 5) & (ew > 0)
    slope_l = (np.polyfit(dleft[msk], np.log(ew[msk]), 1)[0] if msk.sum() > 3
               else None)
    msk2 = (dright >= 3) & (dright <= S - 5) & (ew > 0)
    slope_r = (np.polyfit(dright[msk2], np.log(ew[msk2]), 1)[0]
               if msk2.sum() > 3 else None)
    p1d = dict(window=f"[{w0},{w1}]", max_err=float(ew.max()),
               err_at_center=float(ew[len(ew) // 2]),
               err_at_left_edge=float(ew[0]), err_at_right_edge=float(ew[-1]),
               decay_base_left=None if slope_l is None else float(math.exp(slope_l)),
               decay_base_right=None if slope_r is None else float(math.exp(slope_r)),
               err_vs_dist=[dict(d=int(dd), err=float(ee))
                            for dd, ee in zip(dmin[::10], ew[::10])])
    RES["p1d_window"] = p1d
    print(f"[P1d] window [{w0},{w1}]: edge_err={ew[0]:.3e}/{ew[-1]:.3e} "
          f"center_err={ew[len(ew)//2]:.3e} "
          f"decay_base_l={p1d['decay_base_left']} "
          f"decay_base_r={p1d['decay_base_right']}")
    dump()

    # ---------------- P1e/P1f: CORRECT host DSBT (owned-only writes,
    # mirroring the v4 DEVICE kernel semantics) ----------------
    def _window_thomas_copy(d, l, u, r, tmpx, tmpc, w0, w1):
        cprev = 0.0
        xprev = 0.0
        for i in range(w0, w1 + 1):
            denom = d[i] - l[i] * cprev
            cnew = 0.0 if i == w1 else u[i] / denom
            xnew = (r[i] - l[i] * xprev) / denom
            tmpc[i - w0] = cnew
            tmpx[i - w0] = xnew
            cprev = cnew
            xprev = xnew
        for i in range(w1 - 1, w0 - 1, -1):
            tmpx[i - w0] = tmpx[i - w0] - tmpc[i - w0] * tmpx[i - w0 + 1]

    def dsbt_solve_correct(d, l, u, r, B=250, S=30):
        n = len(d)
        x = np.zeros(n)
        tmpx = np.zeros(2 * B + 2 * S + 2)
        tmpc = np.zeros(2 * B + 2 * S + 2)
        nb = (n + B - 1) // B
        for blk in range(nb):
            o0 = blk * B
            o1 = min(n, o0 + B) - 1
            w0 = max(0, o0 - S)
            w1 = min(n - 1, o1 + S)
            _window_thomas_copy(d, l, u, r, tmpx, tmpc, w0, w1)
            x[o0:o1 + 1] = tmpx[o0 - w0:o1 - w0 + 1]
        return x

    M_ds_c = dsbt_solve_correct(d, l, u, r, B=B, S=S)
    err_c = np.abs(M_ds_c - M_ex)
    RES["p1e_correct_host_dsbt"] = dict(
        max_abs_err=float(err_c.max()),
        max_rel_err=float(err_c.max() / scale))
    print(f"[P1e] CORRECT host DSBT (owned-only writes): "
          f"max_abs_err={err_c.max():.4e} (buggy mirror: {err.max():.4e})")
    dump()

    # P1f: full EMD E=1 with the correct DSBT solver
    import exp_mechanism as EM
    EM.dsbt_solve = dsbt_solve_correct
    imf_dc, res_dc, st_dc, traj_dc = emd_1d_rep(x, t, "dsbt")
    rd = [float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
          for a, b in zip(imf_dc, imf_e)]
    RES["p1f_emd_correct_dsbt"] = dict(
        n_modes=int(st_dc["n_modes"]),
        per_mode_sifts=[int(v) for v in st_dc["per_mode_sifts"]],
        per_mode_relmax_vs_exact=[round(v, 12) for v in rd],
        max_relmax=float(max(rd)))
    print(f"[P1f] EMD E=1 with CORRECT DSBT: n_modes={st_dc['n_modes']} "
          f"(exact: {st_e['n_modes']}), max_relmax={max(rd):.3e}")
    dump()

    # ================= Part 2: CPU<->GPU first divergence =================
    import emd_gpu_dbg as DBG
    t0 = __import__("time").time()
    g = DBG.gpu_emd_dbg(x, tau=0.25, max_sifts=50, max_imf=32)
    w_dbg = __import__("time").time() - t0
    print(f"[P2] debug GPU: nmode={g['nmode']} nsift={g['nsift']} "
          f"({w_dbg:.1f}s), sifts dumped={len(g['hv'])}")

    # sanity: debug kernel must reproduce the production GPU result bit-wise
    rg = M.decompose(x, method="gpu", E=1, tau=0.25, max_imf=32, seed=0)
    n_common = min(len(g["imf"]), rg.n_modes)
    bit_gpu = all(np.array_equal(g["imf"][k], rg.imfs[k]) for k in range(n_common))
    print(f"[P2] debug vs production GPU: nmode {g['nmode']} vs {rg.n_modes}, "
          f"bit-identical={bit_gpu}")

    # CPU trajectory flattened (state at each sift)
    cpu_states = []
    h_start = x
    for m, mode_traj in enumerate(traj_e):
        for s, st in enumerate(mode_traj):
            cpu_states.append(dict(
                mode=m + 1, sift=s + 1, h_start=h_start,
                nmax=st["nmax"], nmin=st["nmin"],
                kmax=st["solves"][0]["knots"].copy(),
                kmin=st["solves"][1]["knots"].copy(),
                Mmax=st["solves"][0]["M"].copy(),
                Mmin=st["solves"][1]["M"].copy(),
                sd=st["sd"], h_end=st["h"].copy(), stop=st["stop"]))
            h_start = st["h"]
    gpu_n = len(g["hv"])
    print(f"[P2] lockstep: cpu sifts={len(cpu_states)} gpu sifts={gpu_n}")

    first_div = None
    report = []
    for idx in range(min(len(cpu_states), gpu_n)):
        c = cpu_states[idx]
        nmax_g = int(g["nmnn"][idx, 0])
        nmin_g = int(g["nmnn"][idx, 1])
        kmax_g = g["kt"][idx, 0, :nmax_g]
        kmin_g = g["kt"][idx, 1, :nmin_g]
        kmax_c = c["kmax"][:min(len(c["kmax"]), 24000)]
        kmin_c = c["kmin"][:min(len(c["kmin"]), 24000)]
        count_ok = (nmax_g == len(kmax_c)) and (nmin_g == len(kmin_c))
        same_max = count_ok and np.array_equal(kmax_g[:len(kmax_c)], kmax_c)
        same_min = count_ok and np.array_equal(kmin_g[:len(kmin_c)], kmin_c)
        dh = float(np.max(np.abs(g["hv"][idx] - c["h_end"])))
        sd_c = c["sd"]
        sd_g = float(g["sd"][idx])
        sd_diff = None if sd_c is None else abs(sd_g - sd_c)
        dMmax = None
        if same_max:
            dMmax = float(np.max(np.abs(g["Mf"][idx, 0, :nmax_g] -
                                        c["Mmax"][:nmax_g])))
        entry = dict(sift=idx + 1, cpu_mode=c["mode"], cpu_sift=c["sift"],
                     nmax=(len(kmax_c), nmax_g), nmin=(len(kmin_c), nmin_g),
                     same_knots=(bool(same_max), bool(same_min)),
                     max_dh=dh, sd=(sd_c, sd_g), sd_diff=sd_diff,
                     max_dM_max_env=dMmax)
        report.append(entry)
        if first_div is None and (not count_ok or not same_max or not same_min):
            first_div = entry
            # margin analysis at the flipped knots
            h_c = c["h_start"]
            h_g = (x if idx == 0 else g["hv"][idx - 1])
            flips = []
            if count_ok:
                sym = np.concatenate([kmax_g, kmax_c])
                for i in np.unique(sym):
                    is_max_c = i in set(kmax_c.tolist())
                    is_max_g = i in set(kmax_g.tolist())
                    if is_max_c != is_max_g:
                        flips.append(dict(
                            i=int(i), max_cpu=bool(is_max_c),
                            max_gpu=bool(is_max_g),
                            cpu=(float(h_c[i - 1]), float(h_c[i]),
                                 float(h_c[i + 1])),
                            gpu=(float(h_g[i - 1]), float(h_g[i]),
                                 float(h_g[i + 1])),
                            dH=(float(h_g[i - 1] - h_c[i - 1]),
                                float(h_g[i] - h_c[i]),
                                float(h_g[i + 1] - h_c[i + 1]))))
            else:
                # counts differ: report the first index where the sorted
                # knot lists stop matching
                a = np.concatenate([kmax_g, kmin_g])
                b2 = np.concatenate([kmax_c, kmin_c])
                la, lb = min(len(a), len(b2)), max(len(a), len(b2))
                for j in range(la):
                    if a[j] != b2[j]:
                        flips.append(dict(first_mismatch_pos=int(j),
                                          gpu_knot=int(a[j]),
                                          cpu_knot=int(b2[j])))
                        break
            first_div["flips"] = flips[:20]
            first_div["n_flips_max"] = len(flips)
            break
    RES["p2_first_divergence"] = dict(
        debug_vs_production_bit_identical=bool(bit_gpu),
        n_cpu_sifts=len(cpu_states), n_gpu_sifts=gpu_n,
        first_divergence=first_div,
        comparison_table=report[:max(1, (first_div or {}).get('sift', 1) + 3)])
    if first_div:
        print(f"[P2] FIRST DIVERGENCE at global sift {first_div['sift']} "
              f"(cpu mode {first_div['cpu_mode']}, sift {first_div['cpu_sift']}): "
              f"nmax={first_div['nmax']} nmin={first_div['nmin']} "
              f"n_flips={first_div.get('n_flips_max')}")
        for fl in first_div.get("flips", [])[:5]:
            print("     ", fl)
    else:
        print("[P2] no knot divergence in common sifts; "
              f"cpu total={len(cpu_states)} gpu total={gpu_n}")
    dump()
    print(f"\ndone -> {OUT / 'bisect_results.json'}")


if __name__ == "__main__":
    main()
