"""test_phase3_gates.py -- Phase 3 integration verification (BLOCK_THOMAS_PLAN §6).

Gates the DSBT kernel (parallel tile-scan + seam-window Thomas) against the
PRE-CHANGE GPU baseline (baseline_r3.npz, frozen by capture_baseline_r3.py
before the emd_gpu.py surgery) and the CPU reference:

  G1  per-IMF max|d| vs pre-change GPU baseline, configs A & B   <= 5e-13
  G2  nmode agreement (new vs pre-change, & vs CPU)  equal, +/-1 explained
  G3  per-trial total_sifts  |diff| vs pre-change (and vs CPU)    <= 1
  G4  partition of unity (recon)                                 <= 1e-14
  G5  determinism (config B, same seed twice)  bit-identical     max|d| == 0
  G8  pathological-spacing E2E: strongly modulated chirp (knot
      spacing CV ~ 1.5) through the GPU vs the CPU reference     per-IMF <= 1e-12

Configs (identical to capture_baseline_r3.py, so the noise streams match the
frozen baseline exactly):
  A: E=1, N=110250, plain reference sweep (E=1 = plain EMD, no noise)
  B: E=8, N=110250, seed-7 perturbed  x + 0.2*std(x)*N(0,1)

G6 (test_emd_gpu.py full battery) and G7 (test_review3_fixes.py +
final_module_test.py) run as separate scripts.

Run:  .venv/Scripts/python.exe test_phase3_gates.py
Exit: 0 = all gates pass; 1 = at least one failure.
"""
import sys
import time
import warnings
import numpy as np

from numba.core import errors as _numba_errors
warnings.filterwarnings('ignore', category=_numba_errors.NumbaPerformanceWarning)

import emd_ref as R
import emd_gpu as G
import emd_par as M
from refsignal import make_ref_signal

FAILS = []
MAXIMF = 16          # identical to capture_baseline_r3.py
N = 110250

def check(name, cond, detail=""):
    print("  [%s] %s  %s" % ("PASS" if cond else "FAIL", name, detail))
    if not cond:
        FAILS.append(name)

def per_imf_max_d(new, base):
    """Worst per-IMF max|d| over the common depth of two (depth, N) stacks."""
    m = min(new.shape[0], base.shape[0])
    if m == 0:
        return 0.0
    return float(max(np.max(np.abs(new[i] - base[i])) for i in range(m)))

def main():
    t_start = time.time()
    xref, _ = make_ref_signal()
    x2 = xref[:N]
    print("Phase 3 integration gates   (N=%d, MAXIMF=%d)" % (N, MAXIMF))

    # ---------------- frozen pre-change baseline ----------------
    base = np.load('baseline_r3.npz', allow_pickle=False)
    A_imfs_b = base['A_imfs']          # (1, mA, N)
    A_nsift_b = base['A_nsift']        # (1,)
    A_nmode_b = base['A_nmode']        # (1,)
    B_imfs_b = base['B_imfs']          # (8, mB, N)
    B_nsift_b = base['B_nsift']        # (8,)
    B_nmode_b = base['B_nmode']        # (8,)
    print("baseline: A nmode=%d nsift=%d | B nmode=%s nsift=%s"
          % (int(A_nmode_b[0]), int(A_nsift_b[0]),
             list(B_nmode_b), list(B_nsift_b)))

    # ---------------- new kernel (DSBT) ----------------
    print("\n--- running new DSBT kernel (config A) ---")
    t0 = time.time()
    A_imfs, A_nsift, A_nmode = G.gpu_eemd_full(
        x2[None, :], E=1, tau=0.25, max_sifts=50, max_imf=MAXIMF)
    dtA = time.time() - t0
    print("  A: nmode=%d nsift=%d  (%.2fs)"
          % (int(A_nmode[0]), int(A_nsift[0]), dtA))

    print("--- running new DSBT kernel (config B, run 1) ---")
    rng = np.random.default_rng(7)
    X = x2[None, :] + 0.2 * np.std(x2) * rng.standard_normal((8, N))
    t0 = time.time()
    B_imfs, B_nsift, B_nmode = G.gpu_eemd_full(
        X, E=8, tau=0.25, max_sifts=50, max_imf=MAXIMF)
    dtB = time.time() - t0
    print("  B: nmode=%s nsift=%s  (%.2fs)"
          % (list(B_nmode), list(B_nsift), dtB))

    # ---------------- CPU references ----------------
    print("\n--- CPU reference (config A, E=1) ---")
    t0 = time.time()
    cA = M.decompose(x2, method='cpu', E=1, tau=0.25, max_imf=MAXIMF)
    dtcA = time.time() - t0
    print("  n_modes=%d total_sifts=%d recon=%.2e  (%.1fs)"
          % (cA.n_modes, cA.stats['total_sifts'], cA.recon_error(), dtcA))

    print("--- CPU reference (config B, E=8) ---")
    t0 = time.time()
    cB = M.decompose(x2, method='cpu', E=8, eps=0.2, tau=0.25, seed=7,
                     max_imf=MAXIMF)
    dtcB = time.time() - t0
    cB_sifts = cB.stats['per_trial_total_sifts']
    print("  n_modes=%d per-trial sifts=%s recon=%.2e  (%.1fs)"
          % (cB.n_modes, list(cB_sifts), cB.recon_error(), dtcB))

    # ================= G1: per-IMF vs pre-change baseline =================
    print("\n=== G1: per-IMF max|d| vs pre-change GPU baseline (<= 5e-13) ===")
    dA = per_imf_max_d(A_imfs[0], A_imfs_b[0])
    check("G1 config A per-IMF vs baseline", dA <= 5e-13,
          "max|d|=%.2e over %d IMFs" % (dA, min(A_imfs.shape[1],
                                                A_imfs_b.shape[1])))
    dB_worst, dB_worst_e = 0.0, -1
    for e in range(8):
        m = min(B_imfs[e].shape[0] if B_imfs.ndim == 3 else B_imfs.shape[1],
                B_imfs_b[e].shape[1])
        d = per_imf_max_d(B_imfs[e, :m], B_imfs_b[e, :m])
        if d > dB_worst:
            dB_worst, dB_worst_e = d, e
    check("G1 config B per-IMF vs baseline (worst of 8 trials)",
          dB_worst <= 5e-13,
          "max|d|=%.2e (trial %d)" % (dB_worst, dB_worst_e))

    # ================= G2: nmode agreement =================
    print("\n=== G2: nmode agreement (equal; +/-1 must be explained) ===")
    dA_n = int(A_nmode[0]) - int(A_nmode_b[0])
    check("G2 config A nmode new vs baseline", dA_n == 0,
          "new=%d baseline=%d (diff %+d)" % (int(A_nmode[0]),
                                              int(A_nmode_b[0]), dA_n))
    check("G2 config A nmode new vs CPU",
          abs(int(A_nmode[0]) - cA.n_modes) <= 1,
          "new=%d CPU=%d" % (int(A_nmode[0]), cA.n_modes))
    dB_n = [int(a) - int(b) for a, b in zip(B_nmode, B_nmode_b)]
    check("G2 config B nmode new vs baseline (per trial)",
          all(abs(d) == 0 for d in dB_n),
          "new=%s baseline=%s (diffs=%s)" % (list(B_nmode), list(B_nmode_b),
                                             dB_n))
    check("G2 config B nmode vs CPU ensemble (reported)",
          all(abs(int(a) - cB.n_modes) <= 1 for a in B_nmode),
          "new per-trial=%s CPU ensemble n_modes=%d" % (list(B_nmode),
                                                        cB.n_modes))

    # ================= G3: per-trial total_sifts =================
    print("\n=== G3: per-trial total_sifts (|diff| <= 1) ===")
    dA_s = int(A_nsift[0]) - int(A_nsift_b[0])
    check("G3 config A total_sifts new vs baseline", abs(dA_s) <= 1,
          "new=%d baseline=%d (diff %+d)" % (int(A_nsift[0]),
                                             int(A_nsift_b[0]), dA_s))
    check("G3 config A total_sifts new vs CPU",
          abs(int(A_nsift[0]) - cA.stats['total_sifts']) <= 1,
          "new=%d CPU=%d" % (int(A_nsift[0]), cA.stats['total_sifts']))
    dB_s = [int(a) - int(b) for a, b in zip(B_nsift, B_nsift_b)]
    check("G3 config B total_sifts new vs baseline (per trial)",
          all(abs(d) <= 1 for d in dB_s),
          "new=%s baseline=%s (diffs=%s)" % (list(B_nsift), list(B_nsift_b),
                                             dB_s))
    dB_s_cpu = [int(a) - int(b) for a, b in zip(B_nsift, cB_sifts)]
    check("G3 config B total_sifts new vs CPU (per trial, reported)",
          all(abs(d) <= 1 for d in dB_s_cpu),
          "new=%s CPU=%s (diffs=%s)" % (list(B_nsift), list(cB_sifts),
                                        dB_s_cpu))

    # ================= G4: partition of unity =================
    print("\n=== G4: partition of unity (<= 1e-14) ===")
    rec_cA = cA.recon_error()
    check("G4 CPU A recon", rec_cA <= 1e-14, "recon=%.2e" % rec_cA)
    rec_cB = cB.recon_error()
    check("G4 CPU B recon", rec_cB <= 1e-14, "recon=%.2e" % rec_cB)
    # GPU implied residual: resid = x - sum(imfs); check it is consistent
    # with the CPU residual at the G1 level (residual diff = sum of the
    # per-IMF diffs, each <= 5e-13).
    resid_gA = x2 - A_imfs[0].sum(axis=0)
    dresA = float(np.max(np.abs(resid_gA - cA.residual)))
    check("G4 GPU A implied residual vs CPU residual", dresA <= 1e-12,
          "max|d|=%.2e" % dresA)
    resid_gB = x2 - B_imfs.sum(axis=0) / 8.0   # ensemble mean, like the module
    dresB = float(np.max(np.abs(resid_gB - cB.residual)))
    print("  [info] GPU B implied residual vs CPU ensemble residual: %.2e"
          % dresB)

    # ================= G5: determinism =================
    print("\n=== G5: determinism (config B twice, bit-identical) ===")
    rng = np.random.default_rng(7)
    X2 = x2[None, :] + 0.2 * np.std(x2) * rng.standard_normal((8, N))
    B_imfs2, B_nsift2, B_nmode2 = G.gpu_eemd_full(
        X2, E=8, tau=0.25, max_sifts=50, max_imf=MAXIMF)
    m5 = min(B_imfs.shape[1], B_imfs2.shape[1])
    d5 = float(np.max(np.abs(B_imfs[:, :m5] - B_imfs2[:, :m5])))
    check("G5 bit-identical (max|d| == 0)", d5 == 0.0,
          "max|d|=%.1e over %d IMFs x 8 trials" % (d5, m5))
    check("G5 nsift/nmode identical",
          np.array_equal(B_nsift, B_nsift2) and np.array_equal(B_nmode,
                                                               B_nmode2),
          "nsift=%s nmode=%s" % (list(B_nsift2), list(B_nmode2)))

    # ================= G8: pathological spacing E2E =================
    print("\n=== G8: pathological-spacing E2E (modulated chirp, CV ~ 1.5) ===")
    N8 = 441000
    t8 = np.arange(N8) / 44100.0
    f0, a, Tm = 50.0, 0.95, 8.0
    f = f0 * (1.0 + a * np.sin(2 * np.pi * t8 / Tm))
    phi = 2 * np.pi * np.cumsum(f) / 44100.0
    x8 = np.sin(phi)
    x8 = x8 + 0.01 * np.random.default_rng(0).standard_normal(N8)

    # measure the actual knot-spacing CV of the constructed signal
    kmax = np.where((x8[1:-1] > x8[:-2]) & (x8[1:-1] > x8[2:]))[0] + 1
    kmin = np.where((x8[1:-1] < x8[:-2]) & (x8[1:-1] < x8[2:]))[0] + 1
    k = np.sort(np.concatenate([kmax, kmin]))
    hsp = np.diff(k.astype(float))
    cv8 = float(hsp.std() / hsp.mean())
    print("  signal: N=%d  f(t)=%.0f*(1+%.2f*sin(2*pi*t/%.0fs))"
          % (N8, f0, a, Tm))
    print("  extrema=%d  knot-spacing CV=%.3f (target ~1.5)" % (len(k), cv8))
    check("G8 signal is pathological (CV >= 1.2)", cv8 >= 1.2,
          "measured CV=%.3f" % cv8)

    t0 = time.time()
    g8, g8_ns, g8_nm = G.gpu_eemd_full(x8[None, :], E=1, tau=0.25,
                                       max_sifts=50, max_imf=MAXIMF)
    dtg8 = time.time() - t0
    print("  GPU: nmode=%d nsift=%d  (%.1fs)" % (int(g8_nm[0]),
                                                 int(g8_ns[0]), dtg8))
    t0 = time.time()
    c8_imfs, c8_resid, c8_st = R.emd_1d(x8, tau=0.25, max_sifts=50)
    dtc8 = time.time() - t0
    c8_imfs = np.array(c8_imfs)
    print("  CPU: n_modes=%d total_sifts=%d  (%.1fs)"
          % (c8_st['n_modes'], c8_st['total_sifts'], dtc8))

    d8 = per_imf_max_d(g8[0], c8_imfs)
    check("G8 per-IMF GPU vs CPU (<= 1e-12)", d8 <= 1e-12,
          "max|d|=%.2e over %d IMFs" % (d8, min(g8.shape[1],
                                                c8_st['n_modes'])))
    check("G8 nmode GPU vs CPU (reported)",
          abs(int(g8_nm[0]) - c8_st['n_modes']) <= 1,
          "GPU=%d CPU=%d" % (int(g8_nm[0]), c8_st['n_modes']))

    # ================= summary =================
    print("\n=== SUMMARY (wall %.1fs) ===" % (time.time() - t_start))
    if FAILS:
        print("  FAILED gates: %s" % ", ".join(FAILS))
        return 1
    print("  All Phase-3 gates G1-G5, G8 PASSED.")
    return 0

if __name__ == '__main__':
    sys.exit(main())
