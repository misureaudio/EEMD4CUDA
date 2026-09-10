"""
test_emd_gpu.py -- definitive test + benchmark for emd_gpu.py
==============================================================

Part A -- CORRECTNESS (hard assertions; non-zero exit on failure)
  A1  Per-sift operator: GPU single sift vs CPU single sift     <= 1e-13
  A2  Full-IMF EMD (E=1): GPU vs CPU emd_1d
        - per-IMF agreement                                   <= 1e-8
        - |nmode_gpu - nmode_cpu|                               <= 1
        - CPU partition of unity (sanity)                      <= 1e-10
  A3  Determinism: same seed, twice  ->  bit-identical (max|d| == 0)
  A4  Different seed  ->  different result (ensemble not frozen)
  A5  API guard: 1-D input with E>1 raises ValueError
      (regression for the silent-E=1 bug that corrupted the early
       concurrency benchmarks)

Part B -- BENCHMARK (report only; medians over repeats)
  B1  Per-block time T_block (E=1, warm, median of 3) at N=110250, 441000
  B2  Saturation-concurrency sweep: wall vs E, C_est = E*T_block/wall
      (plateau of C_est = the saturation concurrency C_sat)
  B3  Device footprint per trial and per E
      (decides memory vs occupancy: which constraint caps concurrency?)
  B4  Host overhead: emd_par module vs raw kernel at E=8, N=441000

Run:  .venv/Scripts/python.exe test_emd_gpu.py
Exit: 0 = all correctness checks pass; 1 = at least one failure.
"""
import sys
import time
import warnings
import numpy as np

# EMD kernels launch one block per trial (grid = E); small E is intentional,
# so silence numba's "low occupancy" performance warning.
from numba.core import errors as _numba_errors
warnings.filterwarnings('ignore', category=_numba_errors.NumbaPerformanceWarning)

import emd_ref as R
import emd_gpu as G
import emd_par as M
from refsignal import make_ref_signal

FAILS = []
_XREF = None
MAXIMF = 16          # headroom above the sweep's 12-13 modes (no silent cap)

def check(name, cond, detail=""):
    print("  [%s] %s  %s" % ("PASS" if cond else "FAIL", name, detail))
    if not cond:
        FAILS.append(name)

def perturbed(E, N, seed=7):
    x = _XREF[:N]
    r = np.random.default_rng(seed)
    return x[None, :] + 0.2 * np.std(x) * r.standard_normal((E, N))

def med(xs):
    return float(np.median(xs))

def free_pool():
    """Return the cupy pool's cached (high-water) blocks to the driver.

    WHY: cp.cuda.runtime.memGetInfo() reports driver-level free memory.
    cupy's default memory pool does NOT release its high-water mark, so a
    large allocation followed by a memGetInfo() read sees ~0 bytes free.
    emd_par._gpu_decompose auto-batches from exactly that reading, so a
    dirty pool silently degrades it to batch=1 (E sequential single-block
    kernels). emd_par now calls free_all_blocks() itself (permanent fix),
    so these calls are belt-and-braces for the raw-kernel sweep timings.
    """
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()

def footprint_bytes(N, max_imf):
    """Per-trial device footprint: S, resid, h, eu (4 x N) + out (max_imf x N)
    + W (10 x N/2) doubles."""
    return N * 8 * (4 + max_imf) + 10 * (N // 2) * 8

def main():
    global _XREF
    xref, _ = make_ref_signal()
    _XREF = xref
    Nref = len(xref)
    print("Reference sweep: N=%d   (max_imf=%d)" % (Nref, MAXIMF))

    # ================= Part A: correctness =================
    print("\n=== Part A: correctness ===")

    # A1 -- per-sift operator
    N1 = 2048
    t1 = np.arange(N1)
    rng = np.random.default_rng(1)
    f1 = 0.01 + 0.05 * t1 / N1
    xs = (np.sin(2*np.pi*np.cumsum(f1)) + 0.3*np.sin(2*np.pi*0.013*t1)
          + 0.1*rng.standard_normal(N1))
    imf_g, ns1 = G.gpu_emd_single(xs, tau=0.25, max_sifts=1)
    eu, _ = R.envelope(t1, xs, 'max', 'natural')
    el, _ = R.envelope(t1, xs, 'min', 'natural')
    h1 = xs - 0.5 * (eu + el)
    d1 = float(np.max(np.abs(imf_g - h1)))
    check("A1 per-sift GPU vs CPU", d1 <= 1e-13,
          "max|d|=%.2e (n_sift=%d)" % (d1, ns1))

    # A2 -- full-IMF EMD, E=1, GPU vs CPU
    N2 = 110250
    x2 = _XREF[:N2]
    imfs_g, ns2, nm2 = G.gpu_eemd_full(x2[None, :], E=1, tau=0.25,
                                       max_sifts=50, max_imf=MAXIMF)
    imfs_c, resid_c, st_c = R.emd_1d(x2, tau=0.25, max_sifts=50)
    imfs_c = np.array(imfs_c)
    m = min(imfs_g.shape[1], imfs_c.shape[0])
    d2 = max(float(np.max(np.abs(imfs_g[0, i] - imfs_c[i]))) for i in range(m))
    dn = abs(imfs_g.shape[1] - imfs_c.shape[0])
    rec_c = float(np.max(np.abs(x2 - (imfs_c.sum(0) + resid_c))))
    check("A2 full-EMF per-IMF GPU vs CPU", d2 <= 1e-8,
          "max|d|=%.2e over %d IMFs" % (d2, m))
    check("A2 nmode agreement", dn <= 1,
          "GPU nmode=%d  CPU n_modes=%d" % (imfs_g.shape[1], imfs_c.shape[0]))
    check("A2 CPU partition of unity", rec_c <= 1e-10, "recon=%.2e" % rec_c)

    # A3 -- determinism (same seed, twice)
    a, _, _ = G.gpu_eemd_full(perturbed(8, N2), E=8, tau=0.25, max_imf=MAXIMF)
    b, _, _ = G.gpu_eemd_full(perturbed(8, N2), E=8, tau=0.25, max_imf=MAXIMF)
    d3 = float(np.max(np.abs(a - b)))
    check("A3 determinism (same seed)", d3 == 0.0, "max|d|=%.1e" % d3)

    # A4 -- different seed differs.
    # NOTE: gpu_eemd_full trims its output to max(nmode) across the batch, so
    # different seeds (different per-trial stop depths) yield DIFFERENT depths
    # (e.g. 10 vs 11 IMFs) -> compare over the common depth.
    c, _, _ = G.gpu_eemd_full(perturbed(8, N2, seed=8), E=8, tau=0.25,
                              max_imf=MAXIMF)
    mc = min(a.shape[1], c.shape[1])
    d4 = float(np.max(np.abs(a[:, :mc] - c[:, :mc])))
    check("A4 different seed differs", d4 > 0.0,
          "max|d|=%.2e over %d common IMFs (seed7 depth=%d, seed8 depth=%d)"
          % (d4, mc, a.shape[1], c.shape[1]))

    # A5 -- API guard (1-D input, E>1 must raise)
    try:
        G.gpu_eemd_full(x2, E=4, tau=0.25, max_imf=MAXIMF)
        check("A5 1-D + E>1 raises ValueError", False, "NO exception raised")
    except ValueError:
        check("A5 1-D + E>1 raises ValueError", True, "raised as expected")

    # ================= Part B: benchmark =================
    print("\n=== Part B: benchmark (warm) ===")
    ft = footprint_bytes(Nref, MAXIMF)
    print("  per-trial footprint: N=%d -> %.1f MB/trial (max_imf=%d)"
          % (Nref, ft / 1024**2, MAXIMF))

    for N, Es in [(110250, [8, 32, 128, 256]), (Nref, [8, 32, 64, 128])]:
        # B1 -- T_block (E=1, warm, median of 3)
        G.gpu_eemd_full(perturbed(1, N), E=1, tau=0.25, max_imf=MAXIMF)  # JIT
        tb = []
        for _ in range(3):
            t0 = time.perf_counter()
            G.gpu_eemd_full(perturbed(1, N), E=1, tau=0.25, max_imf=MAXIMF)
            tb.append(time.perf_counter() - t0)
        T = med(tb)
        print("\n  N=%d : T_block (E=1, warm, median of 3) = %.2fs   [runs: %s]"
              % (N, T, " ".join("%.2f" % v for v in tb)))
        # B2/B3 -- saturation sweep
        print("    E     wall(s)   C_est=E*T/wall   footprint(GB)")
        for E in Es:
            t0 = time.perf_counter()
            try:
                imfs, ns, nm = G.gpu_eemd_full(perturbed(E, N), E=E, tau=0.25,
                                               max_imf=MAXIMF)
            except Exception as e:
                # gpu_eemd_full has no internal batching: E=128 at N=441000
                # needs ~11.3 GB device memory -> CUDA OOM on an 8 GB card.
                # Record the ceiling instead of crashing the sweep.
                print("    %4d   %8s   %6s          %6.2f   %s"
                      % (E, "OOM", "-----",
                         E * footprint_bytes(N, MAXIMF) / 1024**3,
                         type(e).__name__))
                continue
            wall = time.perf_counter() - t0
            Cest = E * T / wall
            print("    %4d   %8.2f   %6.1f          %6.2f"
                  % (E, wall, Cest, E * footprint_bytes(N, MAXIMF) / 1024**3))
            free_pool()

    # B4 -- host overhead: module vs raw kernel (E=8, Nref).
    # free_pool() first: emd_par auto-batches from memGetInfo free, and the
    # default cupy pool keeps its high-water mark, so a dirty pool would make
    # the module run E sequential single-block kernels (batch=1) and the
    # "overhead" would be pool accounting, not real host work.
    free_pool()
    print("\n  B4 host overhead (E=8, N=%d):" % Nref)
    G.gpu_eemd_full(perturbed(8, Nref), E=8, tau=0.25, max_imf=MAXIMF)  # JIT
    kw, mw = [], []
    for _ in range(2):
        t0 = time.perf_counter()
        G.gpu_eemd_full(perturbed(8, Nref), E=8, tau=0.25, max_imf=MAXIMF)
        kw.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        r = M.decompose(_XREF, method='gpu', E=8, tau=0.25, seed=7,
                        max_imf=MAXIMF)
        mw.append(time.perf_counter() - t0)
    print("    raw kernel (median of 2): %.2fs" % med(kw))
    print("    emd_par module (median of 2): %.2fs   host overhead = %.2fs"
          % (med(mw), med(mw) - med(kw)))
    print("    module n_modes=%d  recon=%.2e" % (r.n_modes, r.recon_error()))

    # ================= summary =================
    print("\n=== SUMMARY ===")
    if FAILS:
        print("  FAILED checks: %s" % ", ".join(FAILS))
        return 1
    print("  All Part A correctness checks PASSED.")
    return 0

if __name__ == '__main__':
    sys.exit(main())
