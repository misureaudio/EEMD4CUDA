"""
dbg_serial_split.py  --  PHASE-0 (throwaway, read-only vs shipped code)
=====================================================================
Measure the per-phase time split of the production kernel at N=441000, E=1.

numba 0.67 has no device-side clock64, so instead of in-kernel timestamps we
use PHASE ABLATION: each phase's exact production loop body is extracted into
its own micro-kernel and timed in isolation on REAL data (the reference
signal + its true knots/envelopes from emd_ref). A host-instrumented copy of
emd_1d records the per-sift knot counts of the actual run, so the isolated
phase costs can be weighted into a full T_block model:

    T_model = sum_sifts [scanU + buildU + thomasU + mfU
                         + scanL + buildL + thomasL + mfL
                         + evalSD + store]
            + sum_imfs [residCount]

If T_model closes with the measured T_block (4.62 s) within ~10-15%, the
breakdown is trustworthy and the Amdahl projection for the block-Thomas +
parallel-scan plan follows directly.

NO shipped file is modified. Run:  .venv/Scripts/python.exe dbg_serial_split.py
"""
import time
import numpy as np
from numba import cuda

import emd_ref as R
from refsignal import make_ref_signal

N_THREADS = 128


# ----------------------------------------------------------------------
# host-instrumented EMD: per-sift knot counts of the REAL run
# ----------------------------------------------------------------------
def profile_emd(x, tau=0.25, max_sifts=50):
    """Mirror emd_ref.emd_1d EXACTLY (incl. the h_prev SD stop), but record
    the per-sift (nmax, nmin) knot counts of the real run."""
    t = np.arange(len(x), dtype=float)
    x = np.asarray(x, dtype=float)
    resid = x.copy()
    per_sift = []          # (nm_max, nm_min) in sift order
    n_imfs = 0
    total_sifts = 0
    while True:
        nmax = int(np.sum((resid[1:-1] > resid[:-2]) & (resid[1:-1] > resid[2:])))
        nmin = int(np.sum((resid[1:-1] < resid[:-2]) & (resid[1:-1] < resid[2:])))
        if nmax + nmin < 4:
            break
        h = resid.copy()
        h_prev = None
        for _ in range(max_sifts):
            nmax = int(np.sum((h[1:-1] > h[:-2]) & (h[1:-1] > h[2:])))
            nmin = int(np.sum((h[1:-1] < h[:-2]) & (h[1:-1] < h[2:])))
            if nmax < 1 or nmin < 1:
                break
            eu, _ = R.envelope(t, h, 'max')
            el, _ = R.envelope(t, h, 'min')
            m = 0.5 * (eu + el)
            h_new = h - m
            total_sifts += 1
            per_sift.append((nmax, nmin))
            if h_prev is not None:
                num = np.sum((h_prev - h_new)**2)
                den = np.sum(h_prev**2)
                sd = num/den if den > 0 else 0.0
                if sd < tau:
                    h = h_new
                    break
            h_prev = h_new
            h = h_new
        n_imfs += 1
        resid = resid - h
    return per_sift, n_imfs, total_sifts


# ----------------------------------------------------------------------
# micro-kernels: EXACT production loop bodies, one phase each
# ----------------------------------------------------------------------
@cuda.jit(device=True, inline=True)
def _thomas(d, l, u, r, x, c_, n):
    if n <= 0:
        return
    c_[0] = (u[0] / d[0]) if n > 1 else 0.0
    x[0] = r[0] / d[0]
    for i in range(1, n):
        c_[i] = (u[i] / (d[i] - l[i]*c_[i-1])) if i < n-1 else 0.0
        x[i] = (r[i] - l[i]*x[i-1]) / (d[i] - l[i]*c_[i-1])
    for i in range(n-2, -1, -1):
        x[i] = x[i] - c_[i]*x[i+1]


@cuda.jit(device=True, inline=True)
def _spline_eval_bin(tm, ym, M, t, p):
    if p < 2:
        return ym[0] if p == 1 else 0.0
    lo = 0
    hi = p - 1
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if tm[mid] <= t:
            lo = mid
        else:
            hi = mid
    i = lo
    if i < 0:
        i = 0
    if i > p - 2:
        i = p - 2
    h = float(tm[i+1] - tm[i])
    if h <= 0.0:
        return ym[i]
    a = (float(tm[i+1]) - t) / h
    b = (t - float(tm[i])) / h
    return (a*ym[i] + b*ym[i+1]
            + ((a*a*a - a)*M[i] + (b*b*b - b)*M[i+1]) * (h*h) / 6.0)


def _k_scan(N, K):
    @cuda.jit
    def _kern(hv, kt, ky, K):
        if cuda.threadIdx.x == 0:
            nmax = 0
            for i in range(1, N-1):
                if (hv[i] > hv[i-1]) and (hv[i] > hv[i+1]):
                    if nmax < K:
                        kt[nmax] = np.int32(i)
                        ky[nmax] = hv[i]
                    nmax += 1
    return _kern


def _k_count(N):
    @cuda.jit
    def _kern(rs, nm_s, nn_s):
        if cuda.threadIdx.x == 0:
            nm = 0; nn = 0
            for i in range(1, N-1):
                if rs[i] > rs[i-1]:
                    if rs[i] > rs[i+1]:
                        nm += 1
                if rs[i] < rs[i-1]:
                    if rs[i] < rs[i+1]:
                        nn += 1
            nm_s[0] = nm; nn_s[0] = nn
    return _kern


def _k_build(K):
    @cuda.jit
    def _kern(kt, ky, H, dd, ll, uu, rr, nm):
        if cuda.threadIdx.x == 0:
            for j in range(nm-1):
                H[j] = float(kt[j+1] - kt[j])
            nint = nm - 2
            for q in range(nint):
                dd[q] = 2.0*(H[q] + H[q+1])
                ll[q] = 0.0 if q == 0 else H[q]
                uu[q] = 0.0 if q == nint-1 else H[q+1]
                rr[q] = 6.0*((ky[q+2]-ky[q+1])/H[q+1] - (ky[q+1]-ky[q])/H[q])
    return _kern


def _k_thomas(K):
    @cuda.jit
    def _kern(dd, ll, uu, rr, xx, cc, nint):
        if cuda.threadIdx.x == 0:
            _thomas(dd, ll, uu, rr, xx, cc, nint)
    return _kern


def _k_mf(K):
    @cuda.jit
    def _kern(xx, Mf, nm, nint):
        if cuda.threadIdx.x == 0:
            Mf[0] = 0.0; Mf[nm-1] = 0.0
            for i in range(nint):
                Mf[i+1] = xx[i]
    return _kern


def _k_evalsd(N, K):
    @cuda.jit
    def _kern(hv, eur, kt, ky, Mf, red, nn):
        tid = cuda.threadIdx.x
        num = 0.0; den = 0.0
        for i in range(tid, N, N_THREADS):
            el_i = _spline_eval_bin(kt, ky, Mf, i, nn)
            mval = 0.5*(eur[i] + el_i)
            hnew = hv[i] - mval
            num += mval*mval
            den += hv[i]*hv[i]
            hv[i] = hnew
        red[tid] = num
        cuda.syncthreads()
        step = N_THREADS // 2
        while step >= 1:
            if tid < step:
                red[tid] += red[tid+step]
            cuda.syncthreads()
            step //= 2
    return _kern


def _k_store(N):
    @cuda.jit
    def _kern(hv, rs, outrow):
        tid = cuda.threadIdx.x
        for i in range(tid, N, N_THREADS):
            outrow[i] = hv[i]
            rs[i] = rs[i] - hv[i]
    return _kern


def time_kernel(launch, reps=3):
    launch()                      # warm (JIT + L2)
    cuda.synchronize()
    best = float('inf')
    for _ in range(reps):
        t0 = time.perf_counter()
        launch()
        cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    xref, _ = make_ref_signal()
    N = len(xref)
    K = max(16, N // 2)
    x = xref.astype(np.float64)

    print("=== host profile of the real E=1 run (emd_ref) ===")
    per_sift, n_imfs, total_sifts = profile_emd(x, tau=0.25, max_sifts=50)
    nmax_seq = [a for a, b in per_sift]
    nmin_seq = [b for a, b in per_sift]
    print("  n_imfs=%d  total_sifts=%d" % (n_imfs, total_sifts))
    print("  per-sift nmax: %s" % nmax_seq)

    # representative knot data (worst case = first sift, most knots)
    t = np.arange(N, dtype=float)
    kmax = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1
    kmin = np.where((x[1:-1] < x[:-2]) & (x[1:-1] < x[2:]))[0] + 1
    nm = len(kmax); nn = len(kmin)
    eu, _ = R.envelope(t, x, 'max')
    el, _ = R.envelope(t, x, 'min')
    # M arrays: recompute from the envelope system (host)
    def M_of(k, y):
        tm, ym = t[k], y[k]
        d, l, u, H, n = R._build_interior_system(tm, ym)
        Mint = R.thomas(d, l, u, R._rhs(tm, ym, H, n))
        Mf = np.zeros(len(tm)); Mf[1:-1] = Mint
        return Mf
    Mmax = M_of(kmax, x)
    Mmin = M_of(kmin, x)

    # device buffers
    hv = cuda.to_device(x.copy())
    rs = cuda.to_device(x.copy())
    eur = cuda.to_device(eu.copy())
    kt = cuda.to_device(np.ascontiguousarray(kmax.astype(np.int32)))
    ky = cuda.to_device(x[kmax].copy())
    Mf = cuda.to_device(Mmin.copy())
    W = cuda.to_device(np.zeros(9*K, dtype=np.float64))
    red = cuda.to_device(np.zeros(N_THREADS, dtype=np.float64))
    nm_s = cuda.to_device(np.zeros(1, dtype=np.int32))
    outrow = cuda.to_device(np.zeros(N, dtype=np.float64))

    k_scan = _k_scan(N, K)
    k_count = _k_count(N)
    k_build = _k_build(K)
    k_thomas = _k_thomas(K)
    k_mf = _k_mf(K)
    k_evalsd = _k_evalsd(N, K)
    k_store = _k_store(N)

    # ---- measure each phase (worst-case knot counts) ----
    ts_scan = time_kernel(lambda: k_scan[1, N_THREADS](hv, kt, ky, K))
    ts_count = time_kernel(lambda: k_count[1, N_THREADS](rs, nm_s, nm_s))
    ts_build = time_kernel(lambda: k_build[1, N_THREADS](kt, ky, W[0:K], W[2*K:3*K],
                                                        W[4*K:5*K], W[5*K:6*K],
                                                        W[6*K:7*K], nm))
    ts_thomas = time_kernel(lambda: k_thomas[1, N_THREADS](W[2*K:3*K], W[4*K:5*K],
                                                          W[5*K:6*K], W[6*K:7*K],
                                                          W[6*K:7*K], W[7*K:8*K],
                                                          nm - 2))
    ts_mf = time_kernel(lambda: k_mf[1, N_THREADS](W[6*K:7*K], W[8*K:9*K], nm, nm - 2))
    ts_evalsd = time_kernel(lambda: k_evalsd[1, N_THREADS](hv, eur, kt, ky, Mf, red, nn))
    ts_store = time_kernel(lambda: k_store[1, N_THREADS](hv, rs, outrow))

    print("\n=== isolated phase costs (worst-case knots: nm=%d nn=%d) ===" % (nm, nn))
    phases = [("scan (1 thr, N)", ts_scan, "serial"),
              ("count (1 thr, N)", ts_count, "serial"),
              ("build (1 thr, nm)", ts_build, "serial"),
              ("thomas (1 thr, nm-2)", ts_thomas, "serial"),
              ("mf fill (1 thr, nm)", ts_mf, "serial"),
              ("evalSD (128 thr, N)", ts_evalsd, "parallel"),
              ("store (128 thr, N)", ts_store, "parallel")]
    for name, tsec, tag in phases:
        print("  %-22s %10.4f s   [%s]" % (name, tsec, tag))

    # ---- weight into the full run (per-sift actual knot counts) ----
    # scan/count/evalSD/store: ~constant per sift (scale with N)
    # build/thomas/mf: scale with nint = nm-2 -> unit cost per knot x actual count
    sum_nint_max = sum(max(0, a - 2) for a, b in per_sift)
    sum_nint_min = sum(max(0, b - 2) for a, b in per_sift)
    unit_build = (ts_build / max(1, nm - 2))
    unit_thomas = (ts_thomas / max(1, nm - 2))
    unit_mf = (ts_mf / max(1, nm - 2))
    T_serial = (2 * total_sifts * ts_scan          # MAX + MIN scan per sift
                + n_imfs * ts_count                # residual count per IMF
                + (sum_nint_max + sum_nint_min)
                * (unit_build + unit_thomas + unit_mf))
    T_parallel = total_sifts * (ts_evalsd + ts_store)
    T_model = T_serial + T_parallel
    print("\n=== T_block model (weighted by real per-sift knot counts) ===")
    print("  total_sifts=%d  n_imfs=%d  sum(nint_max)=%d  sum(nint_min)=%d"
          % (total_sifts, n_imfs, sum_nint_max, sum_nint_min))
    print("  SERIAL   (2 scans/sift + counts + build+thomas+mf x nint): %7.3f s  (%5.1f%%)"
          % (T_serial, 100*T_serial/T_model))
    print("  PARALLEL (evalSD+store, 128 thr)                        : %7.3f s  (%5.1f%%)"
          % (T_parallel, 100*T_parallel/T_model))
    print("  T_model total                                           : %7.3f s" % T_model)
    print("  (measured T_block = 4.62 s; gap = launch/syncthreads overhead)")
    print("\n  Amdahl: serial fraction = %.1f%%" % (100*T_serial/T_model))
    print("  block-Thomas+parallel-scan removes ~all of it;")
    print("  new floor = parallel evalSD+store = %.2f s" % T_parallel)
    print("  projected T_block_new ~= %.2f-%.2f s  (%.1f-%.1fx speedup)"
          % (T_parallel, T_parallel*1.3,
             4.62/(T_parallel*1.3), 4.62/T_parallel))


if __name__ == '__main__':
    main()
