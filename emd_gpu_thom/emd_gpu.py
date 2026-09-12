"""
emd_gpu.py  --  GPU (cupy + numba.cuda) parallel 1-D EMD / EEMD.

Full-IMF version: each thread block runs a COMPLETE EMD (all IMFs) of one
perturbed signal; the EEMD ensemble is averaged host-side. This is what the
reusable module (emd_par.py) needs to return the entire decomposition.

Performance-critical fixes vs the earlier draft:
  * O(N) spline evaluation: the knot interval is found by BINARY SEARCH
    (O(log p)) instead of a linear scan (O(p)). At 72k knots on 441k samples
    the linear scan was the whole bottleneck (92 s); binary search is ~17
    comparisons per sample.
  * Full-IMF loop inside the block (the irreducible sequential critical path),
    with per-trial working buffers in global memory (10*K doubles would exceed
    the 48 KB shared limit, so knots/system arrays are global).
  * The cubic-spline tridiagonal system is the EXACT 2-dominant system from
    part (a): factor-2 diagonally dominant, NO pivot, Thomas.

Determinism: for a fixed seed the result is bit-identical across runs
(verified: same-seed max|d| = 0).
"""
from __future__ import annotations
import os
import numpy as np

try:
    from numba import cuda
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False


def _bootstrap_nvvm():
    """Preload cudart.dll + nvvm.dll by full path so numba's bare-name CDLL
    resolves and it can read the toolkit version -> supported compute caps."""
    if not _HAVE_NUMBA or os.name != 'nt':
        return
    try:
        import ctypes
        cands = [os.path.join(os.getcwd(), 'cuda_libs', 'cudart.dll'),
                 os.path.join(os.getcwd(), 'cuda_libs', 'nvvm.dll')]
        ch = os.environ.get('CUDA_HOME')
        if ch:
            cands += [os.path.join(ch, 'bin', 'cudart.dll'),
                      os.path.join(ch, 'nvvm', 'bin', 'nvvm.dll'),
                      os.path.join(ch, 'nvvm', 'lib64', 'nvvm.dll')]
        for p in cands:
            if os.path.isfile(p):
                try:
                    ctypes.CDLL(p)
                except OSError:
                    pass
    except Exception:
        pass


_bootstrap_nvvm()


# ---- device helpers (independent of N/K) ---------------------------------
@cuda.jit(device=True, inline=True)
def _thomas(d, l, u, r, x, c_, n):
    """No-pivot Thomas for the 2-dominant tridiagonal system (part (a))."""
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
    """Cubic-spline value at t; knots tm[0..p-1] (time order), 2nd derivs M.
    Interval found by BINARY SEARCH -> O(log p). Natural BC (M[0]=M[p-1]=0)."""
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
    h = tm[i+1] - tm[i]
    if h <= 0.0:
        return ym[i]
    a = (tm[i+1] - t) / h
    b = (t - tm[i]) / h
    return (a*ym[i] + b*ym[i+1]
            + ((a*a*a - a)*M[i] + (b*b*b - b)*M[i+1]) * (h*h) / 6.0)


# ---- kernel factory: bakes N, K, MAXIMF, nthreads as compile-time consts --
_KERNEL_CACHE = {}


def _make_full_kernel(N, K, MAXIMF, nthreads):
    """Build (and cache) a CUDA kernel that runs a FULL EMD per block.
    Buffers (all global, per block b):
      S      [E,N]  perturbed input signal (read-only)
      resid  [E,N]  working residual (in place)
      h      [E,N]  working h during sifting
      out    [E,MAXIMF,N]  extracted IMFs
      eu     [E,N]  upper-envelope scratch
      W      [E,10*K]  knot/system scratch (2 knot + 7 system + M)
      nsift  [E]     total siftings per trial
      nmode  [E]     number of IMFs per trial
    """

    @cuda.jit
    def _kern(S, resid, h, out, eu, W, nsift, nmode,
              tau, max_sifts, min_extrema):
        b = cuda.blockIdx.x
        tid = cuda.threadIdx.x
        sig = S[b, :]
        rs  = resid[b, :]
        hv  = h[b, :]
        eur = eu[b, :]
        # per-block working arrays (GLOBAL), 10 slabs of K from W[b, :]
        kt = W[b, 0*K:1*K]
        ky = W[b, 1*K:2*K]
        H  = W[b, 2*K:3*K]
        dd = W[b, 3*K:4*K]
        ll = W[b, 4*K:5*K]
        uu = W[b, 5*K:6*K]
        rr = W[b, 6*K:7*K]
        xx = W[b, 7*K:8*K]
        cc = W[b, 8*K:9*K]
        Mf = W[b, 9*K:10*K]
        red = cuda.shared.array(nthreads, dtype=np.float64)
        nm_s = cuda.shared.array(1, dtype=np.int32)
        nn_s = cuda.shared.array(1, dtype=np.int32)
        stop = cuda.shared.array(1, dtype=np.int32)

        # init: resid = S, h = S
        for i in range(tid, N, nthreads):
            rs[i] = sig[i]
            hv[i] = sig[i]

        total_sifts = 0
        nmode_b = 0
        for m in range(MAXIMF):
            cuda.syncthreads()
            # ---- count extrema of residual (thread 0) ----
            if tid == 0:
                nm = 0; nn = 0
                for i in range(1, N-1):
                    if rs[i] > rs[i-1]:
                        if rs[i] > rs[i+1]:
                            nm += 1
                    if rs[i] < rs[i-1]:
                        if rs[i] < rs[i+1]:
                            nn += 1
                nm_s[0] = nm; nn_s[0] = nn
            cuda.syncthreads()
            nm = nm_s[0]; nn = nn_s[0]
            if (nm + nn) < min_extrema or nm < 2 or nn < 2:
                break

            # ---- sift hv (a copy of rs) to an IMF ----
            # hv must equal rs at the start of the sift
            for i in range(tid, N, nthreads):
                hv[i] = rs[i]
            stop[0] = 0
            nsift_local = 0
            for s in range(max_sifts):
                cuda.syncthreads()
                if stop[0] != 0:
                    break
                # collect MAX knots (thread 0, time order)
                if tid == 0:
                    nmax = 0
                    for i in range(1, N-1):
                        if (hv[i] > hv[i-1]) and (hv[i] > hv[i+1]):
                            if nmax < K:
                                kt[nmax] = i; ky[nmax] = hv[i]
                            nmax += 1
                    nm_s[0] = nmax
                cuda.syncthreads()
                nm = nm_s[0]
                if nm < 2:
                    break
                # UPPER envelope (thread 0)
                if tid == 0:
                    for j in range(nm-1):
                        H[j] = kt[j+1] - kt[j]
                    nint = nm - 2
                    for q in range(nint):
                        dd[q] = 2.0*(H[q] + H[q+1])
                        ll[q] = 0.0 if q == 0 else H[q]
                        uu[q] = 0.0 if q == nint-1 else H[q+1]
                        rr[q] = 6.0*((ky[q+2]-ky[q+1])/H[q+1] - (ky[q+1]-ky[q])/H[q])
                    _thomas(dd, ll, uu, rr, xx, cc, nint)
                    Mf[0] = 0.0; Mf[nm-1] = 0.0
                    for i in range(nint):
                        Mf[i+1] = xx[i]
                cuda.syncthreads()
                for i in range(tid, N, nthreads):
                    eur[i] = _spline_eval_bin(kt, ky, Mf, i, nm)
                cuda.syncthreads()
                # collect MIN knots (thread 0, overwrites kt,ky)
                if tid == 0:
                    nmin = 0
                    for i in range(1, N-1):
                        if (hv[i] < hv[i-1]) and (hv[i] < hv[i+1]):
                            if nmin < K:
                                kt[nmin] = i; ky[nmin] = hv[i]
                            nmin += 1
                    nn_s[0] = nmin
                cuda.syncthreads()
                nn = nn_s[0]
                if nn < 2:
                    break
                # LOWER envelope (thread 0)
                if tid == 0:
                    for j in range(nn-1):
                        H[j] = kt[j+1] - kt[j]
                    nint = nn - 2
                    for q in range(nint):
                        dd[q] = 2.0*(H[q] + H[q+1])
                        ll[q] = 0.0 if q == 0 else H[q]
                        uu[q] = 0.0 if q == nint-1 else H[q+1]
                        rr[q] = 6.0*((ky[q+2]-ky[q+1])/H[q+1] - (ky[q+1]-ky[q])/H[q])
                    _thomas(dd, ll, uu, rr, xx, cc, nint)
                    Mf[0] = 0.0; Mf[nn-1] = 0.0
                    for i in range(nint):
                        Mf[i+1] = xx[i]
                cuda.syncthreads()
                # eval lower + mean + subtract + SD accumulate
                num = 0.0; den = 0.0
                for i in range(tid, N, nthreads):
                    el_i = _spline_eval_bin(kt, ky, Mf, i, nn)
                    mval = 0.5*(eur[i] + el_i)
                    hnew = hv[i] - mval
                    num += mval*mval
                    den += hv[i]*hv[i]
                    hv[i] = hnew
                # block-reduce num
                red[tid] = num
                cuda.syncthreads()
                step = nthreads // 2
                while step >= 1:
                    if tid < step:
                        red[tid] += red[tid+step]
                    cuda.syncthreads()
                    step //= 2
                numsum = red[0]
                cuda.syncthreads()
                red[tid] = den
                cuda.syncthreads()
                step = nthreads // 2
                while step >= 1:
                    if tid < step:
                        red[tid] += red[tid+step]
                    cuda.syncthreads()
                    step //= 2
                densum = red[0]
                nsift_local += 1
                if s >= 1:
                    sd = (numsum/densum) if densum > 0 else 0.0
                    if sd < tau:
                        if tid == 0:
                            stop[0] = 1
                        break
            total_sifts += nsift_local
            # store IMF, update residual
            outrow = out[b, m, :]
            for i in range(tid, N, nthreads):
                outrow[i] = hv[i]
                rs[i] = rs[i] - hv[i]
            nmode_b += 1

        if tid == 0:
            nsift[b] = total_sifts
            nmode[b] = nmode_b

    return _kern


def get_full_kernel(N, K, MAXIMF, nthreads):
    key = (N, K, MAXIMF, nthreads)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = _make_full_kernel(N, K, MAXIMF, nthreads)
    return _KERNEL_CACHE[key]


def _pick_K(N):
    return max(16, N // 2)


def gpu_eemd_full(x, E=64, tau=0.25, max_sifts=50, max_imf=12,
                  seed=0, K=None, nthreads=128, batch=None):
    """GPU EEMD, FULL decomposition. x: [E, N] perturbed signals (or [N]).
    Returns (imfs [E,MAXIMF,N], nsift [E], nmode [E])."""
    import cupy as cp
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        # 1-D input is a SINGLE trial. Refuse E>1 with a 1-D signal: the
        # previous version silently reset E=1 here, which corrupted
        # concurrency benchmarks (they ran one trial, not E).
        if E != 1:
            raise ValueError(
                "1-D input x is a single trial (E=1). For E>1 pass a 2-D "
                "(E, N) array of perturbed signals, e.g. x[None,:] + noises.")
        x = x[None, :]
    E, N = x.shape
    K = K or _pick_K(N)
    kern = get_full_kernel(N, K, max_imf, nthreads)
    S = cp.asarray(x)
    resid = cp.empty((E, N), dtype=np.float64)
    h = cp.empty((E, N), dtype=np.float64)
    out = cp.zeros((E, max_imf, N), dtype=np.float64)
    eu = cp.empty((E, N), dtype=np.float64)
    W = cp.empty((E, 10*K), dtype=np.float64)
    nsift = cp.zeros(E, dtype=np.int32)
    nmode = cp.zeros(E, dtype=np.int32)
    kern[E, nthreads](S, resid, h, out, eu, W, nsift, nmode,
                      tau, max_sifts, 4)
    imfs = cp.asnumpy(out)
    nsift_np = cp.asnumpy(nsift)
    nmode_np = cp.asnumpy(nmode)
    # trim to max nmode across trials
    maxm = int(nmode_np.max()) if nmode_np.size else 0
    return imfs[:, :maxm, :], nsift_np, nmode_np


# keep the old single-IMF API for backward compat (used by earlier tests)
def gpu_eemd(x, E=64, eps=0.2, tau=0.25, max_sifts=50, seed=0,
             K=None, nthreads=128):
    """GPU EEMD, FIRST IMF only (legacy). Returns (avg, nsift, raw)."""
    import cupy as cp
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    K = K or _pick_K(N)
    kern = get_full_kernel(N, K, 1, nthreads)
    rng = np.random.default_rng(seed)
    noises = rng.standard_normal((E, N)) * (eps * np.std(x))
    S = cp.asarray(x[None, :] + noises)
    resid = cp.empty((E, N), dtype=np.float64)
    h = cp.empty((E, N), dtype=np.float64)
    out = cp.zeros((E, 1, N), dtype=np.float64)
    eu = cp.empty((E, N), dtype=np.float64)
    W = cp.empty((E, 10*K), dtype=np.float64)
    nsift = cp.zeros(E, dtype=np.int32)
    nmode = cp.zeros(E, dtype=np.int32)
    kern[E, nthreads](S, resid, h, out, eu, W, nsift, nmode, tau, max_sifts, 4)
    imfs = cp.asnumpy(out)[:, 0, :]
    return imfs.mean(axis=0), cp.asnumpy(nsift), imfs


def gpu_emd_single(x, t=None, tau=0.25, max_sifts=50, K=None, nthreads=128):
    """GPU EMD of a single signal, FIRST IMF. Returns (imf, n_sift)."""
    import cupy as cp
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    K = K or _pick_K(N)
    kern = get_full_kernel(N, K, 1, nthreads)
    S = cp.asarray(x[None, :])
    resid = cp.empty((1, N), dtype=np.float64)
    h = cp.empty((1, N), dtype=np.float64)
    out = cp.zeros((1, 1, N), dtype=np.float64)
    eu = cp.empty((1, N), dtype=np.float64)
    W = cp.empty((1, 10*K), dtype=np.float64)
    nsift = cp.zeros(1, dtype=np.int32)
    nmode = cp.zeros(1, dtype=np.int32)
    kern[1, nthreads](S, resid, h, out, eu, W, nsift, nmode, tau, max_sifts, 4)
    return cp.asnumpy(out[0, 0, :]), int(cp.asnumpy(nsift)[0])


if __name__ == '__main__':
    print("emd_gpu module loaded. numba.cuda available:", _HAVE_NUMBA)
