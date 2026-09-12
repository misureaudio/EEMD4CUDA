"""
emd_gpu.py  --  GPU (cupy + numba.cuda) parallel 1-D EMD / EEMD.
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


@cuda.jit(device=True, inline=True)
def _thomas(d, l, u, r, x, c_, n):
    if n <= 0: return
    c_[0] = (u[0] / d[0]) if n > 1 else 0.0
    x[0] = r[0] / d[0]
    for i in range(1, n):
        c_[i] = (u[i] / (d[i] - l[i]*c_[i-1])) if i < n-1 else 0.0
        x[i] = (r[i] - l[i]*x[i-1]) / (d[i] - l[i]*c_[i-1])
    for i in range(n-2, -1, -1):
        x[i] = x[i] - c_[i]*x[i+1]


@cuda.jit(device=True, inline=True)
def _window_thomas(dd, ll, uu, rr, Mf, buf, w0, w1, o0, o1, B2S):
    """Decay-Seamed Block Thomas: solve the window [w0..w1]. 
    buf is a 1D private thread slice of size 2*(B+2S)."""
    cprev = 0.0
    xprev = 0.0
    for i in range(w0, w1 + 1):
        denom = dd[i] - ll[i]*cprev
        cnew = 0.0 if i == w1 else uu[i] / denom
        xnew = (rr[i] - ll[i]*xprev) / denom
        buf[i - w0] = xnew
        buf[B2S + (i - w0)] = cnew
        cprev = cnew
        xprev = xnew
        
    xip1 = buf[w1 - w0]
    if o0 <= w1 <= o1:
        Mf[w1 + 1] = xip1
        
    for i in range(w1 - 1, w0 - 1, -1):
        xi = buf[i - w0] - buf[B2S + (i - w0)] * xip1
        if o0 <= i <= o1:
            Mf[i + 1] = xi
        xip1 = xi


@cuda.jit(device=True, inline=True)
def _scan_extrema(arr, kt, ky, A, Bm, N, K, tid, nthreads, want_max, collect):
    """Tile-Based parallel extrema detect + compact (Preserves Time-Order)."""
    total = 0
    for base in range(1, N - 1, nthreads):
        i = base + tid
        hit = 0
        if i < N - 1:
            if want_max:
                if (arr[i] > arr[i - 1]) and (arr[i] > arr[i + 1]): hit = 1
            else:
                if (arr[i] < arr[i - 1]) and (arr[i] < arr[i + 1]): hit = 1
                    
        A[tid] = hit
        cuda.syncthreads()
        
        for k in range(7):
            step = 1 << k
            if k & 1:
                if tid >= step: A[tid] = Bm[tid] + Bm[tid - step]
                else: A[tid] = Bm[tid]
            else:
                if tid >= step: Bm[tid] = A[tid] + A[tid - step]
                else: Bm[tid] = A[tid]
            cuda.syncthreads()
            
        tile_total = Bm[nthreads - 1]
        
        if collect and hit == 1:
            tile_off = Bm[tid] - 1
            pos = total + tile_off
            if pos < K:
                kt[pos] = i
                ky[pos] = arr[i]
                
        total += tile_total
        cuda.syncthreads()
        
    return total


@cuda.jit(device=True, inline=True)
def _spline_eval_bin(tm, ym, M, t, p):
    if p < 2: return ym[0] if p == 1 else 0.0
    lo = 0
    hi = p - 1
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if tm[mid] <= t: lo = mid
        else: hi = mid
    i = lo
    if i < 0: i = 0
    if i > p - 2: i = p - 2
    h = float(tm[i+1] - tm[i])
    if h <= 0.0: return ym[i]
    a = (float(tm[i+1]) - t) / h
    b = (t - float(tm[i])) / h
    return (a*ym[i] + b*ym[i+1] + ((a*a*a - a)*M[i] + (b*b*b - b)*M[i+1]) * (h*h) / 6.0)


_KERNEL_CACHE = {}

def _make_full_kernel(N, K, MAXIMF, nthreads, B, S, EXACT):

    @cuda.jit
    def _kern(S_arr, resid, h, out, eu, W, W_int, Wwin, nsift, nmode,
              tau, max_sifts, min_extrema):
        b = cuda.blockIdx.x
        tid = cuda.threadIdx.x
        sig = S_arr[b, :]
        rs  = resid[b, :]
        hv  = h[b, :]
        eur = eu[b, :]
        
        kt = W_int[b, :]
        ky = W[b, 0*K:1*K]
        H  = W[b, 1*K:2*K]
        dd = W[b, 2*K:3*K]
        ll = W[b, 3*K:4*K]
        uu = W[b, 4*K:5*K]
        rr = W[b, 5*K:6*K]
        xx = W[b, 6*K:7*K]
        cc = W[b, 7*K:8*K]
        Mf = W[b, 8*K:9*K]
        
        win = Wwin[b, :]
        B2S = B + 2 * S
        
        red = cuda.shared.array(nthreads, dtype=np.float64)
        cntA = cuda.shared.array(nthreads, dtype=np.int32)
        cntB = cuda.shared.array(nthreads, dtype=np.int32)
        stop = cuda.shared.array(1, dtype=np.int32)

        for i in range(tid, N, nthreads):
            rs[i] = sig[i]
            hv[i] = sig[i]

        total_sifts = 0
        nmode_b = 0
        for m in range(MAXIMF):
            cuda.syncthreads()
            nm = _scan_extrema(rs, kt, ky, cntA, cntB, N, K, tid, nthreads, True, False)
            nn = _scan_extrema(rs, kt, ky, cntA, cntB, N, K, tid, nthreads, False, False)
            if (nm + nn) < min_extrema or nm < 2 or nn < 2:
                break

            for i in range(tid, N, nthreads):
                hv[i] = rs[i]
            stop[0] = 0
            nsift_local = 0
            
            for s in range(max_sifts):
                cuda.syncthreads()
                if stop[0] != 0: break
                
                nm = _scan_extrema(hv, kt, ky, cntA, cntB, N, K, tid, nthreads, True, True)
                cuda.syncthreads()
                if nm < 2: break
                
                if EXACT:
                    if tid == 0:
                        for j in range(nm - 1): H[j] = float(kt[j + 1] - kt[j])
                        nint = nm - 2
                        for q in range(nint):
                            dd[q] = 2.0 * (H[q] + H[q + 1])
                            ll[q] = 0.0 if q == 0 else H[q]
                            uu[q] = 0.0 if q == nint - 1 else H[q + 1]
                            rr[q] = 6.0 * ((ky[q + 2] - ky[q + 1]) / H[q + 1] - (ky[q + 1] - ky[q]) / H[q])
                        _thomas(dd, ll, uu, rr, xx, cc, nint)
                        Mf[0] = 0.0
                        Mf[nm - 1] = 0.0
                        for i in range(nint): Mf[i + 1] = xx[i]
                else:
                    for j in range(tid, nm - 1, nthreads): H[j] = float(kt[j + 1] - kt[j])
                    nint = nm - 2
                    for q in range(tid, nint, nthreads):
                        dd[q] = 2.0 * (H[q] + H[q + 1])
                        ll[q] = 0.0 if q == 0 else H[q]
                        uu[q] = 0.0 if q == nint - 1 else H[q + 1]
                        rr[q] = 6.0 * ((ky[q + 2] - ky[q + 1]) / H[q + 1] - (ky[q + 1] - ky[q]) / H[q])
                    cuda.syncthreads()
                    
                    if tid == 0:
                        Mf[0] = 0.0
                        Mf[nm - 1] = 0.0
                        
                    nb = (nint + B - 1) // B
                    for blk in range(tid, nb, nthreads):
                        o0 = blk * B
                        o1 = o0 + B
                        if o1 > nint: o1 = nint
                        o1 = o1 - 1
                        
                        w0 = o0 - S
                        if w0 < 0: w0 = 0
                            
                        w1 = o1 + S
                        if w1 > nint - 1: w1 = nint - 1
                            
                        my_win = win[tid, :]
                        _window_thomas(dd, ll, uu, rr, Mf, my_win, w0, w1, o0, o1, B2S)
                    cuda.syncthreads()
                    
                cuda.syncthreads()
                for i in range(tid, N, nthreads):
                    eur[i] = _spline_eval_bin(kt, ky, Mf, i, nm)
                cuda.syncthreads()
                
                nn = _scan_extrema(hv, kt, ky, cntA, cntB, N, K, tid, nthreads, False, True)
                cuda.syncthreads()
                if nn < 2: break
                
                if EXACT:
                    if tid == 0:
                        for j in range(nn - 1): H[j] = float(kt[j + 1] - kt[j])
                        nint = nn - 2
                        for q in range(nint):
                            dd[q] = 2.0 * (H[q] + H[q + 1])
                            ll[q] = 0.0 if q == 0 else H[q]
                            uu[q] = 0.0 if q == nint - 1 else H[q + 1]
                            rr[q] = 6.0 * ((ky[q + 2] - ky[q + 1]) / H[q + 1] - (ky[q + 1] - ky[q]) / H[q])
                        _thomas(dd, ll, uu, rr, xx, cc, nint)
                        Mf[0] = 0.0
                        Mf[nn - 1] = 0.0
                        for i in range(nint): Mf[i + 1] = xx[i]
                else:
                    for j in range(tid, nn - 1, nthreads): H[j] = float(kt[j + 1] - kt[j])
                    nint = nn - 2
                    for q in range(tid, nint, nthreads):
                        dd[q] = 2.0 * (H[q] + H[q + 1])
                        ll[q] = 0.0 if q == 0 else H[q]
                        uu[q] = 0.0 if q == nint - 1 else H[q + 1]
                        rr[q] = 6.0 * ((ky[q + 2] - ky[q + 1]) / H[q + 1] - (ky[q + 1] - ky[q]) / H[q])
                    cuda.syncthreads()
                    
                    if tid == 0:
                        Mf[0] = 0.0
                        Mf[nn - 1] = 0.0
                        
                    nb = (nint + B - 1) // B
                    for blk in range(tid, nb, nthreads):
                        o0 = blk * B
                        o1 = o0 + B
                        if o1 > nint: o1 = nint
                        o1 = o1 - 1
                        
                        w0 = o0 - S
                        if w0 < 0: w0 = 0
                            
                        w1 = o1 + S
                        if w1 > nint - 1: w1 = nint - 1
                            
                        my_win = win[tid, :]
                        _window_thomas(dd, ll, uu, rr, Mf, my_win, w0, w1, o0, o1, B2S)
                    cuda.syncthreads()
                    
                cuda.syncthreads()
                num = 0.0; den = 0.0
                for i in range(tid, N, nthreads):
                    el_i = _spline_eval_bin(kt, ky, Mf, i, nn)
                    mval = 0.5 * (eur[i] + el_i)
                    hnew = hv[i] - mval
                    num += mval * mval
                    den += hv[i] * hv[i]
                    hv[i] = hnew
                    
                red[tid] = num
                cuda.syncthreads()
                step = nthreads // 2
                while step >= 1:
                    if tid < step: red[tid] += red[tid + step]
                    cuda.syncthreads()
                    step //= 2
                numsum = red[0]
                
                cuda.syncthreads()
                red[tid] = den
                cuda.syncthreads()
                step = nthreads // 2
                while step >= 1:
                    if tid < step: red[tid] += red[tid + step]
                    cuda.syncthreads()
                    step //= 2
                densum = red[0]
                
                nsift_local += 1
                if s >= 1:
                    sd = (numsum / densum) if densum > 0 else 0.0
                    if sd < tau:
                        if tid == 0: stop[0] = 1
                        break
                        
            total_sifts += nsift_local
            outrow = out[b, m, :]
            for i in range(tid, N, nthreads):
                outrow[i] = hv[i]
                rs[i] = rs[i] - hv[i]
            nmode_b += 1

        if tid == 0:
            nsift[b] = total_sifts
            nmode[b] = nmode_b

    return _kern


def get_full_kernel(N, K, MAXIMF, nthreads, B=250, S=30, EXACT=False):
    key = (N, K, MAXIMF, nthreads, B, S, EXACT)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = _make_full_kernel(N, K, MAXIMF, nthreads, B, S, EXACT)
    return _KERNEL_CACHE[key]


def _pick_K(N):
    return max(16, N // 2)


def gpu_eemd_full(x, E=64, tau=0.25, max_sifts=50, max_imf=12,
                  seed=0, K=None, nthreads=128, batch=None,
                  B=250, S=30, EXACT=False):
    import cupy as cp
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        if E != 1:
            raise ValueError("1-D input x is a single trial (E=1). For E>1 pass a 2-D array.")
        x = x[None, :]
    E, N = x.shape
    K = K or _pick_K(N)
    kern = get_full_kernel(N, K, max_imf, nthreads, B, S, EXACT)
    
    S_arr = cp.asarray(x)
    resid = cp.empty((E, N), dtype=np.float64)
    h = cp.empty((E, N), dtype=np.float64)
    out = cp.zeros((E, max_imf, N), dtype=np.float64)
    eu = cp.empty((E, N), dtype=np.float64)
    W = cp.empty((E, 9 * K), dtype=np.float64)
    W_int = cp.empty((E, K), dtype=np.int32)
    Wwin = cp.empty((E, nthreads, 2 * (B + 2 * S)), dtype=np.float64)
    nsift = cp.zeros(E, dtype=np.int32)
    nmode = cp.zeros(E, dtype=np.int32)
    
    kern[E, nthreads](S_arr, resid, h, out, eu, W, W_int, Wwin, nsift, nmode, tau, max_sifts, 4)
    
    imfs = cp.asnumpy(out)
    nsift_np = cp.asnumpy(nsift)
    nmode_np = cp.asnumpy(nmode)
    maxm = int(nmode_np.max()) if nmode_np.size else 0
    return imfs[:, :maxm, :], nsift_np, nmode_np


def gpu_eemd(x, E=64, eps=0.2, tau=0.25, max_sifts=50, seed=0,
             K=None, nthreads=128, B=250, S=30, EXACT=False):
    import cupy as cp
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    K = K or _pick_K(N)
    kern = get_full_kernel(N, K, 1, nthreads, B, S, EXACT)
    rng = np.random.default_rng(seed)
    noises = rng.standard_normal((E, N)) * (eps * np.std(x))
    
    S_arr = cp.asarray(x[None, :] + noises)
    resid = cp.empty((E, N), dtype=np.float64)
    h = cp.empty((E, N), dtype=np.float64)
    out = cp.zeros((E, 1, N), dtype=np.float64)
    eu = cp.empty((E, N), dtype=np.float64)
    W = cp.empty((E, 9 * K), dtype=np.float64)
    W_int = cp.empty((E, K), dtype=np.int32)
    Wwin = cp.empty((E, nthreads, 2 * (B + 2 * S)), dtype=np.float64)
    nsift = cp.zeros(E, dtype=np.int32)
    nmode = cp.zeros(E, dtype=np.int32)
    
    kern[E, nthreads](S_arr, resid, h, out, eu, W, W_int, Wwin, nsift, nmode, tau, max_sifts, 4)
    imfs = cp.asnumpy(out)[:, 0, :]
    return imfs.mean(axis=0), cp.asnumpy(nsift), imfs


def gpu_emd_single(x, t=None, tau=0.25, max_sifts=50, K=None, nthreads=128,
                   B=250, S=30, EXACT=False):
    import cupy as cp
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    K = K or _pick_K(N)
    kern = get_full_kernel(N, K, 1, nthreads, B, S, EXACT)
    
    S_arr = cp.asarray(x[None, :])
    resid = cp.empty((1, N), dtype=np.float64)
    h = cp.empty((1, N), dtype=np.float64)
    out = cp.zeros((1, 1, N), dtype=np.float64)
    eu = cp.empty((1, N), dtype=np.float64)
    W = cp.empty((1, 9 * K), dtype=np.float64)
    W_int = cp.empty((1, K), dtype=np.int32)
    Wwin = cp.empty((1, nthreads, 2 * (B + 2 * S)), dtype=np.float64)
    nsift = cp.zeros(1, dtype=np.int32)
    nmode = cp.zeros(1, dtype=np.int32)
    
    kern[1, nthreads](S_arr, resid, h, out, eu, W, W_int, Wwin, nsift, nmode, tau, max_sifts, 4)
    return cp.asnumpy(out[0, 0, :]), int(cp.asnumpy(nsift)[0])

if __name__ == '__main__':
    print("emd_gpu module loaded. numba.cuda available:", _HAVE_NUMBA)