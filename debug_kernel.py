"""Instrument the GPU kernel: dump per-sift upper envelope, lower envelope,
knot counts, and max|h|, to locate the divergence vs the CPU reference."""
import numpy as np, os
import ctypes
ctypes.CDLL(os.path.join(os.getcwd(), 'cuda_libs', 'cudart.dll'))
ctypes.CDLL(os.path.join(os.getcwd(), 'cuda_libs', 'nvvm.dll'))
from numba import cuda
import emd_ref as R

N = 1024
K = 256
nthreads = 128

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
def _spline_eval(tm, ym, M, t, p):
    i = 0
    while i < p-2 and t > tm[i+1]:
        i += 1
    if i >= p-2:
        i = p-2
    if i < 0:
        i = 0
    h = tm[i+1] - tm[i]
    a = (tm[i+1] - t) / h
    b = (t - tm[i]) / h
    return (a*ym[i] + b*ym[i+1]
            + ((a*a*a - a)*M[i] + (b*b*b - b)*M[i+1]) * (h*h) / 6.0)

@cuda.jit
def dbg(S, T, eur_out, el_out, nmax_out, nmin_out, h1_out):
    b = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    nthreads = cuda.blockDim.x
    sig = S[b, :]
    eur = eur_out[b, :]
    el  = el_out[b, :]
    kt = cuda.shared.array(K, dtype=np.float64)
    ky = cuda.shared.array(K, dtype=np.float64)
    H = cuda.shared.array(K, dtype=np.float64)
    dd = cuda.shared.array(K, dtype=np.float64)
    ll = cuda.shared.array(K, dtype=np.float64)
    uu = cuda.shared.array(K, dtype=np.float64)
    rr = cuda.shared.array(K, dtype=np.float64)
    xx = cuda.shared.array(K, dtype=np.float64)
    cc = cuda.shared.array(K, dtype=np.float64)
    Mf = cuda.shared.array(K, dtype=np.float64)
    cnt = cuda.shared.array(1, dtype=np.int32)
    nm_s = cuda.shared.array(1, dtype=np.int32)
    nn_s = cuda.shared.array(1, dtype=np.int32)
    s = 0
    # ONE sift: upper envelope
    cnt[0] = 0
    cuda.syncthreads()
    if tid == 0:
        for i in range(1, N-1):
            if (sig[i] > sig[i-1]) and (sig[i] > sig[i+1]):
                kk = cnt[0]
                if kk < K:
                    kt[kk] = T[i]; ky[kk] = sig[i]
                cnt[0] += 1
    cuda.syncthreads()
    if tid == 0:
        nm_s[0] = cnt[0] if cnt[0] < K else K
        nmax_out[b] = nm_s[0]
    cuda.syncthreads()
    nm = nm_s[0]
    if tid == 0:
        for j in range(nm-1):
            H[j] = kt[j+1] - kt[j]
        nint = nm - 2
        for m in range(nint):
            dd[m] = 2.0*(H[m] + H[m+1])
            ll[m] = 0.0 if m == 0 else H[m]
            uu[m] = 0.0 if m == nint-1 else H[m+1]
            rr[m] = 6.0*((ky[m+2]-ky[m+1])/H[m+1] - (ky[m+1]-ky[m])/H[m])
    cuda.syncthreads()
    _thomas(dd, ll, uu, rr, xx, cc, nm-2)
    if tid == 0:
        Mf[0] = 0.0
        Mf[nm-1] = 0.0
    for i in range(tid, nm-2, nthreads):
        Mf[i+1] = xx[i]
    cuda.syncthreads()
    for i in range(tid, N, nthreads):
        eur[i] = _spline_eval(kt, ky, Mf, T[i], nm)
    # min knots
    cnt[0] = 0
    cuda.syncthreads()
    if tid == 0:
        for i in range(1, N-1):
            if (sig[i] < sig[i-1]) and (sig[i] < sig[i+1]):
                kk = cnt[0]
                if kk < K:
                    kt[kk] = T[i]; ky[kk] = sig[i]
                cnt[0] += 1
    cuda.syncthreads()
    if tid == 0:
        nn_s[0] = cnt[0] if cnt[0] < K else K
        nmin_out[b] = nn_s[0]
    cuda.syncthreads()
    nn = nn_s[0]
    if tid == 0:
        for j in range(nn-1):
            H[j] = kt[j+1] - kt[j]
        nint = nn - 2
        for m in range(nint):
            dd[m] = 2.0*(H[m] + H[m+1])
            ll[m] = 0.0 if m == 0 else H[m]
            uu[m] = 0.0 if m == nint-1 else H[m+1]
            rr[m] = 6.0*((ky[m+2]-ky[m+1])/H[m+1] - (ky[m+1]-ky[m])/H[m])
    cuda.syncthreads()
    _thomas(dd, ll, uu, rr, xx, cc, nn-2)
    if tid == 0:
        Mf[0] = 0.0
        Mf[nn-1] = 0.0
    for i in range(tid, nn-2, nthreads):
        Mf[i+1] = xx[i]
    cuda.syncthreads()
    for i in range(tid, N, nthreads):
        el[i] = _spline_eval(kt, ky, Mf, T[i], nn)
    # h1 = sig - 0.5*(eur+el), store to global for host comparison
    for i in range(tid, N, nthreads):
        mval = 0.5*(eur[i] + el[i])
        h1_out[b, i] = sig[i] - mval

import cupy as cp
nthreads = 128
t = np.arange(N, dtype=float)
x = np.sin(2*np.pi*0.05*t)+0.5*np.sin(2*np.pi*0.013*t+0.7)+0.1*np.random.default_rng(3).standard_normal(N)

S = cp.asarray(x[None, :])
T = cp.asarray(t)
eur = cp.empty((1, N)); el = cp.empty((1, N))
h1 = cp.empty((1, N))
nmax_o = cp.zeros(1, dtype=np.int32); nmin_o = cp.zeros(1, dtype=np.int32)
dbg[1, nthreads](S, T, eur, el, nmax_o, nmin_o, h1)
eur_np = cp.asnumpy(eur[0]); el_np = cp.asnumpy(el[0]); h1_np = cp.asnumpy(h1[0])
print('GPU nmax=%d nmin=%d h1_max=%.4f'%(int(cp.asnumpy(nmax_o)[0]), int(cp.asnumpy(nmin_o)[0]), np.max(np.abs(h1_np))))

# CPU first sift
eu_c,_ = R.envelope(t, x, 'max')
el_c,_ = R.envelope(t, x, 'min')
h1_c = x - 0.5*(eu_c + el_c)
print('CPU nmax=%d nmin=%d h1_max=%.4f'%(len(np.where((x[1:-1]>x[:-2])&(x[1:-1]>x[2:]))[0]), len(np.where((x[1:-1]<x[:-2])&(x[1:-1]<x[2:]))[0]), np.max(np.abs(h1_c))))
print('EUR  max|d| = %.3e'%np.max(np.abs(eur_np - eu_c)))
print('EL   max|d| = %.3e'%np.max(np.abs(el_np - el_c)))
print('H1   max|d| = %.3e'%np.max(np.abs(h1_np - h1_c)))
i = np.argmax(np.abs(h1_np - h1_c))
print('  H1 argmax idx=%d t=%s gpu=%.5f cpu=%.5f  (eur_g=%.4f el_g=%.4f)'%(i, t[i], h1_np[i], h1_c[i], eur_np[i], el_np[i]))
