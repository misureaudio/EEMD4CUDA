import numpy as np, emd_gpu as G, emd_ref as R, cupy as cp
from numba import cuda

N = 512
t = np.arange(N, dtype=float)
x = np.sin(2*np.pi*0.05*t) + 0.5*np.sin(2*np.pi*0.013*t + 0.7)
eu_c, _ = R.envelope(t, x, 'max')
print('CPU eu finite:', np.isfinite(eu_c).all(), 'eu[:3] =', np.array2string(eu_c[:3], precision=4))
nmx = len(np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0])
print('CPU n_max =', nmx)


def _make(N, K):
    @cuda.jit
    def kern(S, T, out, outM):
        tid = cuda.threadIdx.x
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
        cnt[0] = 0
        cuda.syncthreads()
        for i in range(1, N - 1):
            if (S[i] > S[i - 1]) and (S[i] > S[i + 1]):
                kk = cuda.atomic.add(cnt, 0, 1)
                if kk < K:
                    kt[kk] = T[i]; ky[kk] = S[i]
        cuda.syncthreads()
        nm = cnt[0]
        if nm < 2:
            if tid == 0:
                outM[0] = -1.0
            return
        if tid == 0:
            for j in range(nm - 1):
                H[j] = kt[j + 1] - kt[j]
            nint = nm - 2
            for m in range(nint):
                dd[m] = 2.0 * (H[m] + H[m + 1])
                ll[m] = 0.0 if m == 0 else H[m]
                uu[m] = 0.0 if m == nint - 1 else H[m + 1]
                rr[m] = 6.0 * ((ky[m + 2] - ky[m + 1]) / H[m + 1] - (ky[m + 1] - ky[m]) / H[m])
        cuda.syncthreads()
        G._thomas(dd, ll, uu, rr, xx, cc, nm - 2)
        if tid == 0:
            Mf[0] = 0.0; Mf[nm - 1] = 0.0
        for i in range(tid, nm - 2, cuda.blockDim.x):
            Mf[i + 1] = xx[i]
        cuda.syncthreads()
        for i in range(tid, N, cuda.blockDim.x):
            out[i] = G._spline_eval(kt, ky, Mf, T[i], nm)
        if tid == 0:
            outM[0] = float(nm)
    return kern


K = 128
kern = _make(N, K)
S = cp.asarray(x); T = cp.asarray(t)
out = cp.zeros(N, dtype=np.float64); outM = cp.zeros(1, dtype=np.float64)
kern[1, 128](S, T, out, outM)
gpu_eu = cp.asnumpy(out)
nm = int(cp.asnumpy(outM)[0])
print('GPU nm =', nm)
print('GPU eu finite:', np.isfinite(gpu_eu).all())
print('GPU eu[:5] =', np.array2string(gpu_eu[:5], precision=4))
print('CPU eu[:5] =', np.array2string(eu_c[:5], precision=4))
print('max|diff| =', float(np.nanmax(np.abs(gpu_eu - eu_c))))
# also dump M to check Thomas
print('GPU eu finite count =', int(np.isfinite(gpu_eu).sum()), '/', N)
