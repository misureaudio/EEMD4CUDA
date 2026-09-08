import numpy as np, cupy as cp, emd_gpu as G, emd_ref as R
from numba import cuda

N = 512
t = np.arange(N, dtype=float)
x = np.sin(2*np.pi*0.05*t) + 0.5*np.sin(2*np.pi*0.013*t + 0.7)

# CPU: one sift
eu_c, nm_c = R.envelope(t, x, 'max')
el_c, nn_c = R.envelope(t, x, 'min')
m_c = 0.5*(eu_c + el_c)
h_c = x - m_c
print('CPU nmax=%d nmin=%d'%(nm_c,nn_c))
print('CPU eu[:4]=',np.array2string(eu_c[:4],precision=4))
print('CPU el[:4]=',np.array2string(el_c[:4],precision=4))
print('CPU h [:4]=',np.array2string(h_c[:4],precision=4))


def _make(N, K):
    @cuda.jit
    def k(S, T, eu, el, m, h, cnts):
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
        # collect max
        cnt[0] = 0
        cuda.syncthreads()
        if tid == 0:
            for i in range(1, N-1):
                if (S[i] > S[i-1]) and (S[i] > S[i+1]):
                    if cnt[0] < K:
                        kt[cnt[0]] = T[i]; ky[cnt[0]] = S[i]
                    cnt[0] += 1
        cuda.syncthreads()
        nm = cnt[0]
        if tid == 0:
            cnts[0] = float(nm)
        if nm < 2:
            return
        if tid == 0:
            for j in range(nm-1):
                H[j] = kt[j+1]-kt[j]
            nint = nm-2
            for mm in range(nint):
                dd[mm] = 2.0*(H[mm]+H[mm+1])
                ll[mm] = 0.0 if mm==0 else H[mm]
                uu[mm] = 0.0 if mm==nint-1 else H[mm+1]
                rr[mm] = 6.0*((ky[mm+2]-ky[mm+1])/H[mm+1]-(ky[mm+1]-ky[mm])/H[mm])
        cuda.syncthreads()
        G._thomas(dd, ll, uu, rr, xx, cc, nm-2)
        if tid == 0:
            Mf[0]=0.0; Mf[nm-1]=0.0
        for i in range(tid, nm-2, cuda.blockDim.x):
            Mf[i+1] = xx[i]
        cuda.syncthreads()
        for i in range(tid, N, cuda.blockDim.x):
            eu[i] = G._spline_eval(kt, ky, Mf, T[i], nm)
        cuda.syncthreads()
        # collect min (overwrite kt,ky)
        cnt[0] = 0
        cuda.syncthreads()
        if tid == 0:
            for i in range(1, N-1):
                if (S[i] < S[i-1]) and (S[i] < S[i+1]):
                    if cnt[0] < K:
                        kt[cnt[0]] = T[i]; ky[cnt[0]] = S[i]
                    cnt[0] += 1
        cuda.syncthreads()
        nn = cnt[0]
        if tid == 0:
            cnts[1] = float(nn)
        if nn < 2:
            return
        if tid == 0:
            for j in range(nn-1):
                H[j] = kt[j+1]-kt[j]
            nint = nn-2
            for mm in range(nint):
                dd[mm] = 2.0*(H[mm]+H[mm+1])
                ll[mm] = 0.0 if mm==0 else H[mm]
                uu[mm] = 0.0 if mm==nint-1 else H[mm+1]
                rr[mm] = 6.0*((ky[mm+2]-ky[mm+1])/H[mm+1]-(ky[mm+1]-ky[mm])/H[mm])
        cuda.syncthreads()
        G._thomas(dd, ll, uu, rr, xx, cc, nn-2)
        if tid == 0:
            Mf[0]=0.0; Mf[nn-1]=0.0
        for i in range(tid, nn-2, cuda.blockDim.x):
            Mf[i+1] = xx[i]
        cuda.syncthreads()
        for i in range(tid, N, cuda.blockDim.x):
            eli = G._spline_eval(kt, ky, Mf, T[i], nn)
            el[i] = eli
            mv = 0.5*(eu[i]+eli)
            m[i] = mv
            h[i] = S[i]-mv
    return k

K = 128
k = _make(N, K)
S = cp.asarray(x); T = cp.asarray(t)
eu = cp.zeros(N); el = cp.zeros(N); m = cp.zeros(N); h = cp.zeros(N)
cnts = cp.zeros(2)
k[1, 128](S, T, eu, el, m, h, cnts)
cnts = cp.asnumpy(cnts)
print('GPU nmax=%d nmin=%d'%(cnts[0],cnts[1]))
print('GPU eu[:4]=',np.array2string(cp.asnumpy(eu)[:4],precision=4))
print('GPU el[:4]=',np.array2string(cp.asnumpy(el)[:4],precision=4))
print('GPU h [:4]=',np.array2string(cp.asnumpy(h)[:4],precision=4))
print('max|eu diff|=',np.max(np.abs(cp.asnumpy(eu)-eu_c)))
print('max|el diff|=',np.max(np.abs(cp.asnumpy(el)-el_c)))
print('max|h  diff|=',np.max(np.abs(cp.asnumpy(h)-h_c)))
