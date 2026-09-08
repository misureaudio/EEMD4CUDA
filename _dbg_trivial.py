import numpy as np, cupy as cp
import emd_gpu as G   # triggers _force_cc + _bootstrap_nvvm
from numba import cuda

# TEST 1: trivial identity kernel
@cuda.jit
def ident(S, out, N):
    i = cuda.grid(1)
    if i < N:
        out[i] = S[i] * 2.0

N = 128
x = np.arange(N, dtype=np.float64)
S = cp.asarray(x); out = cp.zeros(N, dtype=np.float64)
ident[1, 128](S, out, N)
ok1 = np.allclose(cp.asnumpy(out), x * 2.0)
print('TEST 1 identity kernel:', 'PASS' if ok1 else 'FAIL', cp.asnumpy(out)[:3])

# TEST 2: shared array + atomic count (no spline)
def _mk(N, K):
    @cuda.jit
    def cntk(S, out, outn):
        tid = cuda.threadIdx.x
        buf = cuda.shared.array(K, dtype=np.float64)
        cnt = cuda.shared.array(1, dtype=np.int32)
        cnt[0] = 0
        cuda.syncthreads()
        for i in range(1, N - 1):
            if (S[i] > S[i - 1]) and (S[i] > S[i + 1]):
                kk = cuda.atomic.add(cnt, 0, 1)
                if kk < K:
                    buf[kk] = S[i]
        cuda.syncthreads()
        if tid == 0:
            outn[0] = float(cnt[0])
            out[0] = buf[0]
    return cntk

t = np.arange(N, dtype=np.float64)
xx = np.sin(2*np.pi*0.05*t)
cntk = _mk(N, K=64)
S2 = cp.asarray(xx); out2 = cp.zeros(N, dtype=np.float64); outn = cp.zeros(1, dtype=np.float64)
cntk[1, 128](S2, out2, outn)
print('TEST 2 shared+atomic: PASS (ran)', 'n_max(gpu)=', int(cp.asnumpy(outn)[0]))
