"""VALID concurrency measurement: real (E, N) 2-D arrays. Bounded to fit 12GB
and run < ~200s. Finds the resident-block capacity C (where wall time stops
being ~1 wave) and confirms T_block."""
import numpy as np, time
import emd_gpu as G
from refsignal import make_ref_signal

x0, _ = make_ref_signal()

def perturbed(E, N, seed=7):
    x = x0[:N]
    r = np.random.default_rng(seed)
    return x[None, :] + 0.2*np.std(x)*r.standard_normal((E, N))

for N, Es in [(110250, [1, 8, 32, 128, 256]), (441000, [1, 8, 32, 64])]:
    print("=== N=%d ===" % N)
    # per-block time (E=1): cold then warm
    t0=time.time(); G.gpu_eemd_full(perturbed(1,N), E=1, tau=0.25, max_imf=13); dtJ=time.time()-t0
    t0=time.time(); G.gpu_eemd_full(perturbed(1,N), E=1, tau=0.25, max_imf=13); dt1=time.time()-t0
    print("  E=1 cold(incl JIT)=%6.2fs   E=1 WARM=T_block=%6.2fs" % (dtJ, dt1))
    for E in Es:
        if E == 1: continue
        t0=time.time(); imfs,ns,nm = G.gpu_eemd_full(perturbed(E,N), E=E, tau=0.25, max_imf=13); dt=time.time()-t0
        print("  E=%4d : wall=%7.2fs  (wall/T_block=%5.2f waves)  sifts[%d,%d]"
              % (E, dt, dt/dt1, ns.min(), ns.max()))
    print()
