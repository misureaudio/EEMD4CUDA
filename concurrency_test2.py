"""CORRECT concurrency measurement: pass a real (E, N) 2-D array of
perturbed signals so E is honored. Find where wall time stops being flat
(the resident-block capacity C) and the per-block time T_block."""
import numpy as np, time
import emd_gpu as G
from refsignal import make_ref_signal

x0, _ = make_ref_signal()
rng = np.random.default_rng(7)

def perturbed(E, N, seed=7):
    x = x0[:N]
    r = np.random.default_rng(seed)
    return x[None, :] + 0.2*np.std(x)*r.standard_normal((E, N))

print("=== N=441000: per-block time (E=1) ===")
N=441000
t0=time.time(); G.gpu_eemd_full(perturbed(1,N), E=1, tau=0.25, max_imf=13); dtJ=time.time()-t0
t0=time.time(); G.gpu_eemd_full(perturbed(1,N), E=1, tau=0.25, max_imf=13); dt1=time.time()-t0
print("  E=1 cold(incl JIT): %.2fs   E=1 WARM=T_block: %.2fs"%(dtJ,dt1))
for E in [8, 16, 32, 64, 128]:
    t0=time.time(); imfs,ns,nm = G.gpu_eemd_full(perturbed(E,N), E=E, tau=0.25, max_imf=13); dt=time.time()-t0
    print("  E=%4d : wall=%7.2fs  (wall/T_block=%.2f waves)  sifts[%d,%d]"
          % (E, dt, dt/dt1, ns.min(), ns.max()))

print("\n=== N=110250: per-block time (E=1) ===")
N=110250
t0=time.time(); G.gpu_eemd_full(perturbed(1,N), E=1, tau=0.25, max_imf=13); dtJ=time.time()-t0
t0=time.time(); G.gpu_eemd_full(perturbed(1,N), E=1, tau=0.25, max_imf=13); dt1=time.time()-t0
print("  E=1 cold(incl JIT): %.2fs   E=1 WARM=T_block: %.2fs"%(dtJ,dt1))
for E in [64, 128, 256, 512]:
    t0=time.time(); imfs,ns,nm = G.gpu_eemd_full(perturbed(E,N), E=E, tau=0.25, max_imf=13); dt=time.time()-t0
    print("  E=%5d : wall=%7.2fs  (wall/T_block=%.2f waves)  sifts[%d,%d]"
          % (E, dt, dt/dt1, ns.min(), ns.max()))
