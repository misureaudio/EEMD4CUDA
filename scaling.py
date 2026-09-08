"""Profile the GPU kernel on a chirp + measure CPU/GPU scaling vs N and E.
Goal: find why GPU is 320x slower than vectorized CPU at N=4096, and where
the crossover is for long signals."""
import numpy as np, time
import emd_ref as R
import emd_gpu as G

def chirp(N, seed=0):
    t = np.arange(N, dtype=float)
    f = 0.01 + 0.09*(t/N)
    ph = 2*np.pi*np.cumsum(f)
    return np.sin(ph) + 0.5*np.sin(2*np.pi*0.02*t + 0.3)

print("=== GPU EEMD on chirp: E=1 vs E=64 (isolate parallelism) ===")
N=4096
x = chirp(N)
for E in [1, 8, 64]:
    t0=time.time(); avg,ns,raw = G.gpu_eemd(x, E=E, eps=0.2, tau=0.25, max_sifts=50, seed=7); dt=time.time()-t0
    print("  E=%3d : %8.3fs   per-trial n_sift max=%d"%(E, dt, ns.max()))

print("\n=== CPU vectorized single EMD vs N (chirp) ===")
for N in [1024, 4096, 16384, 65536, 262144, 1048576]:
    x = chirp(N)
    t0=time.time(); imfs,resid,stats = R.emd_1d(x, tau=0.25, max_sifts=50); dt=time.time()-t0
    print("  N=%7d : %8.3fs   n_modes=%d total_sifts=%d  (per-mode sifts=%s)"
          % (N, dt, stats['n_modes'], stats['total_sifts'], stats['per_mode_sifts']))

print("\n=== CPU EEMD (parallel) vs N, E=64 (chirp) ===")
import os
P = max(1, os.cpu_count()-1)
for N in [4096, 65536, 262144]:
    x = chirp(N)
    t0=time.time(); avg,resid,res = R.eemd(x, E=64, eps=0.2, parallel=True, n_workers=P, seed=7); dt=time.time()-t0
    ns = np.array([r[2]['total_sifts'] for r in res])
    print("  N=%7d : %8.3fs   per-trial total_sifts min=%d max=%d (imbalance x%.2f)"
          % (N, dt, ns.min(), ns.max(), ns.max()/max(1,ns.min())))

print("\n=== GPU EEMD vs N, E=64 (chirp) ===")
for N in [4096, 16384, 65536, 262144]:
    x = chirp(N)
    try:
        t0=time.time(); avg,ns,raw = G.gpu_eemd(x, E=64, eps=0.2, tau=0.25, max_sifts=50, seed=7); dt=time.time()-t0
        print("  N=%7d : %8.3fs   per-trial n_sift max=%d"%(N, dt, ns.max()))
    except Exception as e:
        print("  N=%7d : FAILED %s"%(N, type(e).__name__))
