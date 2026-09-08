"""Warm-call GPU timing (2nd call, JIT already compiled) + EEMD scaling on
the 441k reference sweep. This is the number for repeated use."""
import numpy as np, time
import emd_par as M
import emd_gpu as G
from refsignal import make_ref_signal

x, t = make_ref_signal()
N = len(x)
print("Reference sweep N=%d" % N)

# warm the exact (N,K,MAXIMF,nthreads) kernel
G.gpu_eemd_full(x, E=8, tau=0.25, max_imf=13)
print("  [JIT warm for N=%d done]" % N)

# warm-call module timing (JIT now compiled)
for E in [8, 64, 128]:
    t0=time.time(); r = M.decompose(x, method='gpu', E=E, tau=0.25, seed=7); dt=time.time()-t0
    ns = r.stats['per_trial_total_sifts']
    print("  GPU warm E=%3d : %7.3fs  n_modes=%d  recon=%.2e  sifts[min,max]=[%d,%d]"
          % (E, dt, r.n_modes, r.recon_error(), ns.min(), ns.max()))

# CPU warm for comparison (E=8)
print("\n  CPU E=8 (7 workers) for comparison:")
t0=time.time(); r = M.decompose(x, method='cpu', E=8, tau=0.25, seed=7, n_workers=7); dt=time.time()-t0
print("  CPU warm E=8 : %7.3fs  n_modes=%d  recon=%.2e" % (dt, r.n_modes, r.recon_error()))
