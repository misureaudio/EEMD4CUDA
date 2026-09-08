import numpy as np, time
import emd_par as M
from refsignal import make_ref_signal
x, t = make_ref_signal()
N=len(x)
# single warm JIT
t0=time.time(); r=M.decompose(x, method='gpu', E=8, tau=0.25, seed=7); dt=time.time()-t0
print("GPU E=8 (incl 1st JIT): %.2fs  n_modes=%d  recon=%.2e"%(dt,r.n_modes,r.recon_error()))
# warm call
t0=time.time(); r=M.decompose(x, method='gpu', E=8, tau=0.25, seed=8); dt=time.time()-t0
print("GPU E=8 (WARM):          %.2fs  n_modes=%d  recon=%.2e"%(dt,r.n_modes,r.recon_error()))
t0=time.time(); r=M.decompose(x, method='gpu', E=64, tau=0.25, seed=8); dt=time.time()-t0
print("GPU E=64 (WARM):         %.2fs  n_modes=%d  recon=%.2e"%(dt,r.n_modes,r.recon_error()))
