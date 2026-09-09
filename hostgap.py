"""Clean, VALID module-vs-kernel gap (host overhead: H2D transfer of (E,N)
noise + signal, host-side accumulation). E=8, N=441k, warm."""
import numpy as np, time
import emd_par as M
import emd_gpu as G
from refsignal import make_ref_signal

x, _ = make_ref_signal()
N = len(x)
rng = np.random.default_rng(7)
def perturbed(E, seed=7):
    r = np.random.default_rng(seed)
    return x[None, :] + 0.2*np.std(x)*r.standard_normal((E, N))

# warm JIT
G.gpu_eemd_full(perturbed(8), E=8, tau=0.25, max_imf=13)

# raw kernel (warm)
t0=time.time(); imfs,ns,nm = G.gpu_eemd_full(perturbed(8), E=8, tau=0.25, max_imf=13); kdt=time.time()-t0
# module (warm) -- does H2D transfer + host accumulation
t0=time.time(); r = M.decompose(x, method='gpu', E=8, tau=0.25, seed=7); mdt=time.time()-t0
print("N=%d E=8  warm:" % N)
print("  raw kernel : %7.2fs" % kdt)
print("  module     : %7.2fs  (host overhead = %7.2fs)" % (mdt, mdt-kdt))
print("  module n_modes=%d  recon=%.2e" % (r.n_modes, r.recon_error()))
