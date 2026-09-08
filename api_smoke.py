"""Final public-API smoke test: decompose() default (auto), save/load round
trip, and a 1M-sample signal (the long-signal path the user specified)."""
import numpy as np, time, os
import emd_par as M

# --- reference sweep, default API (auto method) ---
x, t = M.make_reference_sweep()
print("Reference sweep N=%d" % len(x))
t0=time.time(); r = M.decompose(x, tau=0.25); dt=time.time()-t0
print("decompose(x, tau=0.25)  [E=64 default, auto=%s]: %.2fs  n_modes=%d  recon=%.2e"
      % (r.stats['method'], dt, r.n_modes, r.recon_error()))
print("  stats:", {k:v for k,v in r.stats.items() if k!='x'})

# --- save/load round trip ---
p = 'sweep_result.npz'
r.save(p)
r2 = M.EMDResult.load(p)
print("\nsave/load round trip: imfs identical = %s  residual identical = %s"
      % (np.array_equal(r.imfs, r2.imfs), np.array_equal(r.residual, r2.residual)))
os.remove(p)

# --- 1M-sample signal (long-signal path) ---
print("\n1M-sample signal:")
N = 1_000_000
tt = np.arange(N)/44100.0
xx = np.sin(2*np.pi*np.cumsum(0.01+0.05*tt/(N/44100.0))) + 0.2*np.sin(2*np.pi*0.013*tt)
t0=time.time(); r1 = M.decompose(xx, method='cpu', E=1, tau=0.25); dt=time.time()-t0
print("  CPU E=1 (plain EMD) on N=%d: %.2fs  n_modes=%d  recon=%.2e"
      % (len(xx), dt, r1.n_modes, r1.recon_error()))

# --- 2M-sample signal ---
N2 = 2_000_000
tt2 = np.arange(N2)/44100.0
xx2 = np.sin(2*np.pi*np.cumsum(0.01+0.05*tt2/(N2/44100.0))) + 0.2*np.sin(2*np.pi*0.013*tt2)
t0=time.time(); r2 = M.decompose(xx2, method='cpu', E=1, tau=0.25); dt=time.time()-t0
print("  CPU E=1 (plain EMD) on N=%d: %.2fs  n_modes=%d  recon=%.2e"
      % (len(xx2), dt, r2.n_modes, r2.recon_error()))
