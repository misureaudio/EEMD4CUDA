"""Ground-truth: single CPU EMD on the FULL reference signal (441k samples).
Reports mode count, per-mode sifts, total time, reconstruction error, and
per-sift cost breakdown (where the time goes)."""
import numpy as np, time
import emd_ref as R
from refsignal import make_ref_signal

x, t = make_ref_signal()
N = len(x)
print("Reference signal: N=%d, extrema=%d (max=%d, min=%d)"
      % (N, 2*72394, 72394, 72393))

# time a single sift's components at the low end (few knots) vs full
t0 = time.time()
eu, nmax = R.envelope(t, x, 'max', 'natural')
el, nmin = R.envelope(t, x, 'min', 'natural')
dt_env = time.time()-t0
print("  one envelope pair (N=%d, knots~%d): %.3fs  (max=%d min=%d)"
      % (N, nmax+nmin, dt_env, nmax, nmin))

t0 = time.time()
imfs, resid, stats = R.emd_1d(x, tau=0.25, max_sifts=50)
dt = time.time()-t0
rec = sum(imfs) + resid
print("  FULL EMD: %.3fs   n_modes=%d   total_sifts=%d" % (dt, stats['n_modes'], stats['total_sifts']))
print("  per-mode sifts = %s" % stats['per_mode_sifts'])
print("  reconstruction err = %.3e" % np.max(np.abs(x - rec)))
print("  residual std = %.3e  (should be ~trend)" % np.std(resid))
print("  per-IMF energy (std): %s" % ["%.3f" % np.std(a) for a in imfs])
