"""verify_cpu_gpu_1.py -- small-signal core checks: CPU (emd_ref) vs GPU (emd_gpu).

Question: does emd_gpu compute the same EMD as the CPU reference?
  (1) chaos-vs-bug discriminator: per-sift error growth (a spline/Thomas BUG
      shows up already at k=1; CHAOS starts at ~1e-16 and grows with k)
  (2) full E=1 EMD agreement + partition-of-unity + IMF (Huang) property
  (3) GPU determinism
"""
import numpy as np, time
import emd_ref as R
import emd_gpu as G

N = 8192
t = np.arange(N, dtype=float)
rng = np.random.default_rng(123)
x = (np.sin(2*np.pi*0.03*t) + 0.5*np.sin(2*np.pi*0.011*t + 0.7)
     + 0.2*np.sin(2*np.pi*0.1*t) + 0.1*rng.standard_normal(N))

print("=== 1. chaos-vs-bug discriminator: per-sift error growth (E=1) ===")
for k in (1, 2, 3, 5, 10):
    imf_c, _ = R.emd_1imf(x, tau=0.25, max_sifts=k)
    imf_g, _ = G.gpu_emd_single(x, tau=0.25, max_sifts=k)
    print("  k=%2d : max|GPU-CPU| = %.3e" % (k, np.max(np.abs(imf_g - imf_c))))
print("  (a bug would show >>1e-9 already at k=1; chaos starts ~1e-16 and grows)")

print("\n=== 2. full EMD E=1: CPU vs GPU (identical input) ===")
imfs_c, resid_c, st_c = R.emd_1d(x, tau=0.25, max_sifts=50)
imfs_c = np.array(imfs_c)
t0 = time.time()
imfs_g, ns, nm = G.gpu_eemd_full(x[None, :], E=1, tau=0.25, max_sifts=50,
                                 max_imf=st_c['n_modes'])
dt = time.time() - t0
imfs_g = imfs_g[0]
print("  CPU: n_modes=%d  total_sifts=%d  per_mode=%s"
      % (st_c['n_modes'], st_c['total_sifts'], st_c['per_mode_sifts']))
print("  GPU: nmode=%d  nsift=%d  time=%.2fs" % (nm[0], ns[0], dt))
rec_c = np.max(np.abs(x - (imfs_c.sum(0) + resid_c)))
print("  partition of unity: CPU recon=%.2e  (GPU residual is x-sum(imfs) by construction)" % rec_c)
m = min(imfs_c.shape[0], imfs_g.shape[0])
print("  per-IMF agreement (same input, single trial):")
for i in range(m):
    d = np.abs(imfs_g[i] - imfs_c[i])
    print("    IMF %2d: max|d|=%.3e  relL2=%.3e  corr=%.12f" % (
        i + 1, d.max(),
        np.linalg.norm(d) / np.linalg.norm(imfs_c[i]),
        np.corrcoef(imfs_g[i], imfs_c[i])[0, 1]))

print("\n=== 3. IMF property (Huang): |nmax-nmin| <= 1 for every IMF ===")
for tag, imfs in (("CPU", imfs_c), ("GPU", imfs_g)):
    bad = []
    for i in range(len(imfs)):
        y = imfs[i]
        nmax = int(np.sum((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])))
        nmin = int(np.sum((y[1:-1] < y[:-2]) & (y[1:-1] < y[2:])))
        if abs(nmax - nmin) > 1:
            bad.append((i + 1, nmax, nmin))
    print("  %s: %d IMFs, violations: %s" % (tag, len(imfs), bad or "none"))

print("\n=== 4. GPU determinism (same input, two calls) ===")
a, _, _ = G.gpu_eemd_full(x[None, :], E=1, tau=0.25, max_sifts=50,
                          max_imf=st_c['n_modes'])
b, _, _ = G.gpu_eemd_full(x[None, :], E=1, tau=0.25, max_sifts=50,
                          max_imf=st_c['n_modes'])
print("  max|d| = %.3e   bitwise_identical = %s" % (
    np.max(np.abs(a - b)), np.array_equal(a, b)))
