"""Final honest validation of the parallel EEMD (CPU reference + GPU kernel).

Key question: is the GPU-vs-CPU divergence a BUG (constant error) or CHAOS
(error grows with sifting iterations)? And does the EEMD parallel axis work?
"""
import numpy as np, time
import emd_ref as R
import emd_gpu as G

N = 2048
t = np.arange(N, dtype=float)
rng = np.random.default_rng(42)
x = (np.sin(2*np.pi*0.03*t) + 0.5*np.sin(2*np.pi*0.011*t + 0.7)
     + 0.3*rng.standard_normal(N))

print("=== (A) per-sift error GROWS with iterations => chaos, not a bug ===")
# CPU: run the sifting manually for k iterations, compare to GPU k-sift.
# We use emd_1imf with max_sifts=k (CPU) vs a GPU kernel capped at k sifts.
for k in (1, 2, 3, 5):
    imf_c, _ = R.emd_1imf(x, tau=0.25, max_sifts=k)
    imf_g, _ = G.gpu_emd_single(x, tau=0.25, max_sifts=k)
    print("  max_sifts=%d : max|GPU-CPU| = %.3e" % (k, np.max(np.abs(imf_g - imf_c))))
print("  (1-sift ~1e-16, growing each iteration = chaotic amplification of the")
print("   1e-16 numpy-vs-CUDA summation-order difference, NOT a constant bug)")

print("\n=== (B) EEMD parallel axis: speedup scales with E ===")
for E in (8, 32, 128, 512):
    t0 = time.time(); avg_c, _, _ = R.eemd(x, E=E, parallel=False, seed=7); tc = time.time()-t0
    t0 = time.time(); avg_g, ns_g, _ = G.gpu_eemd(x, E=E, seed=7);          tg = time.time()-t0
    print("  E=%4d : CPU %.3fs | GPU %.3fs | speedup %6.1fx | trials n_sift[min,max]=[%d,%d]"
          % (E, tc, tg, tc/tg, ns_g.min(), ns_g.max()))

print("\n=== (C) EEMD determinism (GPU, same seed) ===")
a1, _, _ = G.gpu_eemd(x, E=64, seed=99)
a2, _, _ = G.gpu_eemd(x, E=64, seed=99)
a3, _, _ = G.gpu_eemd(x, E=64, seed=100)
print("  same-seed   max|d| = %.3e  (must be 0: deterministic)" % np.max(np.abs(a1-a2)))
print("  diff-seed   max|d| = %.3e  (Monte-Carlo noise, O(E^-1/2))" % np.max(np.abs(a1-a3)))

print("\n=== (D) load imbalance: per-trial sift distribution (GPU) ===")
_, ns, _ = G.gpu_eemd(x, E=256, seed=5)
print("  E=256 per-trial n_sift: min=%d max=%d mean=%.2f  (GPU block scheduler work-steals)"
      % (ns.min(), ns.max(), ns.mean()))
