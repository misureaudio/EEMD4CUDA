"""Verify the new full-IMF GPU kernel:
  (1) small signal: GPU full EMD vs CPU full EMD (recon + per-IMF agreement)
  (2) 441k reference sweep: GPU timing (binary search fixed the 92s bottleneck?)
      + CPU-vs-GPU per-IMF agreement + load-imbalance (per-trial sift dist)
"""
import numpy as np, time
import emd_ref as R
import emd_gpu as G
from refsignal import make_ref_signal

print("=== (1) small signal: GPU full EMD vs CPU (N=2048 chirp+noise) ===")
N = 2048
t = np.arange(N, dtype=float)
rng = np.random.default_rng(1)
x = np.sin(2*np.pi*np.cumsum(0.01+0.05*t/N)) + 0.3*np.sin(2*np.pi*0.013*t) \
    + 0.1*rng.standard_normal(N)
imfs_c, resid_c, st_c = R.emd_1d(x, tau=0.25, max_sifts=50)
imfs_c = np.array(imfs_c)
# GPU full EMD, E=1 (no noise -> should match CPU plain EMD closely)
imfs_g, ns, nm = G.gpu_eemd_full(x, E=1, tau=0.25, max_sifts=50, max_imf=imfs_c.shape[0])
print("  CPU n_modes=%d  GPU nmode=%d  GPU total_sifts=%d" % (imfs_c.shape[0], nm[0], ns[0]))
rec_c = imfs_c.sum(0)+resid_c
rec_g = imfs_g[0].sum(0) + (x - imfs_g[0].sum(0))
print("  CPU recon=%.2e" % np.max(np.abs(x-rec_c)))
# per-IMF agreement (chaotic: expect ~1e-2..1e-0 drift, not 1e6)
for i in range(min(imfs_c.shape[0], imfs_g[0].shape[0])):
    d = np.abs(imfs_g[0,i]-imfs_c[i])
    print("    IMF %d : max|GPU-CPU|=%.3e  (bulk=%.2e ends=%.2e)"
          % (i+1, d.max(), d[N//10:9*N//10].max(), max(d[:N//10].max(),d[9*N//10:].max())))

print("\n=== (2) 441k reference sweep: GPU timing + agreement ===")
x, t = make_ref_signal()
N = len(x)
# warm JIT
G.gpu_eemd_full(np.sin(2*np.pi*0.03*np.arange(4096)), E=1, max_imf=3)
# CPU full EMD (ground truth)
t0=time.time(); imfs_c,resid_c,st_c = R.emd_1d(x, tau=0.25, max_sifts=50); cpu_t=time.time()-t0
imfs_c = np.array(imfs_c)
print("  CPU full EMD: %.3fs  n_modes=%d  total_sifts=%d  recon=%.2e"
      % (cpu_t, imfs_c.shape[0], st_c['total_sifts'], np.max(np.abs(x-(imfs_c.sum(0)+resid_c)))))
# GPU E=8 (a few trials to see load imbalance)
E=8
t0=time.time(); imfs_g,ns,nm = G.gpu_eemd_full(x, E=E, tau=0.25, max_sifts=50, max_imf=imfs_c.shape[0], seed=7); gpu_t=time.time()-t0
print("  GPU EEMD E=%d: %.3fs  per-trial total_sifts: min=%d max=%d mean=%.1f (imbalance x%.2f)"
      % (E, gpu_t, ns.min(), ns.max(), ns.mean(), ns.max()/max(1,ns.min())))
print("  GPU per-trial nmode: %s" % list(nm))
# GPU ensemble-averaged IMF1 vs CPU IMF1 (EEMD smooths mode mixing)
avg1 = imfs_g[:,0,:].mean(0)
d = np.abs(avg1 - imfs_c[0])
print("  avg(GPU IMF1) vs CPU IMF1: max|d|=%.3e (bulk=%.2e ends=%.2e)"
      % (d.max(), d[N//10:9*N//10].max(), max(d[:N//10].max(),d[9*N//10:].max())))
# GPU per-IMF reconstruction (single trial 0)
rec0 = imfs_g[0].sum(0) + (x - imfs_g[0].sum(0))
print("  GPU trial0 recon=%.2e" % np.max(np.abs(x-rec0)))
