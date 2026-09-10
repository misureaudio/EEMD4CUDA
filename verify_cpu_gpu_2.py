"""verify_cpu_gpu_2.py -- the 441k reference signal (ref_signal.npy).

  A. E=1 plain EMD on 441k: CPU vs GPU (per-IMF agreement, sift counts)
  B. SAME-NOISE EEMD E=8: the identical 2-D perturbed array is fed to both
     backends -> per-trial and ensemble-average agreement. This isolates
     emd_gpu from the host-side orchestration (strongest kernel check).
"""
import os
import numpy as np, time
import emd_ref as R
import emd_gpu as G

x = np.load('ref_signal.npy')
N = len(x)
print("ref_signal.npy: N=%d  std=%.4f  cpu_count=%s" % (N, x.std(), os.cpu_count()))

print("\n=== A. E=1 plain EMD on 441k: CPU vs GPU ===")
t0 = time.time(); imfs_c, resid_c, st_c = R.emd_1d(x, tau=0.25, max_sifts=50); tc = time.time() - t0
imfs_c = np.array(imfs_c)
t0 = time.time(); imfs_g, ns1, nm1 = G.gpu_eemd_full(x[None, :], E=1, tau=0.25,
                                                     max_sifts=50,
                                                     max_imf=st_c['n_modes']); tg = time.time() - t0
imfs_g = imfs_g[0]
print("  CPU: %.1fs  n_modes=%d  total_sifts=%d  per_mode=%s  recon=%.2e" % (
    tc, st_c['n_modes'], st_c['total_sifts'], st_c['per_mode_sifts'],
    np.max(np.abs(x - (imfs_c.sum(0) + resid_c)))))
print("  GPU: %.1fs  nmode=%d  nsift=%d" % (tg, nm1[0], ns1[0]))
print("  sift-count diff |CPU-GPU| = %d  (0 or 1 expected: identical stop criterion, chaotic boundary)"
      % abs(st_c['total_sifts'] - int(ns1[0])))
m = min(imfs_c.shape[0], imfs_g.shape[0])
for i in range(m):
    d = np.abs(imfs_g[i] - imfs_c[i])
    print("    IMF %2d: max|d|=%.3e  relL2=%.3e  corr=%.12f" % (
        i + 1, d.max(), np.linalg.norm(d) / np.linalg.norm(imfs_c[i]),
        np.corrcoef(imfs_g[i], imfs_c[i])[0, 1]))

print("\n=== B. same-noise EEMD E=8 (identical perturbed inputs) ===")
E = 8
sigma = 0.2 * np.std(x)
noises = np.empty((E, N))
for e in range(E):
    rng = np.random.default_rng((7 * 1000003 + e) & 0xFFFFFFFFFFFFFFFF)
    noises[e] = rng.standard_normal(N) * sigma
X = x[None, :] + noises

t0 = time.time()
cpu_imfs, per_cpu = [], []
for e in range(E):
    imfs_e, resid_e, st_e = R.emd_1d(X[e], tau=0.25, max_sifts=50)
    cpu_imfs.append(np.array(imfs_e))
    per_cpu.append(st_e['total_sifts'])
t_cpu = time.time() - t0
ncpu_max = max(len(a) for a in cpu_imfs)
acc = np.zeros((ncpu_max, N)); cnt = np.zeros(ncpu_max)
for a in cpu_imfs:
    for i, imf in enumerate(a):
        acc[i] += imf; cnt[i] += 1
ncpu = int((cnt > 0).sum())
avg_c = acc[:ncpu] / cnt[:ncpu, None]

t0 = time.time()
imfs_g8, ns8, nm8 = G.gpu_eemd_full(X, E=E, tau=0.25, max_sifts=50,
                                    max_imf=ncpu_max + 8)
t_gpu = time.time() - t0
maxm = imfs_g8.shape[1]
accg = np.zeros((maxm, N)); cntg = np.zeros(maxm)
for mm in range(maxm):
    mask = nm8 > mm
    accg[mm] = imfs_g8[mask, mm, :].sum(0); cntg[mm] = int(mask.sum())
ngpu = int((cntg > 0).sum())
avg_g = accg[:ngpu] / cntg[:ngpu, None]

print("  CPU serial: %.1fs  per-trial n_modes=%s" % (t_cpu, [len(a) for a in cpu_imfs]))
print("  GPU blocks: %.1fs  per-trial n_modes=%s" % (t_gpu, list(nm8)))
print("  per-trial total_sifts (CPU vs GPU) -- identical stop criterion => |diff|<=1:")
for e in range(E):
    print("    e=%d: cpu=%d gpu=%d  diff=%+d" % (e, per_cpu[e], int(ns8[e]), int(ns8[e]) - per_cpu[e]))
print("  per-trial per-IMF max|GPU-CPU|:")
for e in range(E):
    row = ["%.1e" % np.max(np.abs(imfs_g8[e, i] - cpu_imfs[e][i]))
           for i in range(min(len(cpu_imfs[e]), nm8[e]))]
    print("    e=%d: %s" % (e, row))
mm = min(ncpu, ngpu)
print("  ensemble average, per mode:")
for i in range(mm):
    d = np.abs(avg_g[i] - avg_c[i])
    print("    mode %2d: max|d|=%.3e  relL2=%.3e  corr=%.12f" % (
        i + 1, d.max(), np.linalg.norm(d) / np.linalg.norm(avg_c[i]),
        np.corrcoef(avg_g[i], avg_c[i])[0, 1]))
