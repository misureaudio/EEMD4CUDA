"""verify_cpu_gpu_3.py -- public API level (emd_par.decompose):

  A. quirk audit: min_extrema is IGNORED on the GPU path (hardcoded 4)
  B. quirk audit: max_imf honored on both paths
  C. EEMD E=64 via the API, cpu vs gpu -- the two paths use DIFFERENT noise
     streams for the same seed, so agreement here is statistical only;
     the difference must sit at the Monte-Carlo scale ~ eps*std(x)/sqrt(E)
  D. determinism: GPU same-seed bitwise; CPU parallel run-to-run
  E. ground truth: two-tone signal -> IMF dominant frequencies
"""
import os
import numpy as np, time
import emd_par as M

x = np.load('ref_signal.npy')
N = len(x)
xs = x[:20000]
print("cpu_count=%s" % os.cpu_count())

print("\n=== A. quirk audit: min_extrema on the GPU path ===")
g_hi = M.decompose(xs, method='gpu', E=1, min_extrema=100, seed=7)
g_lo = M.decompose(xs, method='gpu', E=1, min_extrema=4, seed=7)
print("  gpu min_extrema=100 vs 4: identical = %s  (n_modes %d vs %d)" % (
    np.array_equal(g_hi.imfs, g_lo.imfs), g_hi.n_modes, g_lo.n_modes))
c_hi = M.decompose(xs, method='cpu', E=1, min_extrema=100, seed=7)
c_lo = M.decompose(xs, method='cpu', E=1, min_extrema=4, seed=7)
print("  cpu min_extrema=100 vs 4: identical = %s  (n_modes %d vs %d)" % (
    np.array_equal(c_hi.imfs, c_lo.imfs), c_hi.n_modes, c_lo.n_modes))

print("\n=== B. quirk audit: max_imf=2 on both paths ===")
r2g = M.decompose(xs, method='gpu', E=1, max_imf=2, seed=7)
r2c = M.decompose(xs, method='cpu', E=1, max_imf=2, seed=7)
print("  n_modes: gpu=%d  cpu=%d   (expect 2, 2)" % (r2g.n_modes, r2c.n_modes))

print("\n=== C. API EEMD E=64 on 441k: cpu vs gpu (independent noise streams) ===")
t0 = time.time(); rc = M.decompose(x, method='cpu', E=64, eps=0.2, tau=0.25, seed=7); tc = time.time() - t0
t0 = time.time(); rg = M.decompose(x, method='gpu', E=64, eps=0.2, tau=0.25, seed=7); tg = time.time() - t0
print("  cpu: %.1fs  n_modes=%d  recon=%.2e" % (tc, rc.n_modes, rc.recon_error()))
print("  gpu: %.1fs  n_modes=%d  recon=%.2e" % (tg, rg.n_modes, rg.recon_error()))
mm = min(rc.n_modes, rg.n_modes)
print("  per-mode ensemble averages (same seed, DIFFERENT noise draws):")
for i in range(mm):
    a, b = rc.imfs[i], rg.imfs[i]
    print("    mode %2d: max|d|=%.3e  relL2=%.3e  corr=%.12f" % (
        i + 1, np.max(np.abs(a - b)),
        np.linalg.norm(a - b) / np.linalg.norm(a), np.corrcoef(a, b)[0, 1]))
print("  MC scale reference: eps*std(x)/sqrt(E) = %.4f" % (0.2 * np.std(x) / np.sqrt(64)))

print("\n=== D. determinism ===")
a1 = M.decompose(xs, method='gpu', E=8, seed=7); a2 = M.decompose(xs, method='gpu', E=8, seed=7)
print("  gpu same seed, 2 runs: max|d|=%.3e  bitwise=%s" % (
    np.max(np.abs(a1.imfs - a2.imfs)), np.array_equal(a1.imfs, a2.imfs)))
b1 = M.decompose(xs, method='cpu', E=16, seed=7); b2 = M.decompose(xs, method='cpu', E=16, seed=7)
print("  cpu parallel, 2 runs:  max|d|=%.3e  (0 = bitwise; ~1e-16 = FP summation-order noise from as_completed)"
      % np.max(np.abs(b1.imfs - b2.imfs)))

print("\n=== E. ground truth: two-tone signal (500 Hz + AM 50 Hz, 2 Hz AM) ===")
Fs = 44100.0
N2 = int(Fs * 2.0)
tt = np.arange(N2) / Fs
x2 = (np.sin(2 * np.pi * 500 * tt)
      + 0.5 * (1 + 0.5 * np.sin(2 * np.pi * 2 * tt)) * np.sin(2 * np.pi * 50 * tt))

def dom_freq(imf):
    w = np.hanning(len(imf))
    X = np.abs(np.fft.rfft(imf * w))
    fr = np.fft.rfftfreq(len(imf), d=1.0 / Fs)
    return fr[int(np.argmax(X))]

for meth in ('cpu', 'gpu'):
    r = M.decompose(x2, method=meth, E=1, tau=0.25, seed=7)
    print("  %s: n_modes=%d  dominant freqs = %s Hz" % (
        meth, r.n_modes, [int(dom_freq(r.imfs[i])) for i in range(r.n_modes)]))
