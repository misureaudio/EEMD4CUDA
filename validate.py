"""Full validation: (1) locate the full-EMD GPU-vs-CPU divergence (spike vs
systematic), (2) EEMD statistical agreement GPU vs CPU (same seed), (3) the
load-imbalance per-trial sift distribution."""
import numpy as np, time
import emd_ref as R
import emd_gpu as G

N = 2048
t = np.arange(N, dtype=float)
rng = np.random.default_rng(42)
x = (np.sin(2*np.pi*0.03*t) + 0.5*np.sin(2*np.pi*0.011*t + 0.7)
     + 0.3*rng.standard_normal(N))

# ---- (1) full EMD: where does the divergence live? ----
imf_c, ns_c = R.emd_1imf(x, tau=0.25, max_sifts=50)
imf_g, ns_g = G.gpu_emd_single(x, tau=0.25, max_sifts=50)
d = np.abs(imf_g - imf_c)
print("=== (1) full EMD single signal (natural BC) ===")
print("CPU n_sift=%d GPU n_sift=%d  max|d|=%.3e"%(ns_c, ns_g, d.max()))
i = int(np.argmax(d))
lo = max(0, i-1); hi = min(N-1, i+1)
print("  argmax idx=%d  frac=%.3f  (neighbors: d[%d]=%.1e d[%d]=%.1e)"
      % (i, i/N, lo, d[lo], hi, d[hi]))
print("  #samples with |d|>1e-3 : %d / %d"%(int((d > 1e-3).sum()), N))
# end-effect check: compare BULK (middle 80%) vs ENDS (outer 10% each)
bulk = d[N//10 : 9*N//10]
ends = np.concatenate([d[:N//10], d[9*N//10:]])
print("  BULK (mid 80%%) max|d| = %.3e"%bulk.max())
print("  ENDS (outer 20%%) max|d| = %.3e   <- natural-BC end-effect"%ends.max())

# ---- (2) EEMD statistical agreement (the actual parallel axis) ----
E = 32
t0 = time.time()
avg_c, res_c, raw_c = R.eemd(x, E=E, parallel=False, seed=123)
tc = time.time() - t0
t0 = time.time()
avg_g, nsift_g, raw_g = G.gpu_eemd(x, E=E, eps=0.2, seed=123)
tg = time.time() - t0
dE = np.abs(avg_g - avg_c)
print("\n=== (2) EEMD  E=%d  (same seed, independent trials) ===" % E)
print("  CPU serial: %.2fs | GPU: %.2fs | speedup %.2fx" % (tc, tg, tc/tg))
print("  max|avg_gpu - avg_cpu| = %.3e" % dE.max())
print("  rms|avg_gpu - avg_cpu| = %.3e" % np.sqrt(np.mean(dE**2)))
print("  per-trial n_sift: min=%d max=%d mean=%.1f  (load imbalance)"
      % (nsift_g.min(), nsift_g.max(), nsift_g.mean()))

# ---- (3) load-imbalance: per-trial sift distribution (CPU, serial record) ----
li = R.load_imbalance_demo(x, E=E, n_workers=4, seed=123)
print("\n=== (3) load imbalance (per-trial sifts, CPU) ===")
print("  n_sift min=%d max=%d mean=%.1f  imbalance(max/mean)=%.2f"
      % (li['n_sift_min'], li['n_sift_max'], li['n_sift_mean'],
         li['imbalance_ratio']))
print("  static-RR makespan=%.3f  dynamic lower-bound=%.3f  serial=%.3f"
      % (li['static_makespan'], li['dynamic_lb'], li['serial']))
print("  static/dynamic_lb = %.2f  (gap = the cost of NOT work-stealing)"
      % (li['static_makespan']/li['dynamic_lb']))
