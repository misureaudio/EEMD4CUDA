"""MANDATORY genuinely-non-stationary test.
Signals: (1) linear chirp (sweeping instantaneous freq), (2) AM (slow
amplitude modulation), (3) chirp+AM+noise. These produce mode mixing and
variable per-trial IMF/sift counts -> real load imbalance.
Measures: per-trial n_sift distribution, CPU-vs-GPU single-IMF agreement,
and per-trial timing (imbalance)."""
import numpy as np, time
import emd_ref as R
import emd_gpu as G

def make_signal(N, kind, seed=0):
    t = np.arange(N, dtype=float)
    rng = np.random.default_rng(seed)
    if kind == 'chirp':
        f = 0.01 + 0.09 * (t / N)                 # 0.01 -> 0.10 cyc/sample
        ph = 2*np.pi*np.cumsum(f)
        x = np.sin(ph) + 0.5*np.sin(2*np.pi*0.02*t + 0.3)
    elif kind == 'am':
        x = (1 + 0.7*np.cos(2*np.pi*0.004*t)) * np.sin(2*np.pi*0.03*t)
        x += 0.4*np.sin(2*np.pi*0.007*t + 1.0)
    elif kind == 'chirp_am_noise':
        f = 0.01 + 0.06*(t/N)
        ph = 2*np.pi*np.cumsum(f)
        x = (1 + 0.6*np.cos(2*np.pi*0.003*t)) * np.sin(ph)
        x += 0.3*np.sin(2*np.pi*0.013*t + 0.5)
        x += 0.15*rng.standard_normal(N)
    else:
        raise ValueError(kind)
    return x

for kind in ['chirp', 'am', 'chirp_am_noise']:
    N = 4096
    x = make_signal(N, kind)
    # --- GPU single-IMF EEMD, E=64 ---
    E = 64
    t0 = time.time()
    avg, nsift, raw = G.gpu_eemd(x, E=E, eps=0.2, tau=0.25, max_sifts=50, seed=7)
    gpu_t = time.time()-t0
    # --- CPU single-IMF (first IMF) reference, same seed ---
    t0 = time.time()
    imfs_c, resid_c, _ = R.emd_1d(x, tau=0.25, max_sifts=50)   # full EMD serial
    cpu_t = time.time()-t0
    first_imf_cpu = imfs_c[0]
    d = np.abs(avg - first_imf_cpu)
    i = int(np.argmax(d))
    print("=== %s  (N=%d, E=%d) ===" % (kind, N, E))
    print("  GPU: %.3fs  |  CPU full-EMD serial: %.3fs" % (gpu_t, cpu_t))
    print("  GPU per-trial n_sift: min=%d max=%d mean=%.2f  (imbalance x%.2f)"
          % (nsift.min(), nsift.max(), nsift.mean(), nsift.max()/max(1,nsift.min())))
    print("  GPU n_modes (CPU full): %d  per-mode sifts=%s"
          % (len(imfs_c), [len(a) for a in imfs_c]))
    print("  single-IMF max|GPU-CPU|=%.3e  argmax frac=%.3f" % (d.max(), i/N))
    print("  BULK=%.2e ENDS=%.2e" % (d[N//10:9*N//10].max(),
          max(d[:N//10].max(), d[9*N//10:].max())))
    # reconstruction (CPU)
    rec = sum(imfs_c) + resid_c
    print("  CPU recon err=%.2e" % np.max(np.abs(x-rec)))
    print()
