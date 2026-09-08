"""Definitive scaling study on the 441k reference sweep (the mandated
non-stationary test):
  (A) GPU EEMD vs E  -- does it scale with E, or is thread-0 serial work
      the wall? (E=1,8,32,64,128)
  (B) CPU parallel EEMD vs E and n_workers  -- the workhorse path
  (C) per-trial load imbalance (sift distribution) on the sweep
"""
import numpy as np, time, os
import emd_ref as R
import emd_gpu as G
from refsignal import make_ref_signal

def main():
    x, t = make_ref_signal()
    N = len(x)
    P = max(1, (os.cpu_count() or 2) - 1)
    print("N=%d  cpu_count=%d  (P=%d workers)" % (N, os.cpu_count(), P))

    # warm JIT
    G.gpu_eemd_full(np.sin(2*np.pi*0.03*np.arange(4096)), E=1, max_imf=3)

    print("\n=== (A) GPU EEMD vs E (441k sweep) ===")
    for E in [1, 8, 32, 64, 128]:
        t0=time.time(); imfs,ns,nm = G.gpu_eemd_full(x, E=E, tau=0.25,
             max_sifts=50, max_imf=12, seed=7); dt=time.time()-t0
        print("  E=%4d : %8.3fs  (%.3f ms/trial)  per-trial sifts min=%d max=%d"
              % (E, dt, 1000*dt/E, ns.min(), ns.max()))

    print("\n=== (B) CPU parallel EEMD vs E (441k sweep), P=%d workers ===" % P)
    for E in [8, 32, 64, 128]:
        t0=time.time(); avg,resid,res = R.eemd(x, E=E, eps=0.2, parallel=True,
             n_workers=P, seed=7, tau=0.25); dt=time.time()-t0
        ns = np.array([r[2]['total_sifts'] for r in res])
        print("  E=%4d : %8.3fs  (%.3f ms/trial)  per-trial sifts min=%d max=%d (imbalance x%.2f)"
              % (E, dt, 1000*dt/E, ns.min(), ns.max(), ns.max()/max(1,ns.min())))

    print("\n=== (C) load imbalance: per-trial total_sifts (CPU, E=64) ===")
    avg,resid,res = R.eemd(x, E=64, eps=0.2, parallel=True, n_workers=P, seed=7, tau=0.25)
    ns = np.array([r[2]['total_sifts'] for r in res])
    print("  E=64 per-trial total_sifts: min=%d max=%d mean=%.2f  (imbalance x%.2f)"
          % (ns.min(), ns.max(), ns.mean(), ns.max()/max(1,ns.min())))
    print("  histogram:", dict((int(k),int(v)) for k,v in zip(*np.unique(ns,return_counts=True))))

if __name__ == '__main__':
    main()
