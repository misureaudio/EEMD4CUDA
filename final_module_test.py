"""Final module validation on the 441k reference sweep (the mandated test):
  * decompose() API end-to-end (CPU E=1, CPU E=8, CPU E=64 streaming, GPU E=64)
  * reconstruction (partition of unity) for each
  * CPU E=64 vs old in-memory result (streaming correctness) on a small signal
  * load-imbalance (per-trial sift distribution)
  * mode-mixing reduction: does EEMD (large E) reduce IMF1's IF spread?
"""
import numpy as np, time
import emd_par as M
import emd_ref as R
from refsignal import make_ref_signal
from scipy.signal import hilbert

def main():
    x, t = make_ref_signal()
    N = len(x)
    print("Reference sweep: N=%d" % N)

    # --- small-signal streaming correctness (CPU E=8 via module vs in-memory) ---
    print("\n=== streaming correctness (small signal, N=4096) ===")
    xs = x[:4096]
    r_stream = M.decompose(xs, method='cpu', E=8, eps=0.2, tau=0.25, seed=7)
    # in-memory reference using the SAME per-trial noise as the module worker
    sigma = 0.2*np.std(xs)
    acc = np.zeros((30, 4096)); cnt = np.zeros(30)
    for e in range(8):
        rng = np.random.default_rng((7 * 1000003 + e) & 0xFFFFFFFFFFFFFFFF)
        noise = rng.standard_normal(4096) * sigma
        imfs,_,_ = R.emd_1d(xs+noise, tau=0.25)
        for i,imf in enumerate(imfs): acc[i]+=imf; cnt[i]+=1
    n_modes = int((cnt>0).sum())
    ref = acc[:n_modes]/cnt[:n_modes,None]
    d = np.max(np.abs(r_stream.imfs - ref))
    print("  streaming vs in-memory max|d| = %.3e (should be ~0)" % d)
    print("  streaming recon err = %.3e" % r_stream.recon_error())

    # --- full 441k through the module API ---
    print("\n=== decompose() on 441k reference sweep ===")
    for meth, E in [('cpu',1), ('cpu',8), ('cpu',64), ('gpu',64)]:
        t0=time.time()
        r = M.decompose(x, method=meth, E=E, eps=0.2, tau=0.25, seed=7)
        dt=time.time()-t0
        ns = r.stats.get('per_trial_total_sifts')
        imb = ("sifts min=%d max=%d (x%.2f)" % (ns.min(), ns.max(), ns.max()/max(1,ns.min()))) if ns is not None and ns.size>1 else "single trial"
        print("  %-4s E=%3d : %7.3fs  n_modes=%d  recon_err=%.2e  %s"
              % (meth, E, dt, r.n_modes, r.recon_error(), imb))

    # --- mode-mixing reduction: IMF1 IF spread vs E ---
    print("\n=== mode-mixing reduction (IMF1 Hilbert IF spread) ===")
    def if_spread(imf):
        z = hilbert(imf)
        ifq = np.gradient(np.unwrap(np.angle(z)))/(2*np.pi)/(1/44100.0)
        a = np.abs(ifq)
        return np.percentile(a,5), np.percentile(a,50), np.percentile(a,95)
    for E in [1, 8, 64]:
        r = M.decompose(x, method='cpu' if E<64 else 'gpu', E=E, eps=0.2, tau=0.25, seed=7)
        lo,med,hi = if_spread(r.imfs[0])
        print("  E=%3d : IMF1 IF  5%%=%6.0f  med=%6.0f  95%%=%6.0f Hz  (spread 5%%-95%% = %.0f Hz)"
              % (E, lo, med, hi, hi-lo))

if __name__ == '__main__':
    main()
