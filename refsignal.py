"""Reference test signal (user-specified):
  * frequency sine sweep covering ten audio octaves, 20 Hz -> 20 kHz
  * sweep time 10 s
  * Fs = 44.1 kHz  ->  N = 441,000 samples
  * + white noise at 1/100 of the sweep max level (peak 1.0)

Log sweep:  f(t) = f1 * (f2/f1)^(t/T);  phase phi(t)=2*pi*f1*T*((f2/f1)^(t/T)-1)/ln(f2/f1)
"""
import numpy as np

FS = 44100
T  = 10.0
F1 = 20.0
F2 = 20000.0

def make_ref_signal(noise_level=0.01, seed=0):
    N = int(FS * T)
    t = np.arange(N) / FS
    ratio = F2 / F1
    phi = 2*np.pi * F1 * T * (ratio**(t/T) - 1.0) / np.log(ratio)
    x = np.sin(phi)                       # peak 1.0
    rng = np.random.default_rng(seed)
    x = x + noise_level * rng.standard_normal(N)   # white noise, 1/100 max level
    return x, t

def characterize(x, t):
    N = len(x)
    # instantaneous frequency via zero crossings (robust)
    # local extrema count (what EMD sees)
    nmax = int(np.sum((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])))
    nmin = int(np.sum((x[1:-1] < x[:-2]) & (x[1:-1] < x[2:])))
    # total cycles from the analytic phase
    ratio = F2/F1
    total_cycles = F1*T*(ratio-1.0)/np.log(ratio)
    # samples per cycle at the two ends
    spc_lo = FS/F1     # 2205
    spc_hi = FS/F2     # 2.205
    return dict(N=N, nmax=nmax, nmin=nmin, n_extrema=nmax+nmin,
                total_cycles=total_cycles, spc_lo=spc_lo, spc_hi=spc_hi,
                peak=float(np.max(np.abs(x))), std=float(np.std(x)))

if __name__ == '__main__':
    x, t = make_ref_signal()
    c = characterize(x, t)
    print("Reference signal (10-octave log sweep, 20Hz-20kHz, 10s, Fs=44.1k):")
    for k, v in c.items():
        print("  %-14s = %s" % (k, ("%.3f" % v) if isinstance(v, float) else v))
    print("  Nyquist = %.1f kHz ; sweep top = 20 kHz (%.1f%% of Nyquist)"
          % (FS/2/1000, 100*F2/(FS/2)))
    np.save('ref_signal.npy', x)
    print("  saved -> ref_signal.npy")
