"""Confirm the GPU linear-scan bottleneck on the 441k reference signal,
and check the high-end (20kHz, 2.2 samples/cycle) spectral content of the
extracted IMFs (is the top octave even resolvable by EMD?)."""
import numpy as np, time
import emd_ref as R
from refsignal import make_ref_signal

x, t = make_ref_signal()
N = len(x)

# --- spectral check on CPU EMD IMFs: does the top octave survive? ---
imfs, resid, stats = R.emd_1d(x, tau=0.25, max_sifts=50)
print("CPU EMD: n_modes=%d total_sifts=%d" % (stats['n_modes'], stats['total_sifts']))
def dom_freq(imf):
    X = np.fft.rfft(imf * np.hanning(len(imf)))
    freqs = np.fft.rfftfreq(len(imf), d=1/44100.0)
    # median of the top-10 spectral peaks (robust to leakage)
    order = np.argsort(np.abs(X))[::-1][:10]
    return np.median(freqs[order])
for i, imf in enumerate(imfs):
    print("  IMF %2d : std=%.3f  dominant_freq=%.0f Hz" % (i+1, np.std(imf), dom_freq(imf)))

# --- GPU timing on the reference signal (warm, E=1 and E=8) ---
try:
    import emd_gpu as G
    # warm up JIT on a small signal
    G.gpu_eemd(np.sin(2*np.pi*0.03*np.arange(4096)), E=1, seed=0)
    for E in [1, 8]:
        t0 = time.time()
        avg, ns, raw = G.gpu_eemd(x, E=E, eps=0.2, tau=0.25, max_sifts=50, seed=7)
        dt = time.time()-t0
        print("GPU E=%d on N=%d : %.3fs  per-trial n_sift max=%d" % (E, N, dt, ns.max()))
except Exception as e:
    print("GPU failed:", type(e).__name__, e)
