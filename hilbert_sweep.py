"""Rigorous characterization of EMD on the 10-octave sweep (the mandated
genuinely-non-stationary test). Uses the Hilbert instantaneous frequency
(the Huang-Hilbert connection) rather than a whole-signal FFT, which is the
wrong tool for a chirp.

Questions answered:
  (1) Where is the raw signal's spectral energy (cycles per octave)?
  (2) What instantaneous-frequency range does each IMF track over time?
  (3) Does EMD resolve all 10 octaves, or does the top (20kHz, 2.2
      samples/cycle) get smeared / fail to separate?
"""
import numpy as np
from scipy.signal import hilbert
import emd_ref as R
from refsignal import make_ref_signal, FS, T, F1, F2

x, t = make_ref_signal()
N = len(x)

# ---- (1) cycles per octave ----
print("=== (1) signal structure: cycles per octave (log sweep) ===")
total = 0
for k in range(10):
    f_lo = F1 * 2**k; f_hi = F1 * 2**(k+1)
    # time spent in this octave
    t_lo = T*np.log(f_lo/F1)/np.log(F2/F1)
    t_hi = T*np.log(f_hi/F1)/np.log(F2/F1)
    dur = t_hi - t_lo
    cycles = f_lo*dur*np.log(2)/np.log(f_hi/f_lo) * (f_hi/f_lo)  # exact integral
    # exact: integral f dt = f1*T*( (f_hi/f1)^(t_hi/T) - (f_lo/f1)^(t_lo/T) )/ln(F2/F1)
    cycles = F1*T*( (f_hi/F1) - (f_lo/F1) )/np.log(F2/F1)
    total += cycles
    print("  oct %2d : %7.0f-%7.0f Hz  dur=%.2fs  cycles=%7.0f  (%.1f%%)"
          % (k+1, f_lo, f_hi, dur, cycles, 100*cycles/28924))
print("  total cycles = %.0f" % total)

# ---- (2)+(3) Hilbert instantaneous frequency per IMF ----
print("\n=== (2)+(3) Hilbert instantaneous frequency per IMF ===")
imfs, resid, stats = R.emd_1d(x, tau=0.25, max_sifts=50)
print("n_modes=%d total_sifts=%d" % (stats['n_modes'], stats['total_sifts']))
for i, imf in enumerate(imfs[:8]):
    # instantaneous frequency via Hilbert
    z = hilbert(imf)
    inst_phase = np.unwrap(np.angle(z))
    ifreq = np.gradient(inst_phase) / (2*np.pi) / (1.0/FS)   # Hz
    # robust range: 5-95 percentile, ignore sign flips (imf can go negative)
    pos = ifreq[np.abs(ifreq) > 0]
    f_lo = np.percentile(np.abs(ifreq), 5)
    f_hi = np.percentile(np.abs(ifreq), 95)
    f_med = np.percentile(np.abs(ifreq), 50)
    print("  IMF %2d : std=%.3f  IF med=%7.0f Hz  [5%%=%6.0f, 95%%=%7.0f] Hz"
          % (i+1, np.std(imf), f_med, f_lo, f_hi))

# ---- where did the 20kHz content go? ----
print("\n=== (4) is the top octave (10-20kHz) recovered? ===")
# reconstruct the sum of all IMFs and residual = x (partition of unity), so
# the 20kHz content is SOMEWHERE. Check the raw signal's high-freq band energy
# vs where it lands.
X = np.abs(np.fft.rfft(x * np.hanning(N)))
freqs = np.fft.rfftfreq(N, d=1/FS)
band = (freqs > 10000) & (freqs < 20000)
print("  raw signal energy in 10-20kHz band: %.3f (frac of total: %.3f)"
      % (np.sum(X[band]**2), np.sum(X[band]**2)/np.sum(X**2)))
# per-IMF energy in the 10-20kHz band
for i, imf in enumerate(imfs):
    Xi = np.abs(np.fft.rfft(imf * np.hanning(N)))
    e = np.sum(Xi[band]**2)/max(1e-30, np.sum(Xi**2))
    print("    IMF %2d : 10-20kHz band energy frac = %.4f" % (i+1, e))
