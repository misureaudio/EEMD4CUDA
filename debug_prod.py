"""Directly compare the PRODUCTION kernel (emd_gpu) first sift vs CPU single
sift on the 'failing' signal, dumping envelopes + h1 at the argmax location."""
import numpy as np
import emd_ref as R
import emd_gpu as G

N = 2048
t = np.arange(N, dtype=float)
rng = np.random.default_rng(42)
x = (np.sin(2*np.pi*0.03*t) + 0.5*np.sin(2*np.pi*0.011*t + 0.7)
     + 0.3*rng.standard_normal(N))

# production kernel, single sift
imf_g, ns_g = G.gpu_emd_single(x, tau=0.25, max_sifts=1)
# CPU single sift
eu_c, nmax_c = R.envelope(t, x, 'max')
el_c, nmin_c = R.envelope(t, x, 'min')
h1_c = x - 0.5*(eu_c + el_c)

d = np.abs(imf_g - h1_c)
i = int(np.argmax(d))
print("production kernel max_sifts=1 vs CPU single sift:")
print("  max|d|=%.3e  argmax idx=%d frac=%.3f"%(d.max(), i, i/N))
print("  at idx %d: gpu=%.6f cpu=%.6f"%(i, imf_g[i], h1_c[i]))
print("  CPU nmax=%d nmin=%d"%(nmax_c, nmin_c))
# compare the envelopes directly via a 1-sift that also returns them:
# re-run kernel but we only get out; so compare h1 implied:
# recompute CPU h1 with explicit print of envelopes at i
print("  eu_c[%.0f]=%.6f  el_c[%.0f]=%.6f  x[%.0f]=%.6f"%(t[i],eu_c[i],t[i],el_c[i],t[i],x[i]))
# bulk vs ends
print("  BULK max|d|=%.3e  ENDS max|d|=%.3e"%(d[N//10:9*N//10].max(),
      max(d[:N//10].max(), d[9*N//10:].max())))
# is it a SINGLE spike or widespread?
print("  #|d|>1e-3 = %d / %d"%(int((d>1e-3).sum()), N))
print("  neighbors: d[%d]=%.2e d[%d]=%.2e d[%d]=%.2e"%(max(0,i-1),d[max(0,i-1)],
      i,d[i],min(N-1,i+1),d[min(N-1,i+1)]))
