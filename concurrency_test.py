"""Pin down the A2000 concurrency model (corrects the essay's 'saturates at
E~8' claim):
  (1) N=441k: E=1 cold (incl JIT) and E=1 WARM -> per-block time T_block.
  (2) N=110250 (1/4 size, fits E=512 in 12GB): wall vs E=64,128,256,512 ->
      find where the wall stops being flat (the resident-block capacity C).
If wall is flat for E <= C and grows in waves beyond, per-block time is
constant and blocks are fully concurrent (not capped at 8)."""
import numpy as np, time
import emd_gpu as G
from refsignal import make_ref_signal

def chirp_like(N, seed=7):
    t = np.arange(N, dtype=float)
    f = 0.01 + 0.09*(t/N)
    return np.sin(2*np.pi*np.cumsum(f)) + 0.5*np.sin(2*np.pi*0.02*t + 0.3)

print("=== (1) N=441k: per-block time (E=1) ===")
x = make_ref_signal()[0]
N = len(x)
t0=time.time(); G.gpu_eemd_full(x, E=1, tau=0.25, max_imf=13); dt1=time.time()-t0
t0=time.time(); G.gpu_eemd_full(x, E=1, tau=0.25, max_imf=13); dt2=time.time()-t0
print("  E=1 cold (incl JIT): %.2fs" % dt1)
print("  E=1 WARM  = T_block: %.2fs" % dt2)

print("\n=== (2) N=110250: wall vs E (find resident capacity C) ===")
x4 = x[:110250]
N4 = len(x4)
t0=time.time(); G.gpu_eemd_full(x4, E=1, tau=0.25, max_imf=13); dtJ=time.time()-t0
print("  E=1 cold (incl JIT): %.2fs" % dtJ)
t0=time.time(); G.gpu_eemd_full(x4, E=1, tau=0.25, max_imf=13); dt1b=time.time()-t0
print("  E=1 WARM  = T_block(110k): %.2fs" % dt1b)
for E in [64, 128, 256, 512]:
    t0=time.time(); imfs,ns,nm = G.gpu_eemd_full(x4, E=E, tau=0.25, max_imf=13); dt=time.time()-t0
    print("  E=%4d : wall=%7.2fs  (wall/T_block=%.1f waves)  sifts[%d,%d]"
          % (E, dt, dt/dt1b, ns.min(), ns.max()))
