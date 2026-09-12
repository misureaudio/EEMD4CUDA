"""bench_tblock.py -- T_block (E=1, N=441000, plain reference sweep) median of 3.
Used for the Phase-4 before/after comparison (run before and after the DSBT
change). Not a test: prints numbers only.
"""
import time
import numpy as np
import emd_gpu as G
from refsignal import make_ref_signal

N = 441000
MAXIMF = 16

def main():
    x, _ = make_ref_signal()
    x = x[:N]
    G.gpu_eemd_full(x[None, :], E=1, tau=0.25, max_sifts=50, max_imf=MAXIMF)  # JIT warm
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        G.gpu_eemd_full(x[None, :], E=1, tau=0.25, max_sifts=50, max_imf=MAXIMF)
        ts.append(time.perf_counter() - t0)
    print("T_block N=%d E=1: %s  median=%.3f s"
          % (N, " ".join("%.3f" % v for v in ts), float(np.median(ts))))

if __name__ == '__main__':
    main()
