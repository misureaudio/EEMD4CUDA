import numpy as np
rho = 2 - np.sqrt(3)
print('decay base rho = 2-sqrt(3) = %.12f' % rho)
for s in (20, 25, 30, 40, 50):
    print('  seam width s=%2d : rho^s = %.3e' % (s, rho**s))
print()
# Empirical decay base on a NON-uniform knot set (mirror the essay's check)
rng = np.random.default_rng(0)
n = 4096
t = np.cumsum(1.0 + 0.39*rng.standard_normal(n))  # spacing CV ~0.39
t = np.sort(t)
# build the 2-dominant tridiagonal (natural spline interior system)
H = np.diff(t); nint = n-2
d = 2.0*(H[:nint]+H[1:nint+1]); l=np.zeros(nint); u=np.zeros(nint)
l[1:]=H[1:nint]; u[:nint-1]=H[1:nint]
A = np.diag(d) + np.diag(u,1) + np.diag(l,-1)
G = np.linalg.inv(A)
# decay of column 0 away from the diagonal
col = G[:, nint//2]
r = np.abs(col[1:200])
# fit log slope
slope = np.polyfit(np.arange(1,200), np.log(r+1e-300), 1)
rho_emp = float(np.exp(slope[0]))
print('empirical decay base (non-uniform, CV~0.39): rho_emp = %.6f  (theory %.6f)' % (rho_emp, rho))
print()
# block-Thomas parallelism model at N=441000
N=441000; nint=N//2-2; b=250; s=30
nblocks = int(np.ceil((nint+2*s)/ (b)))  # blocks of b owned, s halo each side
print('N=%d : nint=%d, block b=%d halo s=%d -> ~%d blocks, %d threads/block' % (N, nint, b, s, nblocks, 128))
print('  serial Thomas (1 thread): %d steps' % (2*nint))
print('  block-Thomas (128 thr)  : ~%.0f steps/thread' % (nblocks*(b+2*s)/128))
print('  serial fraction ratio (Thomas only): %.1fx' % ((2*nint)/(nblocks*(b+2*s)/128)))
# whole-sift serial fraction
serial_per_sift = 2*N + 2*(nint) + 2*(2*nint) + 2*nint  # 2 scans + 2 (H+sys) + 2 thomas + 2 Mf
print()
print('per-sift thread-0 serial iterations ~ %.2f M' % (serial_per_sift/1e6))