import numpy as np

# =====================================================================
# (a) Sifting operator linear algebra -- FAST version
#     - exact cubic-spline tridiagonal matrix + 2-dominance (proven + measured)
#     - Green's function A^{-1}: local sub-block inverse, decay base rho
#     - block-method (local-window) error decay vs overlap width w
# =====================================================================

def build_interior_system(tm, ym):
    """Interior tridiagonal system for cubic-spline 2nd derivs M (natural BC).
    Returns (d,l,u,H,n) with A[m,m]=d[m], A[m,m+1]=u[m], A[m,m-1]=l[m]."""
    p = len(tm)
    H = np.diff(tm)                 # H[j]=tm[j+1]-tm[j] = h_{j+1} (1-based)
    n = p - 2
    d = 2.0 * (H[0:n] + H[1:n+1])   # d[m]=2(H[m]+H[m+1])=2(h_{k-1}+h_k), k=m+2
    l = np.zeros(n); u = np.zeros(n)
    l[1:]     = H[1:n]             # l[m]=H[m]   (h_{k-1})
    u[0:n-1]  = H[1:n]             # u[m]=H[m+1] (h_k)
    return d, l, u, H, n

def rhs_vector(tm, ym, H, n):
    h_prev = H[0:n];  h_next = H[1:n+1]
    y_prev = ym[0:n]; y_k    = ym[1:n+1]; y_next = ym[2:n+2]
    return 6.0 * ((y_next - y_k)/h_next - (y_k - y_prev)/h_prev)

def thomas(d, l, u, rhs):
    """O(n) Thomas algorithm (no pivot; valid: 2-dominant)."""
    n = len(d)
    c_ = np.empty(n); d_ = np.empty(n)
    c_[0] = (u[0] if n > 1 else 0.0)/d[0]
    d_[0] = rhs[0]/d[0]
    for i in range(1, n):
        m = l[i]/d[i]
        c_[i] = (u[i] if i < n-1 else 0.0)/(d[i] - l[i]*c_[i-1])
        d_[i] = (rhs[i] - l[i]*d_[i-1])/(d[i] - l[i]*c_[i-1])
    x = np.empty(n)
    x[-1] = d_[-1]
    for i in range(n-2, -1, -1):
        x[i] = d_[i] - c_[i]*x[i+1]
    return x

def solve_M_full(tm, ym):
    d, l, u, H, n = build_interior_system(tm, ym)
    M_int = thomas(d, l, u, rhs_vector(tm, ym, H, n))
    M_full = np.zeros(len(tm)); M_full[1:-1] = M_int
    return M_full

def solve_M_window(tm, ym, k, w):
    lo = max(0, k - w); hi = min(len(tm) - 1, k + w)
    if hi - lo < 2:
        return None
    tloc = tm[lo:hi+1]; yloc = ym[lo:hi+1]
    Mloc = solve_M_full(tloc, yloc)
    return Mloc[k - lo]

def extract_extrema(x):
    mx = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1
    mn = np.where((x[1:-1] < x[:-2]) & (x[1:-1] < x[2:]))[0] + 1
    return mx, mn

def green_subblock(tm, ym, k, half, dmin, dmax):
    """Invert a dense sub-block of the tridiagonal system centered at knot k
    (size 2*half+1, natural BC at sub-block edges). The reference knot k is the
    MIDDLE local index `half`, so both edges are `half` away from it; with
    half >> dmax the boundary does not affect G[half, half+d]. Return the row
    G[half, :] entries for d=0..dmax (i.e. G[half, half+d])."""
    lo = max(0, k - half); hi = min(len(tm) - 1, k + half)
    tloc = tm[lo:hi+1]; yloc = ym[lo:hi+1]
    d, l, u, H, n = build_interior_system(tloc, yloc)
    A = np.diag(d)
    if n > 1:
        A = A + np.diag(l[1:], -1) + np.diag(u[0:n-1], 1)
    G = np.linalg.inv(A)
    # sub-block knots are lo..hi ; interior unknowns lo+1..hi-1
    # reference knot k -> interior index (k) - (lo+1)
    ref = k - (lo + 1)
    return G[ref, ref:ref+dmax+1]

def report(name, t, x, half=200, dmin=6, dmax=18):
    print("\n" + "="*74)
    print(f"SIGNAL: {name}")
    print("="*74)
    mx, mn = extract_extrema(x)
    tm, ym = t[mx], x[mx]                 # UPPER envelope = spline through maxima
    p = len(tm)
    d, l, u, H, n = build_interior_system(tm, ym)
    print(f"  N samples          = {len(x)}")
    print(f"  #maxima (knots) p  = {p}   (interior unknowns n = {n})")
    print(f"  knot spacing H     : mean={H.mean():.5f}  min={H.min():.5f}  "
          f"max={H.max():.5f}  cv={H.std()/H.mean():.3f}")
    off = (np.abs(l) + np.abs(u))
    ratio = off / d
    print(f"  (|l|+|u|)/d        : max={ratio.max():.6f}  interior max="
          f"{ratio[1:-1].max():.6f}   [2-dominant iff <= 0.5]")
    # condition of the (small) sub-block, as a stability proxy
    k = p // 2
    g = green_subblock(tm, ym, k, half, dmin, dmax)
    # fit log|G[0,d]| = a + b d over d in [dmin, dmax]
    ds = np.arange(dmin, dmax+1)
    vals = g[dmin:dmax+1]
    mask = vals > 0
    b, a = np.polyfit(ds[mask], np.log(vals[mask]), 1)
    rho, C = float(np.exp(b)), float(np.exp(a))
    print(f"  Green's fn (local sub-block, ref knot k={k}):")
    print(f"    rho_emp = {rho:.6f}    C_emp = {C:.5f}    (fit d in [{dmin},{dmax}])")
    print(f"    theory (const 4,1,1): rho = 2-sqrt(3) = {2-np.sqrt(3):.6f}   "
          f"C = 1/(2 sqrt(3)) = {1/(2*np.sqrt(3)):.5f}")
    print("    G[0,d]  d=0..12 (SIGNED):", np.array2string(g[:13], precision=4, suppress_small=True))
    print("    |G[0,d]| d=0..12 (MAGN) :", np.array2string(np.abs(g[:13]), precision=4, suppress_small=True))
    # ---- block-method (local-window) error decay ----
    M_global = solve_M_full(tm, ym)
    ws = np.arange(2, 61)
    errs = []
    for w in ws:
        Mk = solve_M_window(tm, ym, k, int(w))
        errs.append(abs(M_global[k] - Mk) if Mk is not None else np.nan)
    errs = np.array(errs)
    good = np.isfinite(errs) & (errs > 0)
    if good.sum() > 8:
        bw, aw = np.polyfit(ws[good], np.log(errs[good]), 1)
        print(f"  block-method err |M_k^global - M_k^(w)| at knot k={k}:")
        print("    w         :", " ".join(f"{int(w):>4}" for w in ws[::10]))
        print("    err       :", np.array2string(errs[::10], precision=3, suppress_small=True))
        print(f"    fitted decay base of the ERROR = {np.exp(bw):.6f}   (compare rho_emp={rho:.6f})")
        # smallest w giving err < 1e-6 (relative to |M_global[k]|)
        scale = max(abs(M_global[k]), 1e-12)
        rel = errs/scale
        idx = np.where(np.isfinite(rel) & (rel < 1e-6))[0]
        if len(idx):
            print(f"    smallest w with rel err < 1e-6 : w = {int(ws[idx[0]])}")
    return rho

# ---------------------------------------------------------------------
# CASE 1: REGULAR (pure sinusoid -> exactly equal knot spacing)
# ---------------------------------------------------------------------
f, T, N = 50.0, 20.0, 400000
t = np.linspace(0, T, N)
x_reg = np.sin(2*np.pi*f*t)
rho_reg = report("REGULAR: pure 50 Hz sinusoid (equal knot spacing)", t, x_reg)

# ---------------------------------------------------------------------
# CASE 2: IRREGULAR / realistic non-stationary (chirp 5->20 Hz + slow AM)
#   NO noise: spacing variation comes purely from the chirp (f changes 4x)
#   and the AM (amplitude modulates the spline knot values), so this tests
#   whether rho is robust to *genuinely* non-uniform knot spacing.
# ---------------------------------------------------------------------
T2, N2 = 100.0, 200000
t2 = np.linspace(0, T2, N2)
f0, f1 = 5.0, 20.0
f_inst = f0 + (f1 - f0)*t2/T2
phase = 2*np.pi*np.cumsum(f_inst)*(t2[1]-t2[0])
amp = 1.0 + 0.4*np.sin(2*np.pi*0.02*t2)
x_irr = amp*np.sin(phase)
rho_irr = report("IRREGULAR: chirp 5->20 Hz + slow AM (no noise, varying spacing)", t2, x_irr)

print("\n" + "="*74)
print("SUMMARY")
print("="*74)
print(f"  rho = 2-sqrt(3) theory        = {2-np.sqrt(3):.6f}")
print(f"  rho_emp  REGULAR  (equal spc) = {rho_reg:.6f}")
print(f"  rho_emp  IRREGULAR (chirp)    = {rho_irr:.6f}")
print(f"  overlap width for rel eps=1e-6 (const case): w > "
      f"{np.log(1e-6*2*np.sqrt(3))/np.log(2-np.sqrt(3)):.1f} extrema")
print(f"  overlap width for rel eps=1e-9 (const case): w > "
      f"{np.log(1e-9*2*np.sqrt(3))/np.log(2-np.sqrt(3)):.1f} extrema")
