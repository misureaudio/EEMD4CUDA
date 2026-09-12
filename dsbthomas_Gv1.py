import numpy as np
import time

# =====================================================================
# 1. Exact Thomas Solver
# =====================================================================
def thomas_solve(d, l, u, r):
    """Exact serial Thomas algorithm."""
    n = len(d)
    c_ = np.zeros(n)
    x = np.zeros(n)
    if n == 0: return x
    
    c_[0] = u[0] / d[0] if n > 1 else 0.0
    x[0] = r[0] / d[0]
    
    for i in range(1, n):
        denom = d[i] - l[i] * c_[i-1]
        c_[i] = u[i] / denom if i < n-1 else 0.0
        x[i] = (r[i] - l[i] * x[i-1]) / denom
        
    for i in range(n-2, -1, -1):
        x[i] = x[i] - c_[i] * x[i+1]
        
    return x

# =====================================================================
# 2. Decay-Seamed Block Thomas (DSBT) Solver
# =====================================================================
def dsbt_solve(d, l, u, r, b, s):
    """
    Decay-Seamed Block Thomas Solver.
    Partitions the system into blocks of size `b`.
    Each block grabs a halo/seam of size `s` on both sides.
    Solves the padded block locally (assuming 0 outside the seam),
    then extracts and stitches the valid `b` core.
    """
    N = len(d)
    x_dsbt = np.zeros(N)
    nblocks = int(np.ceil(N / b))
    
    for i in range(nblocks):
        # The core indices this block is responsible for
        start = i * b
        end = min((i + 1) * b, N)
        
        # The padded indices including the seam 's'
        pad_start = max(0, start - s)
        pad_end = min(N, end + s)
        
        # Extract local padded system
        d_loc = d[pad_start:pad_end].copy()
        l_loc = l[pad_start:pad_end].copy()
        u_loc = u[pad_start:pad_end].copy()
        r_loc = r[pad_start:pad_end].copy()
        
        # Local boundaries treat the world outside the seam as 0
        l_loc[0] = 0.0
        if len(u_loc) > 0:
            u_loc[-1] = 0.0
            
        # Solve the local block exactly
        x_loc = thomas_solve(d_loc, l_loc, u_loc, r_loc)
        
        # Extract the core and map it back to global x
        core_start = start - pad_start
        core_end = core_start + (end - start)
        x_dsbt[start:end] = x_loc[core_start:core_end]
        
    return x_dsbt

# =====================================================================
# System Generator (EMD Natural Spline)
# =====================================================================
def generate_system(N_knots, CV):
    rng = np.random.default_rng(42)
    # Generate knot spacings with desired Coefficient of Variation
    spacing = 1.0 + CV * rng.standard_normal(N_knots)
    spacing = np.clip(spacing, 0.01, None) # strictly positive
    t = np.cumsum(spacing)
    
    H = np.diff(t)
    nint = N_knots - 2
    d = 2.0 * (H[:nint] + H[1:nint+1])
    l = np.zeros(nint); l[1:] = H[1:nint]
    u = np.zeros(nint); u[:nint-1] = H[1:nint]
    
    # Generate a random Right-Hand Side
    r = rng.standard_normal(nint)
    return d, l, u, r

# =====================================================================
# Test Suite matching Reviewer's Demands
# =====================================================================
def run_tests():
    print("=== Decay-Seamed Block Thomas (DSBT) Validation Suite ===\n")
    
    N = 100000
    b = 250
    rho = 2 - np.sqrt(3)
    
    # --- TEST 1: Error vs Seam Width Curve ---
    print("1. Error vs Seam-Width (s) [N=%d, block_size=%d, CV=0.4]" % (N, b))
    print("---------------------------------------------------------")
    d, l, u, r = generate_system(N, 0.4)
    x_exact = thomas_solve(d, l, u, r)
    
    print(f"{'Seam (s)':<10} | {'Max Error (L-inf)':<20} | {'Theory (rho^s)':<20}")
    for s in [5, 10, 15, 20, 25, 30]:
        x_dsbt = dsbt_solve(d, l, u, r, b, s)
        err_inf = np.max(np.abs(x_dsbt - x_exact))
        theory = rho**s
        print(f"{s:<10} | {err_inf:<20.3e} | {theory:<20.3e}")
        
        # Error localization check on s=15
        if s == 15:
            err_array = np.abs(x_dsbt - x_exact)
            idx_max = np.argmax(err_array)
            dist_to_boundary = min(idx_max % b, b - (idx_max % b))
            print(f"   -> [Localization Check at s=15]: Max error occurs at index {idx_max}")
            print(f"   -> Distance to nearest block boundary: {dist_to_boundary} elements (Expected: 0)")
    print()

    # --- TEST 2: Non-uniformity (CV) Stress Test ---
    print("2. Non-Uniform Grid Stress Test [s=30, b=250]")
    print("---------------------------------------------")
    print(f"{'CV':<10} | {'Max Error (L-inf)':<20}")
    for cv in [0.0, 0.1, 0.3, 0.5, 1.0]:
        d, l, u, r = generate_system(N, cv)
        x_exact = thomas_solve(d, l, u, r)
        x_dsbt = dsbt_solve(d, l, u, r, b, s=30)
        err_inf = np.max(np.abs(x_dsbt - x_exact))
        print(f"{cv:<10.1f} | {err_inf:<20.3e}")
    print()

if __name__ == '__main__':
    run_tests()