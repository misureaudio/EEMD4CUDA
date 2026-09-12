"""diag_dsbt.py -- diagnose the DSBT seam error profile (throwaway)."""
import numpy as np, math
import emd_ref as R
from test_block_thomas import (build_system, rhs_of, dsbt_solve, _dense,
                               B, S, decay_base, synth_knots)

def diag(name, tm, ym):
    d, l, u, H, n = build_system(tm)
    r = rhs_of(tm, ym, H, n)
    x_ex = np.linalg.solve(_dense(d, l, u, n), r)
    x_th = R.thomas(d, l, u, r)
    x_ds = dsbt_solve(d, l, u, r)
    
    # sanity: does dense == thomas?
    th_err = np.max(np.abs(x_ex - x_th))
    err = np.abs(x_ds - x_ex)
    scale = np.max(np.abs(x_ex))
    i_max = int(np.argmax(err))
    
    # distance of worst error from nearest seam (seams at block boundaries)
    print("\n== %s  n=%d  scale=%.4g  base=%.5f" % (name, n, scale, decay_base(d, l, u, n)))
    print("   dense-vs-thomas (should be ~eps): %.2e" % th_err)
    print("   DSBT max rel err: %.2e at i=%d  (x_ex=%.4g x_ds=%.4g)"
          % (err.max()/scale, i_max, x_ex[i_max], x_ds[i_max]))
    
    # where are the block boundaries?
    nb = (n + B - 1)//B
    print("   nb=%d  err near each owned boundary (blk*B .. blk*B+5):" % nb)
    for blk in range(min(nb, 8)):
        o0 = blk*B
        # Protect against array out-of-bounds at the end
        if o0 < len(err):
            seg = err[o0:min(o0+6, len(err))]
            print("     blk=%2d o0=%5d  err=%s" % (blk, o0,
                  " ".join("%.1e" % e for e in seg)))

# uniform
tm, ym = synth_knots("uniform", p=5000, seed=1)
diag("uniform", tm, ym)

# cv04
tm, ym = synth_knots("cv04", p=5000, seed=2)
diag("cv04", tm, ym)

# cv15 (Stress Test)
tm, ym = synth_knots("cv15", p=5000, seed=3)
diag("cv15", tm, ym)