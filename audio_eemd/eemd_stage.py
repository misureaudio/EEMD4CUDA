"""EEMD stage: wraps eemd4cuda.emd_par.decompose (plan Task 2).

The EMD core is NOT reimplemented here — this module only bootstraps the
import path (relative to this file, idempotent, no cwd dependence), adds
timing, and enforces the partition-of-unity gate.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_EMD4CUDA_DIR = Path(__file__).resolve().parent.parent / "eemd4cuda"


class EEMDError(Exception):
    """EEMD backend failure or partition-of-unity violation."""

    def __init__(self, message: str, code: str = "eemd.failure"):
        super().__init__(message)
        self.code = code


def _bootstrap():
    """Make eemd4cuda importable (idempotent) and return the emd_par module."""
    p = str(_EMD4CUDA_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    import emd_par  # noqa: F401  (cupy/numba imported lazily inside emd_par)
    return emd_par


@dataclass
class Decomp:
    """Result of one EEMD run."""

    x: np.ndarray        # input segment, float64 [N]
    imfs: np.ndarray     # ensemble-averaged IMFs, float64 [n_modes, N]
    residual: np.ndarray # float64 [N]
    n_modes: int
    stats: dict          # verbatim from emd_par (method, E, sifting, ...)
    recon: float         # max|x - (sum imfs + residual)|
    wall_s: float        # wall-clock of the decompose() call


def run_eemd(x: np.ndarray, method: str = "auto", E: int = 64, eps: float = 0.2,
             tau: float = 0.25, max_imf: int = -1, max_sifts: int = 50,
             seed: int = 0, batch: int | None = None) -> Decomp:
    """Run EMD (E=1) or EEMD (E>1) on a 1-D float64 signal.

    batch: GPU memory control — max trials resident at once (None = auto from
    free VRAM); ignored on CPU.
    Raises EEMDError on bad input, backend failure, or if the partition of
    unity is violated beyond 1e-6 * max|x| (plan §3 Tier 1 gate).
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    M = _bootstrap()
    t0 = time.perf_counter()
    try:
        res = M.decompose(x, method=method, E=E, eps=eps, tau=tau,
                          max_imf=max_imf, max_sifts=max_sifts, seed=seed,
                          batch=batch)
    except ValueError as e:
        raise EEMDError(f"EEMD bad input: {e}", "eemd.bad_input") from e
    except Exception as e:
        raise EEMDError(f"EEMD backend failure: {e}", "eemd.backend") from e
    wall = time.perf_counter() - t0

    recon = float(res.recon_error())
    scale = float(np.abs(x).max())
    if recon > 1e-6 * max(scale, 1e-300):
        raise EEMDError(
            f"partition of unity violated: recon_error={recon:.3e} "
            f"(gate 1e-6*max|x| = {1e-6 * scale:.3e})", "eemd.recon_fail")

    imfs = np.asarray(res.imfs, dtype=np.float64)
    if res.n_modes == 0:
        imfs = np.empty((0, x.shape[0]))
    elif imfs.ndim == 1:
        imfs = imfs[None, :]
    return Decomp(
        x=x,
        imfs=imfs,
        residual=np.asarray(res.residual, dtype=np.float64),
        n_modes=int(res.n_modes),
        stats=dict(res.stats),
        recon=recon,
        wall_s=wall,
    )
