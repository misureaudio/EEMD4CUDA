"""Numerical artifact builders (Tiers 0-1).

Tier 0: metadata.csv + report.json (metadata block).
Tier 1: per-mode table, quantization-floor annotation, imfs.npz, optional
per-IMF CSVs, full report.json.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__

# Static caveats (plan §3 / §8)
CAVEAT_NONORTHOGONAL = (
    "EMD/EEMD is not an orthogonal decomposition: per-mode energy fractions "
    "(||imf_k||^2/||x||^2) do not sum to 1 in general; the deviation is the "
    "cross-term energy (see decomposition.energy_sum)."
)
CAVEAT_SEED = (
    "EEMD is chaotic: results depend on the RNG seed. Same seed is "
    "bit-reproducible on one backend; different seeds give different mode "
    "depths; CPU vs GPU results agree to ~1e-8, not bit-exactly."
)
CAVEAT_GPU = (
    "GPU backend: natural-BC splines only, min_extrema=4 hardcoded, "
    "max_imf=-1 caps at 64 (RuntimeWarning + environment.max_imf_reached if "
    "the cap binds)."
)
CAVEAT_HILBERT = (
    "Instantaneous frequency of a noisy/complex IMF is ill-conditioned "
    "(negative/erratic values are expected); this is precisely what the "
    "ensemble averaging in EEMD is for. IF statistics use the positive part "
    "only."
)


def _dominant_freq(x: np.ndarray, sr: int) -> float | None:
    """Periodogram peak frequency (DC excluded); None if not estimable."""
    if len(x) < 16:
        return None
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    if X.sum() <= 0:
        return None
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    return float(f[np.argmax(X[1:]) + 1])


def _zerocross(x: np.ndarray) -> int:
    return int(np.count_nonzero(np.diff(np.signbit(x))))


def quantization_floor(peak: float, bit_depth: int | None) -> float | None:
    """Std of uniform quantization noise for a `bit_depth`-bit code at the
    given full scale (step = peak / 2^(b-1), noise std = step / (2*sqrt(3))).
    None for float formats (bit_depth None or >= 32).
    """
    if not bit_depth or bit_depth >= 32:
        return None
    step = peak / 2.0 ** (bit_depth - 1)
    return float(step / (2.0 * np.sqrt(3.0)))


def mode_table(imfs: np.ndarray, residual: np.ndarray, x: np.ndarray,
               sr: int, bit_depth: int | None = None
               ) -> tuple[list[dict], dict, dict]:
    """Per-IMF statistics table (plan §3 Tier 1).

    Returns (imf_rows, residual_row, quantization_info).
    """
    e_x = float(np.sum(x ** 2))
    qfloor = quantization_floor(float(np.abs(x).max()), bit_depth)

    def row(k, m: np.ndarray) -> dict:
        std = float(np.std(m))
        return {
            "k": k,
            "std": std,
            "energy_frac": (float(np.sum(m ** 2)) / e_x) if e_x > 0 else 0.0,
            "dominant_freq_hz": _dominant_freq(m, sr),
            "n_zerocross": _zerocross(m),
            "sifting": None,  # per-mode sifting not exposed by emd_par
            "quantization_level": bool(qfloor is not None and std < qfloor),
        }

    rows = [row(k, imfs[k]) for k in range(imfs.shape[0])]
    rrow = row("residual", residual)
    qinfo = {
        "bit_depth": bit_depth,
        "floor_std": qfloor,
        "n_flagged": int(sum(r["quantization_level"] for r in rows)),
    }
    return rows, rrow, qinfo


def build_report(meta: dict, segment: dict, parameters: dict, environment: dict,
                 n_modes: int, recon_error: float, mode_rows: list, resid_row: dict,
                 qinfo: dict, audio_stats: dict, timings: dict,
                 caveats: list) -> dict:
    """Assemble the full report.json structure (JSON-serializable)."""
    return {
        "metadata": meta,
        "segment": segment,
        "parameters": parameters,
        "environment": environment,
        "audio_stats": audio_stats,
        "decomposition": {
            "n_modes": n_modes,
            "recon_error": recon_error,
            "imf": mode_rows,
            "residual": resid_row,
            "energy_sum": (sum(r["energy_frac"] for r in mode_rows)
                           + resid_row["energy_frac"]),
            "quantization": qinfo,
        },
        "timings": timings,
        "caveats": caveats,
        "generated": datetime.now(timezone.utc).isoformat(),
        "tool_version": __version__,
    }


# ---------------------------------------------------------------- writers --

def _write_json(outdir: Path, report: dict) -> Path:
    p = outdir / "report.json"
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return p


def _write_metadata_csv(outdir: Path, report: dict) -> Path:
    rows: list[tuple[str, str, str]] = []
    rows += [("file", k, str(v)) for k, v in report["metadata"].items()]
    rows += [("segment", k, str(v)) for k, v in report["segment"].items()]
    if "audio_stats" in report:
        rows += [("audio_stats", k, "" if v is None else str(v))
                 for k, v in report["audio_stats"].items()]
    p = outdir / "metadata.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "key", "value"])
        w.writerows(rows)
    return p


def write_metadata(outdir: str | Path, report: dict) -> list[Path]:
    """Tier-0 writer: report.json + metadata.csv for a get_metadata() result."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    full = {
        "metadata": report["metadata"],
        "segment": report["segment"],
        "audio_stats": report["audio_stats"],
        "generated": datetime.now(timezone.utc).isoformat(),
        "tool_version": __version__,
    }
    return [_write_json(outdir, full), _write_metadata_csv(outdir, full)]


def write_imfs_npz(outdir: Path, x: np.ndarray, t: np.ndarray, imfs: np.ndarray,
                   residual: np.ndarray, n_modes: int, stats: dict) -> Path:
    """imfs.npz: x, t, imfs [n_modes, N], residual, n_modes, scalar stats."""
    payload: dict = {
        "x": x, "t": t, "imfs": imfs, "residual": residual,
        "n_modes": np.int64(n_modes),
    }
    for k, v in stats.items():
        if k == "x":
            continue  # emd_par retains the input in stats; stored as 'x'
        if isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_)):
            payload[f"stat_{k}"] = np.asarray(v)
        elif isinstance(v, (list, tuple)) and all(
                isinstance(i, (int, np.integer)) for i in v):
            payload[f"stat_{k}"] = np.asarray(v, dtype=np.int64)
    p = outdir / "imfs.npz"
    np.savez_compressed(p, **payload)
    return p


def write_stft_npz(outdir: Path, stft: tuple) -> Path:
    """stft.npz: freqs, times, Zxx_orig [F, T], Zxx_imf [k, F, T]."""
    f, t, Zxx_orig, Zxx_imf = stft
    p = outdir / "stft.npz"
    np.savez_compressed(p, freqs=f, times=t, Zxx_orig=Zxx_orig,
                        Zxx_imf=Zxx_imf)
    return p


def write_hilbert_npz(outdir: Path, hilb: list) -> Path:
    """hilbert.npz: per-k inst_freq_k, inst_amp_k [N]."""
    payload: dict = {}
    for k, (w, A) in enumerate(hilb):
        payload[f"inst_freq_{k}"] = w
        payload[f"inst_amp_{k}"] = A
    p = outdir / "hilbert.npz"
    np.savez_compressed(p, **payload)
    return p


def write_mode_csvs(outdir: Path, t: np.ndarray, imfs: np.ndarray,
                    residual: np.ndarray) -> list[Path]:
    """Per-IMF CSV dumps (time_s, value) + residual.csv (Tier 4 --csv)."""
    paths = []
    for k in range(imfs.shape[0]):
        p = outdir / f"imf_{k}.csv"
        np.savetxt(p, np.column_stack([t, imfs[k]]), delimiter=",",
                   header="time_s,value", comments="")
        paths.append(p)
    p = outdir / "residual.csv"
    np.savetxt(p, np.column_stack([t, residual]), delimiter=",",
               header="time_s,value", comments="")
    paths.append(p)
    return paths


def write_artifacts(res, outdir: str | Path) -> list[Path]:
    """Write all Tier-1/2 artifacts for an AnalysisResult (duck-typed).

    Files: report.json, metadata.csv, imfs.npz, stft.npz, hilbert.npz, and
    (if parameters['csv']) per-IMF CSVs. Figures are written by the caller
    (plots.render_figures).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    files = [_write_json(outdir, res.report), _write_metadata_csv(outdir, res.report),
             write_imfs_npz(outdir, res.x, res.t, res.imfs, res.residual,
                            res.n_modes, res.environment.get("stats", {}))]
    if res.stft is not None:
        files.append(write_stft_npz(outdir, res.stft))
    if res.hilbert is not None:
        files.append(write_hilbert_npz(outdir, res.hilbert))
    if res.parameters.get("csv"):
        files += write_mode_csvs(outdir, res.t, res.imfs, res.residual)
    return files
