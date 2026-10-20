"""Public API surface (plan §4.1).

This is the module an MCP server or a Jupyter notebook binds to:

    import audio_eemd as ae
    res = ae.analyze(path="song.flac", limits=[10, 15], unit="s", E=64)
    res.to_dict()            # JSON-serializable, no raw arrays
    res.write_artifacts()    # report.json, metadata.csv, imfs.npz, ...

Contract (plan §4.2): JSON-friendly arguments, JSON-friendly to_dict(),
typed exceptions with stable machine-readable codes, no implicit state
(explicit seed, explicit outdir, idempotent artifact writes), no cwd
dependence, lazy heavy imports (no matplotlib/tftb at import time).

Tier 1: load → segment → EEMD → per-mode table → artifacts.
Tier 2: STFT + per-IMF Hilbert IF (always, when the segment is long enough).
Tier 3: `wavelet=True` → Morlet CWT scalogram; `tftb=True` → gated tftb
    panel (TFTBError with exit-code-5 semantics if gate G1 fails).
Tier 4: entropy/fractal — still NotImplementedError.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio_io import (AudioError, AudioFile, SegmentError, audio_stats,
                       bit_depth, load_audio, resolve_segment, select_segment)
from .eemd_stage import EEMDError, run_eemd
from .report import (CAVEAT_GPU, CAVEAT_HILBERT, CAVEAT_NONORTHOGONAL,
                     CAVEAT_SEED, CAVEAT_TFTB, build_report, mode_table)
from .spectra import (TFTBError, check_tftb_gate, hilbert_if, if_stats,
                      morlet_cwt, stft_block, tftb_panel)

__all__ = [
    "analyze", "AnalysisResult", "TFTBError",
    "AudioError", "SegmentError", "EEMDError",
    "load_audio", "select_segment", "get_metadata", "audio_stats",
]


def _arr_summary(a: np.ndarray) -> dict:
    a = np.asarray(a)
    return {"shape": list(a.shape), "dtype": str(a.dtype),
            "mean": float(a.mean()), "std": float(a.std())}


def _py(v):
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(v, np.ndarray):
        return [_py(i) for i in v.tolist()]
    if isinstance(v, (dict,)):
        return {k: _py(i) for k, i in v.items()}
    if isinstance(v, (list, tuple)):
        return [_py(i) for i in v]
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


@dataclass
class AnalysisResult:
    """All products of one analyze() call (plan §4.1)."""

    meta: dict
    segment: dict
    parameters: dict
    environment: dict
    x: np.ndarray                      # segment, float64 [N]
    t: np.ndarray                      # absolute time axis, float64 [N]
    imfs: np.ndarray                   # float64 [n_modes, N]
    residual: np.ndarray               # float64 [N]
    n_modes: int
    mode_table: list                   # per-IMF rows (plan §3 Tier 1)
    recon_error: float
    residual_row: dict = None
    quantization: dict = None
    stft: tuple | None = None          # Tier 2: (f, t, S)
    hilbert: list | None = None        # Tier 2: per-IMF (if, iq)
    wavelet: tuple | None = None       # Tier 3: (cfs, freqs)
    tftb_panel: dict | None = None     # Tier 3
    entropy: list | None = None        # Tier 4
    fractal: list | None = None        # Tier 4
    timings: dict = field(default_factory=dict)
    caveats: list = field(default_factory=list)
    outdir: Path | None = None
    report: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-serializable view; raw arrays are summarized, never embedded."""
        return {
            "metadata": self.meta,
            "segment": self.segment,
            "parameters": self.parameters,
            "environment": self.environment,
            "audio_stats": self.report.get("audio_stats"),
            "decomposition": {
                "n_modes": self.n_modes,
                "recon_error": self.recon_error,
                "imf": self.mode_table,
                "residual": self.residual_row,
                "energy_sum": self.report.get("decomposition", {}).get("energy_sum"),
                "quantization": self.quantization,
            },
            "arrays": {
                "x": _arr_summary(self.x),
                "imfs": _arr_summary(self.imfs),
                "residual": _arr_summary(self.residual),
            },
            "wavelet": self.report.get("wavelet"),
            "tftb": self.report.get("tftb"),
            "timings": self.timings,
            "caveats": self.caveats,
            "outdir": str(self.outdir) if self.outdir is not None else None,
        }

    def write_artifacts(self, outdir: str | Path | None = None) -> list[Path]:
        """Write report.json, metadata.csv, imfs.npz (+ per-IMF CSVs if
        parameters['csv']). Figures arrive in Tier 2."""
        out = Path(outdir) if outdir is not None else self.outdir
        if out is None:
            raise ValueError(
                "write_artifacts: no outdir given and none stored on the result "
                "(pass outdir= to analyze() or to this call)")
        from .report import write_artifacts
        return write_artifacts(self, out)


def analyze(path: str | Path | None = None,
            x: np.ndarray | None = None, sr: int | None = None,
            limits: list | tuple | None = None, unit: str = "s",
            method: str = "auto", E: int = 64, eps: float = 0.2,
            tau: float = 0.25, max_imf: int = -1, max_sifts: int = 50,
            seed: int = 0, batch: int | None = None,
            nperseg: int = 4096, noverlap: int | None = None,
            wavelet: bool = False, tftb: bool = False,
            csv: bool = False, entropy: bool = False, fractal: bool = False,
            outdir: str | Path | None = None,
            fmt: str = "png", dpi: int = 150,
            render: bool = True) -> AnalysisResult:
    """Full analysis pipeline (plan §4.1). Tier 1: EEMD core only.

    Exactly one of `path` (FLAC/WAV file) or `x` (1-D float array, with `sr`)
    must be given. `limits` = [t0, t1] in `unit` ("s" or "samples"); None →
    whole file if <= 30 s, else the first 10 s (segment["defaulted"] = True).
    """
    if (path is None) == (x is None):
        raise ValueError("analyze: provide exactly one of path= or x= (with sr=)")
    if entropy or fractal:
        raise NotImplementedError("entropy/fractal are Tier-4 features")

    # ---- load / segment ------------------------------------------------
    timings: dict = {}
    t_start = time.perf_counter()
    t0 = time.perf_counter()
    if path is not None:
        a = load_audio(path)
        bitd = bit_depth(a.subtype)
        meta = {
            "path": str(path),
            "format": a.fmt,
            "subtype": a.subtype,
            "bit_depth": bitd,
            "samplerate": a.sr,
            "channels": a.channels,
            "frames": a.frames,
            "duration_s": a.duration,
        }
    else:
        if sr is None:
            raise ValueError("analyze: sr= is required when x= is given")
        xarr = np.asarray(x, dtype=np.float64).ravel()
        a = AudioFile(path=Path("<array>"), sr=int(sr), channels=1,
                      frames=len(xarr), duration=len(xarr) / float(sr),
                      fmt="array", subtype="FLOAT64",
                      x=xarr.reshape(-1, 1), meta={"source": "array"})
        bitd = None
        meta = {"path": "<array>", "format": "array", "subtype": "FLOAT64",
                "bit_depth": None, "samplerate": int(sr), "channels": 1,
                "frames": len(xarr), "duration_s": len(xarr) / float(sr)}
    xseg, limits, unit, defaulted = resolve_segment(a, limits, unit)
    stats = audio_stats(xseg, a.sr)
    timings["load_s"] = round(time.perf_counter() - t0, 3)

    if unit == "samples":
        s0, s1 = int(limits[0]), int(limits[1])
    else:
        s0, s1 = int(round(limits[0] * a.sr)), int(round(limits[1] * a.sr))
    segment = {
        "start_s": s0 / a.sr, "stop_s": s1 / a.sr, "n_samples": len(xseg),
        "sr": a.sr, "unit_used": unit, "defaulted": defaulted,
    }
    t = (np.arange(len(xseg)) + s0) / a.sr

    # ---- EEMD -----------------------------------------------------------
    decomp = run_eemd(xseg, method=method, E=E, eps=eps, tau=tau,
                      max_imf=max_imf, max_sifts=max_sifts, seed=seed,
                      batch=batch)
    timings["eemd_s"] = round(decomp.wall_s, 3)

    # ---- per-mode table + caveats ---------------------------------------
    rows, rrow, qinfo = mode_table(decomp.imfs, decomp.residual, xseg, a.sr, bitd)
    per_mode = decomp.stats.get("per_mode_sifts")
    if per_mode is not None:  # E=1 CPU exposes per-mode sifting counts
        for k in range(min(len(per_mode), len(rows))):
            rows[k]["sifting"] = int(per_mode[k])

    caveats = [CAVEAT_NONORTHOGONAL, CAVEAT_SEED]

    # ---- Tier 2: STFT + Hilbert IF --------------------------------------
    stft = None
    hilb = None
    nperseg_eff = min(nperseg, len(xseg))
    if len(xseg) >= 16 and nperseg_eff >= 16:
        t0 = time.perf_counter()
        f, t_stft, Zxx_orig = stft_block(xseg, a.sr, nperseg=nperseg_eff,
                                         noverlap=noverlap)
        Zxx_imf = (np.stack([stft_block(m, a.sr, nperseg=nperseg_eff,
                                        noverlap=noverlap)[2]
                             for m in decomp.imfs])
                   if decomp.n_modes else np.empty((0,) + Zxx_orig.shape))
        stft = (f, t_stft, Zxx_orig, Zxx_imf)
        timings["stft_s"] = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        hilb = [hilbert_if(m, a.sr) for m in decomp.imfs]
        for k, (w, _A) in enumerate(hilb):
            s = if_stats(w)
            rows[k]["if_median_hz"] = s["median"]
            rows[k]["if_iqr_hz"] = s["iqr"]
            rows[k]["if_n_pos_frac"] = s["n_pos_frac"]
        timings["hilbert_s"] = round(time.perf_counter() - t0, 3)
        caveats.append(CAVEAT_HILBERT)
    else:
        caveats.append("segment too short for STFT/Hilbert (need ≥ 16 samples); "
                       "stft/hilbert outputs are null")
    # ---- Tier 3: wavelet (Morlet CWT) + tftb panel (gated) --------------
    wlet = None
    if wavelet and len(xseg) >= 64:
        t0 = time.perf_counter()
        wlet = morlet_cwt(xseg, a.sr)
        timings["wavelet_s"] = round(time.perf_counter() - t0, 3)
    elif wavelet:
        caveats.append("segment too short for Morlet CWT (< 64 samples); "
                       "wavelet output is null")

    tpanel = None
    if tftb and len(xseg) >= 256:
        t0 = time.perf_counter()
        tpanel = tftb_panel(xseg, a.sr)  # raises TFTBError if gate G1 fails
        timings["tftb_s"] = round(time.perf_counter() - t0, 3)
        caveats.append(CAVEAT_TFTB)
    elif tftb:
        caveats.append("segment too short for the tftb panel (< 256 samples); "
                       "tftb output is null")
    environment = {
        "backend": decomp.stats.get("method", method),
        "E": decomp.stats.get("E", E),
        "n_modes": decomp.n_modes,
        "total_sifts": decomp.stats.get("total_sifts"),
        "per_trial_total_sifts": decomp.stats.get("per_trial_total_sifts"),
        "n_workers": decomp.stats.get("n_workers"),
        "gpu_time_s": decomp.stats.get("gpu_time_s"),
        "max_imf_reached": bool(decomp.stats.get("max_imf_reached", False)),
        "wall_s": round(decomp.wall_s, 3),
        # verbatim backend stats, JSON-sanitized; 'x' (the input) is dropped
        "stats": {k: v for k, v in _py(decomp.stats).items() if k != "x"},
    }
    environment = _py(environment)
    if environment["backend"] == "gpu":
        caveats.append(CAVEAT_GPU)
    if qinfo["n_flagged"]:
        caveats.append(
            f"{qinfo['n_flagged']} IMF(s) sit below the {bitd}-bit quantization "
            f"noise floor (std < {qinfo['floor_std']:.3e}); read them as "
            "quantization, not content.")

    parameters = {
        "method": method, "E": E, "eps": eps, "tau": tau, "max_imf": max_imf,
        "max_sifts": max_sifts, "seed": seed, "batch": batch,
        "nperseg": nperseg, "noverlap": noverlap, "csv": csv,
        "wavelet": wavelet, "tftb": tftb, "entropy": entropy,
        "fractal": fractal,
        "fmt": fmt, "dpi": dpi,
    }
    timings["total_s"] = round(time.perf_counter() - t_start, 3)
    report = build_report(meta, segment, parameters, environment,
                          decomp.n_modes, decomp.recon, rows, rrow, qinfo,
                          stats, timings, caveats)

    # ---- Tier 3 report blocks (JSON-friendly, no raw arrays) ------------
    if wlet is not None:
        wfreqs, wC = wlet
        report["wavelet"] = {
            "n_scales": int(len(wfreqs)),
            "f_lo_hz": float(wfreqs[0]),
            "f_hi_hz": float(wfreqs[-1]),
            "cfs_shape": list(np.asarray(wC).shape),
            "cfs_dtype": str(np.asarray(wC).dtype),
        }
    if tpanel is not None:
        report["tftb"] = {
            "version": tpanel["version"],
            "n_fbins": int(tpanel["n_fbins"]),
            "time_hop": int(tpanel["time_hop"]),
            "gate": check_tftb_gate(),
            "panels": {
                name: {
                    "shape": list(np.asarray(tpanel[name]).shape),
                    "min": float(np.min(tpanel[name])),
                    "max": float(np.max(tpanel[name])),
                    "f_hi_hz": float(tpanel[f"f_{tag}"][-1]),
                }
                for name, tag in (("spectrogram", "stft"),
                                  ("spwvd", "spwvd"),
                                  ("reassigned", "reass"))
            },
        }

    res = AnalysisResult(
        meta=meta, segment=segment, parameters=parameters,
        environment=environment, x=xseg, t=t,
        imfs=decomp.imfs, residual=decomp.residual,
        n_modes=decomp.n_modes, mode_table=rows, recon_error=decomp.recon,
        residual_row=rrow, quantization=qinfo,
        stft=stft, hilbert=hilb,
        wavelet=wlet, tftb_panel=tpanel,
        timings=timings, caveats=caveats,
        outdir=Path(outdir) if outdir is not None else None,
        report=report,
    )

    if render and res.outdir is not None:
        from .plots import render_figures
        render_figures(res, res.outdir, fmt=fmt)
    return res
