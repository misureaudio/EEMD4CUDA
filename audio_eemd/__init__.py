"""audio_eemd — EEMD audio analysis (API-first; the CLI is a thin adapter).

Tier 0: metadata + audio statistics (audio_io).
Tier 1: EEMD core (eemd_stage) + public API (analyze / AnalysisResult)
  + per-mode table + imfs.npz artifacts (report).
Tier 2: STFT + per-IMF Hilbert IF, Morlet CWT, 6 core figures
  (plots), full artifact bundle (report.write_artifacts).
Tier 3 (current): flag-gated `--wavelet` (Morlet scalogram, fig 07) and
  `--tftb` (gated tftb panel, fig 08; gate G1 → TFTBError / exit 5).
Later tiers (per .hermes/plans/2026-10-19_093212-audio-eemd-analysis.md):
  Tier 4 — CSV extras / entropy / fractal.

`import audio_eemd` stays light: no matplotlib, no cupy, no tftb at
import time (all imported lazily inside functions).
"""
__version__ = "0.4.0"

from .audio_io import (
    AudioError,
    AudioFile,
    SegmentError,
    audio_stats,
    get_metadata,
    load_audio,
    select_segment,
)
from .eemd_stage import EEMDError
from .api import AnalysisResult, TFTBError, analyze
from .report import write_artifacts
from .plots import FigCtx, render

__all__ = [
    "AudioError",
    "AudioFile",
    "SegmentError",
    "audio_stats",
    "get_metadata",
    "load_audio",
    "select_segment",
    "EEMDError",
    "AnalysisResult",
    "TFTBError",
    "analyze",
    "write_artifacts",
    "FigCtx",
    "render",
    "__version__",
]
