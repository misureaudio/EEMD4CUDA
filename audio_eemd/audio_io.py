"""Audio I/O (Tier 0): FLAC/WAV loading, metadata, segmentation, audio stats.

Design rules (plan §4.2): JSON-friendly contracts, typed errors with stable
codes, no cwd dependence, no hidden state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf


class AudioError(Exception):
    """Unreadable or unsupported audio file."""

    def __init__(self, message: str, code: str = "audio.unreadable"):
        super().__init__(message)
        self.code = code


class SegmentError(Exception):
    """Bad segment limits or unit."""

    def __init__(self, message: str, code: str = "segment.bad_limits"):
        super().__init__(message)
        self.code = code


# Whole-file policy (plan §2 / open question 1): files up to max_file_sec are
# analyzed whole; longer files default to the first default_segment_sec.
DEFAULT_MAX_FILE_SEC = 30.0
DEFAULT_SEGMENT_SEC = 10.0


@dataclass
class AudioFile:
    """A decoded audio file: container metadata + full float64 samples."""

    path: Path
    sr: int
    channels: int
    frames: int
    duration: float
    fmt: str
    subtype: str
    x: np.ndarray  # float64, shape (N, C) — soundfile convention, original scale
    meta: dict = field(default_factory=dict)

    def to_mono(self) -> np.ndarray:
        """Mean over channels (v1 downmix policy; documented in README)."""
        if self.channels == 1:
            return self.x[:, 0]
        return self.x.mean(axis=1)


def bit_depth(subtype: str) -> int:
    """Best-effort bit depth from a soundfile subtype string."""
    s = subtype.upper()
    for key in ("DOUBLE", "FLOAT", "PCM_16", "PCM_S16", "PCM_24", "PCM_32",
                "PCM_S8", "ULAW", "ALAW"):
        if s.startswith(key):
            return {"DOUBLE": 64, "FLOAT": 32, "PCM_16": 16, "PCM_S16": 16,
                    "PCM_24": 24, "PCM_32": 32, "PCM_S8": 8,
                    "ULAW": 8, "ALAW": 8}[key]
    return 0


def load_audio(path: str | Path) -> AudioFile:
    """Read a FLAC/WAV (or any soundfile-supported) file fully into memory.

    Raises AudioError on any read failure.
    """
    p = Path(path)
    try:
        info = sf.info(p)
        x, sr = sf.read(p, dtype="float64", always_2d=True)
    except Exception as e:  # soundfile raises many types (LibsndfileError, ...)
        raise AudioError(f"cannot read audio file {p}: {e}") from e
    if x.ndim != 2 or x.shape[0] == 0:
        raise AudioError(f"unexpected sample array shape {x.shape} from {p}")
    meta = {
        "format": info.format,
        "subtype": info.subtype,
        "samplerate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_s": float(info.duration),
    }
    return AudioFile(
        path=p, sr=sr, channels=info.channels, frames=info.frames,
        duration=info.duration, fmt=info.format, subtype=info.subtype,
        x=x, meta=meta,
    )


def select_segment(a: AudioFile, limits: Sequence[float], unit: str = "s") -> np.ndarray:
    """Return the mono float64 segment [limits[0], limits[1]) (stop exclusive).

    unit="s"       : limits are seconds (floats allowed)
    unit="samples" : limits are sample indices (ints)
    """
    mono = a.to_mono()
    n = len(mono)
    if unit == "s":
        s0 = int(round(limits[0] * a.sr))
        s1 = int(round(limits[1] * a.sr))
    elif unit == "samples":
        s0, s1 = int(limits[0]), int(limits[1])
    else:
        raise SegmentError(
            f"unit must be 's' or 'samples', got {unit!r}", "segment.bad_unit")
    if s0 < 0 or s1 <= s0 or s1 > n:
        raise SegmentError(
            f"limits [{limits[0]}, {limits[1]}] (unit={unit!r}) out of range "
            f"for {n} samples (stop exclusive)", "segment.out_of_range")
    return mono[s0:s1].copy()


def audio_stats(x: np.ndarray, sr: int) -> dict:
    """Audio statistics of a segment.

    Keys: peak, rms, crest (peak/rms), lufs (integrated, pyloudnorm; None on
    failure), centroid (spectral centroid, Hz), flatness (spectral flatness,
    0..1). LUFS is approximate for clips well under a few seconds.
    """
    peak = float(np.abs(x).max())
    rms = float(np.sqrt(np.mean(x ** 2)))
    crest = (peak / rms) if rms > 0 else float("inf")

    lufs = None
    try:
        from pyloudnorm import Meter
        lufs = float(Meter(sr).integrated_loudness(x))
    except Exception:
        pass

    win = np.hanning(len(x))
    X = np.abs(np.fft.rfft(x * win))
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    xsum = float(X.sum())
    centroid = float(np.sum(f * X) / xsum) if xsum > 0 else 0.0
    geo = float(np.exp(np.mean(np.log(X + 1e-20))))
    flatness = float(geo / (xsum / len(X) + 1e-20)) if xsum > 0 else 0.0

    return {
        "peak": peak,
        "rms": rms,
        "crest": crest,
        "lufs": lufs,
        "centroid": centroid,
        "flatness": flatness,
    }


def resolve_segment(a: AudioFile, limits: Sequence[float] | None = None,
                    unit: str = "s",
                    max_file_sec: float = DEFAULT_MAX_FILE_SEC
                    ) -> tuple[np.ndarray, Sequence[float], str, bool]:
    """Apply the segment policy and return (xseg, limits, unit, defaulted).

    limits=None: whole file if duration <= max_file_sec, else the first
    DEFAULT_SEGMENT_SEC (defaulted=True).
    """
    defaulted = False
    if limits is None:
        if a.duration <= max_file_sec:
            limits, unit = [0.0, a.duration], "s"
        else:
            limits, unit = [0.0, DEFAULT_SEGMENT_SEC], "s"
            defaulted = True
    return select_segment(a, limits, unit), limits, unit, defaulted


def get_metadata(path: str | Path, limits: Sequence[float] | None = None,
                 unit: str = "s",
                 max_file_sec: float = DEFAULT_MAX_FILE_SEC) -> dict:
    """Tier-0 analysis: container metadata + audio stats of a segment.

    limits=None: whole file if duration <= max_file_sec, else the first
    DEFAULT_SEGMENT_SEC (segment["defaulted"] = True).
    Returns a JSON-serializable dict with keys: metadata, segment, audio_stats.
    """
    a = load_audio(path)
    x, limits, unit, defaulted = resolve_segment(a, limits, unit, max_file_sec)
    stats = audio_stats(x, a.sr)
    if unit == "s":
        s0 = int(round(limits[0] * a.sr))
        s1 = int(round(limits[1] * a.sr))
    else:
        s0, s1 = int(limits[0]), int(limits[1])
    return {
        "metadata": {
            "path": str(a.path),
            "format": a.fmt,
            "subtype": a.subtype,
            "bit_depth": bit_depth(a.subtype),
            "samplerate": a.sr,
            "channels": a.channels,
            "frames": a.frames,
            "duration_s": a.duration,
        },
        "segment": {
            "start_s": s0 / a.sr,
            "stop_s": s1 / a.sr,
            "n_samples": int(len(x)),
            "sr": a.sr,
            "unit_used": unit,
            "defaulted": defaulted,
        },
        "audio_stats": stats,
    }
