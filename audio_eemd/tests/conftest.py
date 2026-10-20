import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Deterministic test backend: tests render to files, no GUI needed. Set before
# any test imports matplotlib (env var is read at first pyplot import).
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FLAC = str(ROOT / "PrimeToneNewPhi[Nch 1][Frac 003][00043-16381][Nfr 027]-[44100-16]-[1800s]-[-0.25dB]-[ 8.4809]-[RD].flac")


@pytest.fixture(scope="session")
def flac_path():
    return FLAC


@pytest.fixture(scope="session")
def clip1():
    """1 s at t=10: (mono float64 [44100], sr)."""
    import soundfile as sf
    x, sr = sf.read(FLAC, start=10 * sr, stop=11 * sr,
                    dtype="float64", always_2d=True)
    return x[:, 0], sr


@pytest.fixture(scope="session")
def clip5():
    """5 s at t=10: (mono float64 [220500], sr)."""
    import soundfile as sf
    x, sr = sf.read(FLAC, start=10 * sr, stop=15 * sr,
                    dtype="float64", always_2d=True)
    return x[:, 0], sr


@pytest.fixture
def two_tones():
    """Mode-mixing reference: 100 Hz + 105 Hz, 4 s @ 8 kHz."""
    sr = 8000
    t = np.arange(4 * sr) / sr
    return (np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 105 * t)) / 2, sr
