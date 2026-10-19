"""Tests for the time-frequency building blocks (plan Task 3)."""
import numpy as np
import pytest

from audio_eemd.spectra import (hilbert_if, higuchi_dim, if_stats, morlet_cwt,
                                sample_entropy, stft_block)


def test_stft_tone_peak():
    sr = 8000
    x = np.sin(2 * np.pi * 1000 * np.arange(sr) / sr)
    f, tt, Z = stft_block(x, sr, nperseg=1024)
    assert np.argmax(np.abs(Z).mean(axis=1)) == np.argmin(np.abs(f - 1000))
    assert len(tt) > 1


def test_stft_tone_peak_custom_overlap():
    sr = 8000
    x = np.sin(2 * np.pi * 500 * np.arange(sr) / sr)
    f, tt, Z = stft_block(x, sr, nperseg=512, noverlap=256)
    assert np.argmax(np.abs(Z).mean(axis=1)) == np.argmin(np.abs(f - 500))


def test_hilbert_if_sine():
    sr = 8000
    x = np.sin(2 * np.pi * 1000 * np.arange(sr) / sr)
    w, A = hilbert_if(x, sr)
    s = if_stats(w)
    assert abs(s["median"] - 1000) < 10
    assert np.isclose(A.std(), 0, atol=0.05)  # constant envelope
    assert s["n_pos_frac"] > 0.99


def test_hilbert_if_two_tone_beat():
    """100 + 105 Hz: the analytic phase is exactly 102.5·t, so IF is a
    constant 102.5 Hz; the 5 Hz beat shows up in the amplitude envelope
    (the classic mode-mixing case for EMD)."""
    sr = 8000
    t = np.arange(4 * sr) / sr
    x = (np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 105 * t)) / 2
    w, A = hilbert_if(x, sr)
    s = if_stats(w)
    assert abs(s["median"] - 102.5) < 1.0
    # the beat modulates the envelope at 5 Hz
    a = A - A.mean()
    F = np.abs(np.fft.rfft(a))
    freqs = np.fft.rfftfreq(len(a), 1.0 / sr)
    assert freqs[np.argmax(F[1:]) + 1] == pytest.approx(5.0, abs=1.0)


def test_morlet_chirp_ridge():
    sr = 8000
    t = np.arange(sr) / sr
    f = 100 + 900 * t / t[-1]  # 100 -> 1000 Hz linear chirp
    x = np.sin(2 * np.pi * f * t)
    freqs, C = morlet_cwt(x, sr, f_lo=50, n_scales=32)
    ridge = freqs[np.argmax(np.abs(C), axis=0)]
    assert ridge[0] < 300
    assert ridge[-1] > 700


def test_entropy_ordering():
    rng = np.random.default_rng(0)
    # 10 Hz @ 1000 Hz: period 100 samples >> k_max, so Higuchi sees clean scaling
    tone = np.sin(2 * np.pi * 10 * np.arange(4096) / 1000)
    noise = rng.standard_normal(4096)
    assert sample_entropy(tone) < sample_entropy(noise)
    # white noise SE is O(1); a smooth periodic signal is near 0
    assert sample_entropy(noise) > 1.0
    assert higuchi_dim(noise) > higuchi_dim(tone)
    # reference values: smooth curve ~1, 1-D white noise ~2
    assert 0.9 < higuchi_dim(tone) < 1.2
    assert 1.7 < higuchi_dim(noise) < 2.2
