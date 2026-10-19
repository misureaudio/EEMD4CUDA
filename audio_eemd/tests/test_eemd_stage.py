"""Tests for the EEMD stage (plan Task 2)."""
import numpy as np
import pytest

from audio_eemd.eemd_stage import EEMDError, run_eemd


def test_emd_single_cpu(two_tones):
    x, sr = two_tones
    d = run_eemd(x, method="cpu", E=1, seed=0)
    assert d.n_modes >= 1
    assert d.imfs.shape == (d.n_modes, len(x))
    assert d.residual.shape == (len(x),)
    assert d.recon < 1e-6 * np.abs(x).max()
    assert d.stats["method"] == "cpu"
    assert d.stats["E"] == 1


def test_eemd_ensemble_cpu(two_tones):
    x, sr = two_tones
    d = run_eemd(x, method="cpu", E=8, seed=1)
    assert d.n_modes >= 1
    assert d.imfs.shape[0] == d.n_modes
    assert d.imfs.shape[1] == len(x)
    assert d.recon < 1e-6 * np.abs(x).max()
    assert "per_trial_total_sifts" in d.stats
    assert len(d.stats["per_trial_total_sifts"]) == 8


def test_seed_determinism(two_tones):
    x, sr = two_tones
    a = run_eemd(x, method="cpu", E=4, seed=42)
    b = run_eemd(x, method="cpu", E=4, seed=42)
    assert a.n_modes == b.n_modes
    assert np.array_equal(a.imfs, b.imfs)
    assert np.array_equal(a.residual, b.residual)


def test_two_tone_mode_mixing(two_tones):
    """EEMD on the 100/105 Hz pair (5 Hz separation — the classic
    mode-mixing case): spurious modes are expected, but the carrier must
    stay in-band and the total out-of-band (artifact) energy must stay
    small.
    """
    x, sr = two_tones
    d = run_eemd(x, method="cpu", E=8, seed=1)
    assert d.n_modes >= 2

    from audio_eemd.report import _dominant_freq
    from audio_eemd.spectra import hilbert_if
    dom = [_dominant_freq(imf, sr) for imf in d.imfs]
    in_band = np.array([f is not None and 95.0 <= f <= 110.0 for f in dom])
    e = np.sum(d.imfs ** 2, axis=1) / max(np.sum(x ** 2), 1e-300)

    # 1) carrier energy concentrated in an in-band IMF
    assert e[in_band].max() >= 0.8, e
    # 2) that IMF's instantaneous frequency stays in-band
    i = int(np.argmax(np.where(in_band, e, 0.0)))
    if0, _ = hilbert_if(d.imfs[i], sr)
    assert 95.0 <= float(np.median(if0)) <= 110.0, float(np.median(if0))
    # 3) artifact budget: out-of-band IMFs + residual carry ≤ 15%
    e_res = float(np.sum(d.residual ** 2) / np.sum(x ** 2))
    assert e[~in_band].sum() + e_res <= 0.15, (e[~in_band].sum(), e_res)


def test_bad_input_raises():
    with pytest.raises(EEMDError) as ei:
        run_eemd(np.ones(5), method="cpu", E=1)
    assert ei.value.code == "eemd.bad_input"
