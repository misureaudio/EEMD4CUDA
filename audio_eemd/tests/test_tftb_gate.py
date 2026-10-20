"""Gate G1 tests for the tftb dev0 build (plan Task 9).

If either gate check fails on a given build, the tftb module is dropped
(flag stays, exit 5) — these tests are the acceptance gate.
"""
import numpy as np
import pytest
from scipy import signal as ss

from audio_eemd.spectra import (TFTBError, check_tftb_gate, tftb_holder,
                                tftb_panel)


def test_tftb_stft_matches_scipy(two_tones):
    """tftb spectrogram vs scipy STFT: amplitude correlation > 0.99 on the
    common grid (tftb returns |DFT|², so compare |Z| vs sqrt(|S|))."""
    from tftb.processing.reassigned import spectrogram
    from tftb.processing.utils import tftb_window

    x, sr = two_tones
    win = tftb_window(255, "Hanning")
    f, tt, Z = ss.stft(x, fs=sr, window=win, nperseg=255, noverlap=127,
                       boundary="zeros", padded=False)
    ts = (tt * sr).round().astype(int)
    ts = ts[(ts >= 0) & (ts < len(x))]
    St, _, _ = spectrogram(x, time_samples=ts, n_fbins=256, window=win)
    Spos = np.sqrt(np.abs(St[:129, :]))
    ft = np.arange(129) * sr / 256
    a = np.abs(Z[:, :len(ts)])
    b = np.array([np.interp(f, ft, Spos[:, j]) for j in range(Spos.shape[1])]).T
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    assert corr > 0.99, corr


def test_tftb_spwvd_runs(two_tones):
    from tftb.processing.reassigned import smoothed_pseudo_wigner_ville

    x, sr = two_tones
    ts = np.arange(0, len(x), 32)
    T = smoothed_pseudo_wigner_ville(x, timestamps=ts, n_fbins=256)[0]
    assert np.isfinite(T).all()
    assert T.shape[0] == 256 and T.shape[1] == len(ts)


def test_gate_passes_and_caches(two_tones):
    g = check_tftb_gate()
    assert g["stft_corr"] > 0.99
    assert g["version"]
    assert g["spwvd_shape"][0] == 256
    assert check_tftb_gate() is g  # cached, not re-run


def test_gate_failure_raises_typed():
    import audio_eemd.spectra as sp

    saved = sp._TFTB_GATE
    sp._TFTB_GATE = None

    def _boom():
        raise TFTBError("simulated", code="tftb.import_failed")

    saved_check = sp.check_tftb_gate
    sp.check_tftb_gate = _boom
    try:
        with pytest.raises(TFTBError) as ei:
            tftb_panel(np.ones(512), 8000)
        assert ei.value.code == "tftb.import_failed"
    finally:
        sp.check_tftb_gate = saved_check
        sp._TFTB_GATE = saved


def test_tftb_panel_shapes(two_tones):
    x, sr = two_tones
    p = tftb_panel(x, sr, n_fbins=128, time_hop=64)
    tcol = len(np.arange(0, len(x), 64))
    nf = 128 // 2 + 1
    for name in ("spectrogram", "spwvd", "reassigned"):
        assert p[name].shape == (tcol, nf)
        assert np.isfinite(p[name]).all()
    assert p["f_stft"][-1] == pytest.approx(sr / 2)
    assert len(p["t_stft"]) == tcol
    # spectrogram (|DFT|²) is non-negative; SPWVD (Cohen class) is not
    assert p["spectrogram"].min() >= 0
    assert p["spwvd"].min() < 0


def test_tftb_holder(two_tones):
    x, sr = two_tones
    h = tftb_holder(x, sr)
    assert h is None or np.isfinite(h)
