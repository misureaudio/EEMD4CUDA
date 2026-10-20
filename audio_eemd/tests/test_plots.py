"""Tests for the figure layer (plan Task 5, Tier 2 scope)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audio_eemd.eemd_stage import run_eemd
from audio_eemd.plots import FigCtx, render
from audio_eemd.spectra import hilbert_if, stft_block


def _ctx(two_tones, tmp_path):
    x, sr = two_tones
    d = run_eemd(x, method="cpu", E=4, seed=0)
    f, tt, Zorig = stft_block(x, sr, 1024)
    Zim = np.stack([stft_block(m, sr, 1024)[2] for m in d.imfs])
    return FigCtx(
        x=x, t=np.arange(len(x)) / sr, sr=sr, imfs=d.imfs,
        residual=d.residual, stft=(f, tt, Zorig, Zim),
        hilb=[hilbert_if(m, sr) for m in d.imfs],
        wavelet=None, tftb=None,
        title="test 0-4s", outdir=tmp_path, fmt="png",
    )


def test_core_figures(two_tones, tmp_path):
    ctx = _ctx(two_tones, tmp_path)
    paths = render(ctx, wavelet=False, tftb=False)
    names = {p.name for p in paths}
    assert {"fig_01_signal.png", "fig_02_imfs.png", "fig_03_stft_original.png",
            "fig_04_stft_imfs.png", "fig_05_hilbert.png",
            "fig_06_energy.png"} <= names
    for p in paths:
        assert p.stat().st_size > 10_000  # non-trivial content


def test_render_show_true_writes_nothing(two_tones, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    ctx = _ctx(two_tones, tmp_path)
    paths = render(ctx, show=True)
    assert paths == []
    assert list(tmp_path.glob("*.png")) == []


def test_render_without_outdir_raises(two_tones, tmp_path):
    ctx = _ctx(two_tones, tmp_path)
    ctx.outdir = None
    try:
        render(ctx)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_tier3_figures(two_tones, tmp_path):
    """fig_07 (Morlet CWT) and fig_08 (tftb panel) render when the data
    is present and are skipped otherwise (test_core_figures)."""
    from audio_eemd.spectra import morlet_cwt, tftb_panel

    x, sr = two_tones
    d = run_eemd(x, method="cpu", E=4, seed=0)
    f, tt, Zorig = stft_block(x, sr, 1024)
    Zim = np.stack([stft_block(m, sr, 1024)[2] for m in d.imfs])
    ctx = FigCtx(
        x=x, t=np.arange(len(x)) / sr, sr=sr, imfs=d.imfs,
        residual=d.residual, stft=(f, tt, Zorig, Zim),
        hilb=[hilbert_if(m, sr) for m in d.imfs],
        wavelet=morlet_cwt(x, sr, f_lo=50, n_scales=24),
        tftb=tftb_panel(x, sr, n_fbins=128, time_hop=64),
        title="test tier3", outdir=tmp_path, fmt="png",
    )
    paths = render(ctx, wavelet=True, tftb=True)
    names = {p.name for p in paths}
    assert "fig_07_wavelet.png" in names
    assert "fig_08_tftb.png" in names
    for p in paths:
        assert p.stat().st_size > 10_000


def test_wavelet_dBFS_calibration():
    """Full-scale sine reads 0 dBFS at its own scale; half-scale = -6.02."""
    from audio_eemd.spectra import morlet_cwt, morlet_cwt_gain

    sr = 8000
    freqs = np.logspace(np.log10(50), np.log10(sr / 2 * 0.99), 30)
    j = 15
    f0 = freqs[j]
    n = 4 * sr
    t = np.arange(n) / sr
    for amp, want in ((1.0, 0.0), (0.5, -6.02)):
        x = amp * np.sin(2 * np.pi * f0 * t)
        fr, C = morlet_cwt(x, sr, f_lo=freqs[0], n_scales=len(freqs))
        gains = morlet_cwt_gain(sr, fr)
        dB = 20 * np.log10(np.abs(C) / gains[:, None] + 1e-12)
        assert abs(dB[j, n // 2] - want) < 0.5, (amp, dB[j, n // 2])


def test_dBFS_calibration():
    """0 dBFS must be a full-scale sine (amplitude 1.0); half-scale = -6.02."""
    from audio_eemd.plots import _zxx_dBFS

    sr, n = 8000, 8192
    t = np.arange(n) / sr
    f0 = 500.0  # exactly on-bin: 8000/1024 = 7.8125 Hz/bin -> bin 64
    for amp, want in ((1.0, 0.0), (0.5, -6.02)):
        x = amp * np.sin(2 * np.pi * f0 * t)
        f, tt, Z = stft_block(x, sr, 1024)
        dB = _zxx_dBFS(Z)
        mid = len(tt) // 2
        assert abs(dB[64, mid] - want) < 0.1, (amp, dB[64, mid])


def test_fig04_colorbar_no_overlap(two_tones, tmp_path):
    """Regression: the fig 04 colorbar must not overlap the data panels.

    fig.colorbar(ax=<list of axes>) places the bar from the axes' bounding
    box at call time; a later subplots_adjust(right=...) then stretched the
    panels over the bar, which was drawn on top of the spectra.
    """
    from audio_eemd.plots import fig_04_stft_imfs

    x, sr = two_tones
    d = run_eemd(x, method="cpu", E=4, seed=0)
    f, tt, Zorig = stft_block(x, sr, 1024)
    Zim = np.stack([stft_block(m, sr, 1024)[2] for m in d.imfs])
    ctx = _ctx(two_tones, tmp_path)
    ctx.stft = (f, tt, Zorig, Zim)
    fig = fig_04_stft_imfs(ctx)
    try:
        fig.canvas.draw()
        subplots = [ax for ax in fig.axes if ax.get_subplotspec() is not None]
        cbars = [ax for ax in fig.axes if ax.get_subplotspec() is None]
        assert len(subplots) >= 1 and len(cbars) == 1
        cb = cbars[0].get_window_extent()
        for ax in subplots:
            assert not cb.overlaps(ax.get_window_extent()), \
                "colorbar overlaps a data panel"
    finally:
        plt.close(fig)
