"""Tests for the figure layer (plan Task 5, Tier 2 scope)."""
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
