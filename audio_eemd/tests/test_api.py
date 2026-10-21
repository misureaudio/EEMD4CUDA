"""Tests for the public API surface (plan Task 6, Tier 1 scope)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_eemd import AudioError, EEMDError, SegmentError, analyze
from audio_eemd.api import AnalysisResult

ROOT = Path(__file__).resolve().parents[2]
FLAC = ROOT / "PrimeToneNewPhi[Nch 1][Frac 003][00043-16381][Nfr 027]-[44100-16]-[1800s]-[-0.25dB]-[ 8.4809]-[RD].flac"


@pytest.fixture(scope="module")
def wav2(tmp_path_factory):
    """2 s of the PrimeTone FLAC (t=10..12 s) as a PCM_16 WAV."""
    x, sr = sf.read(str(FLAC), start=10 * 44100, stop=12 * 44100,
                    dtype="float64", always_2d=True)
    p = tmp_path_factory.mktemp("wav") / "seg.wav"
    sf.write(p, x[:, 0], sr, subtype="PCM_16")
    return p, sr


@pytest.fixture(scope="module")
def res2(wav2, tmp_path_factory):
    """Module-scoped 2 s CPU analysis (E=4) shared by several tests."""
    p, sr = wav2
    out = tmp_path_factory.mktemp("art") / "res2"
    return analyze(path=p, limits=[0, 2], unit="s", method="cpu", E=4,
                   seed=0, outdir=out, render=False)


def test_analyze_from_path(res2):
    assert isinstance(res2, AnalysisResult)
    assert res2.n_modes >= 1
    assert res2.imfs.shape == (res2.n_modes, len(res2.x))
    assert res2.recon_error < 1e-6 * np.abs(res2.x).max()
    assert res2.segment["n_samples"] == 2 * 44100
    assert res2.segment["defaulted"] is False
    assert res2.environment["backend"] == "cpu"
    # per-mode table shape
    assert len(res2.mode_table) == res2.n_modes
    for r in res2.mode_table:
        for key in ("k", "std", "energy_frac", "dominant_freq_hz",
                    "n_zerocross", "sifting", "quantization_level",
                    "if_median_hz", "if_iqr_hz", "if_n_pos_frac"):
            assert key in r
        assert 0.0 <= r["energy_frac"] < 1.5
        assert r["std"] >= 0.0
    # Tier 2 products present
    assert res2.stft is not None
    f, tt, Zxx_orig, Zxx_imf = res2.stft
    assert Zxx_imf.shape[0] == res2.n_modes
    assert Zxx_orig.shape[0] == Zxx_imf.shape[1] == f.shape[0]
    assert len(res2.hilbert) == res2.n_modes
    assert res2.hilbert[0][0].shape == (len(res2.x),)
    # JSON-serializable, no raw arrays embedded
    d = json.loads(json.dumps(res2.to_dict()))
    assert d["decomposition"]["n_modes"] == res2.n_modes
    assert "x" in d["arrays"] and "shape" in d["arrays"]["x"]


def test_analyze_from_array_and_sample_limits(res2):
    res = analyze(x=res2.x, sr=44100, limits=[1000, 20000], unit="samples",
                  method="cpu", E=2, seed=0)
    assert res.segment["n_samples"] == 19000
    assert res.segment["unit_used"] == "samples"
    assert res.t[0] == pytest.approx(1000 / 44100)
    assert res.meta["format"] == "array"
    assert res.n_modes >= 1


def test_analyze_input_validation(wav2):
    p, sr = wav2
    with pytest.raises(ValueError):
        analyze()  # neither path nor x
    with pytest.raises(ValueError):
        analyze(x=np.ones(100))  # x without sr
    with pytest.raises(EEMDError) as ei:
        analyze(x=np.ones(5), sr=100, method="cpu", E=1)
    assert ei.value.code == "eemd.bad_input"


def _two_tone_array(n=2048, sr=8000):
    t = np.arange(n) / sr
    return (np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 105 * t)) / 2, sr


def test_wavelet_flag():
    x, sr = _two_tone_array()
    res = analyze(x=x, sr=sr, method="cpu", E=2, seed=0, wavelet=True)
    assert res.wavelet is not None
    freqs, C = res.wavelet
    assert C.shape == (len(freqs), len(x))
    assert freqs[0] < freqs[-1]
    assert res.report["wavelet"]["n_scales"] == len(freqs)
    d = json.loads(json.dumps(res.to_dict()))
    assert d["wavelet"]["n_scales"] == len(freqs)
    # the scalogram must peak near the two tones (interior: CWT edge
    # artifacts from the zero-padded convolution are expected at the ends)
    ridge = freqs[np.argmax(np.abs(C), axis=0)]
    m = len(ridge) // 4
    assert ridge[m:len(ridge) - m].min() > 50
    assert ridge[m:len(ridge) - m].max() < 200


def test_tftb_flag():
    x, sr = _two_tone_array()
    res = analyze(x=x, sr=sr, method="cpu", E=2, seed=0, tftb=True)
    assert res.tftb_panel is not None
    p = res.tftb_panel
    for key in ("spectrogram", "spwvd", "reassigned",
                "t_stft", "f_stft", "t_spwvd", "f_spwvd", "t_reass", "f_reass"):
        assert key in p, key
    assert np.isfinite(p["spectrogram"]).all()
    assert np.isfinite(p["spwvd"]).all()
    assert np.isfinite(p["reassigned"]).all()
    # SPWVD is a Cohen-class TFR: it must contain negative values here
    assert p["spwvd"].min() < 0
    assert res.report["tftb"]["gate"]["stft_corr"] > 0.99
    assert any("tftb" in c for c in res.caveats)
    d = json.loads(json.dumps(res.to_dict()))
    assert d["tftb"]["version"]


def test_tftb_gate_failure(monkeypatch):
    """A failing gate G1 must surface as TFTBError (CLI maps to exit 5)."""
    import audio_eemd.api as api
    from audio_eemd.spectra import TFTBError

    def _fail():
        raise TFTBError("simulated gate failure", code="tftb.gate_failed")

    monkeypatch.setattr(api, "check_tftb_gate", _fail)
    x, sr = _two_tone_array()
    with pytest.raises(TFTBError) as ei:
        analyze(x=x, sr=sr, method="cpu", E=2, seed=0, tftb=True)
    assert ei.value.code == "tftb.gate_failed"


def test_artifacts_written(res2):
    out = Path(res2.outdir)
    files = res2.write_artifacts()
    assert (out / "report.json").exists()
    assert (out / "metadata.csv").exists()
    assert (out / "imfs.npz").exists()
    assert (out / "stft.npz").exists()
    assert (out / "hilbert.npz").exists()
    rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
    for key in ("metadata", "segment", "parameters", "environment",
                "audio_stats", "decomposition", "timings", "caveats",
                "generated", "tool_version"):
        assert key in rep, key
    d = rep["decomposition"]
    assert d["n_modes"] == res2.n_modes
    assert d["recon_error"] == res2.recon_error
    # EMD is not orthogonal: energy_sum is a diagnostic, not a unit check
    assert 0.0 < d["energy_sum"] < 2.0
    assert d["quantization"]["bit_depth"] == 16
    assert d["quantization"]["floor_std"] is not None
    # npz round-trip
    z = np.load(out / "imfs.npz")
    for key in ("x", "t", "imfs", "residual", "n_modes"):
        assert key in z.files
    assert z["imfs"].shape == (res2.n_modes, len(res2.x))
    assert np.array_equal(z["x"], res2.x)
    # stft.npz / hilbert.npz round-trip
    z = np.load(out / "stft.npz")
    for key in ("freqs", "times", "Zxx_orig", "Zxx_imf"):
        assert key in z.files
    assert z["Zxx_imf"].shape == (res2.n_modes, z["Zxx_orig"].shape[0],
                                  z["Zxx_orig"].shape[1])
    z = np.load(out / "hilbert.npz")
    for k in range(res2.n_modes):
        assert f"inst_freq_{k}" in z.files
        assert f"inst_amp_{k}" in z.files
        assert z[f"inst_freq_{k}"].shape == (len(res2.x),)
    # idempotent rewrite
    res2.write_artifacts(out)
    assert (out / "imfs.npz").exists()


def test_render_figures(wav2, tmp_path):
    """analyze(..., render=True) writes the six core figures."""
    p, sr = wav2
    out = tmp_path / "figs"
    res = analyze(path=p, limits=[0, 1], unit="s", method="cpu", E=2,
                  seed=0, outdir=out, render=True)
    for name in ("fig_01_signal", "fig_02_imfs", "fig_03_stft_original",
                 "fig_04_stft_imfs", "fig_05_hilbert", "fig_06_energy"):
        f = out / f"{name}.png"
        assert f.exists(), name
        assert f.stat().st_size > 10_000


def test_typed_errors(wav2):
    p, sr = wav2
    with pytest.raises(AudioError) as ei:
        analyze(path=p.parent / "missing.flac")
    assert ei.value.code == "audio.unreadable"
    with pytest.raises(SegmentError):
        analyze(path=p, limits=[0, 10.0], unit="s")  # 2 s file
    # no outdir stored -> ValueError
    res = analyze(x=np.ones(1000), sr=1000, method="cpu", E=1)
    with pytest.raises(ValueError):
        res.write_artifacts()


def test_csv_flag(wav2, tmp_path):
    p, sr = wav2
    out = tmp_path / "csv"
    res = analyze(path=p, limits=[0, 1], unit="s", method="cpu", E=2,
                  seed=0, csv=True, outdir=out, render=False)
    res.write_artifacts()
    assert (out / "imf_0.csv").exists()
    assert (out / "residual.csv").exists()
    assert (out / f"imf_{res.n_modes - 1}.csv").exists()
    # coarse binned STFT tables (Tier 4): header + 256 log-freq rows,
    # 1 + min(256, n_frames) columns
    for name in ("stft_x_coarse.csv", "stft_imf_0_coarse.csv"):
        f = out / name
        assert f.exists(), name
        lines = f.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 257, name
        T = len(res.stft[1])
        assert len(lines[0].split(",")) == 1 + min(256, T)
        assert lines[0].split(",")[0] == "freq_hz"
        freqs = [float(l.split(",")[0]) for l in lines[1:]]
        assert freqs == sorted(freqs) and freqs[0] > 0


def test_entropy_fractal_flags(wav2, tmp_path):
    """Tier 4: per-IMF sample entropy + Higuchi dimension (no tftb)."""
    p, sr = wav2
    out = tmp_path / "t4"
    res = analyze(path=p, limits=[0, 1], unit="s", method="cpu", E=2,
                  seed=0, entropy=True, fractal=True, outdir=out,
                  render=False)
    assert res.entropy is not None and len(res.entropy) == res.n_modes
    assert res.fractal is not None and len(res.fractal) == res.n_modes
    # 1 s @ 44.1 kHz -> stride 10 -> effective rate 4410 Hz
    for r in res.mode_table:
        assert "sample_entropy" in r and "higuchi_dim" in r
        assert r["decim_rate_hz"] == pytest.approx(4410.0)
        if r["sample_entropy"] is not None:
            assert np.isfinite(r["sample_entropy"])
        if r["higuchi_dim"] is not None:
            # fractal dimension of a 1-D series: 1 (smooth) .. 2 (noise)
            assert 0.9 <= r["higuchi_dim"] < 2.5
    assert any("decimated" in c for c in res.caveats)
    assert "entropy_s" in res.timings and "fractal_s" in res.timings
    # JSON-serializable with the new columns
    d = json.loads(json.dumps(res.to_dict()))
    assert "sample_entropy" in d["decomposition"]["imf"][0]
    assert "higuchi_dim" in d["decomposition"]["imf"][0]


def test_api_import_is_light():
    """audio_eemd.api must import without matplotlib (plan design rule 8)."""
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys; import audio_eemd.api; "
         "print('matplotlib' in sys.modules)"],
        capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False"
