import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "audio_eemd" / "audio_eemd.py"


def run_cli(*args, cwd=ROOT):
    return subprocess.run(
        [sys.executable, str(CLI), *[str(a) for a in args]],
        capture_output=True, text=True, cwd=str(cwd),
    )


def test_help(flac_path):
    r = run_cli("--help")
    assert r.returncode == 0
    assert "--max-file-sec" in r.stdout


def test_missing_file(tmp_path):
    r = run_cli(tmp_path / "nope.flac")
    assert r.returncode == 3
    assert "audio.unreadable" in r.stderr


def test_bad_segment(flac_path):
    r = run_cli(flac_path, "--start", "10", "--stop", "9")
    assert r.returncode == 2


def test_start_without_stop(flac_path):
    r = run_cli(flac_path, "--start", "10")
    assert r.returncode == 2  # argparse error


def test_cli_smoke_1s(flac_path, tmp_path):
    r = run_cli(flac_path, "--start", "10", "--duration", "1",
                "--method", "cpu", "--E", "8", "--seed", "0",
                "--out", tmp_path / "smoke")
    assert r.returncode == 0, r.stderr
    out = tmp_path / "smoke"
    assert (out / "report.json").exists()
    assert (out / "metadata.csv").exists()
    assert (out / "imfs.npz").exists()
    assert (out / "stft.npz").exists()
    assert (out / "hilbert.npz").exists()
    for name in ("fig_01_signal", "fig_02_imfs", "fig_03_stft_original",
                 "fig_04_stft_imfs", "fig_05_hilbert", "fig_06_energy"):
        assert (out / f"{name}.png").exists(), name
    rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert rep["metadata"]["samplerate"] == 44100
    assert rep["metadata"]["format"].startswith("FLAC")
    assert rep["segment"]["n_samples"] == 44100
    assert rep["segment"]["defaulted"] is False
    assert rep["audio_stats"]["peak"] < 1.0
    assert rep["audio_stats"]["lufs"] is None or isinstance(
        rep["audio_stats"]["lufs"], float)
    assert "tool_version" in rep and "generated" in rep
    d = rep["decomposition"]
    assert d["n_modes"] >= 1
    assert d["recon_error"] < 1e-6
    assert len(d["imf"]) == d["n_modes"]
    assert d["quantization"]["bit_depth"] == 16
    assert rep["environment"]["backend"] == "cpu"
    # stdout summary mentions the segment and the mode table
    assert "segment" in r.stdout
    assert "n_modes" in r.stdout
    assert "IF Hz" in r.stdout


def test_cli_wavelet_tftb(flac_path, tmp_path):
    """--wavelet and --tftb add fig 07/08 and their report blocks (Tier 3)."""
    r = run_cli(flac_path, "--start", "10", "--duration", "1",
                "--method", "cpu", "--E", "2", "--seed", "0",
                "--wavelet", "--tftb",
                "--out", tmp_path / "t3")
    assert r.returncode == 0, r.stderr
    out = tmp_path / "t3"
    assert (out / "fig_07_wavelet.png").exists()
    assert (out / "fig_08_tftb.png").exists()
    rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert rep["parameters"]["wavelet"] is True
    assert rep["parameters"]["tftb"] is True
    assert rep["wavelet"]["n_scales"] >= 10
    assert rep["tftb"]["gate"]["stft_corr"] > 0.99
    assert set(rep["tftb"]["panels"]) == {"spectrogram", "spwvd", "reassigned"}
