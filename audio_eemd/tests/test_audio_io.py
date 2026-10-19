import numpy as np
import pytest


def test_load_flac_metadata(flac_path):
    from audio_eemd.audio_io import load_audio
    a = load_audio(flac_path)
    assert a.sr == 44100
    assert a.channels == 1
    assert a.frames == 79_380_000
    assert a.subtype == "PCM_16"
    assert a.fmt.startswith("FLAC")
    assert a.x.dtype == np.float64
    assert a.x.shape == (79_380_000, 1)


def test_segment_shape(flac_path):
    from audio_eemd.audio_io import load_audio, select_segment
    a = load_audio(flac_path)
    x = select_segment(a, [10.0, 15.0])
    assert x.shape == (5 * 44100,)
    assert x.dtype == np.float64


def test_segment_sample_unit(flac_path):
    from audio_eemd.audio_io import load_audio, select_segment
    a = load_audio(flac_path)
    x_s = select_segment(a, [10.0, 11.0], unit="s")
    x_i = select_segment(a, [10 * 44100, 11 * 44100], unit="samples")
    assert x_s.shape == x_i.shape == (44100,)
    assert np.array_equal(x_s, x_i)


def test_segment_bounds(flac_path):
    from audio_eemd.audio_io import SegmentError, load_audio, select_segment
    a = load_audio(flac_path)
    with pytest.raises(SegmentError):
        select_segment(a, [10.0, 1e9])       # stop beyond EOF
    with pytest.raises(SegmentError):
        select_segment(a, [-1.0, 5.0])       # negative start
    with pytest.raises(SegmentError):
        select_segment(a, [5.0, 5.0])        # empty
    with pytest.raises(SegmentError):
        select_segment(a, [1.0, 2.0], unit="bogus")


def test_stats_keys(flac_path):
    from audio_eemd.audio_io import load_audio, select_segment, audio_stats
    a = load_audio(flac_path)
    s = audio_stats(select_segment(a, [10.0, 15.0]), a.sr)
    for k in ("peak", "rms", "crest", "lufs", "centroid", "flatness"):
        assert k in s
    for k in ("peak", "rms", "crest", "centroid", "flatness"):
        assert np.isfinite(s[k])
    assert s["peak"] < 1.0 and s["rms"] < s["peak"]
    assert s["lufs"] is None or np.isfinite(s["lufs"])
    assert 0.0 <= s["flatness"] <= 1.0


def test_get_metadata_default_segment(flac_path):
    from audio_eemd.audio_io import get_metadata
    rep = get_metadata(flac_path)           # 1800 s file -> first 10 s, defaulted
    assert rep["segment"]["defaulted"] is True
    assert rep["segment"]["n_samples"] == 10 * 44100
    assert rep["metadata"]["samplerate"] == 44100
    assert rep["metadata"]["bit_depth"] == 16
    import json
    json.dumps(rep)                          # must be JSON-serializable


def test_get_metadata_explicit_limits(tmp_path, flac_path):
    import soundfile as sf
    from audio_eemd.audio_io import get_metadata
    # short wav -> whole-file default (no clipping)
    p = tmp_path / "short.wav"
    sf.write(p, 0.1 * np.sin(2 * np.pi * 440 * np.arange(44100) / 44100), 44100)
    rep = get_metadata(str(p))
    assert rep["segment"]["defaulted"] is False
    assert rep["segment"]["n_samples"] == 44100
    rep2 = get_metadata(flac_path, limits=[12.0, 13.0])
    assert rep2["segment"]["start_s"] == pytest.approx(12.0)
    assert rep2["segment"]["n_samples"] == 44100


def test_load_missing_raises(flac_path):
    from audio_eemd.audio_io import AudioError, load_audio
    with pytest.raises(AudioError) as ei:
        load_audio(str(flac_path) + ".missing")
    assert ei.value.code == "audio.unreadable"
