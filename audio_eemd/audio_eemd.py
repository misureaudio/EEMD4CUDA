#!/usr/bin/env python3
"""audio_eemd CLI — thin adapter over the audio_eemd API.

Tier 3: metadata + audio statistics + EEMD decomposition + STFT/Hilbert
time-frequency layer + (flag-gated) Morlet CWT scalogram (fig 07) and
tftb panel (fig 08, gate G1); writes report.json, metadata.csv, imfs.npz,
stft.npz, hilbert.npz and figures fig_01..06 (+07/08 when flagged).
Flags: --method auto|cpu|gpu, --E, --eps, --tau, --max-imf, --max-sifts,
       --seed, --batch, --nperseg, --noverlap, --wavelet, --tftb,
       --csv (per-IMF CSV dumps + coarse STFT tables),
       --entropy (per-IMF sample entropy), --fractal (per-IMF Higuchi
       dimension; tftb Hölder exponent when --tftb is also set).

Exit codes: 0 ok · 2 bad usage/segment · 3 file unreadable · 4 EEMD error
            · 5 tftb gate G1 failed (only with --tftb).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bootstrap: make the package importable regardless of cwd (plan §4.2 rule 4).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_eemd import (AudioError, EEMDError, SegmentError, TFTBError,  # noqa: E402
                        analyze)


class _BadUsage(Exception):
    pass


def _build_limits(args) -> list[float] | None:
    if args.duration is not None:
        start = args.start if args.start is not None else 0.0
        return [start, start + args.duration]
    if args.start is not None or args.stop is not None:
        if args.stop is None:
            raise _BadUsage("--stop is required with --start (or use --duration)")
        start = args.start if args.start is not None else 0.0
        return [start, args.stop]
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="audio_eemd",
        description="Read a FLAC/WAV file, extract metadata, decompose the "
                    "selected segment with EEMD, and write report.json, "
                    "metadata.csv and imfs.npz.")
    ap.add_argument("file", help="path to a .flac or .wav file")
    ap.add_argument("--start", type=float, default=None,
                    help="segment start in seconds (default 0)")
    ap.add_argument("--stop", type=float, default=None,
                    help="segment stop in seconds (exclusive)")
    ap.add_argument("--duration", type=float, default=None,
                    help="segment length in seconds from --start")
    ap.add_argument("--max-file-sec", type=float, default=30.0,
                    help="files longer than this are clipped to the first "
                         "10 s unless a segment is given (default 30)")
    ap.add_argument("--method", choices=["auto", "cpu", "gpu"], default="auto",
                    help="EEMD backend (default auto: GPU if available)")
    ap.add_argument("--E", type=int, default=64,
                    help="ensemble size; E=1 runs plain EMD (default 64)")
    ap.add_argument("--eps", type=float, default=0.2,
                    help="noise amplitude as a fraction of the signal std "
                         "(default 0.2)")
    ap.add_argument("--tau", type=float, default=0.25,
                    help="sifting stopping tolerance (default 0.25)")
    ap.add_argument("--max-imf", type=int, default=-1,
                    help="max number of IMFs; -1 = until residual is monotone "
                         "(default -1)")
    ap.add_argument("--max-sifts", type=int, default=50,
                    help="max sifting iterations per IMF (default 50)")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed (default 0)")
    ap.add_argument("--batch", type=int, default=None,
                    help="GPU: max trials resident at once (memory control); "
                         "default auto from free VRAM (ignored on CPU)")
    ap.add_argument("--nperseg", type=int, default=4096,
                    help="STFT window length in samples (default 4096)")
    ap.add_argument("--noverlap", type=int, default=None,
                    help="STFT overlap in samples (default nperseg//2)")
    ap.add_argument("--wavelet", action="store_true",
                    help="also compute the Morlet CWT scalogram "
                         "(fig_07_wavelet)")
    ap.add_argument("--tftb", action="store_true",
                    help="also compute the tftb panel (fig_08_tftb); "
                         "requires gate G1 to pass (exit 5 if not)")
    ap.add_argument("--csv", action="store_true",
                    help="also dump per-IMF CSVs (imf_k.csv, residual.csv) "
                         "and coarse binned STFT tables (stft_*_coarse.csv)")
    ap.add_argument("--entropy", action="store_true",
                    help="per-IMF sample entropy column (Tier 4)")
    ap.add_argument("--fractal", action="store_true",
                    help="per-IMF fractal column (Tier 4): Higuchi dimension, "
                         "or the tftb Hölder exponent when --tftb is set")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir (default ./eemd_out/<stem>_<a>-<b>s/)")
    args = ap.parse_args(argv)

    try:
        limits = _build_limits(args)
    except _BadUsage as e:
        ap.error(str(e))
    if limits is not None and (limits[0] < 0 or limits[1] <= limits[0]):
        print(f"error: bad segment limits {limits}", file=sys.stderr)
        return 2

    try:
        res = analyze(path=args.file, limits=limits, unit="s",
                      method=args.method, E=args.E, eps=args.eps,
                      tau=args.tau, max_imf=args.max_imf,
                      max_sifts=args.max_sifts, seed=args.seed,
                      batch=args.batch, nperseg=args.nperseg,
                      noverlap=args.noverlap, wavelet=args.wavelet,
                      tftb=args.tftb, csv=args.csv, entropy=args.entropy,
                      fractal=args.fractal,
                      outdir=None, render=False)
    except AudioError as e:
        print(f"error[{e.code}]: {e}", file=sys.stderr)
        return 3
    except SegmentError as e:
        print(f"error[{e.code}]: {e}", file=sys.stderr)
        return 2
    except EEMDError as e:
        print(f"error[{e.code}]: {e}", file=sys.stderr)
        return 4
    except TFTBError as e:
        print(f"error[{e.code}]: {e}", file=sys.stderr)
        return 5
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    seg = res.segment
    md = res.meta
    st = res.report["audio_stats"]
    env = res.environment
    d = res.report["decomposition"]
    name = f"{Path(args.file).stem}_{seg['start_s']:.3f}-{seg['stop_s']:.3f}s"
    outdir = args.out if args.out is not None else Path("eemd_out") / name
    files = res.write_artifacts(outdir)
    from audio_eemd.plots import render_figures  # noqa: E402  (lazy: matplotlib)
    files += render_figures(res, outdir)

    lufs = "n/a" if st["lufs"] is None else f"{st['lufs']:.1f}"
    print(f"file:     {md['path']}")
    print(f"format:   {md['format']} ({md['subtype']}, {md['bit_depth']} bit) | "
          f"{md['samplerate']} Hz | {md['channels']} ch | {md['duration_s']:.3f} s")
    print(f"segment:  {seg['start_s']:.3f} -> {seg['stop_s']:.3f} s "
          f"({seg['n_samples']} samples)"
          + (" [defaulted: file > max-file-sec]" if seg["defaulted"] else ""))
    print(f"peak {st['peak']:.4f} | rms {st['rms']:.4f} | crest {st['crest']:.2f} | "
          f"LUFS {lufs} | centroid {st['centroid']:.1f} Hz | flatness {st['flatness']:.4f}")
    print(f"backend:  {env['backend']} | E={env['E']} | seed={args.seed} | "
          f"wall {env['wall_s']:.1f} s")
    print(f"n_modes:  {d['n_modes']} | recon_error: {d['recon_error']:.3e} | "
          f"energy_sum: {d['energy_sum']:.6f}")
    has_ent = any("sample_entropy" in r for r in d["imf"])
    has_frac = any(("higuchi_dim" in r) or ("holder_exp" in r)
                   for r in d["imf"])
    hdr_extra = (f"{'ent':>8}" if has_ent else "") + \
                (f"{'frac':>8}" if has_frac else "")
    print(f"{'mode':>4} {'std':>12} {'energy%':>9} {'dom Hz':>10} {'IF Hz':>10} "
          f"{'zcross':>8}" + hdr_extra + "  flags")
    for r in d["imf"]:
        flag = "Q" if r["quantization_level"] else ""
        df = ("n/a" if r["dominant_freq_hz"] is None
              else f"{r['dominant_freq_hz']:.1f}")
        ifm = ("n/a" if r.get("if_median_hz") is None
               else f"{r['if_median_hz']:.1f}")
        extra = ""
        if has_ent:
            v = r.get("sample_entropy")
            extra += " {:>8}".format("n/a" if v is None else f"{v:.3f}")
        if has_frac:
            v = r.get("higuchi_dim", r.get("holder_exp"))
            extra += " {:>8}".format("n/a" if v is None else f"{v:.3f}")
        print(f"{r['k']:>4} {r['std']:>12.6g} {100 * r['energy_frac']:>9.4f} "
              f"{df:>10} {ifm:>10} {r['n_zerocross']:>8}{extra}  {flag}")
    rr = d["residual"]
    df = ("n/a" if rr["dominant_freq_hz"] is None
          else f"{rr['dominant_freq_hz']:.1f}")
    print(f"{'resid':>4} {rr['std']:>12.6g} {100 * rr['energy_frac']:>9.4f} "
          f"{df:>10} {'—':>10} {rr['n_zerocross']:>8}  "
          + ("Q" if rr["quantization_level"] else ""))
    for f in files:
        print(f"artifact: {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
