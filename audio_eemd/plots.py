"""Figure layer (plan Task 5) — matplotlib imported lazily, Agg backend.

`render(ctx, show=False)`:
  show=False (default) → saves fig_01..06 (core) [+ 07/08 when flagged] to
    ctx.outdir and returns the paths;
  show=True           → displays inline (Jupyter `matplotlib_inline`) and
    writes nothing (plan §4.2 rule 8 — notebook reuse).

Core figures (always): fig_01_signal, fig_02_imfs, fig_03_stft_original,
fig_04_stft_imfs, fig_05_hilbert, fig_06_energy.
Flagged: fig_07_wavelet (Tier 3), fig_08_tftb (Tier 3).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STYLE = {
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "font.size": 9,
}
COLORMAP_SPECTRO = "magma"
COLORMAP_SCALOG = "viridis"
MAX_LINE_PTS = 20_000
_LOG_EPS = 1e-12
_DBFS_FLOOR = -100.0  # display floor for dBFS scales (below: black)


def _mpl():
    """Lazy matplotlib import. An explicitly selected backend (e.g.
    Jupyter's inline backend) is respected so `show=True` can render
    in-notebook (plan §4.2 rule 8); headless contexts — where no backend
    was chosen and matplotlib falls back to 'agg' — stay on Agg so
    savefig works without a display."""
    import matplotlib
    if matplotlib.get_backend().lower() in ("agg", "template"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _down(t: np.ndarray, y: np.ndarray, max_pts: int = MAX_LINE_PTS):
    """Stride-decimate a line plot to ≤ max_pts points."""
    n = len(y)
    if n <= max_pts:
        return t, y
    step = int(math.ceil(n / max_pts))
    sl = slice(0, n, step)
    return t[sl], y[sl]


@dataclass
class FigCtx:
    """Everything render() needs (plan Task 5)."""

    x: np.ndarray              # segment [N]
    t: np.ndarray              # time axis [N]
    sr: int
    imfs: np.ndarray           # [n_modes, N]
    residual: np.ndarray       # [N]
    stft: tuple                # (f, t, Zxx_orig, Zxx_imf [k, F, T])
    hilb: list                 # per mode: (inst_freq [N], inst_amp [N])
    wavelet: tuple | None = None   # (freqs, C) — Tier 3
    tftb: dict | None = None       # panel arrays — Tier 3
    title: str = ""
    outdir: Path | None = None
    fmt: str = "png"


# ------------------------------------------------------------- figures --

def fig_01_signal(ctx: FigCtx):
    """Waveform + residual + running reconstruction (K modes ≥ 95 % energy)."""
    plt = _mpl()
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    t, x = _down(ctx.t, ctx.x)
    axes[0].plot(t, x, lw=0.4)
    axes[0].set_ylabel("x(t)")
    axes[0].set_title(f"fig 01 — signal / residual / reconstruction  |  {ctx.title}")
    t, r = _down(ctx.t, ctx.residual)
    axes[1].plot(t, r, lw=0.4, color="tab:red")
    axes[1].set_ylabel("residual")
    # running reconstruction: smallest K with cumulative energy ≥ 0.95
    # (or all modes when the sum of fractions never reaches 0.95 — EMD is
    # not orthogonal, so the fractions sum to < 1 in general)
    e = np.sum(ctx.imfs ** 2, axis=1) / max(np.sum(ctx.x ** 2), 1e-300)
    cum = np.cumsum(e)
    K = int(np.searchsorted(cum, 0.95) + 1)
    K = max(1, min(K, ctx.imfs.shape[0]))
    recon = ctx.imfs[:K].sum(axis=0)
    t, rc = _down(ctx.t, recon)
    axes[2].plot(t, x, lw=0.3, color="0.7", label="x(t)")
    axes[2].plot(t, rc, lw=0.5, color="tab:blue",
                 label=f"Σ imf₁..{K} (cum. energy {cum[K-1]:.3f})")
    axes[2].set_ylabel("reconstruction")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def _empty_fig(plt, label: str):
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.text(0.5, 0.5, label, ha="center", va="center")
    ax.axis("off")
    return fig


def fig_02_imfs(ctx: FigCtx):
    """Stacked IMFs, one row per mode, shared x, auto y-limits per row."""
    plt = _mpl()
    k = ctx.imfs.shape[0]
    if k == 0:
        return _empty_fig(plt, "no IMFs (degenerate input)")
    fig, axes = plt.subplots(k, 1, figsize=(11, 1.1 * k + 1), sharex=True,
                             squeeze=False)
    for i, ax in enumerate(axes[:, 0]):
        t, y = _down(ctx.t, ctx.imfs[i])
        ax.plot(t, y, lw=0.4)
        ax.set_ylabel(f"IMF {i}", rotation=0, labelpad=14, fontsize=8)
        if i == k - 1:
            ax.set_xlabel("time [s]")
    axes[0, 0].set_title(f"fig 02 — IMFs ({k} modes)  |  {ctx.title}")
    fig.tight_layout()
    return fig


def _zxx_dBFS(Zxx: np.ndarray) -> np.ndarray:
    """STFT bins -> dBFS, where 0 dBFS = full-scale sine (amplitude 1.0).

    scipy stft (scaling='spectrum') normalizes by sum(window), so a
    bin-centered sine of amplitude A reads |Z| = A/2 — hence A = 2|Z|
    (window-independent). dBFS = 20·log10(A): a full-scale tone sits at
    0 dBFS, giving an absolute, cross-panel comparable level reference.
    """
    A = 2.0 * np.abs(Zxx)
    return 20.0 * np.log10(A + 1e-12)


def fig_03_stft_original(ctx: FigCtx):
    """Log-magnitude STFT of the segment (dBFS, log frequency)."""
    plt = _mpl()
    f, t, Zxx, _ = ctx.stft
    dB = _zxx_dBFS(Zxx)[1:]          # drop the DC bin (log axis)
    f1 = f[1:]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    mesh = ax.pcolormesh(t, f1, dB, shading="auto", cmap=COLORMAP_SPECTRO,
                         vmin=max(dB.min(), _DBFS_FLOOR),
                         vmax=max(dB.max(), 0.0))
    ax.set_yscale("log")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("frequency [Hz] (log)")
    ax.set_title(f"fig 03 — STFT of segment (dBFS)  |  {ctx.title}")
    fig.colorbar(mesh, ax=ax, label="dBFS")
    fig.tight_layout()
    return fig


def fig_04_stft_imfs(ctx: FigCtx):
    """Per-IMF STFT grid, shared dBFS scale, log frequency.

    0 dBFS = full-scale sine (amplitude 1.0); the shared scale keeps
    relative levels comparable across modes (significant vs faint).
    """
    plt = _mpl()
    k = ctx.imfs.shape[0]
    f, t, _, Zim = ctx.stft
    if k == 0:
        return _empty_fig(plt, "no IMFs (degenerate input)")
    f1 = f[1:]
    dBs = [_zxx_dBFS(Zim[i])[1:] for i in range(k)]
    vmin = max(float(min(dB.min() for dB in dBs)), _DBFS_FLOOR)
    vmax = max(float(max(dB.max() for dB in dBs)), 0.0)
    nrows = max(k, 4)
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 1.6 * nrows + 1),
                             squeeze=False)
    mesh0 = None
    for i in range(nrows):
        ax = axes[i, 0]
        if i < k:
            m = ax.pcolormesh(t, f1, dBs[i], shading="auto",
                              cmap=COLORMAP_SPECTRO, vmin=vmin, vmax=vmax)
            if i == 0:
                mesh0 = m
            ax.set_yscale("log")
            ax.set_ylabel(f"IMF {i}", rotation=0, labelpad=14, fontsize=8)
        else:
            ax.axis("off")
        if i == nrows - 1:
            ax.set_xlabel("time [s]")
    axes[0, 0].set_ylabel("IMF 0 / frequency [Hz] (log)", rotation=0,
                          labelpad=14, fontsize=8)
    axes[0, 0].set_title(f"fig 04 — per-IMF STFT, shared dBFS  |  {ctx.title}")
    # Explicit colorbar axes: fig.colorbar(ax=<list>) places the bar from the
    # axes' bounding box AT CALL TIME, so a later subplots_adjust(right=...)
    # would stretch the panels over the bar (colorbar then drawn on top of
    # the data). A fixed cax is immune to layout adjustments.
    fig.subplots_adjust(left=0.07, right=0.90, top=0.97, bottom=0.05,
                        hspace=0.05)
    cax = fig.add_axes([0.92, 0.05, 0.015, 0.88])
    fig.colorbar(mesh0, cax=cax, label="dBFS (shared)")
    return fig


def fig_05_hilbert(ctx: FigCtx):
    """Per-IMF instantaneous-frequency ridges + amplitude envelope."""
    plt = _mpl()
    k = ctx.imfs.shape[0]
    if k == 0:
        return _empty_fig(plt, "no IMFs (degenerate input)")
    fig, axes = plt.subplots(k, 1, figsize=(11, 1.3 * k + 1), sharex=True,
                             squeeze=False)
    for i, ax in enumerate(axes[:, 0]):
        w, A = ctx.hilb[i]
        t, wf = _down(ctx.t, w)
        t, af = _down(ctx.t, A)
        ax.plot(t, wf, lw=0.3, color="tab:blue", label="inst. freq" if i == 0 else None)
        anorm = af / max(A.max(), 1e-300)
        ax.plot(t, anorm, lw=0.4, color="tab:orange",
                label="amp (norm.)" if i == 0 else None)
        ax.set_ylabel(f"IMF {i}", rotation=0, labelpad=14, fontsize=8)
        ax.set_ylim(-max(np.abs(wf).max() * 0.1, 1e-3),
                    max(np.abs(wf).max() * 1.05, 1e-3))
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
            ax.set_ylabel("Hz / norm", rotation=0, labelpad=14, fontsize=8)
        if i == k - 1:
            ax.set_xlabel("time [s]")
    axes[0, 0].set_title(f"fig 05 — Hilbert IF ridges + envelopes  |  {ctx.title}")
    fig.tight_layout()
    return fig


def fig_06_energy(ctx: FigCtx):
    """Energy-fraction bars per mode + cumulative line (log y)."""
    plt = _mpl()
    e = np.sum(ctx.imfs ** 2, axis=1) / max(np.sum(ctx.x ** 2), 1e-300)
    er = np.sum(ctx.residual ** 2) / max(np.sum(ctx.x ** 2), 1e-300)
    e_all = np.append(e, er)
    cum = np.cumsum(e_all)
    labels = [f"IMF {i}" for i in range(len(e))] + ["resid"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels, e_all, color="tab:blue", alpha=0.7,
                  label="energy fraction")
    ax.set_yscale("log")
    ax.set_ylabel("energy fraction (log)")
    ax2 = ax.twinx()
    ax2.plot(labels, cum, "o-", color="tab:red", ms=4, lw=1,
             label="cumulative")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("cumulative")
    ax.set_title(f"fig 06 — energy distribution  |  {ctx.title}")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def fig_07_wavelet(ctx: FigCtx):
    """(Tier 3, flag) Morlet scalogram of the segment (dBFS, log frequency).

    Normalized per scale by morlet_cwt_gain so a full-scale sine reads
    0 dBFS at its own frequency — same reference as fig 03/04.
    """
    plt = _mpl()
    from .spectra import morlet_cwt_gain
    freqs, C = ctx.wavelet
    gains = morlet_cwt_gain(ctx.sr, freqs)
    dB = 20.0 * np.log10(np.abs(C) / gains[:, None] + 1e-12)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    mesh = ax.pcolormesh(ctx.t, freqs, dB, shading="auto",
                         cmap=COLORMAP_SCALOG,
                         vmin=max(dB.min(), _DBFS_FLOOR),
                         vmax=max(dB.max(), 0.0))
    ax.set_yscale("log")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("frequency [Hz] (log)")
    ax.set_title(f"fig 07 — Morlet CWT scalogram (dBFS)  |  {ctx.title}")
    fig.colorbar(mesh, ax=ax, label="dBFS")
    fig.tight_layout()
    return fig


def fig_08_tftb(ctx: FigCtx):
    """(Tier 3, flag) tftb panels: spectrogram / SPWVD / reassignment.

    Shared color scale across panels (the TFRs are comparable in magnitude
    after clipping); explicit cax (see fig_04 for why not ax=<list>).
    """
    plt = _mpl()
    panels = [(name, arr, t, f) for name, arr, t, f in (
        ("tftb spectrogram", ctx.tftb.get("spectrogram"),
         ctx.tftb.get("t_stft"), ctx.tftb.get("f_stft")),
        ("SPWVD (neg. clipped)", ctx.tftb.get("spwvd"),
         ctx.tftb.get("t_spwvd"), ctx.tftb.get("f_spwvd")),
        ("reassigned gabor", ctx.tftb.get("reassigned"),
         ctx.tftb.get("t_reass"), ctx.tftb.get("f_reass")),
    ) if arr is not None]
    if not panels:
        return _empty_fig(plt, "no tftb panels")
    clipped = [np.clip(arr, 0, None) for _, arr, _, _ in panels]
    vmax = float(max(np.percentile(a, 99) for a in clipped))
    # Display band: from 0 up to the frequency containing 99 % of the
    # spectrogram energy (clamped to [1 kHz, Nyquist]) — keeps low-band
    # content readable instead of squashed under a mostly-black axis.
    f_spec = panels[0][3]
    e = np.sum(clipped[0], axis=0)  # per-frequency energy, spectrogram panel
    fmax = float(f_spec[np.searchsorted(np.cumsum(e) / max(e.sum(), 1e-300),
                                        0.99)])
    fmax = min(max(fmax * 1.25, 1000.0), f_spec[-1])
    keep = f_spec <= fmax
    fig, axes = plt.subplots(len(panels), 1,
                             figsize=(11, 3.5 * len(panels) + 1),
                             squeeze=False)
    mesh0 = None
    for (name, arr, tt, ff), a, ax in zip(panels, clipped, axes[:, 0]):
        m = ax.pcolormesh(tt, ff[keep], a.T[keep, :], shading="auto",
                          cmap=COLORMAP_SPECTRO, vmin=0.0, vmax=vmax)
        if mesh0 is None:
            mesh0 = m
        ax.set_title(name, fontsize=9, loc="left")
        ax.set_ylabel("frequency [Hz]")
        ax.set_xlabel("time [s]")
    fig.suptitle(f"fig 08 — tftb panels (shared scale)  |  {ctx.title}")
    cax = fig.add_axes([0.92, 0.10, 0.015, 0.84])
    fig.colorbar(mesh0, cax=cax, label="TFR value (shared)")
    return fig


# ------------------------------------------------------------- render --

def render(ctx: FigCtx, wavelet: bool = False, tftb: bool = False,
           show: bool = False) -> list[Path]:
    """Build all figures; save to ctx.outdir (show=False) or display inline
    (show=True). Returns the list of written paths (empty when show=True)."""
    plt = _mpl()
    plt.rcParams.update(STYLE)
    if not show and ctx.outdir is not None:
        Path(ctx.outdir).mkdir(parents=True, exist_ok=True)
    figs: list[tuple[str, object]] = [
        ("fig_01_signal", fig_01_signal(ctx)),
        ("fig_02_imfs", fig_02_imfs(ctx)),
        ("fig_03_stft_original", fig_03_stft_original(ctx)),
        ("fig_04_stft_imfs", fig_04_stft_imfs(ctx)),
        ("fig_05_hilbert", fig_05_hilbert(ctx)),
        ("fig_06_energy", fig_06_energy(ctx)),
    ]
    if wavelet and ctx.wavelet is not None:
        figs.append(("fig_07_wavelet", fig_07_wavelet(ctx)))
    if tftb and ctx.tftb is not None:
        figs.append(("fig_08_tftb", fig_08_tftb(ctx)))
    paths: list[Path] = []
    for name, fig in figs:
        try:
            if show:
                plt.show()
            else:
                if ctx.outdir is None:
                    raise ValueError("render: ctx.outdir is None (and show=False)")
                p = Path(ctx.outdir) / f"{name}.{ctx.fmt}"
                fig.savefig(p, bbox_inches="tight")
                paths.append(p)
        finally:
            plt.close(fig)
    return paths


def render_figures(res, outdir, fmt: str = "png") -> list[Path]:
    """Convenience for the API: build a FigCtx from an AnalysisResult and
    save the core figures (+ flagged ones when computed)."""
    if res.stft is None or res.hilbert is None:
        raise ValueError("render_figures: result has no STFT/Hilbert data "
                         "(analyze() with Tier 2 computes them automatically)")
    params = res.parameters
    title = (f"{res.meta.get('path', '')} · {res.segment['start_s']:.3f}–"
             f"{res.segment['stop_s']:.3f} s · {res.segment['sr']} Hz · "
             f"E={params['E']} τ={params['tau']} seed={params['seed']} "
             f"backend={res.environment['backend']}")
    ctx = FigCtx(x=res.x, t=res.t, sr=res.segment["sr"], imfs=res.imfs,
                 residual=res.residual, stft=res.stft, hilb=res.hilbert,
                 wavelet=res.wavelet, tftb=res.tftb_panel, title=title,
                 outdir=Path(outdir), fmt=fmt)
    return render(ctx, wavelet=res.wavelet is not None,
                  tftb=res.tftb_panel is not None)
