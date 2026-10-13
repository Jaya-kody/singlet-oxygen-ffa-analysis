"""Figure helpers. Colours follow the original figures: UVA purple, fluorescent orange-red, sunlight blue."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Affine2D

from .kinetics import DecayFit

LIGHT_COLOURS = {"UVA": "#4b006e", "Fluorescent": "orangered", "Sunlight": "#1f5fbf"}
SAMPLE_COLOUR = "darkviolet"
BACKGROUND_COLOUR = "grey"

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def plot_decay(ax, time_h, ln_ratio, fit: DecayFit, label=None,
               colour=SAMPLE_COLOUR, show_fit=True):
    """Scatter one FFA decay; points excluded from the fit are drawn hollow."""
    t = np.asarray(time_h)
    y = np.asarray(ln_ratio)
    ax.scatter(t[fit.used], y[fit.used], color=colour, marker="^", zorder=3, label=label)
    if (~fit.used).any():
        ax.scatter(t[~fit.used], y[~fit.used], facecolors="none", edgecolors=colour,
                   marker="^", zorder=3, label="excluded (>25% slope change)")
    if show_fit:
        t_fit = np.linspace(0, t[fit.used].max(), 50)
        ax.plot(t_fit, fit.intercept - fit.k_obs * t_fit * 3600, color="black", lw=1)


def decay_grid(panels, title, ncols=3):
    """Grid of FFA decay plots.

    ``panels`` is a list of dicts with keys: title, sample (t_h, y, fit),
    background (t_h, y, fit) or None.
    """
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.0 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    for ax, p in zip(axes.flat, panels):
        t, y, fit = p["sample"]
        plot_decay(ax, t, y, fit, label="sample")
        if p.get("background") is not None:
            tb, yb, fitb = p["background"]
            plot_decay(ax, tb, yb, fitb, label="background air", colour=BACKGROUND_COLOUR,
                       show_fit=False)
        ax.set_title(p["title"], fontsize=10)
        ax.text(0.04, 0.05, f"$k_{{obs}}$ = {fit.k_obs:.2e} s$^{{-1}}$\n$R^2$ = {fit.r_squared:.3f}",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=8)
        ax.axhline(0, color="black", lw=0.5, alpha=0.4)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Irradiation time (h)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"ln([FFA]$_t$/[FFA]$_0$)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        seen.update(dict(zip(l, h)))
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=len(seen),
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def offset_dot_plot(ax, categories, series, ylabel, log=False):
    """Dot-and-error-bar plot with one horizontally offset series per light source.

    ``series`` maps a light name to (values, errors).
    """
    n = len(series)
    offsets = np.linspace(-0.2, 0.2, n) if n > 1 else [0.0]
    x = np.arange(len(categories))
    for off, (light, (vals, errs)) in zip(offsets, series.items()):
        trans = Affine2D().translate(off, 0) + ax.transData
        ax.errorbar(x, vals, yerr=errs, marker="o", markersize=7, linestyle="none",
                    capsize=4, color="black", markerfacecolor=LIGHT_COLOURS.get(light, "grey"),
                    markeredgecolor="black", label=light, transform=trans)
    ax.set_xticks(x, categories)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="grey", alpha=0.3)
    if log:
        ax.set_yscale("log")
