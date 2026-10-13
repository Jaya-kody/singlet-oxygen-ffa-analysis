"""Full singlet-oxygen workflow: FFA decay fits -> [1O2]ss -> R_abs -> apparent quantum yields.

Usage:
    python scripts/run_analysis.py                       # uses data/example (synthetic)
    python scripts/run_analysis.py --data data/raw       # your own data, same layout

Outputs go to results/ (CSV tables) and figures/ (PNG).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from singlet_oxygen import (  # noqa: E402
    apparent_quantum_yield,
    fit_decay,
    fit_decay_trimmed,
    k_ffa,
    light_conversion_factor,
    quantum_yield_error,
    rate_of_absorbance,
    steady_state_1o2,
)
from singlet_oxygen.plotting import LIGHT_COLOURS, decay_grid, offset_dot_plot  # noqa: E402

LIGHTS = ["UVA", "Fluorescent", "Sunlight"]
DISH_COLOURS = {"Brussels sprouts": "#2e7d32", "Pancake": "#c48a00", "Stir-fry": "#b23a48"}


def load_decay(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")  # tolerate Excel's BOM
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")].dropna(how="all")
    return df


def analyse(data: Path, results: Path, figures: Path, temps: dict[str, float]):
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    samples = pd.read_csv(data / "samples.csv")
    absorbance = pd.read_csv(data / "absorbance.csv")
    irradiance = pd.read_csv(data / "irradiance.csv")
    wl_abs = absorbance["wavelength_nm"].to_numpy()
    wl_irr = irradiance["wavelength_nm"].to_numpy()

    def rabs(column: str, light: str) -> float:
        a = np.interp(wl_irr, wl_abs, absorbance[column].to_numpy(), left=0, right=0)
        return rate_of_absorbance(wl_irr, irradiance[light].to_numpy(), a)

    rows, panels = [], {light: [] for light in LIGHTS}
    for light in LIGHTS:
        ref_path = data / "ffa_decay" / f"REF_{light}.csv"
        if not ref_path.exists():
            continue
        ref = load_decay(ref_path)
        ref_fit = fit_decay(ref["time_s"], ref["ln(FFAt/FFA0)"])
        ref_rabs = rabs("REF", light)

        for s in samples.itertuples():
            path = data / "ffa_decay" / f"{s.sample_id}_{light}.csv"
            if not path.exists():
                continue
            d = load_decay(path)
            fit = fit_decay_trimmed(d["time_s"], d["ln(FFAt/FFA0)"])

            bg_path = data / "ffa_decay" / f"{s.sample_id}_{light}_background.csv"
            bg_panel, bg_k = None, np.nan
            if bg_path.exists():
                b = load_decay(bg_path)
                bg_fit = fit_decay(b["time_s"], b["ln(FFAt/FFA0)"])
                bg_k = bg_fit.k_obs
                bg_panel = (b["time_h"], b["ln(FFAt/FFA0)"], bg_fit)

            r_abs = rabs(s.sample_id, light)
            phi = apparent_quantum_yield(fit.k_obs, r_abs, ref_fit.k_obs, ref_rabs)
            phi_err = quantum_yield_error(phi, [fit.k_obs_se / fit.k_obs,
                                                ref_fit.k_obs_se / ref_fit.k_obs])
            rows.append({
                "sample_id": s.sample_id, "dish": s.dish, "light": light,
                "temperature_C": temps[light], "k_FFA_M-1s-1": k_ffa(temps[light]),
                "k_obs_s-1": fit.k_obs, "k_obs_se_s-1": fit.k_obs_se, "r_squared": fit.r_squared,
                "points_used": fit.n_points, "points_total": len(d),
                "background_k_obs_s-1": bg_k,
                "ss_1O2_M": steady_state_1o2(fit.k_obs, temps[light]),
                "ss_1O2_se_M": steady_state_1o2(fit.k_obs_se, temps[light]),
                "R_abs": r_abs,
                "ref_k_obs_s-1": ref_fit.k_obs, "ref_R_abs": ref_rabs,
                "QY_percent": 100 * phi, "QY_err_percent": 100 * phi_err,
            })
            panels[light].append({
                "title": f"{s.dish} {s.replicate}",
                "sample": (d["time_h"], d["ln(FFAt/FFA0)"], fit),
                "background": bg_panel,
            })

    table = pd.DataFrame(rows)
    table.to_csv(results / "summary.csv", index=False, float_format="%.4g")

    # conversion factors relative to UVA (eqns 9-10), per sample then averaged
    wide = table.pivot(index="sample_id", columns="light", values="R_abs")
    conv = pd.DataFrame({
        f"{light}_to_UVA_h_per_h": [light_conversion_factor(r, u)
                                    for r, u in zip(wide[light], wide["UVA"])]
        for light in LIGHTS if light != "UVA" and light in wide
    }, index=wide.index)
    conv.to_csv(results / "conversion_factors.csv", float_format="%.4g")

    for light, p in panels.items():
        if p:
            fig = decay_grid(p, f"FFA decay under {light.lower()} light")
            fig.savefig(figures / f"ffa_decay_{light.lower()}.png")
            plt.close(fig)

    plot_spectra(absorbance, irradiance, samples, figures / "spectra.png")
    plot_summary(table, samples, figures / "summary.png")

    print(table.groupby(["dish", "light"])[["ss_1O2_M", "QY_percent"]].mean()
          .to_string(float_format=lambda v: f"{v:.3g}"))
    print("\nMean conversion factors (h of UVA per h of light):")
    print(conv.mean().to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nTables in {results.relative_to(ROOT)}/, figures in {figures.relative_to(ROOT)}/")


def plot_spectra(absorbance, irradiance, samples, out: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    wl = irradiance["wavelength_nm"]
    for light in LIGHTS:
        ax1.fill_between(wl, irradiance[light], color=LIGHT_COLOURS[light], alpha=0.45,
                         label=light, edgecolor=LIGHT_COLOURS[light])
    ax1.set(xlim=(280, 600), xlabel="Wavelength (nm)",
            ylabel=r"Spectral irradiance (W m$^{-2}$ nm$^{-1}$)", title="Light sources")
    ax1.legend(frameon=False)

    seen = set()
    for s in samples.itertuples():
        c = DISH_COLOURS.get(s.dish, "grey")
        ax2.plot(absorbance["wavelength_nm"], absorbance[s.sample_id], color=c, lw=1.2,
                 label=s.dish if s.dish not in seen else None)
        seen.add(s.dish)
    in_range = absorbance["wavelength_nm"].between(250, 600)
    top = absorbance.loc[in_range, samples["sample_id"]].to_numpy().max() * 1.1
    ax2.set(xlim=(250, 600), ylim=(0, top), xlabel="Wavelength (nm)",
            ylabel="Absorbance (1 cm)", title="Sample absorbance")
    ax2.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_summary(table, samples, out: Path):
    cats = samples["sample_id"].tolist()
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    specs = [
        ("ss_1O2_M", "ss_1O2_se_M", r"[$^1$O$_2^*$]$_{ss}$ (M)", True),
        ("R_abs", None, r"$R_{abs}$ (arb. units)", True),
        ("QY_percent", "QY_err_percent", r"Apparent $\Phi_{^1O_2}$ (%)", False),
    ]
    for ax, (col, err, label, log) in zip(axes, specs):
        series = {}
        for light in LIGHTS:
            sub = table[table["light"] == light].set_index("sample_id").reindex(cats)
            if sub[col].notna().any():
                series[light] = (sub[col].to_numpy(),
                                 sub[err].to_numpy() if err else None)
        offset_dot_plot(ax, cats, series, label, log=log)
    axes[0].legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.25))
    axes[-1].set_xlabel("Sample")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "example")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--figures", type=Path, default=ROOT / "figures")
    parser.add_argument("--reactor-temp", type=float, default=30.0,
                        help="photoreactor temperature for UVA/fluorescent runs (degC)")
    parser.add_argument("--sunlight-temp", type=float, default=22.0)
    a = parser.parse_args()
    temps = {"UVA": a.reactor_temp, "Fluorescent": a.reactor_temp, "Sunlight": a.sunlight_temp}
    analyse(a.data.resolve(), a.results.resolve(), a.figures.resolve(), temps)
