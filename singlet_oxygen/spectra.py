"""Light absorption by a sample: screening factor, rate of absorbance, photon flux.

    R_abs = 2.303 * sum_lambda( I_lambda * abs_lambda * sf_lambda )     (eqn 4)
    sf_lambda = (1 - 10^-abs_lambda) / (2.303 * abs_lambda)

The screening factor corrects for the sample absorbing part of the light before
it reaches the whole solution (inner-filter effect).
"""

from __future__ import annotations

import numpy as np

PLANCK = 6.626e-34  # J s
LIGHT_SPEED = 2.998e8  # m s^-1
AVOGADRO = 6.022e23  # mol^-1
LN10 = float(np.log(10))  # 2.303


def screening_factor(absorbance) -> np.ndarray:
    """Light screening factor for a given absorbance (dimensionless, 0-1).

    Tends to 1 as absorbance tends to 0, and is returned as exactly 1 for
    zero or negative absorbance (baseline noise).
    """
    a = np.asarray(absorbance, dtype=float)
    out = np.ones_like(a)
    pos = a > 1e-12
    out[pos] = (1.0 - 10.0 ** (-a[pos])) / (LN10 * a[pos])
    return out


def resample(wavelength_src, values, wavelength_dst) -> np.ndarray:
    """Linear interpolation of a spectrum onto another wavelength grid.

    Values outside the source range are set to 0.
    """
    ws = np.asarray(wavelength_src, dtype=float)
    vs = np.asarray(values, dtype=float)
    order = np.argsort(ws)
    return np.interp(wavelength_dst, ws[order], vs[order], left=0.0, right=0.0)


def rate_of_absorbance(wavelength_nm, irradiance, absorbance,
                       wl_min: float = 280.0, wl_max: float = 800.0) -> float:
    """Rate of light absorption (eqn 4) summed on a 1 nm grid.

    ``irradiance`` and ``absorbance`` must be on the same ``wavelength_nm`` grid;
    use :func:`resample` first if they are not. The result has the units of
    ``irradiance`` (per nm) multiplied by nm, i.e. it is directly comparable
    between samples measured under the same light source.
    """
    wl = np.asarray(wavelength_nm, dtype=float)
    grid = np.arange(np.ceil(max(wl_min, wl.min())), np.floor(min(wl_max, wl.max())) + 1)
    i_grid = resample(wl, irradiance, grid)
    a_grid = resample(wl, absorbance, grid)
    a_grid = np.clip(a_grid, 0.0, None)  # negative absorbance is baseline noise
    return float(LN10 * np.sum(i_grid * a_grid * screening_factor(a_grid)))


def irradiance_to_photon_flux(irradiance_w_m2_nm, wavelength_nm) -> np.ndarray:
    """Convert spectral irradiance (W m^-2 nm^-1) to mmol photons cm^-2 s^-1 nm^-1 (eqn 8).

    E_photon = h c / lambda, so photons per joule = lambda / (h c). With lambda in nm
    the factor 1e-10 combines nm -> m (1e-9), m^-2 -> cm^-2 (1e-4) and mol -> mmol (1e3).
    """
    i = np.asarray(irradiance_w_m2_nm, dtype=float)
    wl = np.asarray(wavelength_nm, dtype=float)
    return i * wl / (PLANCK * LIGHT_SPEED * AVOGADRO) * 1e-10
