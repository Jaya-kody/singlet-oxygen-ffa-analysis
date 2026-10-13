"""Apparent singlet-oxygen quantum yields relative to a reference sensitizer.

Perinaphthenone (PN) produces 1O2* with a wavelength-independent quantum yield
close to 1. Irradiating PN and the sample under the same light gives

    Phi_sample = (k_obs,sample / k_obs,PN) * (R_abs,PN / R_abs,sample) * Phi_PN   (eqn 3)

so the absolute light intensity cancels out.
"""

from __future__ import annotations

import math


def apparent_quantum_yield(k_obs_sample: float, rabs_sample: float,
                           k_obs_ref: float, rabs_ref: float,
                           phi_ref: float = 1.0) -> float:
    """Apparent 1O2* quantum yield of a sample (fraction, multiply by 100 for %)."""
    if rabs_sample <= 0 or k_obs_ref <= 0:
        return float("nan")
    return (k_obs_sample / k_obs_ref) * (rabs_ref / rabs_sample) * phi_ref


def quantum_yield_error(phi: float, rel_errors) -> float:
    """Propagate independent relative errors (e.g. se/value of each term) in quadrature."""
    return abs(phi) * math.sqrt(sum(e**2 for e in rel_errors if e == e))  # skip NaNs


def light_conversion_factor(rabs_light: float, rabs_reference_light: float) -> float:
    """Hours of the reference light (UVA) equivalent to 1 h of another light (eqns 9-10)."""
    return rabs_light / rabs_reference_light
