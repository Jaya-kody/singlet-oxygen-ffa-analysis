"""Tools for quantifying singlet oxygen with the furfuryl alcohol probe method."""

from .kinetics import DecayFit, fit_decay, fit_decay_trimmed, k_ffa, steady_state_1o2
from .quantum_yield import apparent_quantum_yield, light_conversion_factor, quantum_yield_error
from .spectra import irradiance_to_photon_flux, rate_of_absorbance, resample, screening_factor

__all__ = [
    "DecayFit",
    "fit_decay",
    "fit_decay_trimmed",
    "k_ffa",
    "steady_state_1o2",
    "apparent_quantum_yield",
    "quantum_yield_error",
    "light_conversion_factor",
    "irradiance_to_photon_flux",
    "rate_of_absorbance",
    "resample",
    "screening_factor",
]
