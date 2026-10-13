"""Unit tests for the analysis functions. Run with:  python -m pytest  (or  python tests/test_singlet_oxygen.py)"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from singlet_oxygen import (  # noqa: E402
    apparent_quantum_yield,
    fit_decay,
    fit_decay_trimmed,
    irradiance_to_photon_flux,
    k_ffa,
    rate_of_absorbance,
    screening_factor,
    steady_state_1o2,
)


def test_k_ffa_temperature_dependence():
    assert k_ffa(22) == 1.0e8
    assert math.isclose(k_ffa(30), 1.168e8)


def test_fit_recovers_known_rate():
    t = np.array([0, 120, 300, 600, 1200, 2400, 4800], dtype=float)
    k = 1.5e-5
    fit = fit_decay(t, -k * t)
    assert math.isclose(fit.k_obs, k, rel_tol=1e-9)
    assert fit.r_squared > 0.999999


def test_steady_state():
    assert math.isclose(steady_state_1o2(1.0e-5, 22), 1.0e-13)


def test_trimming_keeps_linear_data():
    t = np.linspace(0, 3600, 7)
    fit = fit_decay_trimmed(t, -2e-5 * t)
    assert fit.used.all()


def test_trimming_drops_curved_tail():
    t = np.array([0, 600, 1200, 1800, 2400, 3000, 6000, 12000], dtype=float)
    y = -1e-5 * t
    y[-2:] = y[5]  # decay stops: late points deviate from first-order kinetics
    fit = fit_decay_trimmed(t, y)
    assert not fit.used[-1]
    assert math.isclose(fit.k_obs, 1e-5, rel_tol=1e-6)


def test_screening_factor_limits():
    sf = screening_factor([0.0, 1e-6, 1.0])
    assert sf[0] == 1.0
    assert math.isclose(sf[1], 1.0, rel_tol=1e-5)
    assert math.isclose(sf[2], 0.9 / math.log(10), rel_tol=1e-9)


def test_rate_of_absorbance_scales_with_light():
    wl = np.arange(280, 801)
    irr = np.exp(-((wl - 350) / 15) ** 2)
    a = 0.05 * np.exp(-0.01 * (wl - 280))
    r1 = rate_of_absorbance(wl, irr, a)
    r2 = rate_of_absorbance(wl, 2 * irr, a)
    assert r1 > 0 and math.isclose(r2, 2 * r1)


def test_quantum_yield_equals_reference_for_identical_sample():
    assert math.isclose(apparent_quantum_yield(1e-4, 5.0, 1e-4, 5.0), 1.0)
    assert math.isclose(apparent_quantum_yield(1e-5, 10.0, 1e-3, 5.0), 0.005)


def test_photon_flux_conversion():
    # 1 W m^-2 nm^-1 at 400 nm = 400e-9/(h c) photons/s/m2 -> mmol cm^-2 s^-1 nm^-1
    expected = 400e-9 / (6.626e-34 * 2.998e8) / 6.022e23 * 1e-4 * 1e3
    assert math.isclose(float(irradiance_to_photon_flux(1.0, 400.0)), expected, rel_tol=1e-9)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
