"""Pseudo-first-order kinetics of the furfuryl alcohol (FFA) singlet-oxygen probe.

FFA reacts with singlet oxygen (1O2*) much faster than it decays any other way, and
[1O2*] stays at steady state during an irradiation, so FFA decays exponentially:

    ln([FFA]t / [FFA]0) = -k_obs * t                                    (eqn 1)
    [1O2*]ss = k_obs / k_FFA                                            (eqn 2)

with the temperature-dependent second-order rate constant

    k_FFA [M^-1 s^-1] = 1.0e8 + 2.1e6 * (T[degC] - 22)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def k_ffa(temperature_c: float) -> float:
    """Second-order rate constant of FFA + 1O2* (M^-1 s^-1) at a given temperature."""
    return 1.0e8 + 2.1e6 * (temperature_c - 22.0)


@dataclass
class DecayFit:
    """Result of a pseudo-first-order fit to an FFA decay curve."""

    k_obs: float  # s^-1, positive for a decaying probe
    k_obs_se: float  # standard error of k_obs, s^-1
    intercept: float
    r_squared: float
    n_points: int
    used: np.ndarray = field(repr=False)  # boolean mask of points kept in the fit

    @property
    def slope(self) -> float:
        return -self.k_obs


def _linear_fit(t: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Ordinary least squares y = m t + b. Returns (m, b, r^2, standard error of m)."""
    n = len(t)
    m, b = np.polyfit(t, y, 1)
    residuals = y - (m * t + b)
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    if n > 2:
        se = np.sqrt(ss_res / (n - 2) / np.sum((t - t.mean()) ** 2))
    else:
        se = float("nan")
    return float(m), float(b), r2, float(se)


def fit_decay(time_s, ln_ratio, mask=None) -> DecayFit:
    """Fit ln([FFA]t/[FFA]0) against time (seconds) to get k_obs.

    Parameters
    ----------
    time_s : array-like
        Irradiation time in seconds.
    ln_ratio : array-like
        ln([FFA]t / [FFA]0) at each time point (0 at t = 0, negative afterwards).
    mask : array-like of bool, optional
        Points to include. Defaults to all points.
    """
    t = np.asarray(time_s, dtype=float)
    y = np.asarray(ln_ratio, dtype=float)
    used = np.ones_like(t, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if used.sum() < 3:
        raise ValueError("Need at least 3 points for a decay fit.")
    m, b, r2, se = _linear_fit(t[used], y[used])
    return DecayFit(k_obs=-m, k_obs_se=se, intercept=b, r_squared=r2,
                    n_points=int(used.sum()), used=used)


def fit_decay_trimmed(time_s, ln_ratio, max_slope_change: float = 0.25,
                      min_points: int = 4) -> DecayFit:
    """Fit FFA decay, dropping late time points that break pseudo-first-order kinetics.

    Under weak light (e.g. fluorescent tubes) the late points can curve away from a
    straight line. Following the approach used in the paper, the last time point is
    removed if doing so changes the fitted slope by more than ``max_slope_change``
    (25 % by default). This is repeated until removing the last point no longer
    changes the slope by more than the threshold, or only ``min_points`` remain.
    """
    t = np.asarray(time_s, dtype=float)
    y = np.asarray(ln_ratio, dtype=float)
    order = np.argsort(t)
    t, y = t[order], y[order]

    n_keep = len(t)
    while n_keep > min_points:
        m_all = _linear_fit(t[:n_keep], y[:n_keep])[0]
        m_drop = _linear_fit(t[: n_keep - 1], y[: n_keep - 1])[0]
        if m_drop == 0:
            break
        change = abs(m_all - m_drop) / abs(m_drop)
        if change > max_slope_change:
            n_keep -= 1
        else:
            break

    mask_sorted = np.zeros(len(t), dtype=bool)
    mask_sorted[:n_keep] = True
    mask = np.empty_like(mask_sorted)
    mask[order] = mask_sorted
    return fit_decay(time_s, ln_ratio, mask=mask)


def steady_state_1o2(k_obs: float, temperature_c: float) -> float:
    """Steady-state singlet oxygen concentration (M) from k_obs (s^-1), eqn 2."""
    return k_obs / k_ffa(temperature_c)
