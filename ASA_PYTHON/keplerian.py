"""Low-level 2BP and orbital-element helpers for the virtual thrust port."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

FloatArray = NDArray[np.float64]


def _as_float_array(values: Iterable[float] | FloatArray, size: int | None = None) -> FloatArray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if size is not None and array.size != size:
        raise ValueError(f"Expected {size} entries, got {array.size}.")
    return array


def _solve_ivp_checked(*args, **kwargs):
    solution = solve_ivp(*args, **kwargs)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution


def dynamics_2bp_cartesian(t: float, x: Iterable[float] | FloatArray, mu: float) -> FloatArray:
    """Two-body Cartesian dynamics with state shape ``(6,)`` in km and km/s."""

    state = _as_float_array(x, size=6)
    r = state[:3]
    v = state[3:]
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-12:
        raise ValueError("Singular state: ||r|| is too small.")
    accel = -mu * r / (r_norm**3)
    return np.concatenate((v, accel))


def jacobian_2bp_cartesian(t: float, x: Iterable[float] | FloatArray, mu: float) -> FloatArray:
    """Jacobian of the unperturbed two-body Cartesian dynamics."""

    state = _as_float_array(x, size=6)
    r = state[:3]
    r2 = float(r @ r)
    r_norm = np.sqrt(r2)
    if r_norm < 1e-12:
        raise ValueError("Singular state: ||r|| is too small.")
    r3 = r2 * r_norm
    r5 = r2 * r3
    dadr = mu * ((3.0 * np.outer(r, r) / r5) - (np.eye(3) / r3))
    jac = np.zeros((6, 6), dtype=float)
    jac[:3, 3:] = np.eye(3)
    jac[3:, :3] = dadr
    return jac


def augmented_dynamics_2bp_stm(t: float, x_aug: Iterable[float] | FloatArray, mu: float) -> FloatArray:
    """State-plus-STM dynamics using column-major STM packing."""

    y = _as_float_array(x_aug)
    x = y[:6]
    phi = y[6:].reshape((6, 6), order="F")
    x_dot = dynamics_2bp_cartesian(t, x, mu)
    phi_dot = jacobian_2bp_cartesian(t, x, mu) @ phi
    return np.concatenate((x_dot, phi_dot.reshape(-1, order="F")))


def propagate_stm_2bp(
    x0: Iterable[float] | FloatArray,
    mu: float,
    tspan: tuple[float, float] | list[float] | FloatArray,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    method: str = "DOP853",
) -> tuple[FloatArray, FloatArray]:
    """Propagate the 2BP state and final STM over ``tspan``."""

    x0_array = _as_float_array(x0, size=6)
    t0, tf = float(tspan[0]), float(tspan[-1])
    if np.isclose(t0, tf):
        return np.eye(6, dtype=float), x0_array.copy()

    phi0 = np.eye(6, dtype=float)
    y0 = np.concatenate((x0_array, phi0.reshape(-1, order="F")))
    solution = _solve_ivp_checked(
        lambda t, y: augmented_dynamics_2bp_stm(t, y, mu),
        (t0, tf),
        y0,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    y_final = solution.y[:, -1]
    x_final = y_final[:6]
    phi_final = y_final[6:].reshape((6, 6), order="F")
    return phi_final, x_final


def propagate_two_body(
    x0: Iterable[float] | FloatArray,
    t_eval: Iterable[float] | FloatArray,
    mu: float,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    method: str = "DOP853",
) -> tuple[FloatArray, FloatArray]:
    """Propagate a 2BP state history on the requested time grid."""

    x0_array = _as_float_array(x0, size=6)
    t_grid = _as_float_array(t_eval)
    if t_grid.size == 0:
        raise ValueError("t_eval must contain at least one time sample.")
    if t_grid.size == 1:
        return t_grid.copy(), x0_array.reshape(1, 6)

    solution = _solve_ivp_checked(
        lambda t, x: dynamics_2bp_cartesian(t, x, mu),
        (float(t_grid[0]), float(t_grid[-1])),
        x0_array,
        method=method,
        t_eval=t_grid,
        rtol=rtol,
        atol=atol,
    )
    return solution.t, solution.y.T


def solve_keplers_equation(
    mean_anomaly: float,
    eccentricity: float,
    *,
    tol: float = 1e-13,
    max_iter: int = 50,
) -> float:
    """Solve ``E - e sin(E) = M`` with Newton iterations."""

    mean = float(np.mod(mean_anomaly, 2.0 * np.pi))
    if mean > np.pi:
        mean -= 2.0 * np.pi
    ecc = float(eccentricity)
    if ecc < 0.0 or ecc >= 1.0:
        raise ValueError("This helper only supports elliptic orbits with 0 <= e < 1.")

    eccentric = mean if ecc < 0.8 else np.pi
    for _ in range(max_iter):
        residual = eccentric - ecc * np.sin(eccentric) - mean
        slope = 1.0 - ecc * np.cos(eccentric)
        step = residual / slope
        eccentric -= step
        if abs(step) < tol:
            return eccentric
    raise RuntimeError("Kepler solver failed to converge.")


def true_anomaly_from_eccentric_anomaly(eccentricity: float, eccentric_anomaly: float) -> float:
    """Convert eccentric anomaly to true anomaly for elliptic orbits."""

    ecc = float(eccentricity)
    half = 0.5 * float(eccentric_anomaly)
    numerator = np.sqrt(1.0 + ecc) * np.sin(half)
    denominator = np.sqrt(1.0 - ecc) * np.cos(half)
    return 2.0 * np.arctan2(numerator, denominator)


def _pqw_to_ijk_rotation(raan: float, inclination: float, arg_peri: float) -> FloatArray:
    cos_O, sin_O = np.cos(raan), np.sin(raan)
    cos_i, sin_i = np.cos(inclination), np.sin(inclination)
    cos_w, sin_w = np.cos(arg_peri), np.sin(arg_peri)
    return np.array(
        [
            [cos_O * cos_w - sin_O * sin_w * cos_i, -cos_O * sin_w - sin_O * cos_w * cos_i, sin_O * sin_i],
            [sin_O * cos_w + cos_O * sin_w * cos_i, -sin_O * sin_w + cos_O * cos_w * cos_i, -cos_O * sin_i],
            [sin_w * sin_i, cos_w * sin_i, cos_i],
        ],
        dtype=float,
    )


def coe_to_cartesian(
    x_coe: Iterable[float] | FloatArray,
    mu: float,
    *,
    use_true_anomaly: bool = False,
) -> FloatArray:
    """Convert ``[a, e, i, RAAN, omega, M_or_nu]`` to Cartesian state."""

    elements = np.asarray(x_coe, dtype=float)
    squeeze_output = elements.ndim == 1
    elements_2d = np.atleast_2d(elements)
    if elements_2d.shape[1] != 6:
        raise ValueError("Orbital element input must have six columns.")

    states = np.zeros((elements_2d.shape[0], 6), dtype=float)
    for idx, row in enumerate(elements_2d):
        semi_major_axis, eccentricity, inclination, raan, arg_peri, anomaly = row
        if use_true_anomaly:
            true_anomaly = anomaly
        else:
            eccentric_anomaly = solve_keplers_equation(anomaly, eccentricity)
            true_anomaly = true_anomaly_from_eccentric_anomaly(eccentricity, eccentric_anomaly)

        semilatus_rectum = semi_major_axis * (1.0 - eccentricity**2)
        radius = semilatus_rectum / (1.0 + eccentricity * np.cos(true_anomaly))
        r_pqw = np.array(
            [radius * np.cos(true_anomaly), radius * np.sin(true_anomaly), 0.0],
            dtype=float,
        )
        v_pqw = np.sqrt(mu / semilatus_rectum) * np.array(
            [-np.sin(true_anomaly), eccentricity + np.cos(true_anomaly), 0.0],
            dtype=float,
        )
        rotation = _pqw_to_ijk_rotation(raan, inclination, arg_peri)
        states[idx, :3] = rotation @ r_pqw
        states[idx, 3:] = rotation @ v_pqw

    return states[0] if squeeze_output else states
