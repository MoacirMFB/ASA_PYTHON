"""Discrete-time linearization and indirect-method optimal control helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import root

from .keplerian import dynamics_2bp_cartesian, jacobian_2bp_cartesian

FloatArray = NDArray[np.float64]

_B_3D = np.zeros((6, 3), dtype=float)
_B_3D[3:, :] = np.eye(3)


def _as_float_array(values: Iterable[float] | FloatArray, size: int | None = None) -> FloatArray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if size is not None and array.size != size:
        raise ValueError(f"Expected {size} entries, got {array.size}.")
    return array


@dataclass(frozen=True)
class MinFuelSolution:
    """Converged indirect-shooting min-fuel transfer."""

    lam0: FloatArray
    t: FloatArray
    x_hist: FloatArray
    lam_hist: FloatArray
    u_hist: FloatArray
    residual_norm: float
    success: bool
    message: str


def linear_discrete_time_matrices(
    tk: float,
    tk_1: float,
    yk: Iterable[float] | FloatArray,
    A: Iterable[float] | FloatArray,
    B: Iterable[float] | FloatArray,
    c: Iterable[float] | FloatArray,
    uk: Iterable[float] | FloatArray | None,
    szx: int,
    szu: int,
    ode_options: object | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return exact ZOH ``(Ak, Bk, ck, xk_1)`` for constant ``A, B, c``."""

    del ode_options

    xk = np.asarray(yk, dtype=float).reshape(-1)[:szx]
    A_mat = np.asarray(A, dtype=float).reshape((szx, szx))
    B_mat = np.asarray(B, dtype=float).reshape((szx, szu))
    c_vec = np.asarray(c, dtype=float).reshape(szx)
    u_vec = np.zeros(szu, dtype=float) if uk is None else np.asarray(uk, dtype=float).reshape(szu)

    dt = float(tk_1 - tk)
    if np.isclose(dt, 0.0):
        Ak = np.eye(szx, dtype=float)
        Bk = np.zeros((szx, szu), dtype=float)
        ck = np.zeros(szx, dtype=float)
        return Ak, Bk, ck, xk.copy()

    aug = np.zeros((szx + szu + 1, szx + szu + 1), dtype=float)
    aug[:szx, :szx] = A_mat
    aug[:szx, szx : szx + szu] = B_mat
    aug[:szx, -1] = c_vec
    transition = expm(aug * dt)

    Ak = transition[:szx, :szx]
    Bk = transition[:szx, szx : szx + szu]
    ck = transition[:szx, -1]
    xk_1 = Ak @ xk + Bk @ u_vec + ck
    return Ak, Bk, ck, xk_1


def min_fuel_costate_dynamics(
    t: float,
    lam: Iterable[float] | FloatArray,
    x: Iterable[float] | FloatArray,
    mu: float,
) -> FloatArray:
    """Costate dynamics for the 3D min-fuel two-body problem (lam_dot = -J^T lam)."""

    lam_vec = _as_float_array(lam, size=6)
    jac = jacobian_2bp_cartesian(t, x, mu)
    return -(jac.T @ lam_vec)


def min_fuel_optimal_control(
    lam: Iterable[float] | FloatArray,
    umax: float,
    rho: float,
) -> FloatArray:
    """Smoothed bang-off optimal control u* = gamma * phat for one or many costates."""

    lam_array = np.asarray(lam, dtype=float)
    if lam_array.ndim == 1:
        if lam_array.size != 6:
            raise ValueError(f"Expected 6 costate entries, got {lam_array.size}.")
        lam_2d = lam_array.reshape(1, 6)
        squeeze = True
    elif lam_array.ndim == 2 and lam_array.shape[1] == 6:
        lam_2d = lam_array
        squeeze = False
    else:
        raise ValueError("Costate input must have shape (6,) or (N, 6).")

    primer = -lam_2d[:, 3:]
    p_norm = np.linalg.norm(primer, axis=1)
    gamma = 0.5 * float(umax) * (1.0 + np.tanh((p_norm - 1.0) / float(rho)))
    safe_norm = np.where(p_norm > 0.0, p_norm, 1.0)
    phat = primer / safe_norm[:, None]
    u_star = gamma[:, None] * phat
    u_star = np.where((p_norm > 0.0)[:, None], u_star, 0.0)
    return u_star[0] if squeeze else u_star


def min_fuel_state_dynamics(
    t: float,
    x: Iterable[float] | FloatArray,
    u: Iterable[float] | FloatArray,
    mu: float,
) -> FloatArray:
    """3D two-body state dynamics with control acceleration u (km/s^2)."""

    u_vec = _as_float_array(u, size=3)
    return dynamics_2bp_cartesian(t, x, mu) + _B_3D @ u_vec


def min_fuel_combined_dynamics(
    t: float,
    X: Iterable[float] | FloatArray,
    mu: float,
    umax: float,
    rho: float,
) -> FloatArray:
    """Augmented [state; costate] dynamics for indirect-shooting integration."""

    aug = _as_float_array(X, size=12)
    x = aug[:6]
    lam = aug[6:]
    u_star = min_fuel_optimal_control(lam, umax, rho)
    dx = min_fuel_state_dynamics(t, x, u_star, mu)
    dlam = min_fuel_costate_dynamics(t, lam, x, mu)
    return np.concatenate((dx, dlam))


def min_fuel_shooting_residual(
    lam0: Iterable[float] | FloatArray,
    t0: float,
    tf: float,
    x0: Iterable[float] | FloatArray,
    xf: Iterable[float] | FloatArray,
    mu: float,
    umax: float,
    rho: float,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    method: str = "DOP853",
) -> FloatArray:
    """Propagate [x; lam] from t0 to tf and return x(tf) - xf."""

    x0_vec = _as_float_array(x0, size=6)
    xf_vec = _as_float_array(xf, size=6)
    lam0_vec = _as_float_array(lam0, size=6)
    y0 = np.concatenate((x0_vec, lam0_vec))
    solution = solve_ivp(
        lambda t, y: min_fuel_combined_dynamics(t, y, mu, umax, rho),
        (float(t0), float(tf)),
        y0,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y[:6, -1] - xf_vec


def solve_min_fuel_bvp(
    t0: float,
    tf: float,
    x0: Iterable[float] | FloatArray,
    xf: Iterable[float] | FloatArray,
    mu: float,
    umax: float,
    rho: float,
    lam0_guess: Iterable[float] | FloatArray,
    *,
    n_eval: int = 200,
    root_method: str = "hybr",
    root_tol: float = 1e-10,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    ode_method: str = "DOP853",
) -> MinFuelSolution:
    """Solve the indirect-shooting BVP for the 3D min-fuel transfer."""

    x0_vec = _as_float_array(x0, size=6)
    xf_vec = _as_float_array(xf, size=6)
    lam0_vec = _as_float_array(lam0_guess, size=6)

    result = root(
        lambda lam: min_fuel_shooting_residual(
            lam, t0, tf, x0_vec, xf_vec, mu, umax, rho,
            rtol=rtol, atol=atol, method=ode_method,
        ),
        lam0_vec,
        method=root_method,
        tol=root_tol,
    )
    lam0_star = np.asarray(result.x, dtype=float)
    residual_norm = float(np.linalg.norm(result.fun))

    t_grid = np.linspace(float(t0), float(tf), int(n_eval))
    y0 = np.concatenate((x0_vec, lam0_star))
    propagation = solve_ivp(
        lambda t, y: min_fuel_combined_dynamics(t, y, mu, umax, rho),
        (t_grid[0], t_grid[-1]),
        y0,
        t_eval=t_grid,
        method=ode_method,
        rtol=rtol,
        atol=atol,
    )
    if not propagation.success:
        raise RuntimeError(propagation.message)

    x_hist = propagation.y[:6, :].T
    lam_hist = propagation.y[6:, :].T
    u_hist = min_fuel_optimal_control(lam_hist, umax, rho)

    return MinFuelSolution(
        lam0=lam0_star,
        t=propagation.t,
        x_hist=x_hist,
        lam_hist=lam_hist,
        u_hist=u_hist,
        residual_norm=residual_norm,
        success=bool(result.success),
        message=str(result.message),
    )


def min_fuel_hamiltonian(
    u_hist: Iterable[float] | FloatArray,
    lam_hist: Iterable[float] | FloatArray,
    state_hist: Iterable[float] | FloatArray,
    mu: float,
) -> FloatArray:
    """Hamiltonian H = ||u|| + lambda . f along a trajectory."""

    u_arr = np.atleast_2d(np.asarray(u_hist, dtype=float))
    lam_arr = np.atleast_2d(np.asarray(lam_hist, dtype=float))
    x_arr = np.atleast_2d(np.asarray(state_hist, dtype=float))
    if u_arr.shape[1] != 3 or lam_arr.shape[1] != 6 or x_arr.shape[1] != 6:
        raise ValueError("Expected u (N,3), lam (N,6), state (N,6).")

    r = x_arr[:, :3]
    v = x_arr[:, 3:]
    r_norm = np.linalg.norm(r, axis=1)
    if np.any(r_norm < 1e-12):
        raise ValueError("Singular state: ||r|| is too small.")
    accel = -mu * r / (r_norm**3)[:, None]
    f0 = np.concatenate((v, accel), axis=1)
    f = f0 + np.concatenate((np.zeros_like(u_arr), u_arr), axis=1)
    return np.linalg.norm(u_arr, axis=1) + np.einsum("ij,ij->i", lam_arr, f)
