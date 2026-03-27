"""Local virtual thrust helpers for the Python workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .keplerian import dynamics_2bp_cartesian, propagate_stm_2bp

FloatArray = NDArray[np.float64]


def _solve_ivp_checked(*args, **kwargs):
    solution = solve_ivp(*args, **kwargs)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution


@dataclass(frozen=True)
class ConwayBenchmark:
    """Conway-style fixed-impulse STM benchmark result."""

    Phi_tf: FloatArray
    Phi_rr: FloatArray
    Phi_rv: FloatArray
    Phi_vr: FloatArray
    Phi_vv: FloatArray
    M: FloatArray
    lambda_values: FloatArray
    lambda_max: float
    e_opt: FloatArray
    dV_opt_kmps: FloatArray
    dr_opt_km: FloatArray
    dr_max_km: float
    rA_tf_lin_km: FloatArray
    miss_vec_lin_km: FloatArray
    miss_lin_km: float


def compute_dvmax_from_impactors(
    mA: float,
    ms: Iterable[float] | FloatArray,
    vrel: Iterable[float] | FloatArray,
    beta: float = 1.0,
    cosgamma: Iterable[float] | FloatArray | None = None,
) -> float:
    """Upper bound on total asteroid delta-V from multiple impacts, in m/s."""

    masses = np.asarray(ms, dtype=float).reshape(-1)
    rel_speed = np.asarray(vrel, dtype=float).reshape(-1)
    align = np.ones_like(masses) if cosgamma is None else np.asarray(cosgamma, dtype=float).reshape(-1)
    delta_v = float((beta / mA) * np.sum(masses * rel_speed * align))
    return max(delta_v, 0.0)


def compute_dv_per_impact(
    mA: float,
    ms: float,
    vrel: float,
    beta: float = 1.0,
    cosgamma: float = 1.0,
) -> float:
    """Upper bound on asteroid delta-V from a single impact, in m/s."""

    delta_v = float(beta * (ms / mA) * vrel * cosgamma)
    return max(delta_v, 0.0)


def compute_amax_from_cadence(
    mA: float,
    ms: float,
    vrel: float,
    beta: float = 1.0,
    cosgamma: float = 1.0,
    dt_min: float = 1.0,
) -> float:
    """Cadence-based proxy acceleration bound in m/s^2."""

    if dt_min <= 0.0:
        raise ValueError("dt_min must be positive.")
    return compute_dv_per_impact(mA, ms, vrel, beta, cosgamma) / dt_min


def dyn_2bp_zoh(
    t: float,
    x: Iterable[float] | FloatArray,
    uk: Iterable[float] | FloatArray,
    amax: float,
    mu: float,
) -> FloatArray:
    """Two-body dynamics with constant ZOH control acceleration."""

    xdot = dynamics_2bp_cartesian(t, x, mu)
    xdot[3:] += amax * np.asarray(uk, dtype=float).reshape(3)
    return xdot


def propagate_zoh_2bp_control(
    x0: Iterable[float] | FloatArray,
    t_grid: Iterable[float] | FloatArray,
    U: Iterable[Iterable[float]] | FloatArray,
    mu: float,
    amax: float,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    method: str = "DOP853",
) -> FloatArray:
    """Propagate piecewise-constant 2BP dynamics over ``t_grid``."""

    state0 = np.asarray(x0, dtype=float).reshape(6)
    time_grid = np.asarray(t_grid, dtype=float).reshape(-1)
    controls = np.asarray(U, dtype=float)
    if controls.shape != (time_grid.size - 1, 3):
        raise ValueError("U must have shape (len(t_grid)-1, 3).")

    xhist = np.zeros((time_grid.size, 6), dtype=float)
    xhist[0] = state0
    xk = state0.copy()
    for idx in range(controls.shape[0]):
        tk = float(time_grid[idx])
        tk1 = float(time_grid[idx + 1])
        if np.isclose(tk, tk1):
            xhist[idx + 1] = xk
            continue
        solution = _solve_ivp_checked(
            lambda t, x: dyn_2bp_zoh(t, x, controls[idx], amax, mu),
            (tk, tk1),
            xk,
            method=method,
            t_eval=np.array([tk1], dtype=float),
            rtol=rtol,
            atol=atol,
        )
        xk = solution.y[:, -1]
        xhist[idx + 1] = xk
    return xhist


def conway_max_theoretical_deflection_stm(
    xA_0: Iterable[float] | FloatArray,
    rE_tf: Iterable[float] | FloatArray,
    dVmax_kmps: float,
    tf_sec: float,
    t0_sec: float,
    mu_sun_km: float,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    method: str = "DOP853",
) -> ConwayBenchmark:
    """Conway-style fixed-impulse deflection benchmark based on the 2BP STM."""

    x0 = np.asarray(xA_0, dtype=float).reshape(6)                           # Initial state of asteroid at t0_sec                               
    rE = np.asarray(rE_tf, dtype=float).reshape(3)                          # Position of Earth at tf_sec      
    duration = float(tf_sec - t0_sec)                                       # Time of flight in seconds         
    phi_tf, _ = propagate_stm_2bp(                                          # STM from t0_sec to tf_sec under 2BP dynamics, evaluated at tf_sec
        x0,
        mu_sun_km,
        (0.0, duration),
        rtol=rtol,
        atol=atol,
        method=method,
    )
    phi_rr = phi_tf[:3, :3]                             
    phi_rv = phi_tf[:3, 3:]                             
    phi_vr = phi_tf[3:, :3]                             
    phi_vv = phi_tf[3:, 3:]                             

    M = phi_rv.T @ phi_rv                               #+ phi_vv.T @ phi_vv            
    lambda_values, eigenvectors = np.linalg.eigh(M)     # Compute eigenvalues and eigenvectors of M
    idx = int(np.argmax(lambda_values))                 # Find index of largest eigenvalue
    lambda_max = float(lambda_values[idx])              # Largest eigenvalue of M, which determines the maximum deflection growth
    e_opt = eigenvectors[:, idx]                        # Optimal direction of velocity change in the asteroid's velocity space at t0_sec
    e_opt = e_opt / np.linalg.norm(e_opt)               # Normalize the optimal direction vector to have unit length

    dV_opt_kmps = float(dVmax_kmps) * e_opt              # Optimal velocity change vector in km/s, scaled by the maximum allowed delta-V magnitude
    dr_opt_km = phi_rv @ dV_opt_kmps                     # Resulting change in position at tf_sec due to the optimal velocity change, computed using the STM's phi_rv submatrix
    rA_tf_lin_km = x0[:3] + dr_opt_km                    # Linearized final position of the asteroid at tf_sec after applying the optimal velocity change, starting from the initial position x0[:3] and adding the linearized change dr_opt_km  
    miss_vec_lin_km = rA_tf_lin_km - rE                  # Linearized miss vector at tf_sec, computed as the difference between the linearized final position of the asteroid and the position of Earth at tf_sec    

    return ConwayBenchmark(
        Phi_tf=phi_tf,
        Phi_rr=phi_rr,
        Phi_rv=phi_rv,
        Phi_vr=phi_vr,
        Phi_vv=phi_vv,
        M=M,
        lambda_values=lambda_values,
        lambda_max=lambda_max,
        e_opt=e_opt,
        dV_opt_kmps=dV_opt_kmps,
        dr_opt_km=dr_opt_km,
        dr_max_km=float(np.linalg.norm(dr_opt_km)),
        rA_tf_lin_km=rA_tf_lin_km,
        miss_vec_lin_km=miss_vec_lin_km,
        miss_lin_km=float(np.linalg.norm(miss_vec_lin_km)),
    )
