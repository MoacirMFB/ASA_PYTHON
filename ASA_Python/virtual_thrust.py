"""Runnable Python version of the virtual thrust workflow.

This file plays the role of the main workflow body, but split into small
Python helpers so the setup, plotting, and SCP logic are easier to follow.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import cvxpy as cp

# ============================================================================
# Imports and runtime plumbing
# ============================================================================clear
# This top block supports both module execution (`python -m ASA_Python...`)
# and direct execution from the file path.

try:
    from . import bodies, control, interception, keplerian, virtual_thrust_helpers
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from ASA_Python import bodies, control, interception, keplerian, virtual_thrust_helpers


# ============================================================================
# Global constants
# ============================================================================

# Keep the same "30 days per month" convention used throughout this workflow.
SECONDS_PER_MONTH = 30.0 * 86400.0


# ============================================================================
# Configuration and result containers
# ============================================================================


@dataclass
class VirtualThrustConfig:
    """Main scenario/settings container for the workflow.

    This dataclass keeps the main workflow settings grouped in one place.
    """

    reltol: float = 1e-12          # ODE relative tolerance for propagation and nonlinear checks
    abstol: float = 1e-12          # ODE absolute tolerance for propagation and nonlinear checks
    run_scp: bool = False           #  Whether to run the SCP optimization or just do the nominal propagation and plotting
    use_opt_dv_dir: bool = True    # Whether to use the optimal deflection direction from the linearized problem for the nominal propagation (vs. pure anti-velocity)

    asteroid_name: str = "Apophis"  # Name of the asteroid to target, must be in the asteroid catalog
    rho_ast_kg_m3: float = 2400.0   # Assumed asteroid density for mass and deflection calculations
    asteroid_diameter_m: float = 100.0  
    beta: float = 1.0               # Momentum enhancement factor for the kinetic impactor deflection calculation
    force_impact: bool = True       # Whether to force the zero-month state to be an impact, for easier visualization of the deflection
    lead_time_years: int = 3        # Lead time in years for the scenario, used to set the month grid and environment span
    t0_months: int = 24             # Initial month for the nominal propagation, relative to the CA month (0 means start at the CA state, negative means start earlier)

    env_years: float = 4.0          # Environment span in years
    env_step_min: float = 10.0      # Minimum environment step size in days, used for the dense Earth and asteroid histories

    mass_sc_kg: float = 1000.0      # Spacecraft mass in kg, used for the kinetic impactor deflection calculation
    vrel_use_mps: float = 10_000.0  # Relative velocity for the impact, used in the deflection calculation
    n_impactors: int = 5            # Number of impactors to simulate in the deflection calculation, used to compute `dvmax` for the scenario
    cos_gamma: float = 1.0          # Assumed cosine of the angle between the deflection direction and the asteroid velocity for initial proxies
    cadence_days: float = 10.0      # Assumed minimum days between impactors in the simulation

    n_segments: int = 750           # Number of segments in the SCP trajectory discretization
    kmax: int = 50                  # Maximum number of SCP iterations  
    delta_u0: float = 0.75          # Initial trust-region size for the control update in the SCP loop, as a fraction of the max control `amax`
    eta_good: float = 0.25          # Threshold for accepting a candidate control as a "good" step in the trust-region logic
    eta_great: float = 0.75         # Threshold for accepting a candidate control as a "great" step in the trust-region logic
    shrink: float = 0.5             # Factor to shrink the trust region when a step is rejected
    expand: float = 1.5             # Factor to expand the trust region when a step is accepted as "great"

    def months_back(self) -> np.ndarray:
        """Return the month grid used for impact-relative state extraction."""

        return np.arange(0, 12 * self.lead_time_years + 1, dtype=float)

    def ode_kwargs(self) -> dict[str, Any]:
        """Return ODE tolerance settings as one small helper bundle."""

        return {"rtol": self.reltol, "atol": self.abstol}


@dataclass
class VirtualThrustRunResult:
    """Primary outputs from the runnable Python workflow.

    This keeps the main outputs grouped in one explicit result object.
    """

    config: VirtualThrustConfig
    earth: bodies.CelestialBody
    sun: bodies.CelestialBody
    asteroid: interception.AsteroidRecord
    ca: interception.ClosestApproach
    moid: interception.ClosestApproach
    mbi_states: list[interception.MBIState]
    benchmark: virtual_thrust_helpers.ConwayBenchmark
    asteroid_mass_kg: float
    close_approach_distance_km: float
    amax_mps2: float
    amax_kmps2: float
    dvmax_mps: float
    dv1_mps: float
    tau_budget_s: float
    t_grid_s: np.ndarray
    warm_start_control: np.ndarray
    scp: "SCPResult | None"
    figures: dict[str, go.Figure]
    scp_status: str


@dataclass
class DiscreteLinearization:
    """Nominal propagation and interval-wise discrete linearized dynamics.

    This stores the `X_nom`, `Ak`, `Bk`, and `ck` arrays built before solving
    the convex SCP subproblem.
    """

    X_nom: np.ndarray
    Ak: np.ndarray
    Bk: np.ndarray
    ck: np.ndarray


@dataclass
class SCPSubproblemResult:
    """Result of one convex SCP subproblem solve.

    This is the direct CVXPY result of one convex subproblem solve.
    """

    x: np.ndarray
    u: np.ndarray
    tau_used_s: float
    predicted_improvement: float
    status: str
    solver: str


@dataclass
class SCPResult:
    """Final SCP optimization summary.

    This collects the accepted control, final nonlinear propagation, and
    iteration history from the SCP loop.
    """

    success: bool
    converged: bool
    solver: str | None
    status: str
    iterations: list[dict[str, Any]]
    U_opt: np.ndarray | None
    X_opt: np.ndarray | None
    miss_opt_km: float | None
    accepted_steps: int


# ============================================================================
# Small utility helpers
# ============================================================================

def _select_mbi_state(mbi_states: list[interception.MBIState], month: float) -> interception.MBIState:
    """Pick one MBI state by month value."""

    for state in mbi_states:
        if np.isclose(state.month, month):
            return state
    raise ValueError(f"Requested month {month} not found in MBI state set.")


def _make_hover_data(states: np.ndarray, t_hist_s: np.ndarray) -> np.ndarray:
    """Build shared hover values for Plotly trajectory traces."""

    radius_km = np.linalg.norm(states[:, :3], axis=1)
    return np.column_stack((t_hist_s / 86400.0, radius_km, states[:, 2]))


def _add_body_trace(
    fig: go.Figure,
    name: str,
    states: np.ndarray,
    t_hist_s: np.ndarray,
    color: str,
) -> None:
    """Add one body trajectory plus start/end markers to a 2D figure."""

    hover_data = _make_hover_data(states, t_hist_s)
    fig.add_trace(
        go.Scatter(
            x=states[:, 0],
            y=states[:, 1],
            mode="lines",
            name=name,
            line={"color": color, "width": 5},
            customdata=hover_data,
            hovertemplate=(
                f"{name}<br>"
                "t = %{customdata[0]:.2f} days<br>"
                "r = %{customdata[1]:.3e} km<br>"
                "x = %{x:.3e} km<br>"
                "y = %{y:.3e} km<br>"
                "z = %{customdata[2]:.3e} km<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[states[0, 0]],
            y=[states[0, 1]],
            mode="markers",
            name=f"{name} start",
            marker={"color": color, "size": 5, "symbol": "circle"},
            hovertemplate=f"{name} start<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[states[-1, 0]],
            y=[states[-1, 1]],
            mode="markers",
            name=f"{name} end",
            marker={"color": color, "size": 6, "symbol": "diamond"},
            hovertemplate=f"{name} end<extra></extra>",
        )
    )


def _add_state_marker(
    fig: go.Figure,
    label: str,
    state: np.ndarray,
    color: str,
    symbol: str,
) -> None:
    """Add one special marker such as `t0` or close approach."""

    fig.add_trace(
        go.Scatter(
            x=[state[0]],
            y=[state[1]],
            mode="markers",
            name=label,
            marker={"color": color, "size": 7, "symbol": symbol},
            hovertemplate=(
                f"{label}<br>"
                "x = %{x:.3e} km<br>"
                "y = %{y:.3e} km<br>"
                f"z = {state[2]:.3e} km<extra></extra>"
            ),
        )
    )


# ============================================================================
# Plotting helpers
# ============================================================================
# These helpers build the interactive Plotly figures used by the workflow.

def build_trajectory_figure(
    bodies: list[dict[str, object]],
    xA_t0: np.ndarray,
    xE_t0: np.ndarray,
    xE_tf: np.ndarray,
    xA_tf: np.ndarray,
    title: str,
) -> go.Figure:
    """Build the interactive Plotly version of the main orbit plot in 2D."""

    fig = go.Figure()
    colors = ["#1f77b4", "#d95f02", "#2ca02c", "#9467bd"]
    for idx, body in enumerate(bodies):
        _add_body_trace(
            fig,
            str(body["name"]),
            np.asarray(body["X_hist"], dtype=float),
            np.asarray(body["t_hist"], dtype=float),
            colors[idx % len(colors)],
        )

    _add_state_marker(fig, "Earth @ t0", xE_t0, "#1f77b4", "x")
    _add_state_marker(fig, "Asteroid @ t0", xA_t0, "#d95f02", "x")
    _add_state_marker(fig, "Earth @ CA", xE_tf, "#1f77b4", "cross")
    _add_state_marker(fig, "Asteroid @ CA", xA_tf, "#d95f02", "cross")

    fig.update_layout(
        title=title,
        xaxis_title="x [km]",
        yaxis_title="y [km]",
        xaxis={"scaleanchor": "y", "scaleratio": 1},
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
        template="plotly_white",
    )
    return fig


def build_separation_figure(
    earth_states: np.ndarray,
    asteroid_states: np.ndarray,
    t_hist_s: np.ndarray,
    ca: interception.ClosestApproach,
    moid: interception.ClosestApproach,
    *,
    title: str = "Earth-Asteroid Separation History",
    time_axis_label: str = "Time [years]",
) -> go.Figure:
    """Build the interactive Earth-asteroid separation history plot."""

    n_sync = min(earth_states.shape[0], asteroid_states.shape[0], t_hist_s.size)
    separation_km = np.linalg.norm(earth_states[:n_sync, :3] - asteroid_states[:n_sync, :3], axis=1)
    t_years = t_hist_s[:n_sync] / (365.25 * 86400.0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t_years,
            y=separation_km,
            mode="lines",
            name="Sampled separation",
            line={"color": "#4c78a8", "width": 2},
            hovertemplate="t = %{x:.3f} years<br>d = %{y:.3e} km<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[ca.timeEarth / (365.25 * 86400.0)],
            y=[ca.d_km],
            mode="markers",
            name="Closest approach",
            marker={"size": 9, "color": "#e45756", "symbol": "diamond"},
            hovertemplate="CA<br>t = %{x:.3f} years<br>d = %{y:.3e} km<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[moid.timeEarth / (365.25 * 86400.0)],
            y=[moid.d_km],
            mode="markers",
            name="Sampled MOID",
            marker={"size": 9, "color": "#72b7b2", "symbol": "circle"},
            hovertemplate="MOID<br>t = %{x:.3f} years<br>d = %{y:.3e} km<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=time_axis_label,
        yaxis_title="Separation [km]",
        template="plotly_white",
    )
    return fig


def build_control_history_figure(
    t_grid_s: np.ndarray,
    U0: np.ndarray,
    amax_kmps2: float,
    dt_seg_s: float,
    dvmax_mps: float,
    *,
    title: str,
) -> go.Figure:
    """Build the control-history plot for SCP results.

    The three panels mirror the usual post-processing view:
    normalized control, physical acceleration, and accumulated delta-V.
    """

    t_months = t_grid_s / SECONDS_PER_MONTH
    u_stair = np.vstack((U0, U0[-1]))
    a_stair_mmps2 = 1e6 * amax_kmps2 * u_stair
    u_norm = np.linalg.norm(u_stair, axis=1)
    a_norm = np.linalg.norm(a_stair_mmps2, axis=1)

    dv_cum_mps = np.concatenate(
        (
            [0.0],
            1000.0 * np.cumsum(np.linalg.norm(amax_kmps2 * U0, axis=1) * dt_seg_s),
        )
    )

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Normalized control",
            "Physical acceleration",
            "Accumulated delta-V",
        ),
    )

    for idx, axis_label in enumerate(("u_x", "u_y", "u_z")):
        fig.add_trace(
            go.Scatter(
                x=t_months,
                y=u_stair[:, idx],
                mode="lines",
                name=axis_label,
                line={"shape": "hv"},
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=t_months,
            y=u_norm,
            mode="lines",
            name="||u||",
            line={"color": "black", "width": 2, "dash": "dash", "shape": "hv"},
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=1.0, line={"color": "#d62728", "dash": "dot"}, row=1, col=1)

    for idx, axis_label in enumerate(("a_x", "a_y", "a_z")):
        fig.add_trace(
            go.Scatter(
                x=t_months,
                y=a_stair_mmps2[:, idx],
                mode="lines",
                name=axis_label,
                line={"shape": "hv"},
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=t_months,
            y=a_norm,
            mode="lines",
            name="||a|| [mm/s^2]",
            line={"color": "black", "width": 2, "dash": "dash", "shape": "hv"},
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=1e6 * amax_kmps2, line={"color": "#d62728", "dash": "dot"}, row=2, col=1)

    fig.add_trace(
        go.Scatter(
            x=t_months,
            y=dv_cum_mps,
            mode="lines",
            name="delta-V cumulative",
            line={"color": "#4c78a8", "width": 2, "shape": "hv"},
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=dvmax_mps, line={"color": "#d62728", "dash": "dot"}, row=3, col=1)

    fig.update_yaxes(title_text="u [-]", row=1, col=1)
    fig.update_yaxes(title_text="a [mm/s^2]", row=2, col=1)
    fig.update_yaxes(title_text="delta-V [m/s]", row=3, col=1)
    fig.update_xaxes(title_text="Time [months] (CA at 0)", row=3, col=1)
    fig.update_layout(title=title, template="plotly_white", height=900)
    return fig


def build_optimized_trajectory_figure(
    bodies: list[dict[str, object]],
    X_opt: np.ndarray,
    t_grid_s: np.ndarray,
    title: str,
) -> go.Figure:
    """Overlay the optimized asteroid trajectory on the 2D orbit plot."""

    fig = go.Figure()
    colors = ["#1f77b4", "#d95f02", "#2ca02c", "#9467bd"]
    for idx, body in enumerate(bodies):
        _add_body_trace(
            fig,
            str(body["name"]),
            np.asarray(body["X_hist"], dtype=float),
            np.asarray(body["t_hist"], dtype=float),
            colors[idx % len(colors)],
        )

    _add_body_trace(fig, "Optimized asteroid", X_opt, t_grid_s, "#2ca02c")
    fig.add_trace(
        go.Scatter(
            x=X_opt[:-1, 0],
            y=X_opt[:-1, 1],
            mode="markers",
            name="Control samples",
            marker={
                "size": 4,
                "color": t_grid_s[:-1] / SECONDS_PER_MONTH,
                "colorscale": "Turbo",
                "colorbar": {"title": "Time [months]"},
            },
            hovertemplate=(
                "Control sample<br>"
                "t = %{marker.color:.3f} months<br>"
                "x = %{x:.3e} km<br>"
                "y = %{y:.3e} km<br>"
                "xy projection<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="x [km]",
        yaxis_title="y [km]",
        xaxis={"scaleanchor": "y", "scaleratio": 1},
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
        template="plotly_white",
    )
    return fig


# ============================================================================
# Geometry reconstruction helpers
# ============================================================================
# When `force_impact=True`, the zero-month state is forced to be an impact.
# This helper reconstructs the displayed histories backward from that endpoint
# so the plots match the printed close-approach values.

def _build_forced_impact_histories(
    env_tspan_s: np.ndarray,
    xE_tf: np.ndarray,
    xA_tf: np.ndarray,
    mu_sun_km: float,
    *,
    rtol: float,
    atol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct Earth and asteroid histories backward from the forced CA state."""

    t_rel_desc = -np.asarray(env_tspan_s, dtype=float)
    _, earth_desc = keplerian.propagate_two_body(
        xE_tf,
        t_rel_desc,
        mu_sun_km,
        rtol=rtol,
        atol=atol,
    )
    _, asteroid_desc = keplerian.propagate_two_body(
        xA_tf,
        t_rel_desc,
        mu_sun_km,
        rtol=rtol,
        atol=atol,
    )
    return t_rel_desc[::-1], earth_desc[::-1], asteroid_desc[::-1]


# ============================================================================
# SCP linearization / convex subproblem helpers
# ============================================================================
# This section contains the SCP helper chain:
# 1. nonlinear nominal propagation
# 2. interval-by-interval linearization/discretization
# 3. convex subproblem solve
# 4. nonlinear check of the candidate control
# 5. trust-region accept/reject update

def _control_injection_matrix(amax_kmps2: float) -> np.ndarray:
    """Map dimensionless control into physical Cartesian acceleration."""

    return np.vstack((np.zeros((3, 3)), np.eye(3))) * amax_kmps2


def _build_discrete_linearization(
    xA_0: np.ndarray,
    t_grid_s: np.ndarray,
    U_nom: np.ndarray,
    mu_sun_km: float,
    amax_kmps2: float,
    *,
    rtol: float,
    atol: float,
) -> DiscreteLinearization:
    """Propagate the current nominal control and build `Ak`, `Bk`, `ck`.

    This first computes `X_nom`, then builds one discrete linear model per
    ZOH interval.
    """

    del rtol, atol

    X_nom = virtual_thrust_helpers.propagate_zoh_2bp_control(xA_0, t_grid_s, U_nom, mu_sun_km, amax_kmps2)
    n_intervals = U_nom.shape[0]
    szx, szu = 6, 3
    Ak = np.zeros((n_intervals, szx, szx))
    Bk = np.zeros((n_intervals, szx, szu))
    ck = np.zeros((n_intervals, szx))
    Bctrl = _control_injection_matrix(amax_kmps2)

    for k in range(n_intervals):
        xk_nom = X_nom[k]
        uk_nom = U_nom[k]
        Acont = keplerian.jacobian_2bp_cartesian(0.0, xk_nom, mu_sun_km)
        f0 = keplerian.dynamics_2bp_cartesian(0.0, xk_nom, mu_sun_km)
        f0[3:] += amax_kmps2 * uk_nom  # add the nominal ZOH control acceleration to vdot
        ccont = f0 - Acont @ xk_nom - Bctrl @ uk_nom  # affine remainder at the nominal point

        Phi0 = np.eye(szx)
        Y0aug = np.concatenate(
            (
                xk_nom,
                Phi0.reshape(-1, order="F"),  # keep column-major STM packing
                np.zeros(szx * szu),
                np.zeros(szx),
            )
        )
        Ak[k], Bk[k], ck[k], _ = control.linear_discrete_time_matrices(
            float(t_grid_s[k]),
            float(t_grid_s[k + 1]),
            Y0aug,
            Acont,
            Bctrl,
            ccont,
            uk_nom,
            szx,
            szu,
        )
    return DiscreteLinearization(X_nom=X_nom, Ak=Ak, Bk=Bk, ck=ck)


def _candidate_cvxpy_solvers() -> list[tuple[str, dict[str, Any]]]:
    """Return the preferred cvxpy solver order for this problem."""

    if cp is None:
        raise RuntimeError("cvxpy is required for the SCP optimization but is not installed.")

    installed = set(cp.installed_solvers())
    candidates: list[tuple[str, dict[str, Any]]] = []
    if "MOSEK" in installed:
        candidates.append(("MOSEK", {}))
    if "CLARABEL" in installed:
        candidates.append(("CLARABEL", {}))
    if "ECOS" in installed:
        candidates.append(("ECOS", {"abstol": 1e-8, "reltol": 1e-8, "feastol": 1e-8}))
    if "SCS" in installed:
        candidates.append(("SCS", {"eps": 1e-5, "max_iters": 20_000}))
    return candidates


def _solve_problem_with_fallback(problem: cp.Problem) -> tuple[str, str]:
    """Solve with the best available cvxpy solver, with graceful fallback."""

    if cp is None:
        raise RuntimeError("cvxpy is required for the SCP optimization but is not installed.")

    attempts: list[str] = []
    for solver, solver_options in _candidate_cvxpy_solvers():
        try:
            problem.solve(solver=solver, warm_start=True, verbose=False, **solver_options)
        except Exception as exc:
            attempts.append(f"{solver}: {type(exc).__name__}")
            continue

        status = str(problem.status)
        if status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            solver_name = problem.solver_stats.solver_name if problem.solver_stats is not None else solver
            return solver_name or solver, status
        attempts.append(f"{solver}: {status}")

    installed = ", ".join(cp.installed_solvers()) or "none"
    attempt_text = "; ".join(attempts) or "no compatible solver installed"
    raise RuntimeError(f"No cvxpy solver succeeded. Installed: {installed}. Attempts: {attempt_text}")


def _format_cvx_status(status: str) -> str:
    """Map CVXPY status strings to short terminal labels."""

    status_map = {
        cp.OPTIMAL: "Solved",
        cp.OPTIMAL_INACCURATE: "Solved (Inaccurate)",
        cp.UNBOUNDED: "Unbounded",
        cp.UNBOUNDED_INACCURATE: "Unbounded (Inaccurate)",
        cp.INFEASIBLE: "Infeasible",
        cp.INFEASIBLE_INACCURATE: "Infeasible (Inaccurate)",
    }
    return status_map.get(status, str(status))


def _print_scp_iteration(
    iteration: int,
    miss_nom_km: float,
    miss_new_km: float,
    rho: float,
    delta_u_step: float,
    solver_status: str,
) -> None:
    """Print one SCP progress line immediately to the terminal."""

    print(
        f"SCP {iteration:2d} | "
        f"miss_nom={miss_nom_km:.3f} km -> miss_new={miss_new_km:.3f} km | "
        f"rho={rho:.3f} | Delta_u={delta_u_step:.3f} | cvx={_format_cvx_status(solver_status)}",
        flush=True,
    )


def _solve_scp_subproblem(
    Ak: np.ndarray,
    Bk: np.ndarray,
    ck: np.ndarray,
    xA_0: np.ndarray,
    U_nom: np.ndarray,
    R_nom: np.ndarray,
    rnom_tf: np.ndarray,
    dt_seg_s: float,
    tau_budget_s: float,
    delta_u: float,
) -> SCPSubproblemResult:
    """Build and solve one convex SCP subproblem in cvxpy.

    It uses the discrete linearized dynamics plus the same control, budget,
    and trust-region constraints as the workflow.
    """

    n_intervals = U_nom.shape[0]
    x = cp.Variable((n_intervals + 1, 6))
    u = cp.Variable((n_intervals, 3))

    tau_used = cp.sum(cp.norm(u, 2, axis=1)) * dt_seg_s
    objective = cp.Maximize(R_nom @ (x[n_intervals, :3] - rnom_tf))  # linearized terminal objective
    constraints: list[cp.Constraint] = [x[0, :] == xA_0]

    for k in range(n_intervals):
        constraints.append(x[k + 1, :] == Ak[k] @ x[k, :] + Bk[k] @ u[k, :] + ck[k])

    constraints.append(cp.norm(u, 2, axis=1) <= 1.0)
    constraints.append(tau_used <= tau_budget_s)
    constraints.append(cp.norm(u - U_nom, 2, axis=1) <= delta_u)

    problem = cp.Problem(objective, constraints)
    solver_name, status = _solve_problem_with_fallback(problem)
    if x.value is None or u.value is None:
        raise RuntimeError("cvxpy returned no primal solution.")

    x_value = np.asarray(x.value, dtype=float)
    u_value = np.asarray(u.value, dtype=float)
    predicted_improvement = float(R_nom @ (x_value[-1, :3] - rnom_tf))
    tau_used_s = float(np.sum(np.linalg.norm(u_value, axis=1)) * dt_seg_s)
    return SCPSubproblemResult(
        x=x_value,
        u=u_value,
        tau_used_s=tau_used_s,
        predicted_improvement=predicted_improvement,
        status=status,
        solver=solver_name,
    )


def _run_scp_optimization(
    xA_0: np.ndarray,
    rE_tf: np.ndarray,
    t_grid_s: np.ndarray,
    U0: np.ndarray,
    mu_sun_km: float,
    amax_kmps2: float,
    tau_budget_s: float,
    cfg: VirtualThrustConfig,
) -> SCPResult:
    """Run the full SCP loop in Python.

    High-level flow:
    1. nonlinear propagation of the current nominal control
    2. discrete linearization about that propagation
    3. convex CVXPY subproblem solve
    4. nonlinear evaluation of the candidate control
    5. trust-region accept/reject logic using `rho`

    This function runs the main iterative SCP loop.
    """

    if cp is None:
        return SCPResult(
            success=False,
            converged=False,
            solver=None,
            status="SCP failed: cvxpy is not installed.",
            iterations=[],
            U_opt=None,
            X_opt=None,
            miss_opt_km=None,
            accepted_steps=0,
        )

    delta_u = cfg.delta_u0
    dt_seg_s = float(t_grid_s[1] - t_grid_s[0])
    U_nom = U0.copy()
    iterations: list[dict[str, Any]] = []
    solver_used: str | None = None
    accepted_steps = 0
    converged = False

    print("\n=== SCP start (discrete dynamics) ===", flush=True)

    for iteration in range(1, cfg.kmax + 1):
        delta_u_step = float(delta_u)
        linearization = _build_discrete_linearization(
            xA_0,
            t_grid_s,
            U_nom,
            mu_sun_km,
            amax_kmps2,
            rtol=cfg.reltol,
            atol=cfg.abstol,
        )
        x_tf_nom = linearization.X_nom[-1]
        R_nom = x_tf_nom[:3] - rE_tf
        miss_nom = float(np.linalg.norm(R_nom))

        try:
            subproblem = _solve_scp_subproblem(
                linearization.Ak,
                linearization.Bk,
                linearization.ck,
                xA_0,
                U_nom,
                R_nom,
                x_tf_nom[:3],
                dt_seg_s,
                tau_budget_s,
                delta_u,
            )
        except RuntimeError as exc:
            return SCPResult(
                success=False,
                converged=False,
                solver=solver_used,
                status=f"SCP failed: {exc}",
                iterations=iterations,
                U_opt=None,
                X_opt=None,
                miss_opt_km=None,
                accepted_steps=accepted_steps,
            )

        solver_used = subproblem.solver
        X_new = virtual_thrust_helpers.propagate_zoh_2bp_control(
            xA_0,
            t_grid_s,
            subproblem.u,
            mu_sun_km,
            amax_kmps2,
        )
        x_tf_new = X_new[-1]
        R_new = x_tf_new[:3] - rE_tf
        miss_new = float(np.linalg.norm(R_new))

        pred = float(subproblem.predicted_improvement)
        act = 0.5 * (miss_new**2 - miss_nom**2)  # actual-improvement proxy used for acceptance
        rho = act / max(pred, 1e-12)
        accepted = bool(rho > cfg.eta_good and miss_new >= miss_nom)

        _print_scp_iteration(
            iteration,
            miss_nom,
            miss_new,
            rho,
            delta_u_step,
            subproblem.status,
        )

        if accepted:
            U_nom = subproblem.u  # accept the candidate as the next nominal control
            accepted_steps += 1
            if rho > cfg.eta_great:
                delta_u = min(1.0, cfg.expand * delta_u)  # expand trust region after a strong step
        else:
            delta_u = max(1e-3, cfg.shrink * delta_u)  # reject the step and shrink the trust region

        iterations.append(
            {
                "iteration": iteration,
                "miss_nom_km": miss_nom,
                "miss_new_km": miss_new,
                "rho": float(rho),
                "predicted_improvement": pred,
                "actual_improvement": act,
                "delta_u": float(delta_u),
                "tau_used_s": float(subproblem.tau_used_s),
                "accepted": accepted,
                "solver_status": subproblem.status,
                "solver": subproblem.solver,
            }
        )

        if delta_u < 5e-3:
            converged = True
            break

    X_opt = virtual_thrust_helpers.propagate_zoh_2bp_control(xA_0, t_grid_s, U_nom, mu_sun_km, amax_kmps2)
    miss_opt_km = float(np.linalg.norm(X_opt[-1, :3] - rE_tf))
    status = (
        f"SCP completed with solver {solver_used}, "
        f"{len(iterations)} iterations, {accepted_steps} accepted step(s), "
        f"final miss {miss_opt_km:.6f} km."
    )
    if converged:
        status += " Trust region met the stopping threshold."

    return SCPResult(
        success=True,
        converged=converged,
        solver=solver_used,
        status=status,
        iterations=iterations,
        U_opt=U_nom,
        X_opt=X_opt,
        miss_opt_km=miss_opt_km,
        accepted_steps=accepted_steps,
    )


# ============================================================================
# Main workflow driver
# ============================================================================
# `run_virtual_thrust()` is the main workflow driver. It runs the scenario
# setup in order and returns a structured result bundle.

def run_virtual_thrust(config: VirtualThrustConfig | None = None) -> VirtualThrustRunResult:
    """Run the virtual thrust workflow from setup through plotting and SCP.

    In simple terms, this function propagates the scenario, prepares the MBI
    states, computes the benchmark, builds the plots, and optionally runs SCP.
    """

    cfg = VirtualThrustConfig() if config is None else config
    earth = bodies.CelestialBody("Earth")
    sun = bodies.CelestialBody("Sun")

    # Basic asteroid mass model used by the impact / proxy-thrust calculations.
    asteroid_mass_kg = (4.0 / 3.0) * np.pi * (cfg.asteroid_diameter_m / 2.0) ** 3 * cfg.rho_ast_kg_m3

    # Build the nominal environment and propagate Earth + asteroid histories.
    env = interception.make_env(
        years=cfg.env_years,
        step_min=cfg.env_step_min,
        rtol=cfg.reltol,
        atol=cfg.abstol,
    )
    interception.propagate_earth(env)
    asteroids = interception.propagate_asteroids(interception.ast_catalog(cfg.asteroid_name), env)
    asteroid = asteroids[0]
    if env.t_Earth is None or env.X_Earth_hist is None or asteroid.t_hist is None or asteroid.X_hist is None:
        raise RuntimeError("Propagation failed to produce Earth and asteroid histories.")

    nominal_ca, nominal_moid = interception.get_ca_moid(
        env.X_Earth_hist,
        asteroid.X_hist,
        env.t_Earth,
        asteroid.t_hist,
    )

    # Prepare the impact-relative states used to define the control window.
    asteroids, mbi_sets, _ = interception.prepare_mbi(
        asteroids,
        env,
        cfg.months_back(),
        cfg.force_impact,
        rtol=cfg.reltol,
        atol=cfg.abstol,
    )
    asteroid = asteroids[0]
    mbi_states = mbi_sets[0]

    state_t0 = _select_mbi_state(mbi_states, cfg.t0_months)
    state_tf = _select_mbi_state(mbi_states, 0.0)
    xA_t0 = state_t0.stateAst.copy()
    xE_t0 = state_t0.stateEarth.copy()
    xE_tf = state_tf.stateEarth.copy()
    xA_tf = state_tf.stateAst.copy()
    close_approach_distance_km = float(np.linalg.norm(xA_tf[:3] - xE_tf[:3]))

    # Choose whether to display the nominal geometry or the forced-impact geometry.
    if cfg.force_impact:
        display_t_s, display_earth_hist, display_asteroid_hist = _build_forced_impact_histories(
            env.tspan,
            xE_tf,
            xA_tf,
            sun.mu.km,
            rtol=cfg.reltol,
            atol=cfg.abstol,
        )
        ca, moid = interception.get_ca_moid(
            display_earth_hist,
            display_asteroid_hist,
            display_t_s,
            display_t_s,
        )
        plot_bodies = [
            {"name": "Earth", "X_hist": display_earth_hist, "t_hist": display_t_s},
            {"name": asteroid.name, "X_hist": display_asteroid_hist, "t_hist": display_t_s},
        ]
        trajectory_title = "Earth + Asteroid Trajectories (Forced Impact Geometry)"
        separation_title = "Earth-Asteroid Separation History (Forced Impact Geometry)"
        separation_time_axis_label = "Time relative to CA [years]"
    else:
        ca = nominal_ca
        moid = nominal_moid
        display_t_s = env.t_Earth
        display_earth_hist = env.X_Earth_hist
        display_asteroid_hist = asteroid.X_hist
        plot_bodies = interception.make_bodies_for_plot([asteroid], env)
        trajectory_title = "Earth + Asteroid Trajectories"
        separation_title = "Earth-Asteroid Separation History"
        separation_time_axis_label = "Time [years]"

    # Convert impact assumptions into proxy acceleration and total delta-V bounds.
    dt_min_s = cfg.cadence_days * 24.0 * 3600.0
    amax_mps2 = virtual_thrust_helpers.compute_amax_from_cadence(
        asteroid_mass_kg,
        cfg.mass_sc_kg,
        cfg.vrel_use_mps,
        cfg.beta,
        cfg.cos_gamma,
        dt_min_s,
    )
    amax_kmps2 = amax_mps2 / 1000.0

    dvmax_mps = virtual_thrust_helpers.compute_dvmax_from_impactors(
        asteroid_mass_kg,
        cfg.mass_sc_kg * np.ones(cfg.n_impactors),
        cfg.vrel_use_mps * np.ones(cfg.n_impactors),
        cfg.beta,
        cfg.cos_gamma * np.ones(cfg.n_impactors),
    )
    dv1_mps = virtual_thrust_helpers.compute_dv_per_impact(
        asteroid_mass_kg,
        cfg.mass_sc_kg,
        cfg.vrel_use_mps,
        cfg.beta,
        cfg.cos_gamma,
    )
    dvmax_kmps = dvmax_mps / 1000.0
    dv1_kmps = dv1_mps / 1000.0
    tau_budget_s = dvmax_kmps / amax_kmps2

    # Compute the Conway single-impulse STM benchmark used for comparison and
    # for the optional warm-start direction.
    tf_sec = 0.0
    t0_sec = -cfg.t0_months * SECONDS_PER_MONTH
    benchmark = virtual_thrust_helpers.conway_max_theoretical_deflection_stm(
        xA_t0,
        xE_tf[:3],
        dv1_kmps,
        tf_sec,
        t0_sec,
        sun.mu.km,
        rtol=cfg.reltol,
        atol=cfg.abstol,
    )

    # Build the ZOH time grid and the initial nominal control profile.
    total_duration_s = tf_sec - t0_sec
    dt_seg_s = total_duration_s / cfg.n_segments
    t_grid_s = np.linspace(t0_sec, tf_sec, cfg.n_segments + 1)
    u0_mag_max = tau_budget_s / total_duration_s
    u0_mag = 0.5 * min(0.01, u0_mag_max)
    if cfg.use_opt_dv_dir:
        uhat0 = benchmark.e_opt / np.linalg.norm(benchmark.e_opt)
    else:
        uhat0 = xA_t0[3:] / np.linalg.norm(xA_t0[3:])
    warm_start_control = np.tile(u0_mag * uhat0, (cfg.n_segments, 1))

    # Always create the main geometry plots. SCP-specific plots are added only
    # if the optimizer succeeds.
    figures = {
        "trajectories": build_trajectory_figure(
            plot_bodies,
            xA_t0,
            xE_t0,
            xE_tf,
            xA_tf,
            trajectory_title,
        ),
        "separation": build_separation_figure(
            display_earth_hist,
            display_asteroid_hist,
            display_t_s,
            ca,
            moid,
            title=separation_title,
            time_axis_label=separation_time_axis_label,
        ),
    }
    scp_result: SCPResult | None = None
    if cfg.run_scp:
        # Run the SCP section.
        scp_result = _run_scp_optimization(
            xA_t0,
            xE_tf[:3],
            t_grid_s,
            warm_start_control,
            sun.mu.km,
            amax_kmps2,
            tau_budget_s,
            cfg,
        )
        if scp_result.success and scp_result.U_opt is not None and scp_result.X_opt is not None:
            # Only add optimized-control plots when a valid optimized propagation exists.
            figures["optimized_control"] = build_control_history_figure(
                t_grid_s,
                scp_result.U_opt,
                amax_kmps2,
                dt_seg_s,
                dvmax_mps,
                title="Optimized Control Profile (SCP)",
            )
            figures["optimized_trajectory"] = build_optimized_trajectory_figure(
                plot_bodies,
                scp_result.X_opt,
                t_grid_s,
                "Earth + Nominal Asteroid + Optimized Asteroid",
            )
        scp_status = scp_result.status
    else:
        scp_status = "SCP skipped: run_scp=False."

    for fig in figures.values():
        fig.show()

    return VirtualThrustRunResult(
        config=cfg,
        earth=earth,
        sun=sun,
        asteroid=asteroid,
        ca=ca,
        moid=moid,
        mbi_states=mbi_states,
        benchmark=benchmark,
        asteroid_mass_kg=asteroid_mass_kg,
        close_approach_distance_km=close_approach_distance_km,
        amax_mps2=amax_mps2,
        amax_kmps2=amax_kmps2,
        dvmax_mps=dvmax_mps,
        dv1_mps=dv1_mps,
        tau_budget_s=tau_budget_s,
        t_grid_s=t_grid_s,
        warm_start_control=warm_start_control,
        scp=scp_result,
        figures=figures,
        scp_status=scp_status,
    )


# ============================================================================
# Command-line entry point
# ============================================================================
# `main()` is only a thin wrapper for command-line use. The real workflow lives
# in `run_virtual_thrust()`, which is easier to call from tests or notebooks.

def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the minimal command-line interface for this workflow."""

    parser = argparse.ArgumentParser(description=__doc__)
    return parser


def main() -> None:
    """Command-line wrapper around `run_virtual_thrust()`."""

    args = _build_arg_parser().parse_args()
    del args
    config = VirtualThrustConfig()
    result = run_virtual_thrust(config)

    print(f"Asteroid: {result.asteroid.name}")
    asteroid_mass_tons = result.asteroid_mass_kg / 1000.0
    print(f"Asteroid mass: {asteroid_mass_tons:.6f} tons")
    print(f"Sampled CA distance: {result.ca.d_km:.6f} km")
    print(f"Close approach distance at CA epoch: {result.close_approach_distance_km:.6f} km")
    print(f"Sampled MOID distance: {result.moid.d_km:.6f} km")
    print(f"Per-impact delta-V bound: {result.dv1_mps:.6f} m/s")
    print(f"Total delta-V bound: {result.dvmax_mps:.6f} m/s")
    print(f"Proxy acceleration bound: {result.amax_mps2:.6e} m/s^2")
    print(f"Warm-start thrust-time budget: {result.tau_budget_s:.6f} s")
    print(f"Conway max theoretical deflection: {result.benchmark.dr_max_km:.6f} km")
    if result.scp is not None:
        print(f"SCP solver: {result.scp.solver}")
        if result.scp.miss_opt_km is not None:
            print(f"SCP final miss distance: {result.scp.miss_opt_km:.6f} km")
        print(f"SCP accepted steps: {result.scp.accepted_steps}")
    print(result.scp_status)


if __name__ == "__main__":
    main()
