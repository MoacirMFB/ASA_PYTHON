"""Runnable Python version of the MATLAB virtual thrust workflow."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import cvxpy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised only in partial environments
    cp = None

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from ASA_Python.bodies import CelestialBody
    from ASA_Python.interception import (
        AsteroidRecord,
        ClosestApproach,
        MBIState,
        get_ca_moid,
        make_bodies_for_plot,
        make_env,
        prepare_mbi,
        propagate_asteroids,
        propagate_earth,
        ast_catalog,
    )
    from ASA_Python.control import linear_discrete_time_matrices
    from ASA_Python.keplerian import dynamics_2bp_cartesian, jacobian_2bp_cartesian, propagate_two_body
    from ASA_Python.virtual_thrust_helpers import (
        ConwayBenchmark,
        compute_amax_from_cadence,
        compute_dv_per_impact,
        compute_dvmax_from_impactors,
        conway_max_theoretical_deflection_stm,
        rollout_zoh_2bp_control,
    )
else:
    from .bodies import CelestialBody
    from .control import linear_discrete_time_matrices
    from .interception import (
        AsteroidRecord,
        ClosestApproach,
        MBIState,
        ast_catalog,
        get_ca_moid,
        make_bodies_for_plot,
        make_env,
        prepare_mbi,
        propagate_asteroids,
        propagate_earth,
    )
    from .keplerian import dynamics_2bp_cartesian, jacobian_2bp_cartesian, propagate_two_body
    from .virtual_thrust_helpers import (
        ConwayBenchmark,
        compute_amax_from_cadence,
        compute_dv_per_impact,
        compute_dvmax_from_impactors,
        conway_max_theoretical_deflection_stm,
        rollout_zoh_2bp_control,
    )


SECONDS_PER_MONTH = 30.0 * 86400.0


@dataclass
class VirtualThrustConfig:
    """Scenario settings mirrored from the MATLAB virtual thrust script."""

    reltol: float = 1e-12
    abstol: float = 1e-12
    run_scp: bool = True
    use_opt_dv_dir: bool = True

    asteroid_name: str = "Apophis"
    rho_ast_kg_m3: float = 2400.0
    asteroid_diameter_m: float = 100.0
    beta: float = 1.0
    force_impact: bool = True
    lead_time_years: int = 3
    t0_months: int = 24

    env_years: float = 4.0
    env_step_min: float = 10.0

    mass_sc_kg: float = 1000.0
    vrel_use_mps: float = 10_000.0
    n_impactors: int = 5
    cos_gamma: float = 1.0
    cadence_days: float = 10.0

    n_segments: int = 750
    kmax: int = 50
    delta_u0: float = 0.75
    eta_good: float = 0.25
    eta_great: float = 0.75
    shrink: float = 0.5
    expand: float = 1.5

    show_plots: bool = True
    output_dir: Path | None = None

    def months_back(self) -> np.ndarray:
        return np.arange(0, 12 * self.lead_time_years + 1, dtype=float)

    def ode_kwargs(self) -> dict[str, Any]:
        return {"rtol": self.reltol, "atol": self.abstol}


@dataclass
class VirtualThrustRunResult:
    """Primary outputs from the runnable Python workflow."""

    config: VirtualThrustConfig
    earth: CelestialBody
    sun: CelestialBody
    asteroid: AsteroidRecord
    ca: ClosestApproach
    moid: ClosestApproach
    mbi_states: list[MBIState]
    benchmark: ConwayBenchmark
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
    """Nominal rollout and interval-wise discrete linearized dynamics."""

    X_nom: np.ndarray
    Ak: np.ndarray
    Bk: np.ndarray
    ck: np.ndarray


@dataclass
class SCPSubproblemResult:
    """Solution of one convex SCP subproblem."""

    x: np.ndarray
    u: np.ndarray
    tau_used_s: float
    predicted_improvement: float
    status: str
    solver: str


@dataclass
class SCPResult:
    """Final SCP optimization state and iteration history."""

    success: bool
    converged: bool
    solver: str | None
    status: str
    iterations: list[dict[str, Any]]
    U_opt: np.ndarray | None
    X_opt: np.ndarray | None
    miss_opt_km: float | None
    accepted_steps: int


def _select_mbi_state(mbi_states: list[MBIState], month: float) -> MBIState:
    for state in mbi_states:
        if np.isclose(state.month, month):
            return state
    raise ValueError(f"Requested month {month} not found in MBI state set.")


def _make_hover_data(states: np.ndarray, t_hist_s: np.ndarray) -> np.ndarray:
    radius_km = np.linalg.norm(states[:, :3], axis=1)
    return np.column_stack((t_hist_s / 86400.0, radius_km))


def _add_body_trace(
    fig: go.Figure,
    name: str,
    states: np.ndarray,
    t_hist_s: np.ndarray,
    color: str,
) -> None:
    hover_data = _make_hover_data(states, t_hist_s)
    fig.add_trace(
        go.Scatter3d(
            x=states[:, 0],
            y=states[:, 1],
            z=states[:, 2],
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
                "z = %{z:.3e} km<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[states[0, 0]],
            y=[states[0, 1]],
            z=[states[0, 2]],
            mode="markers",
            name=f"{name} start",
            marker={"color": color, "size": 5, "symbol": "circle"},
            hovertemplate=f"{name} start<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[states[-1, 0]],
            y=[states[-1, 1]],
            z=[states[-1, 2]],
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
    fig.add_trace(
        go.Scatter3d(
            x=[state[0]],
            y=[state[1]],
            z=[state[2]],
            mode="markers",
            name=label,
            marker={"color": color, "size": 7, "symbol": symbol},
            hovertemplate=(
                f"{label}<br>"
                "x = %{x:.3e} km<br>"
                "y = %{y:.3e} km<br>"
                "z = %{z:.3e} km<extra></extra>"
            ),
        )
    )


def build_trajectory_figure(
    bodies: list[dict[str, object]],
    xA_t0: np.ndarray,
    xE_t0: np.ndarray,
    xE_tf: np.ndarray,
    xA_tf: np.ndarray,
    title: str,
) -> go.Figure:
    """Build an interactive 3D Earth/asteroid trajectory plot."""

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
        scene={
            "xaxis_title": "x [km]",
            "yaxis_title": "y [km]",
            "zaxis_title": "z [km]",
            "aspectmode": "data",
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )
    return fig


def build_separation_figure(
    earth_states: np.ndarray,
    asteroid_states: np.ndarray,
    t_hist_s: np.ndarray,
    ca: ClosestApproach,
    moid: ClosestApproach,
    *,
    title: str = "Earth-Asteroid Separation History",
    time_axis_label: str = "Time [years]",
) -> go.Figure:
    """Build an interactive Earth-asteroid separation timeline."""

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
    """Build a step-style control history figure for warm-start or optimized controls."""

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
    """Overlay the optimized asteroid trajectory on the Earth/nominal-asteroid plot."""

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
        go.Scatter3d(
            x=X_opt[:-1, 0],
            y=X_opt[:-1, 1],
            z=X_opt[:-1, 2],
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
                "z = %{z:.3e} km<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "x [km]",
            "yaxis_title": "y [km]",
            "zaxis_title": "z [km]",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )
    return fig


def _maybe_write_figure(fig: go.Figure, name: str, output_dir: Path | None) -> None:
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_dir / f"{name}.html", include_plotlyjs="cdn")


def _maybe_show_figure(fig: go.Figure, show_plots: bool) -> None:
    if show_plots:
        fig.show()


def _build_forced_impact_histories(
    env_tspan_s: np.ndarray,
    xE_tf: np.ndarray,
    xA_tf: np.ndarray,
    mu_sun_km: float,
    *,
    rtol: float,
    atol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct Earth and asteroid histories backwards from the forced-impact epoch."""

    t_rel_desc = -np.asarray(env_tspan_s, dtype=float)
    _, earth_desc = propagate_two_body(
        xE_tf,
        t_rel_desc,
        mu_sun_km,
        rtol=rtol,
        atol=atol,
    )
    _, asteroid_desc = propagate_two_body(
        xA_tf,
        t_rel_desc,
        mu_sun_km,
        rtol=rtol,
        atol=atol,
    )
    return t_rel_desc[::-1], earth_desc[::-1], asteroid_desc[::-1]


def _control_injection_matrix(amax_kmps2: float) -> np.ndarray:
    """Map dimensionless control into Cartesian acceleration in km/s^2."""

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
    """Roll out the nominal trajectory and discretize the interval-wise linear model."""

    del rtol, atol

    X_nom = rollout_zoh_2bp_control(xA_0, t_grid_s, U_nom, mu_sun_km, amax_kmps2)
    n_intervals = U_nom.shape[0]
    szx, szu = 6, 3
    Ak = np.zeros((n_intervals, szx, szx))
    Bk = np.zeros((n_intervals, szx, szu))
    ck = np.zeros((n_intervals, szx))
    Bctrl = _control_injection_matrix(amax_kmps2)

    for k in range(n_intervals):
        xk_nom = X_nom[k]
        uk_nom = U_nom[k]
        Acont = jacobian_2bp_cartesian(0.0, xk_nom, mu_sun_km)
        f0 = dynamics_2bp_cartesian(0.0, xk_nom, mu_sun_km)
        f0[3:] += amax_kmps2 * uk_nom
        ccont = f0 - Acont @ xk_nom - Bctrl @ uk_nom

        Phi0 = np.eye(szx)
        Y0aug = np.concatenate(
            (
                xk_nom,
                Phi0.reshape(-1, order="F"),
                np.zeros(szx * szu),
                np.zeros(szx),
            )
        )
        Ak[k], Bk[k], ck[k], _ = linear_discrete_time_matrices(
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
    """Return installed cvxpy solvers in the preferred order for this SOCP."""

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
    """Try MOSEK first, then fallback SOCP solvers that cvxpy supports."""

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
    """Build and solve one convex SCP subproblem in cvxpy."""

    n_intervals = U_nom.shape[0]
    x = cp.Variable((n_intervals + 1, 6))
    u = cp.Variable((n_intervals, 3))

    tau_used = cp.sum(cp.norm(u, 2, axis=1)) * dt_seg_s
    objective = cp.Maximize(R_nom @ (x[n_intervals, :3] - rnom_tf))
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
    """Run the discrete linearized SCP loop from the MATLAB script in cvxpy."""

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

    for iteration in range(1, cfg.kmax + 1):
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
        X_new = rollout_zoh_2bp_control(xA_0, t_grid_s, subproblem.u, mu_sun_km, amax_kmps2)
        x_tf_new = X_new[-1]
        R_new = x_tf_new[:3] - rE_tf
        miss_new = float(np.linalg.norm(R_new))

        pred = float(subproblem.predicted_improvement)
        act = 0.5 * (miss_new**2 - miss_nom**2)
        rho = act / max(pred, 1e-12)
        accepted = bool(rho > cfg.eta_good and miss_new >= miss_nom)

        if accepted:
            U_nom = subproblem.u
            accepted_steps += 1
            if rho > cfg.eta_great:
                delta_u = min(1.0, cfg.expand * delta_u)
        else:
            delta_u = max(1e-3, cfg.shrink * delta_u)

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

    X_opt = rollout_zoh_2bp_control(xA_0, t_grid_s, U_nom, mu_sun_km, amax_kmps2)
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


def run_virtual_thrust(config: VirtualThrustConfig | None = None) -> VirtualThrustRunResult:
    """Run the Python-supported subset of the MATLAB virtual thrust workflow."""

    cfg = VirtualThrustConfig() if config is None else config
    earth = CelestialBody("Earth")
    sun = CelestialBody("Sun")

    asteroid_mass_kg = (4.0 / 3.0) * np.pi * (cfg.asteroid_diameter_m / 2.0) ** 3 * cfg.rho_ast_kg_m3

    env = make_env(
        years=cfg.env_years,
        step_min=cfg.env_step_min,
        rtol=cfg.reltol,
        atol=cfg.abstol,
    )
    propagate_earth(env)
    asteroids = propagate_asteroids(ast_catalog(cfg.asteroid_name), env)
    asteroid = asteroids[0]
    if env.t_Earth is None or env.X_Earth_hist is None or asteroid.t_hist is None or asteroid.X_hist is None:
        raise RuntimeError("Propagation failed to produce Earth and asteroid histories.")

    nominal_ca, nominal_moid = get_ca_moid(env.X_Earth_hist, asteroid.X_hist, env.t_Earth, asteroid.t_hist)

    asteroids, mbi_sets, _ = prepare_mbi(
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

    if cfg.force_impact:
        display_t_s, display_earth_hist, display_asteroid_hist = _build_forced_impact_histories(
            env.tspan,
            xE_tf,
            xA_tf,
            sun.mu.km,
            rtol=cfg.reltol,
            atol=cfg.abstol,
        )
        ca, moid = get_ca_moid(
            display_earth_hist,
            display_asteroid_hist,
            display_t_s,
            display_t_s,
        )
        bodies = [
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
        bodies = make_bodies_for_plot([asteroid], env)
        trajectory_title = "Earth + Asteroid Trajectories"
        separation_title = "Earth-Asteroid Separation History"
        separation_time_axis_label = "Time [years]"

    dt_min_s = cfg.cadence_days * 24.0 * 3600.0
    amax_mps2 = compute_amax_from_cadence(
        asteroid_mass_kg,
        cfg.mass_sc_kg,
        cfg.vrel_use_mps,
        cfg.beta,
        cfg.cos_gamma,
        dt_min_s,
    )
    amax_kmps2 = amax_mps2 / 1000.0

    dvmax_mps = compute_dvmax_from_impactors(
        asteroid_mass_kg,
        cfg.mass_sc_kg * np.ones(cfg.n_impactors),
        cfg.vrel_use_mps * np.ones(cfg.n_impactors),
        cfg.beta,
        cfg.cos_gamma * np.ones(cfg.n_impactors),
    )
    dv1_mps = compute_dv_per_impact(
        asteroid_mass_kg,
        cfg.mass_sc_kg,
        cfg.vrel_use_mps,
        cfg.beta,
        cfg.cos_gamma,
    )
    dvmax_kmps = dvmax_mps / 1000.0
    dv1_kmps = dv1_mps / 1000.0
    tau_budget_s = dvmax_kmps / amax_kmps2

    tf_sec = 0.0
    t0_sec = -cfg.t0_months * SECONDS_PER_MONTH
    benchmark = conway_max_theoretical_deflection_stm(
        xA_t0,
        xE_tf[:3],
        dv1_kmps,
        tf_sec,
        t0_sec,
        sun.mu.km,
        rtol=cfg.reltol,
        atol=cfg.abstol,
    )

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

    figures = {
        "trajectories": build_trajectory_figure(
            bodies,
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
            figures["optimized_control"] = build_control_history_figure(
                t_grid_s,
                scp_result.U_opt,
                amax_kmps2,
                dt_seg_s,
                dvmax_mps,
                title="Optimized Control Profile (SCP)",
            )
            figures["optimized_trajectory"] = build_optimized_trajectory_figure(
                bodies,
                scp_result.X_opt,
                t_grid_s,
                "Earth + Nominal Asteroid + Optimized Asteroid",
            )
        scp_status = scp_result.status
    else:
        scp_status = "SCP skipped: run_scp=False."

    for name, fig in figures.items():
        _maybe_write_figure(fig, name, cfg.output_dir)
        _maybe_show_figure(fig, cfg.show_plots)

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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Create figures without opening interactive browser windows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for Plotly HTML outputs.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    config = VirtualThrustConfig(show_plots=not args.no_show, output_dir=args.output_dir)
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
    if args.output_dir is not None:
        print(f"Plotly HTML outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
