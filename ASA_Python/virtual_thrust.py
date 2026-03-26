"""Runnable Python version of the pre-SCP virtual thrust workflow."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    from ASA_Python.keplerian import propagate_two_body
    from ASA_Python.virtual_thrust_helpers import (
        ConwayBenchmark,
        compute_amax_from_cadence,
        compute_dv_per_impact,
        compute_dvmax_from_impactors,
        conway_max_theoretical_deflection_stm,
    )
else:
    from .bodies import CelestialBody
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
    from .keplerian import propagate_two_body
    from .virtual_thrust_helpers import (
        ConwayBenchmark,
        compute_amax_from_cadence,
        compute_dv_per_impact,
        compute_dvmax_from_impactors,
        conway_max_theoretical_deflection_stm,
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
    figures: dict[str, go.Figure]
    scp_status: str


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


def build_warm_start_control_figure(
    t_grid_s: np.ndarray,
    U0: np.ndarray,
    amax_kmps2: float,
    dt_seg_s: float,
    dvmax_mps: float,
) -> go.Figure:
    """Build an interactive warm-start control and budget plot."""

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
            "Warm-start normalized control",
            "Warm-start physical acceleration",
            "Warm-start accumulated delta-V",
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
    fig.update_layout(
        title="Warm-start Control Profile (SCP Deferred)",
        template="plotly_white",
        height=900,
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
        "warm_start_control": build_warm_start_control_figure(
            t_grid_s,
            warm_start_control,
            amax_kmps2,
            dt_seg_s,
            dvmax_mps,
        ),
    }

    for name, fig in figures.items():
        _maybe_write_figure(fig, name, cfg.output_dir)
        _maybe_show_figure(fig, cfg.show_plots)

    scp_status = (
        "Deferred: the SCP/CVXPY rewrite is not implemented in Python yet. "
        "This script stops after the MATLAB-parity setup, benchmark, and interactive plotting steps."
    )

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
    print(f"Asteroid mass: {result.asteroid_mass_kg:.6f} kg")
    print(f"Sampled CA distance: {result.ca.d_km:.6f} km")
    print(f"Close approach distance at CA epoch: {result.close_approach_distance_km:.6f} km")
    print(f"Sampled MOID distance: {result.moid.d_km:.6f} km")
    print(f"Per-impact delta-V bound: {result.dv1_mps:.6f} m/s")
    print(f"Total delta-V bound: {result.dvmax_mps:.6f} m/s")
    print(f"Proxy acceleration bound: {result.amax_mps2:.6e} m/s^2")
    print(f"Warm-start thrust-time budget: {result.tau_budget_s:.6f} s")
    print(f"Conway max theoretical deflection: {result.benchmark.dr_max_km:.6f} km")
    print(result.scp_status)
    if args.output_dir is not None:
        print(f"Plotly HTML outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
