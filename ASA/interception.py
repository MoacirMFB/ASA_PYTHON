"""Minimal interception workflow port needed by the virtual thrust script."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .bodies import AU_KM, CelestialBody, get_celestial_body
from .keplerian import coe_to_cartesian, propagate_two_body

FloatArray = NDArray[np.float64]


@dataclass
class Environment:
    """Propagation settings and shared constants for the workflow."""

    mu_sun_km: float
    mu_sun_au: float
    au2km: float
    r_earth_km: float
    years: float
    step_min: float
    tspan: FloatArray
    rtol: float = 1e-13
    atol: float = 1e-13
    method: str = "DOP853"
    t_earth: FloatArray | None = None
    x_earth_hist: FloatArray | None = None


@dataclass
class AsteroidRecord:
    """Asteroid catalog entry and propagated state history."""

    name: str
    coe: FloatArray
    t_hist: FloatArray | None = None
    x_hist: FloatArray | None = None
    moid_pre_km: float | None = None
    ca_pre_km: float | None = None


@dataclass
class ClosestApproach:
    """Closest-approach or MOID record."""

    d_km: float
    idx_earth: int
    idx_ast: int
    time_earth: float
    time_ast: float
    state_earth: FloatArray
    state_ast: FloatArray


@dataclass
class MBIState:
    """Earth and asteroid states a given number of months before impact."""

    month: float
    state_earth: FloatArray
    state_ast: FloatArray


_ASTEROID_CATALOG = {
    "2007 DX40": [1.538448416729292, 0.5382979734185152, np.deg2rad(0.4519685), np.deg2rad(329.8163426), np.deg2rad(273.6828503), 0.0],
    "2004 VD17": [1.5080000, 0.5887000, np.deg2rad(4.22), np.deg2rad(223.98), np.deg2rad(90.99), 0.0],
    "2007 FT3": [1.1246634, 0.3057574, np.deg2rad(26.72735), np.deg2rad(9.79108), np.deg2rad(277.56582), 0.0],
    "1979 XB": [2.2199764, 0.7073090, np.deg2rad(24.57497), np.deg2rad(84.74573), np.deg2rad(76.73211), 0.0],
    "1950 DA": [1.6986794, 0.5075054, np.deg2rad(12.15458), np.deg2rad(356.54705), np.deg2rad(224.82640), 0.0],
    "Bennu": [1.1259897, 0.2037311, np.deg2rad(6.03290), np.deg2rad(1.97241), np.deg2rad(66.39438), 0.0],
    "Apophis": [0.9225521, 0.1912907, np.deg2rad(3.33974), np.deg2rad(203.91537), np.deg2rad(126.68323), 0.0],
    "2011 AG5": [1.4241515, 0.3882701, np.deg2rad(3.69422), np.deg2rad(135.59205), np.deg2rad(54.05017), 0.0],
    "2022 AE1": [1.4708823, 0.5461987, np.deg2rad(6.29686), np.deg2rad(102.18992), np.deg2rad(268.32032), 0.0],
    "2000 SG344": [0.9774614, 0.0669332, np.deg2rad(0.11213), np.deg2rad(191.95995), np.deg2rad(275.30264), 0.0],
    "2023 DW": [0.8200000, 0.3970000, np.deg2rad(5.81), np.deg2rad(326.10), np.deg2rad(40.40), 0.0],
}


def _as_row_state(x: Iterable[float] | FloatArray) -> FloatArray:
    return np.asarray(x, dtype=float).reshape(6)


def make_env(
    *,
    years: float = 4.0,
    step_min: float = 10.0,
    rtol: float = 1e-13,
    atol: float = 1e-13,
    method: str = "DOP853",
) -> Environment:
    """Create the minimal propagation environment used by this workflow."""

    sun = get_celestial_body("Sun")
    earth = get_celestial_body("Earth")
    tf_s = years * 365.25 * 86400.0
    dt_s = step_min * 60.0
    tspan = np.arange(0.0, tf_s + dt_s, dt_s, dtype=float)
    if tspan[-1] > tf_s:
        tspan[-1] = tf_s
    return Environment(
        mu_sun_km=sun.mu.km,
        mu_sun_au=sun.mu.au,
        au2km=AU_KM,
        r_earth_km=earth.radius.km,
        years=years,
        step_min=step_min,
        tspan=tspan,
        rtol=rtol,
        atol=atol,
        method=method,
    )


def propagate_earth(env: Environment) -> tuple[FloatArray, FloatArray]:
    """Propagate Earth's heliocentric two-body state history."""

    earth = get_celestial_body("Earth")
    orbit = earth.orbit
    x_coe = np.array(
        [
            orbit.a_km,
            orbit.e,
            np.deg2rad(orbit.i_deg),
            np.deg2rad(orbit.raan_deg),
            np.deg2rad(orbit.arg_peri_deg),
            np.deg2rad(orbit.m0_deg),
        ],
        dtype=float,
    )
    x0 = coe_to_cartesian(x_coe, env.mu_sun_km, use_true_anomaly=False)
    t_earth, x_earth = propagate_two_body(
        x0,
        env.tspan,
        env.mu_sun_km,
        rtol=env.rtol,
        atol=env.atol,
        method=env.method,
    )
    env.t_earth = t_earth
    env.x_earth_hist = x_earth
    return t_earth, x_earth


def get_asteroid(name: str) -> AsteroidRecord:
    """Return one asteroid record from the minimal in-memory catalog."""

    if name not in _ASTEROID_CATALOG:
        raise KeyError(f'Asteroid "{name}" not in catalog.')
    return AsteroidRecord(
        name=name,
        coe=np.asarray(_ASTEROID_CATALOG[name], dtype=float).copy(),
    )


def propagate_asteroid(
    asteroid: AsteroidRecord,
    env: Environment,
) -> AsteroidRecord:
    """Propagate one asteroid heliocentric history from catalog COEs."""

    coe = asteroid.coe.copy()
    x_coe = coe.copy()
    x_coe[0] *= env.au2km
    x0 = coe_to_cartesian(x_coe, env.mu_sun_km, use_true_anomaly=True)
    t_hist, x_hist = propagate_two_body(
        x0,
        env.tspan,
        env.mu_sun_km,
        rtol=env.rtol,
        atol=env.atol,
        method=env.method,
    )
    asteroid.t_hist = t_hist
    asteroid.x_hist = x_hist
    return asteroid


def make_bodies_for_plot(asteroid: AsteroidRecord, env: Environment) -> list[dict[str, object]]:
    """Package Earth and one asteroid trajectory for deferred plotting work."""

    if env.t_earth is None or env.x_earth_hist is None:
        raise ValueError("Earth trajectory has not been propagated.")
    bodies: list[dict[str, object]] = [
        {"name": "Earth", "x_hist": env.x_earth_hist, "t_hist": env.t_earth}
    ]
    if asteroid.t_hist is None or asteroid.x_hist is None:
        raise ValueError(f'Asteroid "{asteroid.name}" has not been propagated.')
    bodies.append({"name": asteroid.name, "x_hist": asteroid.x_hist, "t_hist": asteroid.t_hist})
    return bodies


def get_ca_moid(
    earth_states: Iterable[Iterable[float]] | FloatArray,
    asteroid_states: Iterable[Iterable[float]] | FloatArray,
    time_earth: Iterable[float] | FloatArray,
    time_asteroid: Iterable[float] | FloatArray,
) -> tuple[ClosestApproach, ClosestApproach]:
    """Compute synchronized closest approach and sampled global MOID."""

    earth = np.asarray(earth_states, dtype=float)
    asteroid = np.asarray(asteroid_states, dtype=float)
    t_earth = np.asarray(time_earth, dtype=float).reshape(-1)
    t_asteroid = np.asarray(time_asteroid, dtype=float).reshape(-1)
    if earth.size == 0 or asteroid.size == 0:
        raise ValueError("Earth and asteroid histories must be non-empty.")

    n_sync = min(earth.shape[0], asteroid.shape[0])
    sync_distances = np.linalg.norm(earth[:n_sync, :3] - asteroid[:n_sync, :3], axis=1)
    k_ca = int(np.argmin(sync_distances))
    ca = ClosestApproach(
        d_km=float(sync_distances[k_ca]),
        idx_earth=k_ca,
        idx_ast=k_ca,
        time_earth=float(t_earth[k_ca]),
        time_ast=float(t_asteroid[k_ca]),
        state_earth=earth[k_ca].copy(),
        state_ast=asteroid[k_ca].copy(),
    )

    tree = cKDTree(asteroid[:, :3])
    distances, idx_ast = tree.query(earth[:, :3], k=1)
    idx_earth = int(np.argmin(distances))
    idx_ast_moid = int(idx_ast[idx_earth])
    moid = ClosestApproach(
        d_km=float(distances[idx_earth]),
        idx_earth=idx_earth,
        idx_ast=idx_ast_moid,
        time_earth=float(t_earth[idx_earth]),
        time_ast=float(t_asteroid[idx_ast_moid]),
        state_earth=earth[idx_earth].copy(),
        state_ast=asteroid[idx_ast_moid].copy(),
    )
    return ca, moid


def get_states_at_mbi(
    states_at_moid: ClosestApproach,
    months_back: Iterable[float] | FloatArray,
    *,
    mu_sun_km: float | None = None,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    method: str = "DOP853",
) -> list[MBIState]:
    """Backward-propagate Earth and asteroid from the MOID epoch."""

    mu = get_celestial_body("Sun").mu.km if mu_sun_km is None else float(mu_sun_km)
    months = np.asarray(months_back, dtype=float).reshape(-1)
    mbi_states: list[MBIState] = []
    for month in months:
        dt = float(month * 30.0 * 86400.0)
        if np.isclose(dt, 0.0):
            mbi_states.append(
                MBIState(
                    month=float(month),
                    state_earth=states_at_moid.state_earth.copy(),
                    state_ast=states_at_moid.state_ast.copy(),
                )
            )
            continue

        _, earth_hist = propagate_two_body(
            states_at_moid.state_earth,
            np.array([0.0, -dt], dtype=float),
            mu,
            rtol=rtol,
            atol=atol,
            method=method,
        )
        _, asteroid_hist = propagate_two_body(
            states_at_moid.state_ast,
            np.array([0.0, -dt], dtype=float),
            mu,
            rtol=rtol,
            atol=atol,
            method=method,
        )
        mbi_states.append(
            MBIState(
                month=float(month),
                state_earth=earth_hist[-1].copy(),
                state_ast=asteroid_hist[-1].copy(),
            )
        )
    return mbi_states


def prepare_mbi(
    asteroid: AsteroidRecord,
    env: Environment,
    months_back: Iterable[float] | FloatArray,
    force_impact: bool,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    method: str = "DOP853",
) -> tuple[AsteroidRecord, list[MBIState], ClosestApproach]:
    """Prepare the single-asteroid state set at months-before-impact epochs."""

    if env.t_earth is None or env.x_earth_hist is None:
        raise ValueError("Earth trajectory must be available before calling prepare_mbi.")

    if asteroid.t_hist is None or asteroid.x_hist is None:
        raise ValueError(f'Asteroid "{asteroid.name}" must be propagated before calling prepare_mbi.')

    _, moid = get_ca_moid(env.x_earth_hist, asteroid.x_hist, env.t_earth, asteroid.t_hist)
    asteroid.moid_pre_km = moid.d_km
    asteroid.ca_pre_km = moid.d_km

    if force_impact:
        forced_state_ast = moid.state_ast.copy()
        forced_state_ast[:3] = moid.state_earth[:3]
        asteroid.moid_pre_km = 0.0
        asteroid.ca_pre_km = 0.0
        if not asteroid.name.endswith("-forced"):
            asteroid.name = f"{asteroid.name}-forced"
        state_at_moid = ClosestApproach(
            d_km=0.0,
            idx_earth=moid.idx_earth,
            idx_ast=moid.idx_ast,
            time_earth=moid.time_earth,
            time_ast=moid.time_ast,
            state_earth=moid.state_earth.copy(),
            state_ast=forced_state_ast,
        )
    else:
        state_at_moid = moid

    mbi_states = get_states_at_mbi(
        state_at_moid,
        months_back,
        mu_sun_km=env.mu_sun_km,
        rtol=rtol,
        atol=atol,
        method=method,
    )
    return asteroid, mbi_states, state_at_moid
