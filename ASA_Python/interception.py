"""Minimal interception workflow port needed by the virtual thrust script."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .bodies import AU_KM, CelestialBody, get_celestial_body
from .keplerian import coe_to_cartesian, coe_to_cartesian_elements, coe_to_cartesian_spice, propagate_two_body

FloatArray = NDArray[np.float64]


@dataclass
class Environment:
    """Propagation settings and shared constants for the workflow."""

    muSun_km: float
    muSun_AU: float
    AU2km: float
    R_E_km: float
    years: float
    step_min: float
    tspan: FloatArray
    rtol: float = 1e-13
    atol: float = 1e-13
    method: str = "DOP853"
    t_Earth: FloatArray | None = None
    X_Earth_hist: FloatArray | None = None


@dataclass
class AsteroidRecord:
    """Asteroid catalog entry and propagated state history."""

    name: str
    coe: FloatArray
    t_hist: FloatArray | None = None
    X_hist: FloatArray | None = None
    init_state_km: FloatArray | None = None
    MOID_pre_km: float | None = None
    CA_pre_km: float | None = None


@dataclass
class ClosestApproach:
    """Closest-approach or MOID record."""

    d_km: float
    idxEarth: int
    idxAst: int
    timeEarth: float
    timeAst: float
    stateEarth: FloatArray
    stateAst: FloatArray


@dataclass
class MBIState:
    """Earth and asteroid states a given number of months before impact."""

    month: float
    stateEarth: FloatArray
    stateAst: FloatArray


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
        muSun_km=sun.mu.km,
        muSun_AU=sun.mu.AU,
        AU2km=AU_KM,
        R_E_km=earth.radius.km,
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
            np.deg2rad(orbit.RAAN_deg),
            np.deg2rad(orbit.arg_peri_deg),
            np.deg2rad(orbit.M0_deg),
        ],
        dtype=float,
    )
    x0 = coe_to_cartesian(x_coe, env.muSun_km, use_true_anomaly=False)
    t_earth, x_earth = propagate_two_body(
        x0,
        env.tspan,
        env.muSun_km,
        rtol=env.rtol,
        atol=env.atol,
        method=env.method,
    )
    env.t_Earth = t_earth
    env.X_Earth_hist = x_earth
    return t_earth, x_earth


def ast_catalog(names: str | Sequence[str]) -> list[AsteroidRecord]:
    """Return the minimal asteroid catalog subset used by the workflow."""

    if isinstance(names, str):
        requested = [names]
    else:
        requested = list(names)

    asteroids: list[AsteroidRecord] = []
    for name in requested:
        if name not in _ASTEROID_CATALOG:
            raise KeyError(f'Asteroid "{name}" not in catalog.')
        asteroids.append(
            AsteroidRecord(
                name=name,
                coe=np.asarray(_ASTEROID_CATALOG[name], dtype=float).copy(),
            )
        )
    return asteroids


def propagate_asteroids(
    asteroids: list[AsteroidRecord],
    env: Environment,
    *,
    converter: str = "repo",
) -> list[AsteroidRecord]:
    """Propagate asteroid heliocentric histories from catalog COEs."""

    for asteroid in asteroids:
        coe = asteroid.coe.copy()
        x_coe = coe.copy()
        x_coe[0] *= env.AU2km
        if converter == "repo":
            x0 = coe_to_cartesian(x_coe, env.muSun_km, use_true_anomaly=True)
        elif converter == "spice":
            x0 = coe_to_cartesian_spice(x_coe, env.muSun_km, use_true_anomaly=True)
        elif converter == "elements":
            x0 = coe_to_cartesian_elements(x_coe, env.muSun_km, use_true_anomaly=True)
        else:
            raise ValueError(f"Unsupported asteroid converter: {converter}")
        t_hist, x_hist = propagate_two_body(
            x0,
            env.tspan,
            env.muSun_km,
            rtol=env.rtol,
            atol=env.atol,
            method=env.method,
        )
        asteroid.init_state_km = x0.copy()
        asteroid.t_hist = t_hist
        asteroid.X_hist = x_hist
    return asteroids


def make_bodies_for_plot(asteroids: list[AsteroidRecord], env: Environment) -> list[dict[str, object]]:
    """Package Earth and asteroid trajectories for deferred plotting work."""

    if env.t_Earth is None or env.X_Earth_hist is None:
        raise ValueError("Earth trajectory has not been propagated.")
    bodies: list[dict[str, object]] = [
        {"name": "Earth", "X_hist": env.X_Earth_hist, "t_hist": env.t_Earth}
    ]
    for asteroid in asteroids:
        if asteroid.t_hist is None or asteroid.X_hist is None:
            raise ValueError(f'Asteroid "{asteroid.name}" has not been propagated.')
        bodies.append({"name": asteroid.name, "X_hist": asteroid.X_hist, "t_hist": asteroid.t_hist})
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
        idxEarth=k_ca,
        idxAst=k_ca,
        timeEarth=float(t_earth[k_ca]),
        timeAst=float(t_asteroid[k_ca]),
        stateEarth=earth[k_ca].copy(),
        stateAst=asteroid[k_ca].copy(),
    )

    tree = cKDTree(asteroid[:, :3])
    distances, idx_ast = tree.query(earth[:, :3], k=1)
    idx_earth = int(np.argmin(distances))
    idx_ast_moid = int(idx_ast[idx_earth])
    moid = ClosestApproach(
        d_km=float(distances[idx_earth]),
        idxEarth=idx_earth,
        idxAst=idx_ast_moid,
        timeEarth=float(t_earth[idx_earth]),
        timeAst=float(t_asteroid[idx_ast_moid]),
        stateEarth=earth[idx_earth].copy(),
        stateAst=asteroid[idx_ast_moid].copy(),
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
                    stateEarth=states_at_moid.stateEarth.copy(),
                    stateAst=states_at_moid.stateAst.copy(),
                )
            )
            continue

        _, earth_hist = propagate_two_body(
            states_at_moid.stateEarth,
            np.array([0.0, -dt], dtype=float),
            mu,
            rtol=rtol,
            atol=atol,
            method=method,
        )
        _, asteroid_hist = propagate_two_body(
            states_at_moid.stateAst,
            np.array([0.0, -dt], dtype=float),
            mu,
            rtol=rtol,
            atol=atol,
            method=method,
        )
        mbi_states.append(
            MBIState(
                month=float(month),
                stateEarth=earth_hist[-1].copy(),
                stateAst=asteroid_hist[-1].copy(),
            )
        )
    return mbi_states


def prepare_mbi(
    asteroids: list[AsteroidRecord],
    env: Environment,
    months_back: Iterable[float] | FloatArray,
    force_impact: bool,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    method: str = "DOP853",
) -> tuple[list[AsteroidRecord], list[list[MBIState]], ClosestApproach]:
    """Prepare Earth/asteroid state sets at months-before-impact epochs."""

    if env.t_Earth is None or env.X_Earth_hist is None:
        raise ValueError("Earth trajectory must be available before calling prepare_mbi.")

    mbi_sets: list[list[MBIState]] = []
    state_at_moid_final: ClosestApproach | None = None

    for asteroid in asteroids:
        if asteroid.t_hist is None or asteroid.X_hist is None:
            raise ValueError(f'Asteroid "{asteroid.name}" must be propagated before calling prepare_mbi.')

        _, moid = get_ca_moid(env.X_Earth_hist, asteroid.X_hist, env.t_Earth, asteroid.t_hist)
        asteroid.MOID_pre_km = moid.d_km
        asteroid.CA_pre_km = moid.d_km

        if force_impact:
            forced_state_ast = moid.stateAst.copy()
            forced_state_ast[:3] = moid.stateEarth[:3]
            asteroid.MOID_pre_km = 0.0
            asteroid.CA_pre_km = 0.0
            if not asteroid.name.endswith("-forced"):
                asteroid.name = f"{asteroid.name}-forced"
            state_at_moid_final = ClosestApproach(
                d_km=0.0,
                idxEarth=moid.idxEarth,
                idxAst=moid.idxAst,
                timeEarth=moid.timeEarth,
                timeAst=moid.timeAst,
                stateEarth=moid.stateEarth.copy(),
                stateAst=forced_state_ast,
            )
        else:
            state_at_moid_final = moid

        mbi_sets.append(
            get_states_at_mbi(
                state_at_moid_final,
                months_back,
                mu_sun_km=env.muSun_km,
                rtol=rtol,
                atol=atol,
                method=method,
            )
        )

    if state_at_moid_final is None:
        raise ValueError("At least one asteroid is required.")
    return asteroids, mbi_sets, state_at_moid_final
