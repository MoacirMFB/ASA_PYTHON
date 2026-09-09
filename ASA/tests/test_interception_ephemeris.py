"""The interception workflow, driven from SPICE kernels instead of two-body.

Skipped when the kernel cache is empty; see ``test_ephemeris`` for how to fill it.
"""

import numpy as np
import pytest

from .. import interception
from ..ephemeris import (
    EARTH,
    common_coverage,
    et_to_utc,
    kernel_dir,
    load_kernels,
    state,
    utc_to_et,
)
from ..keplerian import propagate_two_body

KERNEL = "2024_PDC25a-s3-merged-DE441.bsp"
BODY = "-937020"
LEAPSECONDS = "naif0012.tls"
EARTH_RADIUS_KM = 6378.137
MU_SUN = 1.32712440018e11
SECONDS_PER_YEAR = 365.25 * 86400.0


def _environment(years: float = 0.02, step_min: float = 60.0):
    """An Environment whose window ends exactly at the kernel's usable end."""

    for name in (LEAPSECONDS, KERNEL):
        if not (kernel_dir() / name).is_file():
            pytest.skip(f"{name} not in the kernel cache")
    kernel_path = kernel_dir() / KERNEL
    load_kernels([kernel_dir() / LEAPSECONDS, kernel_path])
    end_et_s = common_coverage(kernel_path, [BODY, EARTH]).end_et_s
    environment = interception.make_env(years=years, step_min=step_min)
    environment.ephemeris = interception.EphemerisSource(
        epoch_et_s=end_et_s - years * SECONDS_PER_YEAR
    )
    return environment, end_et_s


def test_propagators_return_exactly_what_the_kernel_holds():
    environment, _ = _environment()
    times_s, earth_states = interception.propagate_earth(environment)
    asteroid = interception.propagate_asteroid(
        interception.get_asteroid_from_ephemeris(
            "2024 PDC25", BODY, environment.ephemeris.epoch_et_s
        ),
        environment,
    )
    assert earth_states.shape == (times_s.size, 6)
    assert asteroid.x_hist.shape == (times_s.size, 6)

    # Not "close to" - the histories are lookups, so they must match bit for bit.
    for index in (0, times_s.size // 3, times_s.size - 1):
        epoch_et_s = environment.ephemeris.epoch_et_s + times_s[index]
        np.testing.assert_array_equal(earth_states[index], state(EARTH, epoch_et_s))
        np.testing.assert_array_equal(asteroid.x_hist[index], state(BODY, epoch_et_s))


def test_osculating_elements_describe_the_published_orbit():
    environment, _ = _environment()
    asteroid = interception.get_asteroid_from_ephemeris(
        "2024 PDC25", BODY, environment.ephemeris.epoch_et_s
    )
    semi_major_axis_au, eccentricity, inclination_rad = asteroid.coe[:3]
    # CNEOS: perihelion 1.0 au, aphelion 2.29 au, period 774 d, inclination ~11 deg.
    assert semi_major_axis_au == pytest.approx(1.65, abs=0.05)
    assert eccentricity == pytest.approx(0.39, abs=0.02)
    assert np.degrees(inclination_rad) == pytest.approx(11.0, abs=0.5)
    assert asteroid.spice_id == BODY


def test_mbi_states_are_sampled_not_back_propagated():
    """The whole point: earlier states come from the kernel, exactly."""

    environment, _ = _environment()
    interception.propagate_earth(environment)
    asteroid = interception.propagate_asteroid(
        interception.get_asteroid_from_ephemeris(
            "2024 PDC25", BODY, environment.ephemeris.epoch_et_s
        ),
        environment,
    )
    asteroid, mbi_states, at_moid = interception.prepare_mbi(
        asteroid, environment, [0.0, 6.0, 12.0], force_impact=False
    )
    assert len(mbi_states) == 3
    for mbi_state in mbi_states:
        offset_s = mbi_state.month * 30.0 * 86400.0
        np.testing.assert_array_equal(
            mbi_state.state_ast,
            state(BODY, environment.ephemeris.epoch_et_s + at_moid.time_ast - offset_s),
        )
        np.testing.assert_array_equal(
            mbi_state.state_earth,
            state(EARTH, environment.ephemeris.epoch_et_s + at_moid.time_earth - offset_s),
        )


def test_asteroid_without_a_spice_id_is_rejected():
    environment, _ = _environment()
    catalog_asteroid = interception.get_asteroid("Apophis")
    with pytest.raises(ValueError, match="no spice_id"):
        interception.propagate_asteroid(catalog_asteroid, environment)


def test_force_impact_is_refused_for_a_real_trajectory():
    environment, _ = _environment()
    interception.propagate_earth(environment)
    asteroid = interception.propagate_asteroid(
        interception.get_asteroid_from_ephemeris(
            "2024 PDC25", BODY, environment.ephemeris.epoch_et_s
        ),
        environment,
    )
    with pytest.raises(ValueError, match="force_impact"):
        interception.prepare_mbi(asteroid, environment, [0.0], force_impact=True)


def test_two_body_error_exceeds_the_whole_impact_risk_chord():
    """Why the ephemeris is needed at all, stated as a test.

    Deflection is measured against Earth's capture cross-section, whose chord in
    the b-plane is about 21,760 km. Two-body propagation of this asteroid drifts
    further than that within four months of the encounter, so a two-body nominal
    cannot resolve the quantity the whole analysis is about.
    """

    for name in (LEAPSECONDS, KERNEL):
        if not (kernel_dir() / name).is_file():
            pytest.skip(f"{name} not in the kernel cache")
    kernel_path = kernel_dir() / KERNEL
    load_kernels([kernel_dir() / LEAPSECONDS, kernel_path])
    end_et_s = common_coverage(kernel_path, [BODY, EARTH]).end_et_s

    # Two days before the encounter, so the flyby's own deflection is excluded.
    target_et_s = end_et_s - 2.0 * 86400.0
    truth = state(BODY, target_et_s)

    start_et_s = utc_to_et("2041 JAN 01 00:00:00")
    _, history = propagate_two_body(
        state(BODY, start_et_s),
        np.array([0.0, target_et_s - start_et_s]),
        MU_SUN,
        rtol=1e-12,
        atol=1e-12,
        method="DOP853",
    )
    error_km = np.linalg.norm(history[-1][:3] - truth[:3])
    assert (target_et_s - start_et_s) / SECONDS_PER_YEAR < 0.35  # under four months
    assert error_km > 21_760.0
