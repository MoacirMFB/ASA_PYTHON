"""Tests for the SPICE ephemeris layer.

Anything needing an SPK is skipped when the kernel cache is empty, so the suite
still runs offline. Populate the cache with::

    python -c "from ASA import ephemeris as e; e.ensure_leapseconds(); \\
        [e.fetch_kernel('https://ssd.jpl.nasa.gov/ftp/xfr/2024_PDC25/' + n) for n in \\
         ('2024_PDC25-s1-merged-DE441.bsp', '2024_PDC25a-s3-merged-DE441.bsp')]"
"""

import numpy as np
import pytest

from .. import bplane
from ..ephemeris import (
    EARTH,
    SUN,
    Coverage,
    bodies_in_kernel,
    common_coverage,
    coverage,
    et_to_utc,
    fetch_kernel,
    kernel_dir,
    load_kernels,
    state,
    utc_to_et,
)

PDC25_KERNEL = "2024_PDC25a-s3-merged-DE441.bsp"
PDC25_BODY = "-937020"
EARTH_RADIUS_KM = 6378.137
EARTH_MU = 398600.4418
LEAPSECONDS = "naif0012.tls"


def _kernels_or_skip():
    """Load the PDC25 kernel and leapseconds, or skip."""

    paths = [kernel_dir() / LEAPSECONDS, kernel_dir() / PDC25_KERNEL]
    for path in paths:
        if not path.is_file():
            pytest.skip(f"{path.name} not in the kernel cache; see module docstring")
    load_kernels(paths)
    return paths[1]


def test_coverage_reports_outer_bounds_and_membership():
    interval = Coverage(body="x", intervals=((0.0, 10.0), (20.0, 25.0)))
    assert interval.start_et_s == 0.0
    assert interval.end_et_s == 25.0
    assert interval.duration_s == 15.0  # gaps are not counted
    assert interval.contains(5.0)
    assert interval.contains(22.0)
    assert not interval.contains(15.0)


def test_fetch_kernel_refuses_insecure_urls():
    with pytest.raises(ValueError, match="https"):
        fetch_kernel("http://example.com/some.bsp")
    with pytest.raises(ValueError, match="https"):
        fetch_kernel("ftp://example.com/some.bsp")


def test_sphere_of_influence_matches_the_known_earth_value():
    radius_km = bplane.sphere_of_influence_radius(EARTH_MU, 1.32712440018e11, 149597870.7)
    assert radius_km == pytest.approx(925_000.0, rel=0.01)


def test_time_conversion_round_trips():
    if not (kernel_dir() / LEAPSECONDS).is_file():
        pytest.skip("leapseconds kernel not cached")
    load_kernels([kernel_dir() / LEAPSECONDS])
    et_s = utc_to_et("2041 APR 24 16:23:47.836")
    assert et_to_utc(et_s, decimals=3).startswith("2041 APR 24 16:23:47.83")


def test_kernel_bundles_planets_alongside_the_asteroid():
    kernel_path = _kernels_or_skip()
    bodies = bodies_in_kernel(kernel_path)
    assert int(PDC25_BODY) in bodies
    assert 399 in bodies  # Earth, from the merged DE441
    assert 10 in bodies  # Sun


def test_common_coverage_is_shorter_than_either_body_alone():
    """The asteroid outlives Earth in this kernel by one second."""

    kernel_path = _kernels_or_skip()
    asteroid = coverage(kernel_path, PDC25_BODY)
    earth = coverage(kernel_path, EARTH)
    both = common_coverage(kernel_path, [PDC25_BODY, EARTH])
    assert both.end_et_s == min(asteroid.end_et_s, earth.end_et_s)
    assert both.end_et_s < asteroid.end_et_s
    # The asteroid's own last epoch is unusable for a relative state.
    with pytest.raises(Exception):
        state(PDC25_BODY, asteroid.end_et_s, observer=EARTH)


def test_state_shapes_follow_the_epoch_argument():
    _kernels_or_skip()
    et_s = utc_to_et("2035 JAN 01 00:00:00")
    assert state(PDC25_BODY, et_s).shape == (6,)
    assert state(PDC25_BODY, [et_s, et_s + 86400.0]).shape == (2, 6)


def test_pdc25_nominal_impacts_earth():
    """The distributed nominal is an impact, and the b-plane says so too."""

    kernel_path = _kernels_or_skip()
    end_et_s = common_coverage(kernel_path, [PDC25_BODY, EARTH]).end_et_s

    # Separation is still falling at the end of coverage, and is already well
    # inside Earth: the kernel stops at the impact.
    separation_km = np.linalg.norm(state(PDC25_BODY, end_et_s, observer=EARTH)[:3])
    assert separation_km < EARTH_RADIUS_KM

    # Six hours out, comfortably inside the sphere of influence, the encounter
    # reduces to a hyperbola whose impact parameter is below the capture radius.
    et_s = end_et_s - 6.0 * 3600.0
    relative = state(PDC25_BODY, et_s, observer=EARTH)
    earth_velocity = state(EARTH, et_s, observer=SUN)[3:]
    assert np.linalg.norm(relative[:3]) < bplane.sphere_of_influence_radius(
        EARTH_MU, 1.32712440018e11, 149597870.7
    )

    encounter = bplane.bplane_coordinates(relative[:3], relative[3:], earth_velocity, EARTH_MU)
    capture_km = bplane.capture_impact_parameter(
        EARTH_RADIUS_KM, encounter.v_infinity_kmps, EARTH_MU
    )
    assert encounter.b_km < capture_km
    assert encounter.periapsis_radius_km < EARTH_RADIUS_KM


def test_pdc25_capture_chord_agrees_with_the_published_value():
    """Independently derived chord must land on the paper's 21850 km."""

    kernel_path = _kernels_or_skip()
    end_et_s = common_coverage(kernel_path, [PDC25_BODY, EARTH]).end_et_s
    et_s = end_et_s - 6.0 * 3600.0
    relative = state(PDC25_BODY, et_s, observer=EARTH)
    earth_velocity = state(EARTH, et_s, observer=SUN)[3:]
    encounter = bplane.bplane_coordinates(relative[:3], relative[3:], earth_velocity, EARTH_MU)

    assert encounter.v_infinity_kmps == pytest.approx(8.09, abs=0.05)
    capture_km = bplane.capture_impact_parameter(
        EARTH_RADIUS_KM, encounter.v_infinity_kmps, EARTH_MU
    )
    assert 2.0 * capture_km == pytest.approx(21850.0, rel=0.01)

    # Deflecting either way must together span the whole capture cross-section,
    # which is what the paper's 0.34 C and 0.70 C requirements encode.
    northward_km = capture_km - encounter.zeta_km
    southward_km = capture_km + encounter.zeta_km
    assert northward_km + southward_km == pytest.approx(2.0 * capture_km, rel=1e-12)
    assert northward_km / (2.0 * capture_km) == pytest.approx(0.34, abs=0.03)
    assert southward_km / (2.0 * capture_km) == pytest.approx(0.70, abs=0.03)


def test_bplane_coordinates_are_stable_inside_the_sphere_of_influence():
    kernel_path = _kernels_or_skip()
    end_et_s = common_coverage(kernel_path, [PDC25_BODY, EARTH]).end_et_s
    values = []
    for hours in (6.0, 3.0, 1.0):
        et_s = end_et_s - hours * 3600.0
        relative = state(PDC25_BODY, et_s, observer=EARTH)
        earth_velocity = state(EARTH, et_s, observer=SUN)[3:]
        values.append(bplane.bplane_coordinates(relative[:3], relative[3:], earth_velocity, EARTH_MU))
    for sampled in values[1:]:
        assert sampled.b_km == pytest.approx(values[0].b_km, rel=2e-3)
        assert sampled.v_infinity_kmps == pytest.approx(values[0].v_infinity_kmps, rel=1e-4)
