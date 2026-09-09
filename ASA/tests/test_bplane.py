import numpy as np
import pytest
from scipy.integrate import solve_ivp

from ..bplane import (
    bplane_coordinates,
    bplane_frame,
    capture_impact_parameter,
    hyperbolic_encounter,
    impact_parameter_from_periapsis,
    periapsis_from_impact_parameter,
)
from ..keplerian import dynamics_2bp_cartesian

EARTH_MU = 398600.4418
EARTH_RADIUS = 6378.137


def _hyperbolic_state(periapsis_km, v_infinity_kmps, mu=EARTH_MU):
    """State at periapsis of a hyperbola in the x-y plane, periapsis along +x."""

    speed = np.sqrt(v_infinity_kmps**2 + 2.0 * mu / periapsis_km)
    return np.array([periapsis_km, 0.0, 0.0]), np.array([0.0, speed, 0.0])


def test_frame_matches_valsecchi_closed_form():
    """The vector construction must reproduce the published theta-phi expressions."""

    planet_velocity = np.array([0.0, 1.0, 0.0])  # Y axis, by definition of the frame
    for theta in (0.3, 1.1, 2.4):
        for phi in (-2.0, 0.0, 0.7, 2.9):
            u_vec = 9.0 * np.array(
                [np.sin(phi) * np.sin(theta), np.cos(theta), np.cos(phi) * np.sin(theta)]
            )
            xi_hat, eta_hat, zeta_hat = bplane_frame(u_vec, planet_velocity)
            np.testing.assert_allclose(
                xi_hat, [np.cos(phi), 0.0, -np.sin(phi)], atol=1e-12
            )
            np.testing.assert_allclose(eta_hat, u_vec / np.linalg.norm(u_vec), atol=1e-12)
            np.testing.assert_allclose(
                zeta_hat,
                [np.sin(phi) * np.cos(theta), -np.sin(theta), np.cos(phi) * np.cos(theta)],
                atol=1e-12,
            )


def test_frame_is_right_handed_orthonormal():
    xi_hat, eta_hat, zeta_hat = bplane_frame([3.0, -5.0, 2.0], [0.4, 8.0, -1.0])
    basis = np.array([xi_hat, eta_hat, zeta_hat])
    np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.cross(xi_hat, eta_hat), zeta_hat, atol=1e-12)


def test_frame_rejects_u_parallel_to_planet_velocity():
    with pytest.raises(ValueError, match="undefined when U is parallel"):
        bplane_frame([0.0, 5.0, 0.0], [0.0, 30.0, 0.0])


def test_impact_parameter_is_perpendicular_to_asymptote():
    position, velocity = _hyperbolic_state(12000.0, 6.0)
    encounter = hyperbolic_encounter(position, velocity, EARTH_MU)
    assert np.dot(encounter.b_vector_km, encounter.s_hat) == pytest.approx(0.0, abs=1e-9)
    assert np.linalg.norm(encounter.b_vector_km) == pytest.approx(encounter.b_km, rel=1e-12)
    assert encounter.periapsis_radius_km == pytest.approx(12000.0, rel=1e-12)
    assert encounter.v_infinity_kmps == pytest.approx(6.0, rel=1e-12)


def test_asymptote_matches_a_numerically_propagated_trajectory():
    """Propagate far upstream; the straight-line miss must converge on b as 1/r.

    A real trajectory only reaches its asymptote at infinity, so a check at
    finite range cannot match exactly. The residual is the leftover curvature,
    of relative size mu / (v_inf^2 r), so verifying that the error follows that
    law - shrinking tenfold for every tenfold in range - tests the analytic
    asymptote far more sharply than any single loose tolerance would.
    """

    position, velocity = _hyperbolic_state(9000.0, 7.5)
    encounter = hyperbolic_encounter(position, velocity, EARTH_MU)

    errors = []
    for duration_s in (-4.0e6, -4.0e7):
        solution = solve_ivp(
            dynamics_2bp_cartesian,
            (0.0, duration_s),
            np.concatenate([position, velocity]),
            args=(EARTH_MU,),
            rtol=1e-12,
            atol=1e-9,
        )
        far_position = solution.y[:3, -1]
        far_velocity = solution.y[3:, -1]

        direction = far_velocity / np.linalg.norm(far_velocity)
        offset = far_position - np.dot(far_position, direction) * direction

        # Direction of the miss, and of travel, converge much faster than its size.
        np.testing.assert_allclose(
            offset / np.linalg.norm(offset), encounter.b_vector_km / encounter.b_km, atol=1e-4
        )
        np.testing.assert_allclose(direction, encounter.s_hat, atol=1e-4)

        predicted = EARTH_MU / (encounter.v_infinity_kmps**2 * np.linalg.norm(far_position))
        observed = abs(np.linalg.norm(offset) - encounter.b_km) / encounter.b_km
        assert observed == pytest.approx(predicted, rel=1e-2)
        errors.append(observed)

    assert errors[1] == pytest.approx(errors[0] / 10.0, rel=1e-2)
    assert errors[1] < 1e-4


def test_coordinates_are_invariant_along_the_trajectory():
    """xi and zeta describe the encounter, so they cannot depend on when we look."""

    position, velocity = _hyperbolic_state(9000.0, 7.5)
    planet_velocity = np.array([0.0, 29.78, 0.0])
    reference = bplane_coordinates(position, velocity, planet_velocity, EARTH_MU)

    solution = solve_ivp(
        dynamics_2bp_cartesian,
        (0.0, -3.0e5),
        np.concatenate([position, velocity]),
        args=(EARTH_MU,),
        rtol=1e-12,
        atol=1e-9,
        t_eval=np.linspace(0.0, -3.0e5, 7),
    )
    for column in solution.y.T:
        sampled = bplane_coordinates(column[:3], column[3:], planet_velocity, EARTH_MU)
        assert sampled.xi_km == pytest.approx(reference.xi_km, rel=1e-6, abs=1e-3)
        assert sampled.zeta_km == pytest.approx(reference.zeta_km, rel=1e-6, abs=1e-3)


def test_coordinates_decompose_the_impact_parameter():
    position, velocity = _hyperbolic_state(15000.0, 5.0)
    coordinates = bplane_coordinates(position, velocity, [1.0, 29.0, 0.3], EARTH_MU)
    assert np.hypot(coordinates.xi_km, coordinates.zeta_km) == pytest.approx(
        coordinates.b_km, rel=1e-12
    )


def test_periapsis_and_impact_parameter_round_trip():
    for periapsis_km in (6600.0, 7089.0, 42164.0):
        for v_infinity_kmps in (3.0, 8.04, 15.0):
            b_km = impact_parameter_from_periapsis(periapsis_km, v_infinity_kmps, EARTH_MU)
            assert b_km > periapsis_km  # gravitational focusing
            assert periapsis_from_impact_parameter(b_km, v_infinity_kmps, EARTH_MU) == pytest.approx(
                periapsis_km, rel=1e-12
            )


def test_pdc25_capture_cross_section_reproduces_the_published_chord():
    """PDC25's impact risk chord fixes v_infinity; check we land on it.

    Franzese et al. (2025) give the Earth gravitational capture cross-section for
    the 2041 encounter as a chord of C = 21850 km, so the capture radius in the
    b-plane is C / 2 = 10925 km. Inverting the focusing relation at Earth's
    radius puts the encounter v_infinity at about 8.04 km/s.
    """

    chord_km = 21850.0
    v_infinity_kmps = 8.0398
    capture_km = capture_impact_parameter(EARTH_RADIUS, v_infinity_kmps, EARTH_MU)
    assert 2.0 * capture_km == pytest.approx(chord_km, abs=5.0)

    # The paper's northward and southward deflection requirements are quoted both
    # as fractions of the chord and as multiples of Earth's radius. That the two
    # spellings agree, and that the pair spans the whole chord, confirms they are
    # deflections *across* the capture cross-section rather than final positions.
    assert 0.34 * chord_km == pytest.approx(1.16 * EARTH_RADIUS, rel=0.01)
    assert 0.70 * chord_km == pytest.approx(2.41 * EARTH_RADIUS, rel=0.01)
    assert 0.34 + 0.70 == pytest.approx(1.0, abs=0.05)


def test_impact_is_exactly_b_below_the_capture_radius():
    v_infinity_kmps = 8.0398
    capture_km = capture_impact_parameter(EARTH_RADIUS, v_infinity_kmps, EARTH_MU)
    assert periapsis_from_impact_parameter(
        capture_km * 0.999, v_infinity_kmps, EARTH_MU
    ) < EARTH_RADIUS
    assert periapsis_from_impact_parameter(
        capture_km * 1.001, v_infinity_kmps, EARTH_MU
    ) > EARTH_RADIUS


def test_bound_and_radial_states_are_rejected():
    with pytest.raises(ValueError, match="hyperbolic"):
        hyperbolic_encounter([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0], EARTH_MU)
    with pytest.raises(ValueError, match="radial"):
        hyperbolic_encounter([7000.0, 0.0, 0.0], [20.0, 0.0, 0.0], EARTH_MU)


def test_unperturbed_converges_on_the_hyperbolic_result_upstream():
    """The two constructions must agree where the planet has not yet bent the path.

    Far upstream the real trajectory is still effectively straight, so the
    straight-line impact parameter approaches the hyperbolic one with the same
    mu / (v_inf^2 r) law that governs the asymptote itself.
    """

    from ..bplane import bplane_coordinates_unperturbed

    position, velocity = _hyperbolic_state(9000.0, 7.5)
    planet_velocity = np.array([0.0, 29.78, 0.0])
    reference = bplane_coordinates(position, velocity, planet_velocity, EARTH_MU)

    errors = []
    for duration_s in (-4.0e6, -4.0e7):
        solution = solve_ivp(
            dynamics_2bp_cartesian,
            (0.0, duration_s),
            np.concatenate([position, velocity]),
            args=(EARTH_MU,),
            rtol=1e-12,
            atol=1e-9,
        )
        far = solution.y[:, -1]
        straight = bplane_coordinates_unperturbed(far[:3], far[3:], planet_velocity, EARTH_MU)
        predicted = EARTH_MU / (reference.v_infinity_kmps**2 * np.linalg.norm(far[:3]))
        observed = abs(straight.b_km - reference.b_km) / reference.b_km
        assert observed == pytest.approx(predicted, rel=2e-2)
        errors.append(observed)
    assert errors[1] == pytest.approx(errors[0] / 10.0, rel=2e-2)


def test_unperturbed_accepts_a_state_the_hyperbolic_form_must_reject():
    """A gravity-free trajectory close in is bound, yet still has a b-plane."""

    from ..bplane import bplane_coordinates_unperturbed

    position = np.array([3000.0, 4000.0, 0.0])   # inside Earth, slow enough to be bound
    velocity = np.array([0.0, 0.0, 8.0])
    planet_velocity = np.array([0.0, 29.78, 0.0])
    with pytest.raises(ValueError, match="hyperbolic"):
        hyperbolic_encounter(position, velocity, EARTH_MU)
    coordinates = bplane_coordinates_unperturbed(position, velocity, planet_velocity)
    assert coordinates.b_km == pytest.approx(5000.0, rel=1e-12)  # offset perpendicular to +z
    assert np.hypot(coordinates.xi_km, coordinates.zeta_km) == pytest.approx(5000.0, rel=1e-12)
