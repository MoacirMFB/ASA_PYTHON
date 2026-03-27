import numpy as np
import pytest

from ..keplerian import (
    coe_to_cartesian,
    coe_to_cartesian_elements,
    coe_to_cartesian_spice,
    dynamics_2bp_cartesian,
    jacobian_2bp_cartesian,
    propagate_stm_2bp,
)


def test_dynamics_2bp_cartesian_matches_expected_axis_acceleration():
    mu = 398600.4418
    state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 1.0])
    xdot = dynamics_2bp_cartesian(0.0, state, mu)
    assert xdot.shape == (6,)
    np.testing.assert_allclose(xdot[:3], state[3:])
    np.testing.assert_allclose(xdot[3:], np.array([-mu / 7000.0**2, 0.0, 0.0]), rtol=1e-12, atol=1e-12)


def test_jacobian_2bp_cartesian_matches_finite_difference():
    mu = 398600.4418
    state = np.array([7000.0, -1200.0, 300.0, 1.2, 7.1, -0.8])
    analytic = jacobian_2bp_cartesian(0.0, state, mu)
    numeric = np.zeros((6, 6))
    eps = 1e-6
    for idx in range(6):
        delta = np.zeros(6)
        delta[idx] = eps
        f_plus = dynamics_2bp_cartesian(0.0, state + delta, mu)
        f_minus = dynamics_2bp_cartesian(0.0, state - delta, mu)
        numeric[:, idx] = (f_plus - f_minus) / (2.0 * eps)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=2e-8)


def test_propagate_stm_2bp_zero_span_returns_identity():
    state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    phi, x_final = propagate_stm_2bp(state, 398600.4418, (0.0, 0.0))
    np.testing.assert_allclose(phi, np.eye(6))
    np.testing.assert_allclose(x_final, state)


def test_coe_to_cartesian_circular_equatorial_case():
    mu = 398600.4418
    state = coe_to_cartesian([7000.0, 0.0, 0.0, 0.0, 0.0, 0.0], mu, use_true_anomaly=True)
    expected_speed = np.sqrt(mu / 7000.0)
    np.testing.assert_allclose(state, np.array([7000.0, 0.0, 0.0, 0.0, expected_speed, 0.0]), rtol=1e-12, atol=1e-12)


def test_coe_to_cartesian_matches_repository_apophis_case():
    mu_sun_km = 132712440018.0
    state = coe_to_cartesian(
        [
            0.9225521 * 149597870.7,
            0.1912907,
            np.deg2rad(3.33974),
            np.deg2rad(203.91537),
            np.deg2rad(126.68323),
            0.0,
        ],
        mu_sun_km,
        use_true_anomaly=True,
    )
    expected = np.array(
        [
            97174473.6704443,
            -54653892.9939707,
            5214352.61923254,
            18.4922263853679,
            32.7541623882571,
            -1.30982313944453,
        ]
    )
    np.testing.assert_allclose(state, expected, rtol=1e-10, atol=1e-8)


def test_coe_to_cartesian_spice_returns_valid_state_when_available():
    pytest.importorskip("spiceypy")
    state = coe_to_cartesian_spice(
        [
            0.9225521 * 149597870.7,
            0.1912907,
            np.deg2rad(3.33974),
            np.deg2rad(203.91537),
            np.deg2rad(126.68323),
            0.0,
        ],
        132712440018.0,
        use_true_anomaly=True,
    )
    assert state.shape == (6,)
    assert np.all(np.isfinite(state))


def test_coe_to_cartesian_elements_matches_aerospace_sign_convention():
    state = coe_to_cartesian_elements(
        [
            0.9225521 * 149597870.7,
            0.1912907,
            np.deg2rad(3.33974),
            np.deg2rad(203.91537),
            np.deg2rad(126.68323),
            0.0,
        ],
        132712440018.0,
        use_true_anomaly=True,
    )
    expected = np.array(
        [
            97174473.6704443,
            -54653892.9939707,
            5214352.61923254,
            18.4922263853679,
            32.7541623882571,
            -1.30982313944453,
        ]
    )
    np.testing.assert_allclose(state, expected, rtol=1e-10, atol=1e-8)
