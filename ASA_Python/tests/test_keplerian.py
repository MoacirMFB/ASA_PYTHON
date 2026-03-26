import numpy as np

from ..keplerian import (
    coe_to_cartesian,
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
