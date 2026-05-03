import numpy as np

from ..control import (
    linear_discrete_time_matrices,
    min_fuel_combined_dynamics,
    min_fuel_costate_dynamics,
    min_fuel_hamiltonian,
    min_fuel_optimal_control,
    min_fuel_state_dynamics,
    solve_min_fuel_bvp,
)
from ..keplerian import jacobian_2bp_cartesian


def test_linear_discrete_time_matrices_double_integrator():
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    c = np.zeros(2)
    x0 = np.array([3.0, -2.0])
    y0 = np.concatenate((x0, np.eye(2).reshape(-1, order="F"), np.zeros(2), np.zeros(2)))
    Ak, Bk, ck, x1 = linear_discrete_time_matrices(0.0, 0.1, y0, A, B, c, np.array([4.0]), 2, 1)

    np.testing.assert_allclose(Ak, np.array([[1.0, 0.1], [0.0, 1.0]]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(Bk, np.array([[0.005], [0.1]]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ck, np.zeros(2), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(x1, np.array([2.82, -1.6]), rtol=1e-12, atol=1e-12)


def test_linear_discrete_time_matrices_handles_affine_term():
    A = np.zeros((2, 2))
    B = np.eye(2)
    c = np.array([1.0, -2.0])
    x0 = np.array([4.0, 5.0])
    y0 = np.concatenate((x0, np.eye(2).reshape(-1, order="F"), np.zeros(4), np.zeros(2)))
    Ak, Bk, ck, x1 = linear_discrete_time_matrices(3.0, 5.0, y0, A, B, c, np.array([0.5, 1.5]), 2, 2)

    np.testing.assert_allclose(Ak, np.eye(2), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(Bk, 2.0 * np.eye(2), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ck, np.array([2.0, -4.0]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(x1, np.array([7.0, 4.0]), rtol=1e-12, atol=1e-12)


MU_SUN = 1.32712440018e11  # km^3/s^2
AU_KM = 1.495978707e8


def _circular_state(radius_km: float, true_anomaly: float, mu: float) -> np.ndarray:
    speed = np.sqrt(mu / radius_km)
    return np.array(
        [
            radius_km * np.cos(true_anomaly),
            radius_km * np.sin(true_anomaly),
            0.0,
            -speed * np.sin(true_anomaly),
            speed * np.cos(true_anomaly),
            0.0,
        ]
    )


def test_min_fuel_costate_dynamics_matches_negative_jacobian_transpose():
    x = np.array([1.2 * AU_KM, 0.3 * AU_KM, -0.1 * AU_KM, 5.0, 25.0, 0.5])
    lam = np.array([1e-3, -2e-3, 4e-4, 1e2, -3e2, 2e2])
    expected = -(jacobian_2bp_cartesian(0.0, x, MU_SUN).T @ lam)
    actual = min_fuel_costate_dynamics(0.0, lam, x, MU_SUN)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_min_fuel_optimal_control_outside_singular_arc():
    umax = 2.0
    rho = 1e-3

    lam_high = np.array([0.0, 0.0, 0.0, -10.0, 0.0, 0.0])  # ||p|| = 10 >> 1
    u_high = min_fuel_optimal_control(lam_high, umax, rho)
    np.testing.assert_allclose(u_high, np.array([umax, 0.0, 0.0]), rtol=1e-9, atol=1e-9)

    lam_low = np.array([0.0, 0.0, 0.0, -1e-3, 0.0, 0.0])  # ||p|| ~ 0
    u_low = min_fuel_optimal_control(lam_low, umax, rho)
    np.testing.assert_allclose(u_low, np.zeros(3), rtol=1e-9, atol=1e-9)


def test_min_fuel_optimal_control_vectorized_matches_loop():
    rng = np.random.default_rng(seed=7)
    lam_hist = rng.standard_normal((5, 6))
    umax = 1.5
    rho = 0.1
    vectorized = min_fuel_optimal_control(lam_hist, umax, rho)
    looped = np.stack([min_fuel_optimal_control(row, umax, rho) for row in lam_hist])
    np.testing.assert_allclose(vectorized, looped, rtol=1e-12, atol=1e-12)


def test_min_fuel_combined_dynamics_concatenates_state_and_costate():
    x = np.array([AU_KM, 0.0, 0.0, 0.0, 30.0, 0.0])
    lam = np.array([1e-4, 2e-4, -1e-4, 50.0, -20.0, 10.0])
    aug = np.concatenate((x, lam))
    umax = 1e-5
    rho = 0.05

    out = min_fuel_combined_dynamics(0.0, aug, MU_SUN, umax, rho)
    assert out.shape == (12,)

    u_star = min_fuel_optimal_control(lam, umax, rho)
    expected_dx = min_fuel_state_dynamics(0.0, x, u_star, MU_SUN)
    expected_dlam = min_fuel_costate_dynamics(0.0, lam, x, MU_SUN)
    np.testing.assert_allclose(out[:6], expected_dx, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out[6:], expected_dlam, rtol=1e-12, atol=1e-12)


def test_min_fuel_hamiltonian_zero_when_control_and_costate_are_zero():
    x = np.array([[AU_KM, 0.0, 0.0, 0.0, 30.0, 0.0], [AU_KM, 0.1 * AU_KM, 0.0, -1.0, 30.0, 0.0]])
    u = np.zeros((2, 3))
    lam = np.zeros((2, 6))
    H = min_fuel_hamiltonian(u, lam, x, MU_SUN)
    np.testing.assert_allclose(H, np.zeros(2), rtol=1e-12, atol=1e-12)


def test_min_fuel_hamiltonian_constant_along_combined_integration():
    from scipy.integrate import solve_ivp

    x0 = _circular_state(AU_KM, 0.0, MU_SUN)
    lam0 = np.array([1e-12, 1e-12, 0.0, 5e-7, -5e-7, 0.0])
    y0 = np.concatenate((x0, lam0))
    umax = 5e-8  # km/s^2, small so the bang-off arc stays smooth
    rho = 0.05
    period = 2.0 * np.pi * np.sqrt(AU_KM**3 / MU_SUN)
    tf = 0.05 * period
    t_eval = np.linspace(0.0, tf, 60)

    sol = solve_ivp(
        lambda t, y: min_fuel_combined_dynamics(t, y, MU_SUN, umax, rho),
        (0.0, tf),
        y0,
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
    )
    assert sol.success

    x_hist = sol.y[:6, :].T
    lam_hist = sol.y[6:, :].T
    u_hist = min_fuel_optimal_control(lam_hist, umax, rho)
    H = min_fuel_hamiltonian(u_hist, lam_hist, x_hist, MU_SUN)
    H_scale = max(abs(H).max(), 1e-30)
    assert (H.max() - H.min()) / H_scale < 1e-6


def test_solve_min_fuel_bvp_recovers_known_lambda0():
    """End-to-end in normalized units: integrate with known lam0 to get xf, recover it via BVP."""
    from scipy.integrate import solve_ivp

    mu = 1.0
    x0 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])  # circular orbit, r=1, v=1
    # Velocity-costate magnitude > 1 so the primer norm exceeds the optimality threshold and
    # the bang-off arc actually fires (otherwise the BVP is degenerate).
    lam0_true = np.array([0.05, -0.02, 0.0, -1.4, 0.3, 0.0])
    umax = 0.05
    rho = 0.1
    tf = 0.4

    forward = solve_ivp(
        lambda t, y: min_fuel_combined_dynamics(t, y, mu, umax, rho),
        (0.0, tf),
        np.concatenate((x0, lam0_true)),
        method="DOP853",
        rtol=1e-13,
        atol=1e-13,
    )
    assert forward.success
    xf = forward.y[:6, -1]

    # Cold-start from a perturbed guess.
    lam0_guess = lam0_true * 0.7
    result = solve_min_fuel_bvp(
        0.0, tf, x0, xf, mu, umax, rho, lam0_guess,
        n_eval=80, root_tol=1e-12, rtol=1e-13, atol=1e-13,
    )
    # scipy hybr can report success=False once residual hits machine epsilon; trust the residual.
    assert result.residual_norm < 1e-9, result.message
    np.testing.assert_allclose(result.x_hist[-1], xf, rtol=1e-9, atol=1e-9)
    # Confirm control actually activated (sanity that the BVP wasn't degenerate).
    assert np.linalg.norm(result.u_hist, axis=1).max() > 0.5 * umax
