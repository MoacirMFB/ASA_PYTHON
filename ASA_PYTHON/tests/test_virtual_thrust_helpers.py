import numpy as np
import pytest

from ..keplerian import propagate_two_body
pytest.importorskip("virtual_thrust")

from virtual_thrust.vt_thrust_helpers import (
    compute_amax_from_cadence,
    compute_dv_per_impact,
    compute_dvmax_from_impactors,
    conway_max_theoretical_deflection_stm,
    propagate_zoh_2bp_control,
)


def test_delta_v_and_cadence_helpers_are_consistent():
    mA = 10_000.0
    ms = 100.0
    vrel = 5_000.0
    beta = 1.5
    cosgamma = 0.8
    dt_min = 200.0

    dv_one = compute_dv_per_impact(mA, ms, vrel, beta, cosgamma)
    dv_total = compute_dvmax_from_impactors(mA, [ms, ms], [vrel, vrel], beta, [cosgamma, cosgamma])
    amax = compute_amax_from_cadence(mA, ms, vrel, beta, cosgamma, dt_min)

    assert dv_one > 0.0
    np.testing.assert_allclose(dv_total, 2.0 * dv_one, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(amax, dv_one / dt_min, rtol=1e-12, atol=1e-12)


def test_propagate_zoh_2bp_control_matches_unforced_two_body_when_control_is_zero():
    mu = 398600.4418
    x0 = np.array([7000.0, 0.0, 0.0, 0.0, np.sqrt(mu / 7000.0), 0.0])
    t_grid = np.array([0.0, 25.0, 50.0, 75.0])
    U = np.zeros((3, 3))

    piecewise = propagate_zoh_2bp_control(x0, t_grid, U, mu, 0.0, rtol=1e-11, atol=1e-11)
    _, direct = propagate_two_body(x0, t_grid, mu, rtol=1e-11, atol=1e-11)

    np.testing.assert_allclose(piecewise, direct, rtol=1e-10, atol=1e-10)


def test_conway_benchmark_matches_eigenvalue_gain_relation():
    mu = 1.32712440041279419e11
    x0 = np.array([1.2e8, 0.0, 0.0, 0.0, 28.0, 0.5])
    bench = conway_max_theoretical_deflection_stm(
        x0,
        np.zeros(3),
        dVmax_kmps=1e-3,
        tf_sec=0.0,
        t0_sec=-10_000.0,
        mu_sun_km=mu,
        n_spacecraft=5,
        rtol=1e-10,
        atol=1e-10,
    )

    assert bench.Phi_tf.shape == (6, 6)
    assert bench.Phi_rv.shape == (3, 3)
    np.testing.assert_allclose(np.linalg.norm(bench.e_opt), 1.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(bench.dr_max_km, np.sqrt(bench.lambda_max) * 1e-3, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(bench.single_sc_impulse_deviation_km, bench.dr_max_km, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(bench.multi_sc_impulsive_deviation_km, 5.0 * bench.single_sc_impulse_deviation_km, rtol=1e-12, atol=1e-12)
