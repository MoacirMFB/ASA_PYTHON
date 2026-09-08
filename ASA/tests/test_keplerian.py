import numpy as np

from ..keplerian import (
    CR3BP,
    cartesian_to_keplerian,
    coe_to_cartesian,
    convert_equinoctial_to_eci,
    convert_milankovitch_to_eci,
    dynamics_2bp_cartesian,
    dynamics_2bp_cartesian_j2,
    dynamics_2bp_equinoctial_j2,
    dynamics_2bp_keplerian_j2,
    dynamics_2bp_milankovitch_j2,
    eci2hci,
    hci2eci,
    jacobian_2bp_cartesian,
    keplerian_to_equinoctial,
    mean_anomaly_from_eccentric_anomaly,
    orbital_elements_to_dcm,
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


def test_element_and_frame_conversions_round_trip():
    mu = 398600.4418
    coe = np.array([8000.0, 0.2, 0.6, 1.1, 0.7, 2.0])
    state = coe_to_cartesian(coe, mu, use_true_anomaly=True)
    recovered = cartesian_to_keplerian(state, mu)
    np.testing.assert_allclose(
        [recovered[name] for name in ("a", "e", "i", "Omega", "omega", "f")],
        coe,
        rtol=1e-12,
        atol=1e-12,
    )

    equinoctial = keplerian_to_equinoctial(*coe)
    r_equinoctial, _ = convert_equinoctial_to_eci(equinoctial)
    np.testing.assert_allclose(r_equinoctial, state[:3], rtol=1e-12, atol=1e-9)

    dcm = orbital_elements_to_dcm(coe[3], coe[2], coe[4] + coe[5])
    h = np.sqrt(mu * coe[0] * (1.0 - coe[1] ** 2))
    h_vec = h * dcm[:, 2]
    e_vec = coe[1] * orbital_elements_to_dcm(coe[3], coe[2], coe[4])[:, 0]
    milankovitch = np.r_[h_vec, e_vec, coe[3] + coe[4] + coe[5]]
    r_milankovitch, _ = convert_milankovitch_to_eci(milankovitch, mu)
    np.testing.assert_allclose(r_milankovitch, state[:3], rtol=1e-12, atol=1e-9)

    np.testing.assert_allclose(hci2eci(eci2hci(state, 0.0), 0.0), state, rtol=1e-12, atol=1e-9)


def test_perturbed_element_dynamics_are_callable_and_finite():
    mu = 398600.4418
    coe_true = np.array([8000.0, 0.2, 0.6, 1.1, 0.7, 2.0])
    state = coe_to_cartesian(coe_true, mu, use_true_anomaly=True)
    assert np.isfinite(dynamics_2bp_cartesian_j2(0.0, state, mu)).all()

    equinoctial = keplerian_to_equinoctial(*coe_true)
    assert np.isfinite(dynamics_2bp_equinoctial_j2(0.0, equinoctial, mu)).all()

    eccentric_anomaly = 2.0 * np.arctan2(
        np.sqrt(1.0 - coe_true[1]) * np.sin(coe_true[5] / 2.0),
        np.sqrt(1.0 + coe_true[1]) * np.cos(coe_true[5] / 2.0),
    )
    coe_mean = coe_true.copy()
    coe_mean[5] = mean_anomaly_from_eccentric_anomaly(coe_true[1], eccentric_anomaly)
    assert np.isfinite(dynamics_2bp_keplerian_j2(0.0, coe_mean, mu)).all()

    dcm = orbital_elements_to_dcm(coe_true[3], coe_true[2], coe_true[4] + coe_true[5])
    h_vec = np.sqrt(mu * coe_true[0] * (1.0 - coe_true[1] ** 2)) * dcm[:, 2]
    e_vec = coe_true[1] * orbital_elements_to_dcm(coe_true[3], coe_true[2], coe_true[4])[:, 0]
    milankovitch = np.r_[h_vec, e_vec, coe_true[3] + coe_true[4] + coe_true[5]]
    assert np.isfinite(dynamics_2bp_milankovitch_j2(0.0, milankovitch, mu)).all()
    assert CR3BP([1.02, 0.0, 0.0, 0.0, 0.15, 0.0], 0.01215058, 1.0).shape == (6,)
