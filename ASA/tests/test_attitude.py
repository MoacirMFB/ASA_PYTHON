import numpy as np
import pytest

from ..attitude import dcm as dcm_mod
from ..attitude import kinematics as kin
from ..attitude import (
    axis_angle_to_quaternion,
    center_of_light,
    check_dcm_constraints,
    compute_w4_configuration,
    davenport_qmethod,
    dcm_from_euler_angle_seq,
    dcm_from_space_rotations,
    dcm_from_space_rotations_sym,
    dcm_to_euler_axis_angle,
    dcm_to_quat,
    dcm_to_quat_2,
    dcm_dot,
    dual_trig_inverse,
    omega_from_euler_rates_space_seq,
    full_quat_from_vec,
    dcm_to_euler_313,
    euler_axis_angle_to_dcm,
    euler_rates_body_seq,
    euler_rates_space_seq,
    inertia_solid_cylinder,
    minimax_method,
    omega_from_euler_rates_body_seq,
    pseudoinverse_method,
    quat_dot,
    quat_dot_2,
    quat_to_dcm,
    quat_to_dcm_2,
    quaternion_kinematics_fixed_omega,
    quaternion_product,
    quat_conjugate,
    quest_method,
    rigid_body_dynamics_quaternions,
    rotation_dyadic_from_axis_angle,
    skew_symmetric,
    triad_method,
)

REFERENCE_SEQ = (3, 1, 3)
REFERENCE_ANGLES = (0.3, 0.4, 0.5)


def reference_dcm():
    return dcm_from_euler_angle_seq(REFERENCE_SEQ, REFERENCE_ANGLES, convention="col")


def test_check_dcm_constraints_separates_the_two_requirements():
    rotation = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    reflection = np.diag([1.0, 1.0, -1.0])
    sheared = np.array([[0.0, 1.0, 0.0], [-1.0, 0.1, 0.0], [0.0, 0.0, 1.0]])

    assert check_dcm_constraints(rotation)["valid"]
    assert not check_dcm_constraints(reflection)["valid"]
    assert check_dcm_constraints(reflection)["orthogonal"]  # fails only on handedness
    assert not check_dcm_constraints(sheared)["valid"]
    assert check_dcm_constraints(sheared)["right_handed"]  # fails only on orthonormality


def test_euler_axis_angle_round_trip():
    axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    theta = np.deg2rad(120.0)
    dcm = euler_axis_angle_to_dcm(axis, theta)

    recovered_axis, recovered_theta = dcm_to_euler_axis_angle(dcm)
    assert np.isclose(recovered_theta, theta)
    assert np.allclose(recovered_axis, axis)
    assert check_dcm_constraints(dcm)["valid"]
    assert np.allclose(rotation_dyadic_from_axis_angle(axis, theta), dcm)


def test_quaternion_and_dcm_conversions_agree():
    axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    q, _, _ = axis_angle_to_quaternion(np.deg2rad(120.0), axis)

    assert np.isclose(np.linalg.norm(q), 1.0)
    assert np.allclose(q, 0.5)  # the 120 deg rotation about [1,1,1] has all-equal parameters
    assert np.allclose(quat_to_dcm(q), quat_to_dcm_2(q))
    assert np.allclose(quat_to_dcm(q), euler_axis_angle_to_dcm(axis, np.deg2rad(120.0)))
    # -q must give the same DCM as +q
    assert np.allclose(quat_to_dcm(-q), quat_to_dcm(q))

    dcm = reference_dcm()
    for converter in (dcm_to_quat, dcm_to_quat_2):
        assert np.allclose(quat_to_dcm(converter(dcm)), dcm)


def test_space_rotation_table_matches_the_symbolic_form():
    angles = (0.3, 0.4, 0.5)
    for seq in ((1, 2, 3), (2, 3, 1), (3, 1, 2), (1, 3, 2), (2, 1, 3), (3, 2, 1),
                (1, 2, 1), (1, 3, 1), (2, 1, 2), (2, 3, 2), (3, 1, 3), (3, 2, 3)):
        numeric = dcm_from_space_rotations(seq, angles)
        symbolic = np.array(dcm_from_space_rotations_sym(seq, angles), dtype=float)
        assert np.allclose(numeric, symbolic), seq
        assert check_dcm_constraints(numeric)["valid"], seq


ALL_SEQUENCES = ((1, 2, 3), (2, 3, 1), (3, 1, 2), (1, 3, 2), (2, 1, 3), (3, 2, 1),
                 (1, 2, 1), (1, 3, 1), (2, 1, 2), (2, 3, 2), (3, 1, 3), (3, 2, 3))


def test_euler_rates_invert_the_angular_velocity_map():
    angles = np.array([0.3, 0.4, 0.5])
    rates = np.array([0.01, 0.02, 0.03])
    for seq in ALL_SEQUENCES:
        omega = omega_from_euler_rates_body_seq(angles, rates, seq)
        assert np.allclose(euler_rates_body_seq(omega, angles, seq), rates), seq


def test_space_euler_rates_match_the_reversed_body_sequence():
    # A space-fixed sequence is the reversed body-fixed sequence with the
    # angles and rates reversed, which pins down the space table independently.
    angles = np.array([0.37, 0.53, 0.71])
    omega = np.array([0.11, -0.23, 0.41])
    for seq in sorted(kin._EULER_RATES_SPACE):
        rates = euler_rates_space_seq(omega, angles, seq)
        reversed_seq = tuple(reversed(seq))
        recovered = omega_from_euler_rates_body_seq(angles[::-1], rates[::-1], reversed_seq)
        assert np.allclose(recovered, omega), seq


def test_quaternion_algebra():
    q = np.array([0.1, 0.2, 0.3, np.sqrt(1 - 0.14)])
    identity = np.array([0.0, 0.0, 0.0, 1.0])

    assert np.allclose(quaternion_product(q, identity), q)
    assert np.allclose(quaternion_product(q, quat_conjugate(q)), identity)
    # all three kinematics forms must agree
    omega = np.array([0.2, 0.1, 1.0])
    assert np.allclose(quat_dot(q, omega), quaternion_kinematics_fixed_omega(q, omega))
    assert np.allclose(quat_dot(q, omega), quat_dot_2(q, omega))
    # unit norm is preserved, so q . q_dot vanishes
    assert np.isclose(q @ quat_dot(q, omega), 0.0)
    assert np.isclose(q @ quat_dot_2(q, omega), 0.0)


def test_skew_symmetric_reproduces_the_cross_product():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([-4.0, 0.5, 2.0])
    assert np.allclose(skew_symmetric(a) @ b, np.cross(a, b))


def test_static_attitude_estimators_recover_a_known_dcm():
    dcm = reference_dcm()
    inertial = np.column_stack([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.3, -0.2, 0.9]])
    inertial /= np.linalg.norm(inertial, axis=0)
    body = dcm @ inertial
    weights = np.array([0.4, 0.35, 0.25])

    assert np.allclose(triad_method(body[:, 0], body[:, 1],
                                    inertial[:, 0], inertial[:, 1]), dcm)

    _, quest_dcm, quest_loss, _, lambda_max = quest_method(weights, body, inertial)
    assert np.allclose(quest_dcm, dcm)
    assert np.isclose(lambda_max, weights.sum())
    assert np.isclose(quest_loss, 0.0, atol=1e-10)  # noise-free data is a perfect fit

    _, qmethod_dcm, qmethod_loss, _ = davenport_qmethod(weights, body, inertial)
    assert np.allclose(qmethod_dcm, dcm)
    assert np.isclose(qmethod_loss, 0.0, atol=1e-10)


def test_torque_free_propagation_conserves_the_invariants():
    from scipy.integrate import solve_ivp

    inertia = np.diag([10.0, 10.0, 5.0])
    state0 = np.array([0.0, 0.0, 0.0, 1.0, 0.2, 0.1, 1.0])
    solution = solve_ivp(rigid_body_dynamics_quaternions, (0.0, 120.0), state0,
                         args=(inertia,), rtol=1e-12, atol=1e-12)

    quaternions = solution.y[:4]
    omega = solution.y[4:7]
    assert np.allclose(np.linalg.norm(quaternions, axis=0), 1.0, atol=1e-9)
    assert np.allclose(omega[2], omega[2, 0])  # axisymmetric body, w3 stays constant

    # Analytical torque-free solution, Eq. (3) of AAE 507 PS2.
    k = (10.0 - 5.0) / 10.0
    phase = k * omega[2, 0] * solution.t
    assert np.allclose(omega[0], 0.2 * np.cos(phase) + 0.1 * np.sin(phase), atol=1e-6)
    assert np.allclose(omega[1], 0.1 * np.cos(phase) - 0.2 * np.sin(phase), atol=1e-6)


def test_wheel_configurations_and_distribution():
    for config, params in (("pyramid", dict(a=0.5, b=0.5, c=0.5, d=0.5)),
                           ("nasa", dict(alpha=0.3, beta=0.4, gamma=0.5))):
        w4, w4_pseudo, null_vector = compute_w4_configuration(config, params)
        assert np.isclose(null_vector @ null_vector, 1.0)
        assert np.allclose(w4 @ null_vector, 0.0)  # the null vector spins up no body momentum
        assert np.allclose(w4 @ w4_pseudo, np.eye(3))

    w4, _, _ = compute_w4_configuration("pyramid", dict(a=0.5, b=0.5, c=0.5, d=0.5))
    body_momentum = np.array([[0.1, 0.2, 0.3], [0.15, 0.25, 0.35]])
    for wheel_momentum in (pseudoinverse_method(w4, body_momentum, 0.01)[0],
                           minimax_method(w4, body_momentum, body_momentum,
                                          np.eye(3), 1.0, 0.01)[0]):
        assert np.allclose(wheel_momentum @ w4.T, body_momentum)


def test_misc_helpers():
    assert np.allclose(dual_trig_inverse("cos", 0.5), [-300.0, -60.0, 60.0, 300.0])
    axial, transverse, mass = inertia_solid_cylinder(2.0, 3.0, 5.0, param_is_mass=True)
    assert np.isclose(mass, 5.0)
    assert np.isclose(axial, 10.0)
    assert np.isclose(transverse, (1 / 12) * 5.0 * (3 * 4.0 + 9.0))

    image = np.zeros((5, 5))
    image[2, 3] = 1.0
    assert center_of_light(image) == (4.0, 3.0)  # 1-based, (column, row)

    assert dcm_mod.angle_between_vectors([1, 0, 0], [0, 1, 0]) == np.pi / 2


def test_space_angular_velocity_covers_every_sequence():
    # A space sequence is the reversed body sequence with reversed angles/rates.
    angles = np.array([0.37, 0.53, 0.71])
    rates = np.array([0.011, -0.023, 0.041])
    for seq in ALL_SEQUENCES:
        expected = omega_from_euler_rates_body_seq(angles[::-1], rates[::-1],
                                                   tuple(reversed(seq)))
        assert np.allclose(omega_from_euler_rates_space_seq(angles, rates, seq), expected), seq


def test_dcm_rate_matches_a_numerical_derivative():
    omega = np.array([0.2, -0.1, 0.4])
    step = 1e-7
    col = reference_dcm()
    turn = euler_axis_angle_to_dcm(omega / np.linalg.norm(omega), np.linalg.norm(omega) * step)
    assert np.allclose((turn @ col - col) / step, dcm_dot(col, omega, "col"), atol=1e-6)
    assert np.allclose(dcm_dot(col.T, omega, "row"), col.T @ skew_symmetric(omega))


def test_angle_extraction_stays_accurate_at_zero_and_half_turns():
    axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    for theta in (0.0, 1e-9, 1e-4, 0.7, np.pi - 1e-4, np.pi - 1e-9, np.pi):
        dcm = euler_axis_angle_to_dcm(axis, theta)
        recovered_axis, recovered_theta = dcm_to_euler_axis_angle(dcm)
        rebuilt = (np.eye(3) if recovered_axis is None
                   else euler_axis_angle_to_dcm(recovered_axis, recovered_theta))
        assert np.allclose(rebuilt, dcm, atol=1e-11), theta

    # Near gimbal lock the extraction reports psi = 0, which reproduces the
    # attitude only to order theta, so the tolerance follows the middle angle.
    for angles in ((0.0, 0.0, 0.0), (0.3, 0.4, 0.5), (0.3, np.pi, 0.5), (1.1, 1e-10, 0.2)):
        dcm = dcm_from_euler_angle_seq((3, 1, 3), angles, "col")
        rebuilt = dcm_from_euler_angle_seq((3, 1, 3), dcm_to_euler_313(dcm), "col")
        assert np.allclose(rebuilt, dcm, atol=max(1e-12, 10.0 * angles[1])), angles


def test_bad_input_is_rejected_rather_than_silently_accepted():
    for call in (lambda c: dcm_from_euler_angle_seq((1, 2, 3), [0.3, 0.4, 0.5], convention=c),
                 lambda c: dcm_from_space_rotations((1, 2, 3), [0.3, 0.4, 0.5], convention=c),
                 lambda c: euler_axis_angle_to_dcm([0, 0, 1], 0.5, convention=c),
                 lambda c: quat_to_dcm([0, 0, 0, 1], convention=c),
                 lambda c: dcm_dot(np.eye(3), [0, 0, 1], convention=c)):
        with pytest.raises(ValueError):
            call("typo")
    with pytest.raises(ValueError):
        full_quat_from_vec([0.9, 0.9, 0.9])  # would otherwise be a silent NaN


def test_axis_angle_constructor_normalizes_its_axis():
    quat, _, _ = axis_angle_to_quaternion(np.pi / 2, [2.0, 0.0, 0.0])
    assert np.allclose(quat_to_dcm(quat), euler_axis_angle_to_dcm([2.0, 0.0, 0.0], np.pi / 2))
