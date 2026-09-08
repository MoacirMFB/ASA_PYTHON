"""Independent attitude checks. Run: python -m pytest ASA/tests/test_attitude_validation.py -q.

SciPy uses active matrices R. ASA's column DCM is C = R.T, mapping
inertial components to body components. All angular velocities below are
body components. Derivatives use central differences, not ASA rate tables.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from ..attitude import dcm as d
from ..attitude import kinematics as k
from ..attitude import quaternions as q
from ..attitude import rigidbody as r


SEQUENCES = ("123", "231", "312", "132", "213", "321",
             "121", "131", "212", "232", "313", "323")
ANGLES = np.array([0.37, 0.53, 0.71])
RATES = np.array([0.11, -0.23, 0.41])


def reference_rotation(sequence, angles, intrinsic):
    axes = sequence.translate(str.maketrans("123", "xyz"))
    return Rotation.from_euler(axes.upper() if intrinsic else axes, angles).as_matrix()


def reference_omega(sequence, angles, rates, intrinsic):
    step = 1e-6
    rotation = reference_rotation(sequence, angles, intrinsic)
    derivative = (reference_rotation(sequence, angles + step * rates, intrinsic)
                  - reference_rotation(sequence, angles - step * rates, intrinsic)) / (2 * step)
    cross = rotation.T @ derivative  # Rdot = R [omega_B x]
    return np.array([cross[2, 1], cross[0, 2], cross[1, 0]])


@pytest.mark.parametrize("sequence", SEQUENCES)
def test_euler_dcms_against_scipy(sequence):
    seq = tuple(map(int, sequence))
    for intrinsic, numeric, symbolic in (
        (True, d.dcm_from_euler_angle_seq, d.dcm_from_euler_angle_seq_sym),
        (False, d.dcm_from_space_rotations, d.dcm_from_space_rotations_sym),
    ):
        expected = reference_rotation(sequence, ANGLES, intrinsic)
        assert_allclose(numeric(seq, ANGLES, "row"), expected, atol=1e-14)
        assert_allclose(numeric(seq, ANGLES, "col"), expected.T, atol=1e-14)
        assert_allclose(np.array(symbolic(seq, ANGLES), dtype=float), expected, atol=1e-14)


@pytest.mark.parametrize("constructor", [d.dcm_from_euler_angle_seq, d.dcm_from_space_rotations])
def test_euler_dcms_reject_invalid_convention(constructor):
    with pytest.raises(ValueError):
        constructor((1, 2, 3), ANGLES, convention="typo")


@pytest.mark.parametrize("sequence", SEQUENCES)
def test_body_euler_rates_against_differentiated_rotation(sequence):
    seq = tuple(map(int, sequence))
    expected = reference_omega(sequence, ANGLES, RATES, True)
    assert_allclose(k.omega_from_euler_rates_body_seq(ANGLES, RATES, seq), expected, atol=2e-9)
    assert_allclose(k.euler_rates_body_seq(expected, ANGLES, seq), RATES, atol=2e-9)


@pytest.mark.parametrize("sequence", SEQUENCES)
def test_space_angular_velocity_against_differentiated_rotation(sequence):
    expected = reference_omega(sequence, ANGLES, RATES, False)
    actual = k.omega_from_euler_rates_space_seq(ANGLES, RATES, tuple(map(int, sequence)))
    assert_allclose(actual, expected, atol=2e-9)


@pytest.mark.parametrize("sequence", SEQUENCES[:8])
def test_implemented_space_euler_rates_against_differentiated_rotation(sequence):
    expected = reference_omega(sequence, ANGLES, RATES, False)
    actual = k.euler_rates_space_seq(expected, ANGLES, tuple(map(int, sequence)))
    assert_allclose(actual, RATES, atol=2e-9)


@pytest.mark.parametrize("convention", ["row", "col"])
def test_dcm_rate_against_differentiated_quaternion(convention):
    quat = Rotation.from_euler("XYZ", ANGLES).as_quat()
    derivative = k.quat_dot(quat, RATES)
    step = 1e-6
    expected = (q.quat_to_dcm(quat + step * derivative, convention)
                - q.quat_to_dcm(quat - step * derivative, convention)) / (2 * step)
    actual = k.dcm_dot(q.quat_to_dcm(quat, convention), RATES, convention)
    assert_allclose(actual, expected, atol=2e-9)


def test_quaternion_algebra_conversions_and_rates_against_rotations():
    rng = np.random.default_rng(507)
    quaternions = Rotation.random(100, random_state=rng).as_quat()
    quaternions = np.vstack((quaternions, [0, 0, 0, 1], np.column_stack((np.eye(3), np.zeros(3)))))
    for quat in quaternions:
        rotation = Rotation.from_quat(quat).as_matrix()
        other = Rotation.random(random_state=rng).as_quat()
        omega = rng.normal(size=3)
        for convention, expected in (("row", rotation), ("col", rotation.T)):
            assert_allclose(q.quat_to_dcm(quat, convention), expected, atol=1e-14)
            assert_allclose(q.quat_to_dcm_2(quat, convention), expected, atol=1e-14)
            recovered = q.dcm_to_quat(expected, convention)
            assert_allclose(abs(recovered @ quat), 1, atol=1e-14)
            if abs(quat[3]) > 0.05:  # the shortcut explicitly excludes half turns
                recovered = q.dcm_to_quat_2(expected, convention)
                assert_allclose(abs(recovered @ quat), 1, atol=1e-12)
        product = q.quaternion_product(quat, other)
        assert_allclose(q.quaternion_cross_matrix(quat) @ other, product, atol=1e-14)
        assert_allclose(q.quat_to_dcm(product), rotation.T @ q.quat_to_dcm(other), atol=1e-14)
        assert_allclose(q.quaternion_product(quat, q.quat_conjugate(quat)), [0, 0, 0, 1], atol=1e-14)
        derivative = k.quat_dot(quat, omega)
        assert_allclose(k.quat_dot_2(quat, omega), derivative, atol=1e-14)
        assert_allclose(k.quaternion_kinematics_fixed_omega(quat, omega), derivative, atol=1e-14)
        assert_allclose(k.quat_dot_to_omega(quat, derivative), np.append(omega, 0), atol=1e-14)
        step = 1e-6
        c_dot = (q.quat_to_dcm(quat + step * derivative)
                 - q.quat_to_dcm(quat - step * derivative)) / (2 * step)
        assert_allclose(c_dot, -d.skew_symmetric(omega) @ rotation.T, atol=2e-9)


@pytest.mark.parametrize("angles", [(0, 0, 0), (0.3, 0.4, 0.5), (0.3, np.pi, 0.5)])
def test_313_extraction_reconstructs_attitude(angles):
    matrix = reference_rotation("313", angles, True).T
    recovered = d.dcm_to_euler_313(matrix)
    assert_allclose(reference_rotation("313", recovered, True).T, matrix, atol=1e-12)


@pytest.mark.parametrize("angle", [0, 1e-4, 1e-3, 0.7, np.pi - 1e-3, np.pi, np.pi + 1e-3])
def test_axis_angle_extraction_reconstructs_attitude(angle):
    axis = np.array([1., 2., 3.]) / np.sqrt(14)
    matrix = Rotation.from_rotvec(axis * angle).as_matrix().T
    recovered_axis, recovered_angle = d.dcm_to_euler_axis_angle(matrix)
    recovered = (np.eye(3) if recovered_axis is None else
                 Rotation.from_rotvec(recovered_axis * recovered_angle).as_matrix().T)
    assert_allclose(recovered, matrix, atol=2e-8)


def test_axis_angle_constructor_normalizes_or_rejects_nonunit_axis():
    axis = [2., 0., 0.]
    try:
        quat, _, _ = q.axis_angle_to_quaternion(np.pi / 2, axis)
    except ValueError:
        return  # Requiring a unit axis explicitly is also a valid contract.
    assert_allclose(q.quat_to_dcm(quat), d.euler_axis_angle_to_dcm(axis, np.pi / 2), atol=1e-14)


def test_remaining_vector_quaternion_and_mass_helpers():
    axis = np.array([1., 2., 3.]) / np.sqrt(14)
    quat, vector, scalar = q.axis_angle_to_quaternion(0.7, axis)
    assert_allclose(q.full_quat_from_vec(vector)[0], quat, atol=1e-14)
    assert_allclose(q.quaternion_scalar(vector)[0], scalar, atol=1e-14)
    assert_allclose(q.quaternion_scalar(vector)[1], quat, atol=1e-14)
    angle, recovered_axis = q.quat_to_euler_axis_angle(quat)
    assert_allclose(angle, 0.7, atol=1e-14)
    assert_allclose(recovered_axis, axis, atol=1e-14)
    expected = Rotation.from_rotvec(axis * 0.7).as_matrix().T
    assert_allclose(d.rotation_dyadic_from_axis_angle(axis, 0.7), expected, atol=1e-14)
    for index in (1, 2, 3):
        expected = Rotation.from_rotvec(np.eye(3)[index - 1] * 0.7).as_matrix()
        assert_allclose(d.dcm_from_single_euler_angle(index, 0.7), expected.T, atol=1e-14)
        assert_allclose(d.dcm_single_euler_angle_2(index, 0.7), expected, atol=1e-14)
    first = Rotation.from_euler("x", 0.4).as_quat()
    second = Rotation.from_euler("y", 0.7).as_quat()
    # Without DCMs the documented assumption is axes in a common frame.
    composed = q.sequential_quat_rotations(first, second)
    expected = Rotation.from_euler("xy", [0.4, 0.7]).as_matrix().T
    assert_allclose(q.quat_to_dcm(composed), expected, atol=1e-14)
    # Supplying row DCMs expresses the intermediate axis in the original frame.
    composed = q.sequential_quat_rotations(first, second, q.quat_to_dcm(first, "row"),
                                          q.quat_to_dcm(second, "row"))
    expected = Rotation.from_euler("XY", [0.4, 0.7]).as_matrix().T
    assert_allclose(q.quat_to_dcm(composed), expected, atol=1e-14)
    assert_allclose(r.center_of_mass([[1, 2], [3, 4]], [2, 1]), [5 / 3, 8 / 3])
    radius, force, outward, magnitude = r.center_of_gravity([[4, 0], [6, 0]], [2, 3], [1, 0], 7, 11)
    expected_force = 7 * 11 * (2 / 3**2 + 3 / 5**2)
    assert_allclose(force, [-expected_force, 0])
    assert_allclose(magnitude, expected_force)
    assert_allclose(outward, [1, 0])
    assert_allclose(radius, np.sqrt(7 * 11 * 5 / expected_force))
    axial, transverse, mass = r.inertia_solid_cylinder(2, 3, 5)
    assert_allclose([axial, transverse, mass], [120 * np.pi, 105 * np.pi, 60 * np.pi])


def test_rigid_body_conservation_and_torque_balance():
    rotation = Rotation.from_euler("XYZ", ANGLES).as_matrix()
    inertia = rotation @ np.diag([2., 3., 4.]) @ rotation.T
    state = np.r_[Rotation.from_euler("ZYX", ANGLES).as_quat(), RATES]
    torque = np.array([0.07, -0.02, 0.03])
    derivative = r.rigid_body_dynamics_quaternions(0, state, inertia, torque)
    omega = state[4:]
    assert_allclose(inertia @ derivative[4:] + np.cross(omega, inertia @ omega), torque, atol=1e-14)
    assert_allclose(omega @ inertia @ derivative[4:], omega @ torque, atol=1e-14)
    solution = solve_ivp(r.rigid_body_dynamics_quaternions, (0, 120), state,
                         args=(inertia,), method="DOP853", rtol=1e-11, atol=1e-12)
    assert solution.success
    omega = solution.y[4:]
    momentum_body = inertia @ omega
    energy = 0.5 * np.sum(omega * momentum_body, axis=0)
    momentum_inertial = Rotation.from_quat(solution.y[:4].T).apply(momentum_body.T)
    assert_allclose(energy, np.full_like(energy, energy[0]), rtol=0, atol=2e-10)
    assert_allclose(momentum_inertial, np.tile(momentum_inertial[0], (len(solution.t), 1)), rtol=0, atol=2e-9)
    assert_allclose(np.linalg.norm(solution.y[:4], axis=0), 1, rtol=0, atol=2e-9)


def test_gravity_force_and_moment_against_point_masses():
    # Symmetric point pairs have zero first and third mass moments.
    body_points = np.vstack((np.diag([1., 2., 3.]), -np.diag([1., 2., 3.])))
    rotation = Rotation.from_euler("XYZ", ANGLES).as_matrix()
    offsets = body_points @ rotation.T
    inertia = sum(np.dot(p, p) * np.eye(3) - np.outer(p, p) for p in body_points)
    errors = []
    for distance in (500., 1000.):
        center = np.array([distance, 0., 0.])
        positions = center + offsets
        forces = -100 * positions / np.linalg.norm(positions, axis=1)[:, None] ** 3
        exact_force = forces.sum(axis=0)
        exact_moment = np.cross(offsets, forces).sum(axis=0)
        approximate, particle = r.approx_gravity_force_f2(100, 6, distance, rotation, *np.diag(inertia))
        moment = r.compute_gravity_moment(100, 6, distance, approximate)[0]
        force_error = np.linalg.norm(approximate - exact_force)
        moment_error = np.linalg.norm(moment - exact_moment)
        assert force_error < 10 * (3 / distance)**2 * np.linalg.norm(exact_force - particle)
        assert moment_error < 10 * (3 / distance)**2 * np.linalg.norm(exact_moment)
        errors.append([force_error, moment_error])
    # The omitted fourth mass moment gives force O(R^-6), torque O(R^-5).
    assert_allclose(np.array(errors[0]) / errors[1], [64, 32], rtol=5e-3)
