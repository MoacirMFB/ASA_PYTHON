"""Quaternion algebra and quaternion/DCM conversions.

Ported from ``AttitudeDeterminationLibrary.m`` (ASA_MATLAB).

Scalar-last convention throughout: ``q = [q1, q2, q3, q4]`` with ``q4`` the
scalar part.  The ``convention`` flag has the same meaning as in
:mod:`ASA.attitude.dcm`.
"""

import numpy as np

from .dcm import _apply_convention

__all__ = [
    "axis_angle_to_quaternion",
    "dcm_to_quat",
    "dcm_to_quat_2",
    "full_quat_from_vec",
    "quat_conjugate",
    "quat_to_dcm",
    "quat_to_dcm_2",
    "quat_to_euler_axis_angle",
    "quaternion_cross_matrix",
    "quaternion_product",
    "quaternion_scalar",
    "sequential_quat_rotations",
    "xi_matrix",
]


def quaternion_product(q, q_prime):
    """Quaternion product q (x) q_prime, scalar-last."""
    q = np.asarray(q, dtype=float).ravel()
    q_prime = np.asarray(q_prime, dtype=float).ravel()
    q_v, q4 = q[:3], q[3]
    p_v, p4 = q_prime[:3], q_prime[3]

    vector_part = q4 * p_v + p4 * q_v - np.cross(q_v, p_v)
    scalar_part = q4 * p4 - np.dot(q_v, p_v)
    return np.append(vector_part, scalar_part)


def quaternion_cross_matrix(q):
    """The 4x4 [q (x)] matrix of a quaternion."""
    q1, q2, q3, q4 = np.asarray(q, dtype=float).ravel()
    return np.array([[q4, q3, -q2, q1],
                     [-q3, q4, q1, q2],
                     [q2, -q1, q4, q3],
                     [-q1, -q2, -q3, q4]])


def quat_conjugate(q):
    """Quaternion conjugate, negating the vector part."""
    q = np.asarray(q, dtype=float).ravel()
    if q.size != 4:
        raise ValueError("Input quaternion must have exactly 4 elements.")
    return np.append(-q[:3], q[3])


def xi_matrix(q):
    """The 4x3 Xi(q) matrix used in the quaternion kinematics equation."""
    q1, q2, q3, q4 = np.asarray(q, dtype=float).ravel()
    return np.array([[q4, -q3, q2],
                     [q3, q4, -q1],
                     [-q2, q1, q4],
                     [-q1, -q2, -q3]])


# --- Quaternion to DCM ------------------------------------------------------
def quat_to_dcm(q, convention="col"):
    """DCM from a quaternion; the quaternion is normalized first."""
    q = np.asarray(q, dtype=float).ravel()
    q1, q2, q3, q4 = q / np.linalg.norm(q)

    dcm = np.array([
        [q1**2 - q2**2 - q3**2 + q4**2, 2 * (q1 * q2 + q3 * q4), 2 * (q1 * q3 - q2 * q4)],
        [2 * (q2 * q1 - q3 * q4), -q1**2 + q2**2 - q3**2 + q4**2, 2 * (q2 * q3 + q1 * q4)],
        [2 * (q3 * q1 + q2 * q4), 2 * (q3 * q2 - q1 * q4), -q1**2 - q2**2 + q3**2 + q4**2],
    ])
    return _apply_convention(dcm, convention)


def quat_to_dcm_2(q, convention="col"):
    """Alternative quaternion-to-DCM formula; assumes ``q`` is already unit."""
    q1, q2, q3, q4 = np.asarray(q, dtype=float).ravel()

    # Built in the row convention, so 'col' is the transpose of this.
    dcm = np.array([
        [1 - 2 * q2**2 - 2 * q3**2, 2 * (q1 * q2 - q3 * q4), 2 * (q3 * q1 + q2 * q4)],
        [2 * (q1 * q2 + q3 * q4), 1 - 2 * q3**2 - 2 * q1**2, 2 * (q2 * q3 - q1 * q4)],
        [2 * (q3 * q1 - q2 * q4), 2 * (q2 * q3 + q1 * q4), 1 - 2 * q1**2 - 2 * q2**2],
    ])
    if convention.lower() == "col":
        return dcm.T
    if convention.lower() == "row":
        return dcm
    raise ValueError('Invalid convention. Use "row" or "col".')


# --- DCM to quaternion ------------------------------------------------------
def dcm_to_quat(dcm, convention="col"):
    """Quaternion from a DCM, picking the best-conditioned of four candidates."""
    dcm = np.asarray(dcm, dtype=float)
    trace = np.trace(dcm)
    sign = 1.0 if convention.lower() == "col" else -1.0
    if convention.lower() not in ("col", "row"):
        raise ValueError('Invalid convention. Use "row" or "col".')

    d23 = sign * (dcm[1, 2] - dcm[2, 1])
    d31 = sign * (dcm[2, 0] - dcm[0, 2])
    d12 = sign * (dcm[0, 1] - dcm[1, 0])

    candidates = [
        np.array([1 + 2 * dcm[0, 0] - trace, dcm[0, 1] + dcm[1, 0], dcm[0, 2] + dcm[2, 0], d23]),
        np.array([dcm[1, 0] + dcm[0, 1], 1 + 2 * dcm[1, 1] - trace, dcm[1, 2] + dcm[2, 1], d31]),
        np.array([dcm[2, 0] + dcm[0, 2], dcm[2, 1] + dcm[1, 2], 1 + 2 * dcm[2, 2] - trace, d12]),
        np.array([d23, d31, d12, 1 + trace]),
    ]
    best = max(candidates, key=np.linalg.norm)
    return best / np.linalg.norm(best)


def dcm_to_quat_2(dcm, convention="col"):
    """Quaternion from a DCM via the scalar-first shortcut; fails near q4 = 0."""
    dcm = np.asarray(dcm, dtype=float)
    q4 = 0.5 * np.sqrt(1.0 + dcm[0, 0] + dcm[1, 1] + dcm[2, 2])

    if convention.lower() == "row":
        q1 = (dcm[2, 1] - dcm[1, 2]) / (4 * q4)
        q2 = (dcm[0, 2] - dcm[2, 0]) / (4 * q4)
        q3 = (dcm[1, 0] - dcm[0, 1]) / (4 * q4)
    elif convention.lower() == "col":
        q1 = (dcm[1, 2] - dcm[2, 1]) / (4 * q4)
        q2 = (dcm[2, 0] - dcm[0, 2]) / (4 * q4)
        q3 = (dcm[0, 1] - dcm[1, 0]) / (4 * q4)
    else:
        raise ValueError('Invalid convention. Use "row" or "col".')

    return np.array([q1, q2, q3, q4])


# --- Axis/angle -------------------------------------------------------------
def axis_angle_to_quaternion(theta, axis):
    """Quaternion for a rotation of ``theta`` about ``axis``.

    Returns ``(q, q_vec, q_scalar)``.
    """
    axis = np.asarray(axis, dtype=float).ravel()
    q_vec = axis * np.sin(theta / 2.0)
    q_scalar = float(np.cos(theta / 2.0))
    return np.append(q_vec, q_scalar), q_vec, q_scalar


def quat_to_euler_axis_angle(q):
    """Euler angle and axis of a quaternion, returned as ``(theta, axis)``.

    The axis is ``None`` for the identity rotation, where it is undefined.
    """
    q = np.asarray(q, dtype=float).ravel()
    theta = 2.0 * float(np.arccos(np.clip(q[3], -1.0, 1.0)))
    half_sine = np.sin(theta / 2.0)
    if np.isclose(half_sine, 0.0):
        return theta, None
    return theta, q[:3] / half_sine


# --- Partial quaternions ----------------------------------------------------
def full_quat_from_vec(q_vec):
    """Full unit quaternion from its vector part, returned as ``(q, q4)``."""
    q_vec = np.asarray(q_vec, dtype=float).ravel()
    q4 = float(np.sqrt(1.0 - np.dot(q_vec, q_vec)))
    return np.append(q_vec, q4), q4


def quaternion_scalar(q_vec):
    """Scalar part implied by a quaternion vector part, as ``(q4, q)``."""
    q, q4 = full_quat_from_vec(q_vec)
    return q4, q


def sequential_quat_rotations(q_abp, q_bpb, c_abp=None, c_bpb=None):
    """Compose two successive rotations A -> B' -> B.

    When both DCMs are given, the intermediate vector part is first expressed
    in the A frame; otherwise both quaternions are assumed to share a frame.
    """
    q_abp = np.asarray(q_abp, dtype=float).ravel()
    q_bpb = np.asarray(q_bpb, dtype=float).ravel()
    v_abp, s_abp = q_abp[:3], q_abp[3]
    v_bpb, s_bpb = q_bpb[:3], q_bpb[3]

    if c_abp is not None and c_bpb is not None:
        v_bpb = np.asarray(c_abp, dtype=float) @ np.asarray(c_bpb, dtype=float) @ v_bpb

    vector_part = v_abp * s_bpb + v_bpb * s_abp + np.cross(v_bpb, v_abp)
    scalar_part = s_abp * s_bpb - np.dot(v_abp, v_bpb)
    return np.append(vector_part, scalar_part)
