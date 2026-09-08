"""Attitude kinematics: DCM, Euler-angle, and quaternion rates of change.

Ported from ``AttitudeDeterminationLibrary.m`` (ASA_MATLAB).

Sequences are given as a 3-tuple of axis numbers, e.g. ``(3, 1, 3)``.  Body
sequences are intrinsic (each rotation about the current axes); space sequences
are extrinsic (each rotation about the original axes).
"""

import numpy as np

from .dcm import skew_symmetric
from .quaternions import xi_matrix

__all__ = [
    "dcm_dot",
    "euler_rates_body_seq",
    "euler_rates_space_seq",
    "omega_from_euler_rates_body_seq",
    "omega_from_euler_rates_space_seq",
    "quat_dot",
    "quat_dot_2",
    "quat_dot_to_omega",
    "quaternion_kinematics_fixed_omega",
]


def _lookup(table, seq, what):
    key = tuple(int(axis) for axis in seq)
    if key not in table:
        raise ValueError(f"{what} sequence {key} not implemented.")
    return table[key]


# --- DCM rate ---------------------------------------------------------------
def dcm_dot(dcm, omega, convention="row"):
    """Poisson's kinematical equation, Cdot = C [omega x] in the row convention."""
    dcm = np.asarray(dcm, dtype=float)
    if convention.lower() == "row":
        return dcm @ skew_symmetric(omega)
    if convention.lower() == "col":
        return -skew_symmetric(omega) @ dcm
    raise ValueError('Invalid convention. Use "row" or "col".')


# --- Euler angle rates, body-fixed sequences --------------------------------
# theta_dot = f(w1, w2, w3, c2, s2, c3, s3)
_EULER_RATES_BODY = {
    (1, 2, 1): lambda w1, w2, w3, c2, s2, c3, s3: (
        (w2 * s3 + w3 * c3) / s2, w2 * c3 - w3 * s3, w1 - ((w2 * s3 + w3 * c3) * c2) / s2),
    (1, 3, 1): lambda w1, w2, w3, c2, s2, c3, s3: (
        (-w2 * c3 + w3 * s3) / s2, w2 * s3 + w3 * c3, w1 + ((w2 * c3 - w3 * s3) * c2) / s2),
    (2, 1, 2): lambda w1, w2, w3, c2, s2, c3, s3: (
        (w1 * s3 - w3 * c3) / s2, w1 * c3 + w3 * s3, ((-w1 * s3 + w3 * c3) * c2) / s2 + w2),
    (2, 3, 2): lambda w1, w2, w3, c2, s2, c3, s3: (
        (w1 * c3 + w3 * s3) / s2, -w1 * s3 + w3 * c3, -((w1 * c3 + w3 * s3) * c2) / s2 + w2),
    (3, 1, 3): lambda w1, w2, w3, c2, s2, c3, s3: (
        (w1 * s3 + w2 * c3) / s2, w1 * c3 - w2 * s3, -((w1 * s3 + w2 * c3) * c2) / s2 + w3),
    (3, 2, 3): lambda w1, w2, w3, c2, s2, c3, s3: (
        (-w1 * c3 + w2 * s3) / s2, w1 * s3 + w2 * c3, ((w1 * c3 - w2 * s3) * c2) / s2 + w3),
    (1, 2, 3): lambda w1, w2, w3, c2, s2, c3, s3: (
        (w1 * c3 - w2 * s3) / c2, w1 * s3 + w2 * c3, ((-w1 * c3 + w2 * s3) * s2) / c2 + w3),
    (2, 3, 1): lambda w1, w2, w3, c2, s2, c3, s3: (
        (w2 * c3 - w3 * s3) / c2, w2 * s3 + w3 * c3, w1 + ((-w2 * c3 + w3 * s3) * s2) / c2),
    (3, 1, 2): lambda w1, w2, w3, c2, s2, c3, s3: (
        (-w1 * s3 + w3 * c3) / c2, w1 * c3 + w3 * s3, ((w1 * s3 - w3 * c3) * s2) / c2 + w2),
    (1, 3, 2): lambda w1, w2, w3, c2, s2, c3, s3: (
        (w1 * c3 + w3 * s3) / c2, -w1 * s3 + w3 * c3, ((w1 * c3 + w3 * s3) * s2) / c2 + w2),
    (2, 1, 3): lambda w1, w2, w3, c2, s2, c3, s3: (
        (w1 * s3 + w2 * c3) / c2, w1 * c3 - w2 * s3, ((w1 * s3 + w2 * c3) * s2) / c2 + w3),
    (3, 2, 1): lambda w1, w2, w3, c2, s2, c3, s3: (
        (w2 * s3 + w3 * c3) / c2, w2 * c3 - w3 * s3, w1 + ((w2 * s3 + w3 * c3) * s2) / c2),
}


def euler_rates_body_seq(omega, euler_angles, seq):
    """Euler-angle rates from body-frame angular velocity, body-fixed sequence."""
    w1, w2, w3 = np.asarray(omega, dtype=float).ravel()
    _, theta2, theta3 = np.asarray(euler_angles, dtype=float).ravel()
    formula = _lookup(_EULER_RATES_BODY, seq, "Body-fixed")
    return np.array(formula(w1, w2, w3, np.cos(theta2), np.sin(theta2),
                            np.cos(theta3), np.sin(theta3)))


# --- Angular velocity from Euler rates, body-fixed sequences ----------------
# omega = f(td1, td2, td3, c2, s2, c3, s3)
_OMEGA_FROM_BODY_RATES = {
    (1, 2, 3): lambda t1, t2, t3, c2, s2, c3, s3: (
        t1 * c2 * c3 + t2 * s3, -t1 * c2 * s3 + t2 * c3, t1 * s2 + t3),
    (2, 3, 1): lambda t1, t2, t3, c2, s2, c3, s3: (
        t1 * s2 + t3, t1 * c2 * c3 + t2 * s3, -t1 * c2 * s3 + t2 * c3),
    (3, 1, 2): lambda t1, t2, t3, c2, s2, c3, s3: (
        -t1 * c2 * s3 + t2 * c3, t1 * s2 + t3, t1 * c2 * c3 + t2 * s3),
    (1, 3, 2): lambda t1, t2, t3, c2, s2, c3, s3: (
        t1 * c2 * c3 - t2 * s3, -t1 * s2 + t3, t1 * c2 * s3 + t2 * c3),
    (2, 1, 3): lambda t1, t2, t3, c2, s2, c3, s3: (
        t1 * c2 * s3 + t2 * c3, t1 * c2 * c3 - t2 * s3, -t1 * s2 + t3),
    (3, 2, 1): lambda t1, t2, t3, c2, s2, c3, s3: (
        -t1 * s2 + t3, t1 * c2 * s3 + t2 * c3, t1 * c2 * c3 - t2 * s3),
    (1, 2, 1): lambda t1, t2, t3, c2, s2, c3, s3: (
        t1 * c2 + t3, t1 * s2 * s3 + t2 * c3, t1 * s2 * c3 - t2 * s3),
    (1, 3, 1): lambda t1, t2, t3, c2, s2, c3, s3: (
        t1 * c2 + t3, -t1 * s2 * c3 + t2 * s3, t1 * s2 * s3 + t2 * c3),
    (2, 1, 2): lambda t1, t2, t3, c2, s2, c3, s3: (
        t1 * s2 * s3 + t2 * c3, t1 * c2 + t3, -t1 * s2 * c3 + t2 * s3),
    (2, 3, 2): lambda t1, t2, t3, c2, s2, c3, s3: (
        t1 * s2 * c3 - t2 * s3, t1 * c2 + t3, t1 * s2 * s3 + t2 * c3),
    (3, 1, 3): lambda t1, t2, t3, c2, s2, c3, s3: (
        t1 * s2 * s3 + t2 * c3, t1 * s2 * c3 - t2 * s3, t1 * c2 + t3),
    (3, 2, 3): lambda t1, t2, t3, c2, s2, c3, s3: (
        -t1 * s2 * c3 + t2 * s3, t1 * s2 * s3 + t2 * c3, t1 * c2 + t3),
}


def omega_from_euler_rates_body_seq(euler_angles, theta_dot, seq):
    """Body-frame angular velocity from Euler rates, body-fixed sequence."""
    _, theta2, theta3 = np.asarray(euler_angles, dtype=float).ravel()
    t1, t2, t3 = np.asarray(theta_dot, dtype=float).ravel()
    formula = _lookup(_OMEGA_FROM_BODY_RATES, seq, "Body-fixed")
    return np.array(formula(t1, t2, t3, np.cos(theta2), np.sin(theta2),
                            np.cos(theta3), np.sin(theta3)))


# --- Space-fixed sequences --------------------------------------------------
def omega_from_euler_rates_space_seq(euler_angles, theta_dot, seq):
    """Angular velocity from Euler rates for a space-fixed rotation sequence.

    A space-fixed sequence is the reversed body-fixed sequence with the angles
    and rates reversed, so the body table serves both cases.
    """
    euler_angles = np.asarray(euler_angles, dtype=float).ravel()
    theta_dot = np.asarray(theta_dot, dtype=float).ravel()
    return omega_from_euler_rates_body_seq(euler_angles[::-1], theta_dot[::-1],
                                           tuple(reversed(tuple(seq))))


# theta_dot = f(w1, w2, w3, c1, s1, c2, s2)
_EULER_RATES_SPACE = {
    (1, 2, 3): lambda w1, w2, w3, c1, s1, c2, s2: (
        w1 + ((w2 * s1 + w3 * c1) * s2) / c2, w2 * c1 - w3 * s1, (w2 * s1 + w3 * c1) / c2),
    (2, 3, 1): lambda w1, w2, w3, c1, s1, c2, s2: (
        ((w1 * c1 + w3 * s1) * s2) / c2 + w2, -w1 * s1 + w3 * c1, (w1 * c1 + w3 * s1) / c2),
    (3, 1, 2): lambda w1, w2, w3, c1, s1, c2, s2: (
        (w1 * s1 + w2 * c1) * s2 / c2 + w3, w1 * c1 - w2 * s1, (w1 * s1 + w2 * c1) / c2),
    (1, 3, 2): lambda w1, w2, w3, c1, s1, c2, s2: (
        w1 + ((-w2 * c1 + w3 * s1) * s2) / c2, w2 * s1 + w3 * c1, (w2 * c1 - w3 * s1) / c2),
    (2, 1, 3): lambda w1, w2, w3, c1, s1, c2, s2: (
        ((w1 * s1 - w3 * c1) * s2) / c2 + w2, w1 * c1 + w3 * s1, (-w1 * s1 + w3 * c1) / c2),
    (3, 2, 1): lambda w1, w2, w3, c1, s1, c2, s2: (
        ((-w1 * c1 + w2 * s1) * s2) / c2 + w3, w1 * s1 + w2 * c1, (w1 * c1 - w2 * s1) / c2),
    (1, 2, 1): lambda w1, w2, w3, c1, s1, c2, s2: (
        w1 - ((w2 * s1 + w3 * c1) * c2) / s2, w2 * c1 - w3 * s1, (w2 * s1 + w3 * c1) / s2),
    (1, 3, 1): lambda w1, w2, w3, c1, s1, c2, s2: (
        w1 + ((w2 * c1 - w3 * s1) * c2) / s2, w2 * s1 + w3 * c1, (-w2 * c1 + w3 * s1) / s2),
}


def euler_rates_space_seq(omega, euler_angles, seq):
    """Euler-angle rates from angular velocity for a space-fixed sequence."""
    w1, w2, w3 = np.asarray(omega, dtype=float).ravel()
    theta1, theta2, _ = np.asarray(euler_angles, dtype=float).ravel()
    formula = _lookup(_EULER_RATES_SPACE, seq, "Space-fixed")
    return np.array(formula(w1, w2, w3, np.cos(theta1), np.sin(theta1),
                            np.cos(theta2), np.sin(theta2)))


# --- Quaternion rates -------------------------------------------------------
def quat_dot(q, omega):
    """Quaternion derivative from the vector/scalar split of the kinematics."""
    q = np.asarray(q, dtype=float).ravel()
    omega = np.asarray(omega, dtype=float).ravel()
    q13, q4 = q[:3], q[3]

    q13_dot = 0.5 * (q4 * omega + np.cross(q13, omega))
    q4_dot = -0.5 * np.dot(omega, q13)
    return np.append(q13_dot, q4_dot)


def quat_dot_2(q, omega):
    """Quaternion derivative written out component by component."""
    q1, q2, q3, q4 = np.asarray(q, dtype=float).ravel()
    w1, w2, w3 = np.asarray(omega, dtype=float).ravel()
    return 0.5 * np.array([
        w1 * q4 - w2 * q3 + w3 * q2,
        w1 * q3 + w2 * q4 - w3 * q1,
        -w1 * q2 + w2 * q1 + w3 * q4,
        -(w1 * q1 + w2 * q2 + w3 * q3),
    ])


def quaternion_kinematics_fixed_omega(q, omega):
    """Quaternion derivative as q_dot = 0.5 Xi(q) omega."""
    return 0.5 * xi_matrix(q) @ np.asarray(omega, dtype=float).ravel()


def quat_dot_to_omega(q, q_dot):
    """Angular velocity from a quaternion and its derivative.

    Returns a 4-vector, as the MATLAB source did; the last entry is zero for a
    consistent (unit-norm) quaternion and its derivative.
    """
    q1, q2, q3, q4 = np.asarray(q, dtype=float).ravel()
    e_matrix = np.array([[q4, -q3, q2, q1],
                         [q3, q4, -q1, q2],
                         [-q2, q1, q4, q3],
                         [-q1, -q2, -q3, q4]])
    return 2.0 * (np.asarray(q_dot, dtype=float).ravel() @ e_matrix)
