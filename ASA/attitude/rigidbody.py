"""Rigid-body rotational dynamics, gravity moments, and inertia properties.

Ported from ``AttitudeDeterminationLibrary.m`` (ASA_MATLAB).
"""

import numpy as np

from .kinematics import quaternion_kinematics_fixed_omega

__all__ = [
    "approx_gravity_force_f2",
    "center_of_gravity",
    "center_of_mass",
    "compute_gravity_moment",
    "inertia_solid_cylinder",
    "rigid_body_dynamics_quaternions",
]


def rigid_body_dynamics_quaternions(t, state, inertia, torque=None):
    """Quaternion kinematics plus Euler's equations, for use with an ODE solver.

    State is ``[q (4), omega (3)]``.  ``torque`` defaults to zero, which gives
    torque-free motion.
    """
    state = np.asarray(state, dtype=float).ravel()
    q, omega = state[:4], state[4:7]
    inertia = np.asarray(inertia, dtype=float)
    torque = np.zeros(3) if torque is None else np.asarray(torque, dtype=float).ravel()

    q_dot = quaternion_kinematics_fixed_omega(q, omega)
    omega_dot = np.linalg.solve(inertia, torque - np.cross(omega, inertia @ omega))
    return np.concatenate([q_dot, omega_dot])


# --- Inertia ----------------------------------------------------------------
def inertia_solid_cylinder(radius, height, param, param_is_mass=False):
    """Axial and transverse inertia of a solid cylinder, plus its mass.

    ``param`` is the density unless ``param_is_mass`` is set.
    """
    mass = param if param_is_mass else param * np.pi * radius**2 * height
    return 0.5 * mass * radius**2, (1.0 / 12.0) * mass * (3 * radius**2 + height**2), mass


# --- Mass and gravity centers -----------------------------------------------
def center_of_mass(pos_vector, m_vector):
    """Planar center of mass of a set of particles, returned as ``(x_cm, y_cm)``."""
    pos_vector = np.asarray(pos_vector, dtype=float)
    m_vector = np.asarray(m_vector, dtype=float).ravel()
    total = m_vector.sum()
    return float(pos_vector[:, 0] @ m_vector / total), float(pos_vector[:, 1] @ m_vector / total)


def center_of_gravity(pos_vector, m_vector, r_p, m_p, g):
    """Effective center of gravity of particles attracted by a body at ``r_p``.

    Returns ``(r_cg, f_net, f_hat, f_mag)``, where ``r_cg`` is the distance from
    the attractor along the line of action of the net force.
    """
    pos_vector = np.asarray(pos_vector, dtype=float)
    m_vector = np.asarray(m_vector, dtype=float).ravel()
    r_p = np.asarray(r_p, dtype=float).ravel()

    f_net = np.zeros(2)
    for position, mass in zip(pos_vector, m_vector):
        r_i = position - r_p
        f_net += -(g * m_p * mass / np.linalg.norm(r_i) ** 3) * r_i

    f_mag = float(np.linalg.norm(f_net))
    r_cg = float(np.sqrt(g * m_p * m_vector.sum() / f_mag))
    return r_cg, f_net, -f_net / f_mag, f_mag


def approx_gravity_force_f2(mu_body, mass, distance, dcm, i1, i2, i3,
                            a1=None, a2=None, a3=None):
    """Gravitational force on an extended body, kept through the f2 term.

    ``dcm`` is the inertial-to-body matrix in the row convention, ``Cij = ai.bj``.
    Returns ``(f_approx, f_particle)``.
    """
    a1 = np.array([1.0, 0.0, 0.0]) if a1 is None else np.asarray(a1, dtype=float).ravel()
    a2 = np.array([0.0, 1.0, 0.0]) if a2 is None else np.asarray(a2, dtype=float).ravel()
    a3 = np.array([0.0, 0.0, 1.0]) if a3 is None else np.asarray(a3, dtype=float).ravel()
    dcm = np.asarray(dcm, dtype=float)

    f_particle = -(mu_body * mass / distance**2) * a1

    term1 = 0.5 * (i1 * (1 - 3 * dcm[0, 0] ** 2) + i2 * (1 - 3 * dcm[0, 1] ** 2)
                   + i3 * (1 - 3 * dcm[0, 2] ** 2))
    term2 = i1 * dcm[1, 0] * dcm[0, 0] + i2 * dcm[1, 1] * dcm[0, 1] + i3 * dcm[1, 2] * dcm[0, 2]
    term3 = i1 * dcm[2, 0] * dcm[0, 0] + i2 * dcm[2, 1] * dcm[0, 1] + i3 * dcm[2, 2] * dcm[0, 2]

    f2 = (3.0 / (mass * distance**2)) * (term1 * a1 + term2 * a2 + term3 * a3)
    return -(mu_body * mass / distance**2) * (a1 + f2), f_particle


def compute_gravity_moment(mu_body, mass, r_cm_scalar, f_approx):
    """Gravity-gradient moment about the center of mass from an approximate force.

    Returns ``(moment, f_approx, f_mag, f_hat, r_cg_vec, r_cm_vec, r_cm_cg)``.
    """
    f_approx = np.asarray(f_approx, dtype=float).ravel()
    f_mag = float(np.linalg.norm(f_approx))
    f_hat = -f_approx / f_mag

    r_cg_vec = np.sqrt(mu_body * mass / f_mag) * f_hat
    r_cm_vec = r_cm_scalar * np.array([1.0, 0.0, 0.0])
    r_cm_cg = r_cg_vec - r_cm_vec
    moment = -np.cross(r_cm_vec, f_approx)
    return moment, f_approx, f_mag, f_hat, r_cg_vec, r_cm_vec, r_cm_cg
