"""Reaction wheels: configuration, PD control, and torque distribution.

Ported from ``AttitudeDeterminationLibrary.m`` (ASA_MATLAB).
"""

import numpy as np

from .dcm import skew_symmetric
from .quaternions import xi_matrix

__all__ = [
    "compute_hiw_hjw_h0",
    "compute_hkw",
    "compute_dij",
    "compute_sijk",
    "compute_vij",
    "compute_w4_configuration",
    "compute_wij",
    "control_torque_rcw_law1",
    "control_torque_rcw_law2",
    "minimax_method",
    "pseudoinverse_method",
    "system_dynamics_controlled_rcw",
]


def compute_w4_configuration(config, params):
    """Four-wheel configuration matrix, its pseudoinverse, and its null vector.

    ``config`` is ``"pyramid"`` (params a, b, c, d) or ``"nasa"`` (params alpha,
    beta, gamma).  Returns ``(w4, w4_pseudo, null_vector)``.
    """
    if config == "pyramid":
        a, b, c, d = (params[k] for k in ("a", "b", "c", "d"))
        w4 = np.array([[a, -a, 0.0, 0.0],
                       [b, b, c, c],
                       [0.0, 0.0, d, -d]])
        null_vector = (1.0 / np.sqrt(2 * (b**2 + c**2))) * np.array([c, c, -b, -b])
    elif config == "nasa":
        alpha, beta, gamma = (params[k] for k in ("alpha", "beta", "gamma"))
        w4 = np.array([[1.0, 0.0, 0.0, alpha],
                       [0.0, 1.0, 0.0, beta],
                       [0.0, 0.0, 1.0, gamma]])
        null_vector = (1.0 / np.sqrt(1 + alpha**2 + beta**2 + gamma**2)) \
            * np.array([alpha, beta, gamma, -1.0])
    else:
        raise ValueError('Invalid configuration. Choose either "pyramid" or "nasa".')

    return w4, np.linalg.pinv(w4), null_vector


# --- Control laws (Crassidis Eq. 7.17) --------------------------------------
def control_torque_rcw_law1(dq, omega, kp, kd):
    """PD control torque with the gain scaled by the quaternion error (7.17b)."""
    dq = np.asarray(dq, dtype=float).ravel()
    omega = np.asarray(omega, dtype=float).ravel()
    dq13, dq4 = dq[:3], dq[3]
    return -kp * np.sign(dq4) * dq13 - kd * (1.0 + dq13 @ dq13) * omega


def control_torque_rcw_law2(dq, omega, kp, kd):
    """Plain PD control torque on the quaternion error (7.17a)."""
    dq = np.asarray(dq, dtype=float).ravel()
    omega = np.asarray(omega, dtype=float).ravel()
    return -kp * np.sign(dq[3]) * dq[:3] - kd * omega


def system_dynamics_controlled_rcw(t, state, inertia, inertia_inv, q_command,
                                   kp, kd, control_law=1):
    """Closed-loop dynamics with wheel momentum in the state.

    State is ``[q (4), omega (3), h (3)]``, where ``h`` is the wheel angular
    momentum in the body frame.
    """
    state = np.asarray(state, dtype=float).ravel()
    q, omega, h = state[:4], state[4:7], state[7:10]

    dq13 = xi_matrix(q_command).T @ q
    dq4 = float(np.asarray(q).ravel() @ np.asarray(q_command).ravel())
    dq = np.append(dq13, dq4)

    if control_law == 1:
        torque = control_torque_rcw_law1(dq, omega, kp, kd)
    elif control_law == 2:
        torque = control_torque_rcw_law2(dq, omega, kp, kd)
    else:
        raise ValueError("Invalid control_law. Use 1 for Law1 or 2 for Law2.")

    q_dot = 0.5 * xi_matrix(q) @ omega
    omega_dot = inertia_inv @ (-skew_symmetric(omega) @ inertia @ omega + torque)
    h_dot = -skew_symmetric(omega) @ h - torque
    return np.concatenate([q_dot, omega_dot, h_dot])


# --- Torque distribution ----------------------------------------------------
def pseudoinverse_method(w, h_values, j_wheel):
    """Minimum-norm wheel momenta, speeds, and torques from the body momentum.

    ``h_values`` is (num_steps, 3).  Returns three (num_steps, N) arrays.
    """
    h_values = np.asarray(h_values, dtype=float)
    w = np.asarray(w, dtype=float)
    w_pinv = np.linalg.pinv(w)

    h_wheels = h_values @ w_pinv.T
    omega_wheels = h_wheels / j_wheel
    # ponytail: unit time step, as in the MATLAB source; pass scaled momenta if dt != 1.
    torques = np.vstack([np.zeros((1, w.shape[1])), np.diff(h_wheels, axis=0)])
    return h_wheels, omega_wheels, torques


def compute_sijk(w_i, w_j, w_k_list):
    """Triple products (w_i x w_j).w_k for the wheels outside the pair (i, j)."""
    cross_ij = np.cross(w_i, w_j)
    return np.asarray(w_k_list, dtype=float).T @ cross_ij


def compute_vij(s_ijk, w_k_list):
    """Signed sum of the remaining spin axes."""
    w_k_list = np.asarray(w_k_list, dtype=float)
    return w_k_list @ np.sign(np.asarray(s_ijk, dtype=float))


def compute_dij(w_i, w_j, v_ij):
    """Distance measure d_ij = (w_i x w_j).v_ij / |w_i x w_j|."""
    cross_ij = np.cross(w_i, w_j)
    return float(cross_ij @ v_ij / np.linalg.norm(cross_ij))


def compute_wij(w_i, w_j, v_ij):
    """Normal w_ij of the momentum-envelope facet, and its denominator."""
    cross_ij = np.cross(w_i, w_j)
    denominator = float(cross_ij @ v_ij)
    return cross_ij / denominator, denominator


def compute_hiw_hjw_h0(w_i, w_j, v_ij, h_body):
    """Wheel momenta for the selected pair plus the common level H0."""
    cross_ij = np.cross(w_i, w_j)
    denominator = float(cross_ij @ v_ij)
    if abs(denominator) < 1.0e-6:
        raise ValueError("Denominator is too small, possible numerical instability.")
    numerator = np.vstack([np.cross(w_j, v_ij), np.cross(v_ij, w_i), cross_ij])
    h_values = (1.0 / denominator) * numerator @ np.asarray(h_body, dtype=float)
    return h_values[0], h_values[1], h_values[2]


def compute_hkw(h0, s_ijk, k_list, n):
    """Momenta of the saturated wheels, all at +/- H0."""
    h_kw = np.zeros(n)
    for index, k in enumerate(k_list):
        h_kw[k] = h0 * np.sign(s_ijk[index])
    return h_kw


def minimax_method(w, omega_values, h_values, inertia, dt, j_wheel):
    """Minimax (L-infinity) wheel momentum distribution over a time history.

    Returns ``(h_wheels, omega_wheels, torques)``, each (num_steps, N).
    """
    w = np.asarray(w, dtype=float)
    h_values = np.asarray(h_values, dtype=float)
    n = w.shape[1]
    num_steps = np.asarray(omega_values, dtype=float).shape[0]

    h_wheels = np.zeros((num_steps, n))
    for step in range(num_steps):
        h_body = h_values[step, :]

        best_dot = -np.inf
        best = None
        for i in range(n - 1):
            for j in range(i + 1, n):
                k_list = [k for k in range(n) if k not in (i, j)]
                w_k_list = w[:, k_list]
                s_ijk = compute_sijk(w[:, i], w[:, j], w_k_list)
                v_ij = compute_vij(s_ijk, w_k_list)
                w_ij, _ = compute_wij(w[:, i], w[:, j], v_ij)
                dot = float(w_ij @ h_body)
                if dot > best_dot:
                    best_dot = dot
                    best = (i, j, v_ij, s_ijk, k_list)

        i, j, v_ij, s_ijk, k_list = best
        h_iw, h_jw, h0 = compute_hiw_hjw_h0(w[:, i], w[:, j], v_ij, h_body)
        h_kw = compute_hkw(h0, s_ijk, k_list, n)
        h_kw[i], h_kw[j] = h_iw, h_jw
        h_wheels[step, :] = h_kw

    omega_wheels = h_wheels / j_wheel
    torques = np.vstack([np.zeros((1, n)), np.diff(h_wheels, axis=0) / dt])
    return h_wheels, omega_wheels, torques
