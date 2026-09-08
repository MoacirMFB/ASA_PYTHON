"""Static attitude determination: TRIAD, QUEST, Davenport's q-method.

Ported from ``AttitudeDeterminationLibrary.m`` (ASA_MATLAB).

Vector sets are passed as 3xN arrays, one measurement per column, and every
routine returns the inertial-to-body DCM ``[BN]``.
"""

import numpy as np

from .quaternions import quat_to_dcm

__all__ = [
    "davenport_qmethod",
    "estimate_covariance_wahba",
    "quest_method",
    "real_covariance_wahba",
    "triad_method",
]


def _adjugate_3x3(matrix):
    """Adjugate (classical adjoint) of a 3x3 matrix, valid even when singular."""
    cofactors = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            minor = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
            cofactors[i, j] = (-1.0) ** (i + j) * np.linalg.det(minor)
    return cofactors.T


def _attitude_profile(weights, body_vectors, inertial_vectors, num_vectors=None):
    """Weighted attitude profile matrix B and its axial vector z."""
    body_vectors = np.asarray(body_vectors, dtype=float)
    inertial_vectors = np.asarray(inertial_vectors, dtype=float)
    weights = np.asarray(weights, dtype=float).ravel()
    if num_vectors is None:
        num_vectors = body_vectors.shape[1]

    b_matrix = np.zeros((3, 3))
    z_vector = np.zeros(3)
    for i in range(num_vectors):
        b_matrix += weights[i] * np.outer(body_vectors[:, i], inertial_vectors[:, i])
        z_vector += weights[i] * np.cross(body_vectors[:, i], inertial_vectors[:, i])
    return b_matrix, z_vector


def triad_method(v1_b, v2_b, v1_i, v2_i):
    """TRIAD algorithm; the first vector pair is taken as the more accurate one."""
    def unit(v):
        v = np.asarray(v, dtype=float).ravel()
        return v / np.linalg.norm(v)

    v1_b, v2_b, v1_i, v2_i = unit(v1_b), unit(v2_b), unit(v1_i), unit(v2_i)

    t1_b = v1_b
    t2_b = np.cross(v1_b, v2_b) / np.linalg.norm(np.cross(v1_b, v2_b))
    t3_b = np.cross(t1_b, t2_b)

    t1_i = v1_i
    t2_i = np.cross(v1_i, v2_i) / np.linalg.norm(np.cross(v1_i, v2_i))
    t3_i = np.cross(t1_i, t2_i)

    return np.column_stack([t1_b, t2_b, t3_b]) @ np.column_stack([t1_i, t2_i, t3_i]).T


def quest_method(weights, body_vectors, inertial_vectors, num_vectors=None,
                 max_iter=50, tol=1.0e-12):
    """QUEST: Newton-Raphson on the characteristic equation of Wahba's problem.

    Returns ``(quaternion, dcm, loss, taste, lambda_max)``.
    """
    weights = np.asarray(weights, dtype=float).ravel()
    b_matrix, z_vector = _attitude_profile(weights, body_vectors, inertial_vectors, num_vectors)

    trace_b = np.trace(b_matrix)
    s_matrix = b_matrix + b_matrix.T
    det_s = np.linalg.det(s_matrix)
    kappa = np.trace(_adjugate_3x3(s_matrix))
    z_norm_sq = float(z_vector @ z_vector)
    zsz = float(z_vector @ s_matrix @ z_vector)
    zs2z = float(z_vector @ (s_matrix @ s_matrix) @ z_vector)

    def characteristic(lam):
        return ((lam**2 - trace_b**2 + kappa) * (lam**2 - trace_b**2 - z_norm_sq)
                - (lam - trace_b) * (zsz + det_s) - zs2z)

    def characteristic_slope(lam):
        return (2 * lam * (lam**2 - trace_b**2 + kappa) - (zsz + det_s)
                - 2 * lam * (-lam**2 + trace_b**2 + z_norm_sq))

    lambda_max = 1.0
    for _ in range(max_iter):
        step = characteristic(lambda_max) / characteristic_slope(lambda_max)
        lambda_max -= step
        if abs(step) < tol:
            break

    rho = lambda_max + trace_b
    q13 = _adjugate_3x3(rho * np.eye(3) - s_matrix) @ z_vector
    q4 = np.linalg.det(rho * np.eye(3) - s_matrix)

    quaternion = np.append(q13, q4)
    quaternion = quaternion / np.linalg.norm(quaternion)

    loss = float(np.sum(weights) - lambda_max)
    return quaternion, quat_to_dcm(quaternion), loss, 2.0 * loss, float(lambda_max)


def davenport_qmethod(weights, body_vectors, inertial_vectors, num_vectors=None):
    """Davenport's q-method: the optimal quaternion is the top eigenvector of K.

    Returns ``(quaternion, dcm, loss, taste)``.
    """
    weights = np.asarray(weights, dtype=float).ravel()
    b_matrix, z_vector = _attitude_profile(weights, body_vectors, inertial_vectors, num_vectors)
    trace_b = np.trace(b_matrix)

    k_matrix = np.zeros((4, 4))
    k_matrix[:3, :3] = b_matrix + b_matrix.T - trace_b * np.eye(3)
    k_matrix[:3, 3] = z_vector
    k_matrix[3, :3] = z_vector
    k_matrix[3, 3] = trace_b

    eigenvalues, eigenvectors = np.linalg.eigh(k_matrix)  # K is symmetric
    index = int(np.argmax(eigenvalues))
    quaternion = eigenvectors[:, index]
    quaternion = quaternion / np.linalg.norm(quaternion)

    loss = float(np.sum(weights) - eigenvalues[index])
    return quaternion, quat_to_dcm(quaternion), loss, 2.0 * loss


def estimate_covariance_wahba(lambda_max, measured_vectors, inertial_vectors,
                              weights, dcm_est, c):
    """Attitude covariance P = c (lambda_max I - B A^T)^-1 from the estimate."""
    measured_vectors = np.asarray(measured_vectors, dtype=float)
    weights = np.asarray(weights, dtype=float).ravel()
    if weights.size != measured_vectors.shape[1]:
        raise ValueError("The number of weights must match the number of vectors.")

    b_matrix, _ = _attitude_profile(weights, measured_vectors, inertial_vectors)
    return c * np.linalg.pinv(lambda_max * np.eye(3) - b_matrix @ np.asarray(dcm_est).T)


def real_covariance_wahba(n, std_dev_deg, true_vector):
    """Covariance from the Fisher information matrix for N equal-variance measurements."""
    sigma = np.deg2rad(std_dev_deg)
    true_vector = np.asarray(true_vector, dtype=float).ravel()
    fisher = n * (1.0 / sigma**2) * (np.eye(3) - np.outer(true_vector, true_vector))
    return np.linalg.pinv(fisher)
