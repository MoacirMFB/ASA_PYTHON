"""Attitude sensor models: rate gyro, star tracker, and center of light.

Ported from ``AttitudeDeterminationLibrary.m`` (ASA_MATLAB).

Every stochastic routine takes an optional ``rng`` (a ``numpy.random.Generator``)
so that simulations can be made repeatable.
"""

import numpy as np

from .quaternions import quaternion_product

__all__ = [
    "center_of_light",
    "gyro_bias_at_next_step",
    "simulate_gyroscope_noised",
    "simulate_star_tracker_measurements",
]

ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)


def simulate_gyroscope_noised(omega_true, s_true, sigma_u, sigma_v, dt, bias_k,
                              bias_kp1, rng=None):
    """Noisy rate-gyro measurement at step k+1 (Crassidis Eq. 4.54a).

    ``s_true`` is the 3x3 scale-factor and misalignment matrix, ``sigma_u`` the
    bias random-walk parameter and ``sigma_v`` the angle random-walk parameter.
    """
    rng = np.random.default_rng() if rng is None else rng
    noise = rng.standard_normal(3)

    omega_true = np.asarray(omega_true, dtype=float).ravel()
    bias_k = np.asarray(bias_k, dtype=float).ravel()
    bias_kp1 = np.asarray(bias_kp1, dtype=float).ravel()

    scale = np.sqrt(sigma_v**2 / dt + (1.0 / 12.0) * sigma_u**2 * dt)
    return (np.eye(3) + np.asarray(s_true, dtype=float)) @ omega_true \
        + 0.5 * (bias_k + bias_kp1) + scale * noise


def gyro_bias_at_next_step(bias_k, sigma_u, dt, rng=None):
    """Gyro bias random walk from step k to k+1 (Crassidis Eq. 4.54b)."""
    rng = np.random.default_rng() if rng is None else rng
    bias_k = np.asarray(bias_k, dtype=float).ravel()
    return bias_k + sigma_u * np.sqrt(dt) * rng.standard_normal(3)


def simulate_star_tracker_measurements(q_truth, covariance_arcsec2, rng=None):
    """Star tracker quaternion measurements from a small-angle noise model.

    ``q_truth`` is 4xN (one quaternion per column) and ``covariance_arcsec2`` is
    the 3x3 attitude error covariance in arcsec^2.
    """
    rng = np.random.default_rng() if rng is None else rng
    q_truth = np.asarray(q_truth, dtype=float)
    if q_truth.ndim == 1:
        q_truth = q_truth.reshape(4, 1)
    if q_truth.shape[0] != 4:
        raise ValueError("q_truth must be a 4xN matrix, one quaternion per column.")

    covariance = np.asarray(covariance_arcsec2, dtype=float) * ARCSEC_TO_RAD**2
    if covariance.shape != (3, 3):
        raise ValueError("The covariance must be a 3x3 matrix.")
    cholesky = np.linalg.cholesky(covariance)

    q_meas = np.zeros_like(q_truth)
    for i in range(q_truth.shape[1]):
        small_angle = cholesky @ rng.standard_normal(3)
        delta_q = np.append(0.5 * small_angle, 1.0)
        delta_q /= np.linalg.norm(delta_q)
        measured = quaternion_product(delta_q, q_truth[:, i])
        q_meas[:, i] = measured / np.linalg.norm(measured)
    return q_meas


def center_of_light(image):
    """Intensity-weighted centroid of an image, returned as ``(col, row)``.

    The coordinates are 1-based to match the MATLAB pixel indexing.
    """
    image = np.asarray(image, dtype=float)
    total = image.sum()
    rows, cols = np.indices(image.shape) + 1  # 1-based pixel coordinates
    return float((image * cols).sum() / total), float((image * rows).sum() / total)
