"""Keplerian orbital mechanics, frame conversions, and propagators.

Distances are in kilometers, time is in seconds, and angles are in radians.
Functions accept array-like inputs and return NumPy arrays where appropriate.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .bodies import CelestialBody

FloatArray = NDArray[np.float64]

EARTH_MU_KM3_S2 = 398600.0
EARTH_J2 = 1.08263e-3
EARTH_RADIUS_KM = 6378.137
AU_KM = 149_597_870.7
SOLAR_FLUX_CONSTANT = 1.02e14  # G0 = G_AU * d_AU^2 / c, in kg*km/s^2


def _as_float_array(values: Iterable[float] | FloatArray, size: int | None = None) -> FloatArray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if size is not None and array.size != size:
        raise ValueError(f"Expected {size} entries, got {array.size}.")
    return array


def _solve_ivp_checked(*args, **kwargs):
    solution = solve_ivp(*args, **kwargs)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution


def vec_cart_to_rot(dcm_inverse: Iterable[float] | FloatArray, vector_eci: Iterable[float] | FloatArray) -> FloatArray:
    """Transform an ECI vector to the rotating radial-transverse-normal frame."""

    return np.asarray(dcm_inverse, dtype=float).reshape(3, 3) @ _as_float_array(vector_eci, 3)


def vec_rot_to_cart(dcm: Iterable[float] | FloatArray, vector_rot: Iterable[float] | FloatArray) -> FloatArray:
    """Transform a radial-transverse-normal vector to ECI."""

    return np.asarray(dcm, dtype=float).reshape(3, 3) @ _as_float_array(vector_rot, 3)


def orbital_elements_to_dcm(raan: float, inclination: float, theta: float) -> FloatArray:
    """Return the RTN-to-ECI direction cosine matrix for ``theta = omega + nu``."""

    cos_o, sin_o = np.cos(raan), np.sin(raan)
    cos_i, sin_i = np.cos(inclination), np.sin(inclination)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [cos_o * cos_t - sin_o * cos_i * sin_t, -cos_o * sin_t - sin_o * cos_i * cos_t, sin_o * sin_i],
            [sin_o * cos_t + cos_o * cos_i * sin_t, -sin_o * sin_t + cos_o * cos_i * cos_t, -cos_o * sin_i],
            [sin_i * sin_t, sin_i * cos_t, cos_i],
        ]
    )


def i_h_xyz(h_vec_norm: Iterable[float] | FloatArray) -> float:
    """Recover inclination from a normalized angular momentum vector."""

    return float(np.arccos(np.clip(_as_float_array(h_vec_norm, 3)[2], -1.0, 1.0)))


def theta_3d_rth_hat(
    r_eci_unit: Iterable[float] | FloatArray,
    theta_eci_unit: Iterable[float] | FloatArray,
    inclination: float,
) -> float:
    """Recover argument of latitude from radial and transverse unit vectors."""

    sin_i = np.sin(inclination)
    if abs(sin_i) < 1e-14:
        raise ValueError("Argument of latitude is undefined for an equatorial orbit.")
    sin_theta = _as_float_array(r_eci_unit, 3)[2] / sin_i
    cos_theta = _as_float_array(theta_eci_unit, 3)[2] / sin_i
    return float(np.mod(np.arctan2(sin_theta, cos_theta), 2.0 * np.pi))


def raan_find_use_h_hat(h_vec_norm_eci: Iterable[float] | FloatArray, inclination: float) -> float:
    """Recover right ascension of the ascending node from ``h_hat``."""

    sin_i = np.sin(inclination)
    if abs(sin_i) < 1e-14:
        raise ValueError("RAAN is undefined for an equatorial orbit.")
    h_hat = _as_float_array(h_vec_norm_eci, 3)
    return float(np.mod(np.arctan2(h_hat[0] / sin_i, -h_hat[1] / sin_i), 2.0 * np.pi))


def rotational_hat(raan: float, inclination: float, theta: float) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return transverse, radial, and normal unit vectors and their DCM."""

    dcm = orbital_elements_to_dcm(raan, inclination, theta)
    r_hat, theta_hat, h_hat = dcm[:, 0], dcm[:, 1], dcm[:, 2]
    return theta_hat, r_hat, h_hat, dcm


def r_vec_rot_to_vnb(fpa: float, r_mag: float) -> FloatArray:
    return np.array([r_mag * np.sin(fpa), 0.0, r_mag * np.cos(fpa)])


def r_vec_ep_a_e_e_b(a: float, eccentric_anomaly: float, eccentricity: float, b: float) -> FloatArray:
    return np.array([a * (np.cos(eccentric_anomaly) - eccentricity), b * np.sin(eccentric_anomaly), 0.0])


def r_vec_rot_frame(r_mag: float) -> FloatArray:
    return np.array([r_mag, 0.0, 0.0])


def r_vec_ep_tar(true_anomaly: float, r_mag: float) -> FloatArray:
    return np.array([r_mag * np.cos(true_anomaly), r_mag * np.sin(true_anomaly), 0.0])


def r_aeta(a: float, eccentricity: float, true_anomaly: float) -> float:
    """Return conic radius from semi-major-axis magnitude, eccentricity, and anomaly."""

    if eccentricity == 1.0:
        raise ValueError("A parabolic orbit requires a semilatus rectum, not a semi-major axis.")
    p = abs(a) * (eccentricity**2 - 1.0) if eccentricity > 1.0 else a * (1.0 - eccentricity**2)
    return float(p / (1.0 + eccentricity * np.cos(true_anomaly)))


def r_peta(p: float, eccentricity: float, true_anomaly: float) -> float:
    return float(p / (1.0 + eccentricity * np.cos(true_anomaly)))


def r_hueta(h: float, mu: float, eccentricity: float, true_anomaly: float) -> float:
    return r_peta(h**2 / mu, eccentricity, true_anomaly)


def vel_inf_ua(mu: float, a: float) -> float:
    return float(np.sqrt(mu / abs(a)))


def v_vec_rot_frame_fpav(fpa: float, speed: float) -> FloatArray:
    return np.array([speed * np.sin(fpa), speed * np.cos(fpa), 0.0])


def v_vec_ep_r_eban(r: Iterable[float] | FloatArray, eccentric_anomaly: float, b: float, a: float, n: float) -> FloatArray:
    r_mag = np.linalg.norm(_as_float_array(r))
    return np.array([-a**2 * n * np.sin(eccentric_anomaly) / r_mag, a * b * n * np.cos(eccentric_anomaly) / r_mag, 0.0])


def v_vec_ep_frame(fpa: float, true_anomaly: float, speed: float) -> FloatArray:
    radial, transverse = speed * np.sin(fpa), speed * np.cos(fpa)
    return np.array(
        [
            radial * np.cos(true_anomaly) - transverse * np.sin(true_anomaly),
            radial * np.sin(true_anomaly) + transverse * np.cos(true_anomaly),
            0.0,
        ]
    )


def vel_ura(mu: float, r: float, a: float, orbit_type: str) -> float:
    kind = orbit_type.lower()
    if kind == "e":
        return float(np.sqrt(2.0 * mu / r - mu / a))
    if kind == "h":
        return float(np.sqrt(2.0 * mu / r + mu / abs(a)))
    raise ValueError("orbit_type must be 'E' for ellipse or 'H' for hyperbola.")


v_mag_ura = vel_ura


def v_circ_ur(mu: float, r: float) -> float:
    return float(np.sqrt(mu / r))


def a_rpe(rp: float, eccentricity: float) -> float:
    if eccentricity == 1.0:
        raise ValueError("A parabolic orbit has no finite semi-major axis.")
    return float(rp / (eccentricity - 1.0) if eccentricity > 1.0 else rp / (1.0 - eccentricity))


def a_rae(ra: float, eccentricity: float) -> float:
    return float(ra / (1.0 + eccentricity))


def a_ruv(r: float, mu: float, speed: float) -> float:
    return float(abs(mu * r / (r * speed**2 - 2.0 * mu)))


def a_rpra(rp: float, ra: float) -> float:
    return float((rp + ra) / 2.0)


def a_espu(specific_energy: float, mu: float) -> float:
    return float(mu / (2.0 * abs(specific_energy)))


def b_ae(a: float, eccentricity: float) -> float:
    return float(abs(a) * np.sqrt(abs(1.0 - eccentricity**2)))


def ra_ae(a: float, eccentricity: float) -> float:
    return float(a * (1.0 + eccentricity))


def rp_ae(a: float, eccentricity: float) -> float:
    return float(abs(a) * abs(1.0 - eccentricity))


def p_hybl_ae(a: float, eccentricity: float) -> float:
    return float(abs(a) * (eccentricity**2 - 1.0))


def p_hu(h: float, mu: float) -> float:
    return float(h**2 / mu)


def p_ae(a: float, eccentricity: float) -> float:
    return p_hybl_ae(a, eccentricity) if eccentricity > 1.0 else float(a * (1.0 - eccentricity**2))


def delta_v_vnb_alpha_beta(delta_v_mag: float, beta: float, alpha: float) -> FloatArray:
    return delta_v_mag * np.array([np.cos(beta) * np.cos(alpha), np.sin(beta), np.cos(beta) * np.sin(alpha)])


def delta_v_rot_beta_phi(delta_v_mag: float, beta: float, phi: float) -> FloatArray:
    return delta_v_mag * np.array([np.cos(beta) * np.sin(phi), np.cos(beta) * np.cos(phi), np.sin(beta)])


def rx_fg_hyperbola(a: float, r0: Iterable[float] | FloatArray, delta_h: float, v0: Iterable[float] | FloatArray, dt: float, mu: float) -> FloatArray:
    r0_array, v0_array = _as_float_array(r0, 3), _as_float_array(v0, 3)
    f = 1.0 - abs(a) * (np.cosh(delta_h) - 1.0) / np.linalg.norm(r0_array)
    g = dt - np.sqrt(abs(a) ** 3 / mu) * (np.sinh(delta_h) - delta_h)
    return f * r0_array + g * v0_array


def f_conic(p: float, r: float, delta_true_anomaly: float) -> float:
    return float(1.0 - r * (1.0 - np.cos(delta_true_anomaly)) / p)


def g_conic(r: float, r0: Iterable[float] | FloatArray, mu: float, p: float, delta_true_anomaly: float) -> float:
    return float(r * np.linalg.norm(_as_float_array(r0)) * np.sin(delta_true_anomaly) / np.sqrt(mu * p))


def rx_fg_conic(p: float, r: float, r0: Iterable[float] | FloatArray, delta_true_anomaly: float, v0: Iterable[float] | FloatArray, mu: float) -> FloatArray:
    return f_conic(p, r, delta_true_anomaly) * _as_float_array(r0, 3) + g_conic(r, r0, mu, p, delta_true_anomaly) * _as_float_array(v0, 3)


def f_rx_ellipse(a: float, r0: Iterable[float] | FloatArray, delta_e: float) -> FloatArray:
    r0_array = _as_float_array(r0, 3)
    return (1.0 - a * (1.0 - np.cos(delta_e)) / np.linalg.norm(r0_array)) * r0_array


def g_rx_ellipse(v0: Iterable[float] | FloatArray, dt: float, delta_e: float, a: float, mu: float) -> FloatArray:
    return (dt - np.sqrt(a**3 / mu) * (delta_e - np.sin(delta_e))) * _as_float_array(v0, 3)


def rx_fg_ellipse(a: float, r0: Iterable[float] | FloatArray, delta_e: float, v0: Iterable[float] | FloatArray, dt: float, mu: float) -> FloatArray:
    return f_rx_ellipse(a, r0, delta_e) + g_rx_ellipse(v0, dt, delta_e, a, mu)


def vx_fg_hyperbola(a: float, r0: Iterable[float] | FloatArray, r: float, mu: float, delta_h: float, v0: Iterable[float] | FloatArray) -> FloatArray:
    r0_array, v0_array = _as_float_array(r0, 3), _as_float_array(v0, 3)
    f_dot = -np.sqrt(mu * abs(a)) * np.sinh(delta_h) / (r * np.linalg.norm(r0_array))
    g_dot = 1.0 - abs(a) * (np.cosh(delta_h) - 1.0) / r
    return f_dot * r0_array + g_dot * v0_array


def f_dot_conics(p: float, r0: Iterable[float] | FloatArray, v0: Iterable[float] | FloatArray, mu: float, delta_true_anomaly: float) -> float:
    r0_array, v0_array = _as_float_array(r0, 3), _as_float_array(v0, 3)
    r0_mag = np.linalg.norm(r0_array)
    return float(
        np.dot(r0_array, v0_array) * (1.0 - np.cos(delta_true_anomaly)) / (p * r0_mag)
        - np.sqrt(mu / p) * np.sin(delta_true_anomaly) / r0_mag
    )


def g_dot_conics(r0: Iterable[float] | FloatArray, p: float, delta_true_anomaly: float) -> float:
    return float(1.0 - np.linalg.norm(_as_float_array(r0)) * (1.0 - np.cos(delta_true_anomaly)) / p)


def vx_fg_conics(p: float, r0: Iterable[float] | FloatArray, v0: Iterable[float] | FloatArray, mu: float, delta_true_anomaly: float) -> FloatArray:
    return f_dot_conics(p, r0, v0, mu, delta_true_anomaly) * _as_float_array(r0, 3) + g_dot_conics(r0, p, delta_true_anomaly) * _as_float_array(v0, 3)


def f_vi_ellipse(a: float, r0: Iterable[float] | FloatArray, r: float, mu: float, delta_e: float) -> FloatArray:
    r0_array = _as_float_array(r0, 3)
    return -np.sqrt(mu * a) * np.sin(delta_e) * r0_array / (r * np.linalg.norm(r0_array))


def g_vi_ellipse(a: float, v0: Iterable[float] | FloatArray, r: float, delta_e: float) -> FloatArray:
    return (1.0 - a * (1.0 - np.cos(delta_e)) / r) * _as_float_array(v0, 3)


def vx_fg_ellipse(a: float, r0: Iterable[float] | FloatArray, r: float, mu: float, delta_e: float, v0: Iterable[float] | FloatArray) -> FloatArray:
    return f_vi_ellipse(a, r0, r, mu, delta_e) + g_vi_ellipse(a, v0, r, delta_e)


def mean_anomaly_from_time(t_since_periapsis: float, mu: float, a: float) -> float:
    return float(t_since_periapsis * np.sqrt(mu / a**3))


def mean_anomaly_from_eccentric_anomaly(eccentricity: float, eccentric_anomaly: float) -> float:
    return float(eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly))


def ta_rvufpa(r: float, speed: float, mu: float, fpa: float) -> FloatArray:
    scale = r * speed**2 / mu
    value = np.arctan2(scale * np.cos(fpa) * np.sin(fpa), scale * np.cos(fpa) ** 2 - 1.0)
    return np.array([value, value - np.pi])


def ta_rep(r: float, eccentricity: float, p: float) -> FloatArray:
    value = np.arccos(np.clip((p / r - 1.0) / eccentricity, -1.0, 1.0))
    return np.array([value, -value])


def ta_inf_e(eccentricity: float) -> float:
    return float(np.arccos(-1.0 / eccentricity))


def eccentric_anomaly_from_radius(a: float, r: float, eccentricity: float) -> float:
    if eccentricity == 0.0:
        raise ValueError("Eccentric anomaly cannot be recovered from radius for a circular orbit.")
    if eccentricity < 1.0:
        return float(np.arccos(np.clip((a - r) / (a * eccentricity), -1.0, 1.0)))
    return float(np.arccosh((abs(a) + r) / (abs(a) * eccentricity)))


def eccentric_anomaly_from_radius_and_true_anomaly(a: float, r: float, eccentricity: float, true_anomaly: float) -> float:
    anomaly = eccentric_anomaly_from_radius(a, r, eccentricity)
    return float(2.0 * np.pi - anomaly if eccentricity < 1.0 and np.mod(true_anomaly, 2.0 * np.pi) > np.pi else anomaly)


def phi_n_tof(transfer_angle: float, mean_motion: float, time_of_flight: float) -> float:
    return float(transfer_angle - mean_motion * time_of_flight)


def flyby_e(eccentricity: float) -> float:
    return float(2.0 * np.arcsin(1.0 / eccentricity))


def fpa_eta(eccentricity: float, true_anomaly: float) -> float:
    return float(np.arctan2(eccentricity * np.sin(true_anomaly), 1.0 + eccentricity * np.cos(true_anomaly)))


def fpa_hrv(h: float, r: float, speed: float) -> FloatArray:
    value = np.arccos(np.clip(h / (r * speed), -1.0, 1.0))
    return np.array([value, -value])


def e_flyby(flyby: float) -> float:
    return float(1.0 / np.sin(flyby / 2.0))


def e_r1r2_ta1_ta2(r1: float, r2: float, ta1: float, ta2: float) -> float:
    return float((r1 - r2) / (r2 * np.cos(ta2) - r1 * np.cos(ta1)))


def e_ellps_arp(a: float, rp: float) -> float:
    return float((a - rp) / a)


def e_hyp_arp(a: float, rp: float) -> float:
    return float(rp / abs(a) + 1.0)


def e_esphu(specific_energy: float, h: float, mu: float) -> float:
    return float(np.sqrt(1.0 + 2.0 * specific_energy * h**2 / mu**2))


e_concs_hespu = e_esphu
e_hypbl_arp = lambda rp, a: e_hyp_arp(a, rp)


def e_ruvfpa(r: float, mu: float, speed: float, fpa: float) -> float:
    scale = r * speed**2 / mu
    return float(np.sqrt((scale - 1.0) ** 2 * np.cos(fpa) ** 2 + np.sin(fpa) ** 2))


def e_ellps_pa(p: float, a: float) -> float:
    return float(np.sqrt(1.0 - p / a))


def e_hyper_pa(p: float, a: float) -> float:
    return float(np.sqrt(1.0 + p / abs(a)))


def e_rpra(rp: float, ra: float) -> float:
    return float((ra - rp) / (ra + rp))


def n_ua(mu: float, a: float) -> float:
    return float(np.sqrt(mu / a**3))


def h_mag_vprp(vp: float, rp: float) -> float:
    return float(vp * rp)


def h_uae(mu: float, a: float, eccentricity: float) -> float:
    return float(np.sqrt(mu * p_ae(a, eccentricity)))


def esp_vur(speed: float, mu: float, r: float) -> float:
    return float(speed**2 / 2.0 - mu / r)


def esp_ua(mu: float, a: float, orbit_type: str) -> float:
    kind = orbit_type.lower()
    if kind == "e":
        return float(-mu / (2.0 * abs(a)))
    if kind == "h":
        return float(mu / (2.0 * abs(a)))
    raise ValueError("orbit_type must be 'E' for ellipse or 'H' for hyperbola.")


def period_ua(mu: float, a: float) -> float:
    return float(2.0 * np.pi * np.sqrt(a**3 / mu))


def txtp_eccan_eua(anomaly: float, eccentricity: float, mu: float, a: float) -> float:
    scale = np.sqrt(mu / abs(a) ** 3)
    if eccentricity > 1.0:
        return float((eccentricity * np.sinh(anomaly) - anomaly) / scale)
    if eccentricity < 1.0:
        return float((anomaly - eccentricity * np.sin(anomaly)) / scale)
    raise ValueError("Parabolic time of flight requires Barker's equation.")


def delta_v_dept_prk(v_inf_departure: float, mu_planet: float, parking_radius: float, parking_speed: float) -> float:
    return float(np.sqrt(v_inf_departure**2 + 2.0 * mu_planet / parking_radius) - parking_speed)


def trngl_hyp_r1r2(r1: float, r2: float) -> float:
    return float(np.hypot(r1, r2))


def trngl_semiper_r1r2c(r1: float, r2: float, chord: float) -> float:
    return float((r1 + r2 + chord) / 2.0)


def trngl_a_min_semiper(semiperimeter: float) -> float:
    return float(semiperimeter / 2.0)


def trngl_dist_fq_fmin(a_min: float, radius: float) -> float:
    return float(2.0 * a_min - radius)


def trngl_alpha(a: float, semiperimeter: float) -> float:
    return float(2.0 * np.arcsin(np.sqrt(semiperimeter / (2.0 * a))))


def trngl_beta(a: float, semiperimeter: float, chord: float) -> float:
    return float(2.0 * np.arcsin(np.sqrt((semiperimeter - chord) / (2.0 * a))))


def trngl_alpha_hyp(a: float, semiperimeter: float) -> float:
    return float(2.0 * np.arcsinh(np.sqrt(semiperimeter / (2.0 * abs(a)))))


def trngl_beta_hyp(a: float, semiperimeter: float, chord: float) -> float:
    return float(2.0 * np.arcsinh(np.sqrt((semiperimeter - chord) / (2.0 * abs(a)))))


def trngl_p_ellipse(a: float, semiperimeter: float, chord: float, r1: float, r2: float, alpha: float, beta: float) -> FloatArray:
    factor = 4.0 * a * (semiperimeter - r1) * (semiperimeter - r2) / chord**2
    return factor * np.sin(np.array([(alpha + beta) / 2.0, (alpha - beta) / 2.0])) ** 2


def trngl_p_hyperb(a: float, semiperimeter: float, chord: float, r1: float, r2: float, alpha: float, beta: float) -> FloatArray:
    factor = 4.0 * abs(a) * (semiperimeter - r1) * (semiperimeter - r2) / chord**2
    return factor * np.sinh(np.array([(alpha + beta) / 2.0, (alpha - beta) / 2.0])) ** 2


def trngl_tof_parab(mu: float, semiperimeter: float, chord: float, transfer_type: int) -> float:
    sign = -1.0 if transfer_type == 1 else 1.0 if transfer_type == 2 else None
    if sign is None:
        raise ValueError("transfer_type must be 1 or 2.")
    return float(np.sqrt(2.0 / mu) * (semiperimeter**1.5 + sign * (semiperimeter - chord) ** 1.5) / 3.0)


def trngl_tof(mu: float, a: float, semiperimeter: float, chord: float, transfer_type: str) -> float:
    kind = transfer_type.upper()
    if kind == "1H":
        alpha = trngl_alpha_hyp(a, semiperimeter)
        beta = trngl_beta_hyp(a, semiperimeter, chord)
        return float(np.sqrt(abs(a) ** 3 / mu) * (np.sinh(alpha) - alpha - np.sinh(beta) + beta))

    alpha = trngl_alpha(a, semiperimeter)
    beta = trngl_beta(a, semiperimeter, chord)
    terms = {
        "1A": alpha - np.sin(alpha) - beta + np.sin(beta),
        "1B": 2.0 * np.pi - alpha + np.sin(alpha) - beta + np.sin(beta),
        "2A": alpha - np.sin(alpha) + beta - np.sin(beta),
        "2B": 2.0 * np.pi - alpha + np.sin(alpha) + beta - np.sin(beta),
    }
    if kind not in terms:
        raise ValueError("transfer_type must be '1A', '1B', '2A', '2B', or '1H'.")
    return float(np.sqrt(a**3 / mu) * terms[kind])


def dynamics_2bp_cartesian(t: float, x: Iterable[float] | FloatArray, mu: float) -> FloatArray:
    """Two-body Cartesian dynamics with state shape ``(6,)`` in km and km/s."""

    state = _as_float_array(x, size=6)
    r = state[:3]
    v = state[3:]
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-12:
        raise ValueError("Singular state: ||r|| is too small.")
    accel = -mu * r / (r_norm**3)
    return np.concatenate((v, accel))


def jacobian_2bp_cartesian(t: float, x: Iterable[float] | FloatArray, mu: float) -> FloatArray:
    """Jacobian of the unperturbed two-body Cartesian dynamics."""

    state = _as_float_array(x, size=6)
    r = state[:3]
    r2 = float(r @ r)
    r_norm = np.sqrt(r2)
    if r_norm < 1e-12:
        raise ValueError("Singular state: ||r|| is too small.")
    r3 = r2 * r_norm
    r5 = r2 * r3
    dadr = mu * ((3.0 * np.outer(r, r) / r5) - (np.eye(3) / r3))
    jac = np.zeros((6, 6), dtype=float)
    jac[:3, 3:] = np.eye(3)
    jac[3:, :3] = dadr
    return jac


def augmented_dynamics_2bp_stm(t: float, x_aug: Iterable[float] | FloatArray, mu: float) -> FloatArray:
    """State-plus-STM dynamics using column-major STM packing."""

    y = _as_float_array(x_aug)
    x = y[:6]
    phi = y[6:].reshape((6, 6), order="F")
    x_dot = dynamics_2bp_cartesian(t, x, mu)
    phi_dot = jacobian_2bp_cartesian(t, x, mu) @ phi
    return np.concatenate((x_dot, phi_dot.reshape(-1, order="F")))


def propagate_stm_2bp(
    x0: Iterable[float] | FloatArray,
    mu: float,
    tspan: tuple[float, float] | list[float] | FloatArray,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    method: str = "DOP853",
) -> tuple[FloatArray, FloatArray]:
    """Propagate the 2BP state and final STM over ``tspan``."""

    x0_array = _as_float_array(x0, size=6)
    t0, tf = float(tspan[0]), float(tspan[-1])
    if np.isclose(t0, tf):
        return np.eye(6, dtype=float), x0_array.copy()

    phi0 = np.eye(6, dtype=float)
    y0 = np.concatenate((x0_array, phi0.reshape(-1, order="F")))
    solution = _solve_ivp_checked(
        lambda t, y: augmented_dynamics_2bp_stm(t, y, mu),
        (t0, tf),
        y0,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    y_final = solution.y[:, -1]
    x_final = y_final[:6]
    phi_final = y_final[6:].reshape((6, 6), order="F")
    return phi_final, x_final


def propagate_two_body(
    x0: Iterable[float] | FloatArray,
    t_eval: Iterable[float] | FloatArray,
    mu: float,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    method: str = "DOP853",
) -> tuple[FloatArray, FloatArray]:
    """Propagate a 2BP state history on the requested time grid."""

    x0_array = _as_float_array(x0, size=6)
    t_grid = _as_float_array(t_eval)
    if t_grid.size == 0:
        raise ValueError("t_eval must contain at least one time sample.")
    if t_grid.size == 1:
        return t_grid.copy(), x0_array.reshape(1, 6)

    solution = _solve_ivp_checked(
        lambda t, x: dynamics_2bp_cartesian(t, x, mu),
        (float(t_grid[0]), float(t_grid[-1])),
        x0_array,
        method=method,
        t_eval=t_grid,
        rtol=rtol,
        atol=atol,
    )
    return solution.t, solution.y.T


def cartesian_to_spherical_state(x_cart: Iterable[float] | FloatArray) -> FloatArray:
    """Convert a Cartesian state to ``[r, phi, theta, r_dot, phi_dot, theta_dot]``.

    Uses the lecture convention S: {e_r, e_phi, e_theta}, where ``phi`` is the
    azimuth about e_z measured from e_x and ``theta`` is the elevation above the
    e_x-e_y plane.
    """

    state = _as_float_array(x_cart, 6)
    x, y, z = state[:3]
    vx, vy, vz = state[3:]
    radius = np.linalg.norm(state[:3])
    elevation = np.arcsin(z / radius)
    azimuth = np.arctan2(y, x)
    radius_rate = np.dot(state[:3], state[3:]) / radius
    azimuth_rate = (x * vy - y * vx) / (x**2 + y**2)
    elevation_rate = (vz - radius_rate * np.sin(elevation)) / (radius * np.cos(elevation))
    return np.array([radius, azimuth, elevation, radius_rate, azimuth_rate, elevation_rate])


def spherical_to_cartesian_position(x_spherical: Iterable[float] | FloatArray) -> FloatArray:
    """Return the Cartesian position from ``[r, phi, theta, ...]`` spherical states."""

    states = np.atleast_2d(np.asarray(x_spherical, dtype=float))
    radius, azimuth, elevation = states[:, 0], states[:, 1], states[:, 2]
    positions = np.column_stack(
        [
            radius * np.cos(elevation) * np.cos(azimuth),
            radius * np.cos(elevation) * np.sin(azimuth),
            radius * np.sin(elevation),
        ]
    )
    return positions[0] if np.ndim(x_spherical) == 1 else positions


def dynamics_2bp_spherical_j2(
    t: float,
    x: Iterable[float] | FloatArray,
    mu: float = EARTH_MU_KM3_S2,
    j2: float = EARTH_J2,
    equatorial_radius: float = EARTH_RADIUS_KM,
) -> FloatArray:
    """J2-perturbed two-body EoMs in spherical coordinates ``[r, phi, theta]``.

    Follows the lecture form, with ``phi`` the azimuth and ``theta`` the
    elevation. The zonal potential is axisymmetric, so the disturbing
    acceleration has no azimuth component.
    """

    radius, _, elevation, radius_rate, azimuth_rate, elevation_rate = _as_float_array(x, 6)
    sin_el, cos_el = np.sin(elevation), np.cos(elevation)

    # disturbing acceleration a_d from the J2 potential, in the spherical basis
    j2_factor = 3.0 * mu * j2 * equatorial_radius**2
    ad_radial = j2_factor / (2.0 * radius**4) * (3.0 * sin_el**2 - 1.0)
    ad_elevation = -j2_factor / radius**4 * sin_el * cos_el

    return np.array(
        [
            radius_rate,
            azimuth_rate,
            elevation_rate,
            radius * elevation_rate**2
            + radius * azimuth_rate**2 * cos_el**2
            - mu / radius**2
            + ad_radial,
            (-2.0 * radius_rate * azimuth_rate * cos_el
             + 2.0 * radius * elevation_rate * azimuth_rate * sin_el) / (radius * cos_el),
            (-2.0 * radius_rate * elevation_rate
             - radius * azimuth_rate**2 * sin_el * cos_el
             + ad_elevation) / radius,
        ]
    )


def propagate_two_body_j2(
    x0: Iterable[float] | FloatArray,
    t_eval: Iterable[float] | FloatArray,
    mu: float,
    *,
    j2: float = EARTH_J2,
    equatorial_radius: float = EARTH_RADIUS_KM,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    method: str = "DOP853",
) -> tuple[FloatArray, FloatArray]:
    """Propagate a J2-perturbed 2BP state history on the requested time grid."""

    x0_array = _as_float_array(x0, size=6)
    t_grid = _as_float_array(t_eval)
    solution = _solve_ivp_checked(
        lambda t, x: dynamics_2bp_cartesian_j2(t, x, mu, j2, equatorial_radius),
        (float(t_grid[0]), float(t_grid[-1])),
        x0_array,
        method=method,
        t_eval=t_grid,
        rtol=rtol,
        atol=atol,
    )
    return solution.t, solution.y.T


def orbital_invariants(states: Iterable[float] | FloatArray, mu: float) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return specific energy, angular momentum vector, and eccentricity vector histories."""

    x = np.atleast_2d(np.asarray(states, dtype=float))
    r_vec, v_vec = x[:, :3], x[:, 3:6]
    r = np.linalg.norm(r_vec, axis=1)
    speed_sq = np.sum(v_vec**2, axis=1)
    h_vec = np.cross(r_vec, v_vec)
    energy = speed_sq / 2.0 - mu / r
    e_vec = (
        (speed_sq - mu / r)[:, None] * r_vec - np.sum(r_vec * v_vec, axis=1)[:, None] * v_vec
    ) / mu
    return energy, h_vec, e_vec


def solve_keplers_equation(
    mean_anomaly: float,
    eccentricity: float,
    *,
    tol: float = 1e-13,
    max_iter: int = 50,
    return_history: bool = False,
) -> float | tuple[float, list[dict[str, float]]]:
    """Solve ``E - e sin(E) = M`` with Newton iterations.

    With ``return_history=True`` the per-iteration records
    ``{"k", "E", "f", "df", "dE"}`` are returned alongside the solution, where
    ``dE = -f(E_k) / f'(E_k)`` is the Newton update applied at iteration ``k``.
    """

    mean_input = float(mean_anomaly)
    revolutions = np.floor((mean_input + np.pi) / (2.0 * np.pi))
    mean = mean_input - revolutions * 2.0 * np.pi
    ecc = float(eccentricity)
    if ecc < 0.0 or ecc >= 1.0:
        raise ValueError("This helper only supports elliptic orbits with 0 <= e < 1.")

    history: list[dict[str, float]] = []
    eccentric = mean if ecc < 0.8 else np.pi
    for k in range(max_iter):
        residual = eccentric - ecc * np.sin(eccentric) - mean
        slope = 1.0 - ecc * np.cos(eccentric)
        step = -residual / slope
        history.append(
            {"k": k, "E": float(eccentric), "f": float(residual), "df": float(slope), "dE": float(step)}
        )
        eccentric += step
        if abs(step) < tol:
            solution = float(eccentric + revolutions * 2.0 * np.pi)
            return (solution, history) if return_history else solution
    raise RuntimeError("Kepler solver failed to converge.")


def true_anomaly_from_eccentric_anomaly(eccentricity: float, eccentric_anomaly: float) -> float:
    """Convert eccentric anomaly to true anomaly for elliptic orbits."""

    ecc = float(eccentricity)
    if ecc < 0.0 or ecc >= 1.0:
        raise ValueError("This helper only supports elliptic orbits with 0 <= e < 1.")
    half = 0.5 * float(eccentric_anomaly)
    numerator = np.sqrt(1.0 + ecc) * np.sin(half)
    denominator = np.sqrt(1.0 - ecc) * np.cos(half)
    return 2.0 * np.arctan2(numerator, denominator)


def _pqw_to_ijk_rotation(raan: float, inclination: float, arg_peri: float) -> FloatArray:
    cos_O, sin_O = np.cos(raan), np.sin(raan)
    cos_i, sin_i = np.cos(inclination), np.sin(inclination)
    cos_w, sin_w = np.cos(arg_peri), np.sin(arg_peri)
    return np.array(
        [
            [cos_O * cos_w - sin_O * sin_w * cos_i, -cos_O * sin_w - sin_O * cos_w * cos_i, sin_O * sin_i],
            [sin_O * cos_w + cos_O * sin_w * cos_i, -sin_O * sin_w + cos_O * cos_w * cos_i, -cos_O * sin_i],
            [sin_w * sin_i, cos_w * sin_i, cos_i],
        ],
        dtype=float,
    )


def coe_to_cartesian(
    x_coe: Iterable[float] | FloatArray,
    mu: float,
    *,
    use_true_anomaly: bool = False,
) -> FloatArray:
    """Convert ``[a, e, i, RAAN, omega, M_or_nu]`` to Cartesian state."""

    elements = np.asarray(x_coe, dtype=float)
    squeeze_output = elements.ndim == 1
    elements_2d = np.atleast_2d(elements)
    if elements_2d.shape[1] != 6:
        raise ValueError("Orbital element input must have six columns.")

    states = np.zeros((elements_2d.shape[0], 6), dtype=float)
    for idx, row in enumerate(elements_2d):
        semi_major_axis, eccentricity, inclination, raan, arg_peri, anomaly = row
        if use_true_anomaly:
            true_anomaly = anomaly
        else:
            eccentric_anomaly = solve_keplers_equation(anomaly, eccentricity)
            true_anomaly = true_anomaly_from_eccentric_anomaly(eccentricity, eccentric_anomaly)

        semilatus_rectum = semi_major_axis * (1.0 - eccentricity**2)
        radius = semilatus_rectum / (1.0 + eccentricity * np.cos(true_anomaly))
        r_pqw = np.array(
            [radius * np.cos(true_anomaly), radius * np.sin(true_anomaly), 0.0],
            dtype=float,
        )
        v_pqw = np.sqrt(mu / semilatus_rectum) * np.array(
            [-np.sin(true_anomaly), eccentricity + np.cos(true_anomaly), 0.0],
            dtype=float,
        )
        rotation = _pqw_to_ijk_rotation(raan, inclination, arg_peri)
        states[idx, :3] = rotation @ r_pqw
        states[idx, 3:] = rotation @ v_pqw

    return states[0] if squeeze_output else states


def cartesian_to_keplerian(x_cart: Iterable[float] | FloatArray, mu: float) -> dict[str, float | FloatArray]:
    """Convert one Cartesian state to classical orbital elements."""

    state = _as_float_array(x_cart, 6)
    r_vec, v_vec = state[:3], state[3:]
    r, speed = np.linalg.norm(r_vec), np.linalg.norm(v_vec)
    if r < 1e-12:
        raise ValueError("Singular state: ||r|| is too small.")
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    if h < 1e-12:
        raise ValueError("Singular state: angular momentum is too small.")
    h_hat = h_vec / h
    e_vec = ((speed**2 - mu / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / mu
    eccentricity = np.linalg.norm(e_vec)
    energy = speed**2 / 2.0 - mu / r
    a = np.inf if abs(energy) < 1e-15 else -mu / (2.0 * energy)
    p = h**2 / mu
    inclination = float(np.arccos(np.clip(h_hat[2], -1.0, 1.0)))
    node = np.cross([0.0, 0.0, 1.0], h_vec)
    node_norm = np.linalg.norm(node)
    raan = float(np.mod(np.arctan2(node[1], node[0]), 2.0 * np.pi)) if node_norm > 1e-14 else 0.0

    if eccentricity > 1e-14:
        if node_norm > 1e-14:
            arg_peri = float(
                np.mod(
                    np.arctan2(np.dot(np.cross(node, e_vec), h_hat), np.dot(node, e_vec)),
                    2.0 * np.pi,
                )
            )
        else:
            arg_peri = float(np.mod(np.arctan2(e_vec[1], e_vec[0]), 2.0 * np.pi))
        true_anomaly = float(
            np.mod(
                np.arctan2(np.dot(np.cross(e_vec, r_vec), h_hat), np.dot(e_vec, r_vec)),
                2.0 * np.pi,
            )
        )
    else:
        arg_peri = 0.0
        true_anomaly = float(
            np.mod(
                np.arctan2(np.dot(np.cross(node, r_vec), h_hat), np.dot(node, r_vec))
                if node_norm > 1e-14
                else np.arctan2(r_vec[1], r_vec[0]),
                2.0 * np.pi,
            )
        )

    if eccentricity < 1.0 and np.isfinite(a):
        eccentric_anomaly = 2.0 * np.arctan2(
            np.sqrt(1.0 - eccentricity) * np.sin(true_anomaly / 2.0),
            np.sqrt(1.0 + eccentricity) * np.cos(true_anomaly / 2.0),
        )
        mean_anomaly = float(np.mod(eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly), 2.0 * np.pi))
        mean_motion = float(np.sqrt(mu / a**3))
    else:
        mean_anomaly = mean_motion = float("nan")

    return {
        "a": float(a),
        "e": float(eccentricity),
        "i": inclination,
        "Omega": raan,
        "omega": arg_peri,
        "f": true_anomaly,
        "M": mean_anomaly,
        "n": mean_motion,
        "p": float(p),
        "h": float(h),
        "r": float(r),
        "v": float(speed),
        "evec": e_vec,
        "hvec": h_vec,
        "hhat": h_hat,
    }


def keplerian_to_equinoctial(a: float, eccentricity: float, inclination: float, raan: float, arg_peri: float, true_anomaly: float) -> FloatArray:
    """Convert classical elements to modified equinoctial elements."""

    longitude_periapsis = arg_peri + raan
    return np.array(
        [
            a * (1.0 - eccentricity**2),
            eccentricity * np.cos(longitude_periapsis),
            eccentricity * np.sin(longitude_periapsis),
            np.tan(inclination / 2.0) * np.cos(raan),
            np.tan(inclination / 2.0) * np.sin(raan),
            longitude_periapsis + true_anomaly,
        ]
    )


def convert_equinoctial_to_eci(x_equinoctial: Iterable[float] | FloatArray) -> tuple[FloatArray, FloatArray]:
    """Convert rows of modified equinoctial elements to ECI positions and COEs."""

    states = np.asarray(x_equinoctial, dtype=float)
    squeeze_output = states.ndim == 1
    states = np.atleast_2d(states)
    if states.shape[1] != 6:
        raise ValueError("Equinoctial states must have six columns.")
    positions = np.empty((len(states), 3))
    coes = np.empty((len(states), 5))
    for index, (p, f, g, h, k, longitude) in enumerate(states):
        q = 1.0 + f * np.cos(longitude) + g * np.sin(longitude)
        raan = np.arctan2(k, h)
        inclination = 2.0 * np.arctan(np.hypot(h, k))
        eccentricity = np.hypot(f, g)
        a = p / (1.0 - eccentricity**2)
        arg_latitude = longitude - raan
        radius = p / q
        positions[index] = orbital_elements_to_dcm(raan, inclination, arg_latitude)[:, 0] * radius
        coes[index] = [a, eccentricity, inclination, raan, arg_latitude]
    return (positions[0], coes[0]) if squeeze_output else (positions, coes)


def _milankovitch_geometry(state: Iterable[float] | FloatArray, mu: float) -> tuple[FloatArray, FloatArray, float, float]:
    values = _as_float_array(state, 7)
    h_vec, e_vec, longitude = values[:3], values[3:6], values[6]
    h, eccentricity = np.linalg.norm(h_vec), np.linalg.norm(e_vec)
    if h < 1e-12:
        raise ValueError("Milankovitch angular momentum is too small.")
    h_hat = h_vec / h
    inclination = np.arccos(np.clip(h_hat[2], -1.0, 1.0))
    node = np.cross([0.0, 0.0, 1.0], h_hat)
    node_norm = np.linalg.norm(node)
    raan = np.arctan2(node[1], node[0]) if node_norm > 1e-14 else 0.0
    if eccentricity > 1e-14:
        e_hat = e_vec / eccentricity
        arg_peri = (
            np.arctan2(np.dot(np.cross(node / node_norm, e_hat), h_hat), np.dot(node / node_norm, e_hat))
            if node_norm > 1e-14
            else np.arctan2(e_hat[1], e_hat[0])
        )
    else:
        arg_peri = 0.0
    true_anomaly = longitude - raan - arg_peri
    p = h**2 / mu
    radius = p / (1.0 + eccentricity * np.cos(true_anomaly))
    dcm = orbital_elements_to_dcm(raan, inclination, arg_peri + true_anomaly)
    r_vec = radius * dcm[:, 0]
    v_rot = np.array([mu * eccentricity * np.sin(true_anomaly) / h, mu * (1.0 + eccentricity * np.cos(true_anomaly)) / h, 0.0])
    return r_vec, dcm @ v_rot, raan, arg_peri


def convert_milankovitch_to_eci(x_milankovitch: Iterable[float] | FloatArray, mu: float) -> tuple[FloatArray, FloatArray]:
    """Convert rows of Milankovitch elements to ECI positions and COEs."""

    states = np.asarray(x_milankovitch, dtype=float)
    squeeze_output = states.ndim == 1
    states = np.atleast_2d(states)
    if states.shape[1] != 7:
        raise ValueError("Milankovitch states must have seven columns.")
    positions = np.empty((len(states), 3))
    coes = np.empty((len(states), 5))
    for index, state in enumerate(states):
        r_vec, _, raan, arg_peri = _milankovitch_geometry(state, mu)
        h, eccentricity = np.linalg.norm(state[:3]), np.linalg.norm(state[3:6])
        positions[index] = r_vec
        coes[index] = [h**2 / mu / (1.0 - eccentricity**2), eccentricity, np.arccos(state[2] / h), raan, arg_peri]
    return (positions[0], coes[0]) if squeeze_output else (positions, coes)


def calc_j2_accel_cartesian(
    r_vec_eci: Iterable[float] | FloatArray,
    mu: float = EARTH_MU_KM3_S2,
    j2: float = EARTH_J2,
    equatorial_radius: float = EARTH_RADIUS_KM,
) -> FloatArray:
    """Return standard J2 perturbation acceleration in ECI coordinates."""

    r_vec = _as_float_array(r_vec_eci, 3)
    radius = np.linalg.norm(r_vec)
    if radius < 1e-12:
        raise ValueError("Singular state: ||r|| is too small.")
    z_ratio = (r_vec[2] / radius) ** 2
    factor = -1.5 * j2 * mu * equatorial_radius**2 / radius**5
    return factor * r_vec * np.array([1.0 - 5.0 * z_ratio, 1.0 - 5.0 * z_ratio, 3.0 - 5.0 * z_ratio])


def calc_srp_accel_cartesian(
    t: float,
    r_vec_eci: Iterable[float] | FloatArray,
    *,
    area_mass_ratio: float = 5.4e-6,
    solar_flux_constant: float = 1.02e14,
    earth_sun_distance: float = 149_597_898.0,
    sun_mu: float = 132_712_440_017.99,
) -> FloatArray:
    """Return the MATLAB library's cannonball solar-radiation acceleration."""

    r_vec = _as_float_array(r_vec_eci, 3)
    theta = np.sqrt(sun_mu / earth_sun_distance**3) * t
    sun_vec = earth_sun_distance * np.array([np.cos(theta), -np.sin(theta), 0.0])
    displacement = r_vec + sun_vec
    distance = np.linalg.norm(displacement)
    if distance < 1e-12:
        raise ValueError("Spacecraft-to-Sun distance is too small.")
    return area_mass_ratio * solar_flux_constant * displacement / distance**3


def calc_third_body_accel_cartesian(
    r_vec_eci: Iterable[float] | FloatArray,
    r_body_eci: Iterable[float] | FloatArray,
    mu_body: float,
) -> FloatArray:
    """Return the third-body gravitational perturbation felt by the spacecraft."""

    r_sc = _as_float_array(r_vec_eci, 3)
    r_body = _as_float_array(r_body_eci, 3)
    relative = r_sc - r_body
    return -mu_body * (
        relative / np.linalg.norm(relative) ** 3 + r_body / np.linalg.norm(r_body) ** 3
    )


def calc_srp_accel_cannonball(
    sun_to_sc_eci: Iterable[float] | FloatArray,
    area_mass_ratio: float,
    reflectivity: float = 1.3,
    solar_flux_constant: float = SOLAR_FLUX_CONSTANT,
) -> FloatArray:
    """Return the cannonball SRP acceleration, pushing away from the Sun.

    ``sun_to_sc_eci`` points from the Sun to the spacecraft in km and
    ``area_mass_ratio`` is in km^2/kg.
    """

    displacement = _as_float_array(sun_to_sc_eci, 3)
    distance = np.linalg.norm(displacement)
    return solar_flux_constant * area_mass_ratio * reflectivity * displacement / distance**3


def atmospheric_density_exponential(
    radius: float,
    density_reference: float = 1.454e-13,
    reference_radius: float = 6978.0,
    scale_height: float = 60.0,
) -> float:
    """Return an exponential-atmosphere density in kg/m^3 at the given radius in km."""

    return density_reference * np.exp(-(radius - reference_radius) / scale_height)


def calc_drag_accel_cartesian(
    v_vec_eci: Iterable[float] | FloatArray,
    density_kg_m3: float,
    drag_coefficient: float,
    area_mass_ratio: float,
) -> FloatArray:
    """Return the cannonball drag acceleration opposing the velocity.

    ``area_mass_ratio`` is in km^2/kg and ``density_kg_m3`` is in kg/m^3.
    """

    v_vec = _as_float_array(v_vec_eci, 3)
    density_kg_km3 = density_kg_m3 * 1e9
    return -0.5 * drag_coefficient * area_mass_ratio * density_kg_km3 * np.linalg.norm(v_vec) * v_vec


def dynamics_2bp_cartesian_j2(
    t: float,
    x: Iterable[float] | FloatArray,
    mu: float = EARTH_MU_KM3_S2,
    j2: float = EARTH_J2,
    equatorial_radius: float = EARTH_RADIUS_KM,
) -> FloatArray:
    state = _as_float_array(x, 6)
    derivative = dynamics_2bp_cartesian(t, state, mu)
    derivative[3:] += calc_j2_accel_cartesian(state[:3], mu, j2, equatorial_radius)
    return derivative


def dynamics_2bp_cartesian_j2_srp(
    t: float,
    x: Iterable[float] | FloatArray,
    mu: float = EARTH_MU_KM3_S2,
    j2: float = EARTH_J2,
    equatorial_radius: float = EARTH_RADIUS_KM,
) -> FloatArray:
    state = _as_float_array(x, 6)
    derivative = dynamics_2bp_cartesian_j2(t, state, mu, j2, equatorial_radius)
    derivative[3:] += calc_srp_accel_cartesian(t, state[:3])
    return derivative


def _equinoctial_dynamics(
    t: float,
    x: Iterable[float] | FloatArray,
    include_srp: bool,
    mu: float,
    j2: float,
    equatorial_radius: float,
) -> tuple[FloatArray, FloatArray]:
    p, f, g, h, k, longitude = _as_float_array(x, 6)
    q = 1.0 + f * np.cos(longitude) + g * np.sin(longitude)
    if p <= 0.0 or abs(q) < 1e-14:
        raise ValueError("Equinoctial state has an invalid p or q value.")
    s_squared = 1.0 + h**2 + k**2
    raan = np.arctan2(k, h)
    inclination = 2.0 * np.arctan(np.hypot(h, k))
    radius = p / q
    dcm = orbital_elements_to_dcm(raan, inclination, longitude - raan)
    r_vec = radius * dcm[:, 0]
    acceleration = calc_j2_accel_cartesian(r_vec, mu, j2, equatorial_radius)
    if include_srp:
        acceleration += calc_srp_accel_cartesian(t, r_vec)
    acceleration_rtn = dcm.T @ acceleration
    hk = h * np.sin(longitude) - k * np.cos(longitude)
    b_matrix = np.sqrt(p / mu) * np.array(
        [
            [0.0, 2.0 * p / q, 0.0],
            [np.sin(longitude), ((q + 1.0) * np.cos(longitude) + f) / q, -g * hk / q],
            [-np.cos(longitude), ((q + 1.0) * np.sin(longitude) + g) / q, f * hk / q],
            [0.0, 0.0, s_squared * np.cos(longitude) / (2.0 * q)],
            [0.0, 0.0, s_squared * np.sin(longitude) / (2.0 * q)],
            [0.0, 0.0, hk / q],
        ]
    )
    unperturbed = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(mu * p) * (q / p) ** 2])
    return unperturbed + b_matrix @ acceleration_rtn, r_vec


def dynamics_2bp_equinoctial_j2(
    t: float,
    x: Iterable[float] | FloatArray,
    mu: float = EARTH_MU_KM3_S2,
    j2: float = EARTH_J2,
    equatorial_radius: float = EARTH_RADIUS_KM,
    *,
    return_position: bool = False,
) -> FloatArray | tuple[FloatArray, FloatArray]:
    result = _equinoctial_dynamics(t, x, False, mu, j2, equatorial_radius)
    return result if return_position else result[0]


def dynamics_2bp_equinoctial_j2_srp(
    t: float,
    x: Iterable[float] | FloatArray,
    mu: float = EARTH_MU_KM3_S2,
    j2: float = EARTH_J2,
    equatorial_radius: float = EARTH_RADIUS_KM,
    *,
    return_position: bool = False,
) -> FloatArray | tuple[FloatArray, FloatArray]:
    result = _equinoctial_dynamics(t, x, True, mu, j2, equatorial_radius)
    return result if return_position else result[0]


def _keplerian_dynamics(
    t: float,
    x: Iterable[float] | FloatArray,
    include_srp: bool,
    mu: float,
    j2: float,
    equatorial_radius: float,
) -> tuple[FloatArray, FloatArray]:
    a, eccentricity, inclination, raan, arg_peri, mean_anomaly = _as_float_array(x, 6)
    if eccentricity <= 1e-14 or abs(np.sin(inclination)) <= 1e-14:
        raise ValueError("Classical-element perturbation equations require nonzero eccentricity and inclination.")
    eccentric_anomaly = solve_keplers_equation(mean_anomaly, eccentricity)
    true_anomaly = true_anomaly_from_eccentric_anomaly(eccentricity, eccentric_anomaly)
    p = p_ae(a, eccentricity)
    radius = r_peta(p, eccentricity, true_anomaly)
    h = np.sqrt(mu * p)
    b = b_ae(a, eccentricity)
    dcm = orbital_elements_to_dcm(raan, inclination, arg_peri + true_anomaly)
    r_vec = radius * dcm[:, 0]
    acceleration = calc_j2_accel_cartesian(r_vec, mu, j2, equatorial_radius)
    if include_srp:
        acceleration += calc_srp_accel_cartesian(t, r_vec)
    acceleration_rtn = dcm.T @ acceleration
    sin_nu, cos_nu = np.sin(true_anomaly), np.cos(true_anomaly)
    b_matrix = np.array(
        [
            [2.0 * a**2 * eccentricity * sin_nu, 2.0 * a**2 * p / radius, 0.0],
            [p * sin_nu, (p + radius) * cos_nu + radius * eccentricity, 0.0],
            [0.0, 0.0, radius * np.cos(true_anomaly + arg_peri)],
            [0.0, 0.0, radius * np.sin(true_anomaly + arg_peri) / np.sin(inclination)],
            [-p * cos_nu / eccentricity, (p + radius) * sin_nu / eccentricity, -radius * np.sin(true_anomaly + arg_peri) / np.tan(inclination)],
            [b * p * cos_nu / (a * eccentricity) - 2.0 * b * radius / a, -b * (p + radius) * sin_nu / (a * eccentricity), 0.0],
        ]
    ) / h
    unperturbed = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(mu / a**3)])
    return unperturbed + b_matrix @ acceleration_rtn, r_vec


def dynamics_2bp_keplerian_j2(
    t: float,
    x: Iterable[float] | FloatArray,
    mu: float = EARTH_MU_KM3_S2,
    j2: float = EARTH_J2,
    equatorial_radius: float = EARTH_RADIUS_KM,
    *,
    return_position: bool = False,
) -> FloatArray | tuple[FloatArray, FloatArray]:
    result = _keplerian_dynamics(t, x, False, mu, j2, equatorial_radius)
    return result if return_position else result[0]


def dynamics_2bp_keplerian_j2_srp(
    t: float,
    x: Iterable[float] | FloatArray,
    mu: float = EARTH_MU_KM3_S2,
    j2: float = EARTH_J2,
    equatorial_radius: float = EARTH_RADIUS_KM,
    *,
    return_position: bool = False,
) -> FloatArray | tuple[FloatArray, FloatArray]:
    result = _keplerian_dynamics(t, x, True, mu, j2, equatorial_radius)
    return result if return_position else result[0]


def _skew(vector: FloatArray) -> FloatArray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _milankovitch_dynamics(
    t: float,
    x: Iterable[float] | FloatArray,
    include_srp: bool,
    mu: float,
    j2: float,
    equatorial_radius: float,
) -> tuple[FloatArray, FloatArray]:
    state = _as_float_array(x, 7)
    h_vec = state[:3]
    r_vec, v_vec, _, _ = _milankovitch_geometry(state, mu)
    h, radius = np.linalg.norm(h_vec), np.linalg.norm(r_vec)
    denominator = h * (h + h_vec[2])
    if abs(denominator) < 1e-14:
        raise ValueError("Milankovitch longitude is singular for this angular momentum vector.")
    acceleration = calc_j2_accel_cartesian(r_vec, mu, j2, equatorial_radius)
    if include_srp:
        acceleration += calc_srp_accel_cartesian(t, r_vec)
    b_matrix = np.vstack(
        (
            _skew(r_vec),
            (_skew(v_vec) @ _skew(r_vec) - _skew(h_vec)) / mu,
            r_vec[2] * h_vec / denominator,
        )
    )
    unperturbed = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, h / radius**2])
    return unperturbed + b_matrix @ acceleration, r_vec


def dynamics_2bp_milankovitch_j2(
    t: float,
    x: Iterable[float] | FloatArray,
    mu: float = EARTH_MU_KM3_S2,
    j2: float = EARTH_J2,
    equatorial_radius: float = EARTH_RADIUS_KM,
    *,
    return_position: bool = False,
) -> FloatArray | tuple[FloatArray, FloatArray]:
    result = _milankovitch_dynamics(t, x, False, mu, j2, equatorial_radius)
    return result if return_position else result[0]


def dynamics_2bp_milankovitch_j2_srp(
    t: float,
    x: Iterable[float] | FloatArray,
    mu: float = EARTH_MU_KM3_S2,
    j2: float = EARTH_J2,
    equatorial_radius: float = EARTH_RADIUS_KM,
    *,
    return_position: bool = False,
) -> FloatArray | tuple[FloatArray, FloatArray]:
    result = _milankovitch_dynamics(t, x, True, mu, j2, equatorial_radius)
    return result if return_position else result[0]


def cr3bp(x: Iterable[float] | FloatArray, mu: float, n: float = 1.0) -> FloatArray:
    """Circular restricted three-body dynamics in the synodic frame."""

    state = _as_float_array(x, 6)
    px, py, pz, vx, vy, vz = state
    primary_distance = np.sqrt((px + mu) ** 2 + py**2 + pz**2)
    secondary_distance = np.sqrt((px - 1.0 + mu) ** 2 + py**2 + pz**2)
    if min(primary_distance, secondary_distance) < 1e-12:
        raise ValueError("CR3BP state lies on a primary body.")
    acceleration = np.array(
        [
            2.0 * n * vy + n**2 * px - (1.0 - mu) * (px + mu) / primary_distance**3 - mu * (px - 1.0 + mu) / secondary_distance**3,
            -2.0 * n * vx + n**2 * py - (1.0 - mu) * py / primary_distance**3 - mu * py / secondary_distance**3,
            -(1.0 - mu) * pz / primary_distance**3 - mu * pz / secondary_distance**3,
        ]
    )
    return np.concatenate(([vx, vy, vz], acceleration))


def _state_matrix(states: Iterable[float] | FloatArray) -> tuple[FloatArray, bool]:
    array = np.asarray(states, dtype=float)
    squeeze_output = array.ndim == 1
    array = np.atleast_2d(array)
    if array.shape[1] != 6:
        raise ValueError("States must have six columns.")
    return array, squeeze_output


def _broadcast_ephemeris(values: Iterable[float] | FloatArray, rows: int, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.shape == (3,):
        return np.broadcast_to(array, (rows, 3))
    if array.shape != (rows, 3):
        raise ValueError(f"{name} must have shape (3,) or ({rows}, 3).")
    return array


def eci2hci_known_earth_ics(
    states_eci: Iterable[float] | FloatArray,
    earth_position_hci: Iterable[float] | FloatArray,
    earth_velocity_hci: Iterable[float] | FloatArray,
) -> FloatArray:
    states, squeeze_output = _state_matrix(states_eci)
    earth_position = _broadcast_ephemeris(earth_position_hci, len(states), "earth_position_hci")
    earth_velocity = _broadcast_ephemeris(earth_velocity_hci, len(states), "earth_velocity_hci")
    result = states + np.hstack((earth_position, earth_velocity))
    return result[0] if squeeze_output else result


def hci2eci_known_earth_ics(
    states_hci: Iterable[float] | FloatArray,
    earth_position_hci: Iterable[float] | FloatArray,
    earth_velocity_hci: Iterable[float] | FloatArray,
) -> FloatArray:
    states, squeeze_output = _state_matrix(states_hci)
    earth_position = _broadcast_ephemeris(earth_position_hci, len(states), "earth_position_hci")
    earth_velocity = _broadcast_ephemeris(earth_velocity_hci, len(states), "earth_velocity_hci")
    result = states - np.hstack((earth_position, earth_velocity))
    return result[0] if squeeze_output else result


def _circular_earth_ephemeris(times: Iterable[float] | FloatArray) -> tuple[FloatArray, FloatArray]:
    time = _as_float_array(times)
    earth = CelestialBody("Earth")
    mean_motion = 2.0 * np.pi / earth.orbit.period_sec
    theta = mean_motion * time
    position = earth.orbit.a_km * np.column_stack((np.cos(theta), np.sin(theta), np.zeros_like(theta)))
    velocity = mean_motion * earth.orbit.a_km * np.column_stack((-np.sin(theta), np.cos(theta), np.zeros_like(theta)))
    return position, velocity


def eci2hci(states_eci: Iterable[float] | FloatArray, t: Iterable[float] | FloatArray | float) -> FloatArray:
    states, squeeze_output = _state_matrix(states_eci)
    times = np.atleast_1d(np.asarray(t, dtype=float))
    if len(times) != len(states):
        raise ValueError("Time vector length must match the number of states.")
    earth_position, earth_velocity = _circular_earth_ephemeris(times)
    result = eci2hci_known_earth_ics(states, earth_position, earth_velocity)
    return result[0] if squeeze_output else result


def hci2eci(states_hci: Iterable[float] | FloatArray, t: Iterable[float] | FloatArray | float) -> FloatArray:
    states, squeeze_output = _state_matrix(states_hci)
    times = np.atleast_1d(np.asarray(t, dtype=float))
    if len(times) != len(states):
        raise ValueError("Time vector length must match the number of states.")
    earth_position, earth_velocity = _circular_earth_ephemeris(times)
    result = hci2eci_known_earth_ics(states, earth_position, earth_velocity)
    return result[0] if squeeze_output else result


def dcm_from_euler_angle_seq(seq: Iterable[int], angles: Iterable[float], convention: str = "row") -> FloatArray:
    """Compose three active axis rotations using the MATLAB library convention."""

    axes = list(seq)
    rotations = list(angles)
    if len(axes) != 3 or len(rotations) != 3:
        raise ValueError("seq and angles must each contain three values.")
    dcm = np.eye(3)
    for axis, angle in zip(axes, rotations):
        c, s = np.cos(angle), np.sin(angle)
        if axis == 1:
            rotation = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        elif axis == 2:
            rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        elif axis == 3:
            rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        else:
            raise ValueError("Each axis must be 1, 2, or 3.")
        dcm = dcm @ rotation
    if convention.lower() == "col":
        return dcm.T
    if convention.lower() != "row":
        raise ValueError("convention must be 'row' or 'col'.")
    return dcm


def sine_law_angle(angle1: float, side1: float, side2: float) -> FloatArray:
    angle = np.arcsin(np.clip(side2 * np.sin(angle1) / side1, -1.0, 1.0))
    return np.array([angle, np.pi - angle])


def cos_law_side(included_angle: float, side1: float, side2: float) -> float:
    return float(np.sqrt(side1**2 + side2**2 - 2.0 * side1 * side2 * np.cos(included_angle)))


def cos_law_angle_s1s2(side1: float, side2: float, side3: float) -> float:
    return float(np.arccos(np.clip((side1**2 + side2**2 - side3**2) / (2.0 * side1 * side2), -1.0, 1.0)))


def sec_to_hours(seconds):
    return np.asarray(seconds) / 3600.0


def sec_to_days(seconds):
    return np.asarray(seconds) / 86400.0


def sec_to_weeks(seconds):
    return np.asarray(seconds) / (86400.0 * 7.0)


def sec_to_months(seconds):
    return np.asarray(seconds) / (86400.0 * 30.0)


def sec_to_years(seconds):
    return np.asarray(seconds) / (86400.0 * 365.0)


def au_to_km(distance):
    return np.asarray(distance) * AU_KM


def km_to_au(distance):
    return np.asarray(distance) / AU_KM


# MATLAB-compatible names retained for direct migration of existing scripts.
OrbitalElementsToDCM = orbital_elements_to_dcm
theta_3D_rth_hat = theta_3d_rth_hat
RAAN_find_use_h_hat = raan_find_use_h_hat
r_vec_ep_aEeb = r_vec_ep_a_e_e_b
r_Peta = r_peta
v_vec_ep_rEban = v_vec_ep_r_eban
E_Me2 = solve_keplers_equation
E_Me = solve_keplers_equation
M_ttp_ua = mean_anomaly_from_time
M_eE = mean_anomaly_from_eccentric_anomaly
ta_eE = true_anomaly_from_eccentric_anomaly
EccA_are = eccentric_anomaly_from_radius
EccA_areta = eccentric_anomaly_from_radius_and_true_anomaly
txtp_EccAn_eua = txtp_eccan_eua
trngl_dist_FQ_fmin = trngl_dist_fq_fmin
trngl_alpha_Hyp = trngl_alpha_hyp
trngl_beta_Hyp = trngl_beta_hyp
trngl_P_ellipse = trngl_p_ellipse
trngl_P_hyperb = trngl_p_hyperb
trngl_TOF_parab = trngl_tof_parab
trngl_TOF = trngl_tof
dynamics_2BP_cartesian = dynamics_2bp_cartesian
jacobian_2BP_cartesian = jacobian_2bp_cartesian
propagateSTM_2BP = propagate_stm_2bp
augmentedDynamics2BP_STM = augmented_dynamics_2bp_stm
dynamics_2BP_cartesian_J2 = dynamics_2bp_cartesian_j2
dynamics_2BP_cartesian_J2_SRP = dynamics_2bp_cartesian_j2_srp
dynamics_2BP_milankovitch_J2 = dynamics_2bp_milankovitch_j2
dynamics_2BP_milankovitch_J2_SRP = dynamics_2bp_milankovitch_j2_srp
dynamics_2BP_equinoctial_J2 = dynamics_2bp_equinoctial_j2
dynamics_2BP_equinoctial_J2_SRP = dynamics_2bp_equinoctial_j2_srp
dynamics_2BP_keplerian_J2 = dynamics_2bp_keplerian_j2
dynamics_2BP_keplerian_J2_SRP = dynamics_2bp_keplerian_j2_srp
calc_J2_accel_cartesian = calc_j2_accel_cartesian
calc_SRP_accel_cartesian = calc_srp_accel_cartesian
CR3BP = cr3bp
cart2kep = cartesian_to_keplerian
convertEquinoctialToECI = convert_equinoctial_to_eci
convertMilankovitchToECI = convert_milankovitch_to_eci
keplerianToEquinoctial = keplerian_to_equinoctial
eci2hci_knownEarthICs = eci2hci_known_earth_ics
hci2eci_knownEarthICs = hci2eci_known_earth_ics
dcmFromEulerAngleSeq = dcm_from_euler_angle_seq
secToHours = sec_to_hours
secToDays = sec_to_days
secToWeeks = sec_to_weeks
secToMonths = sec_to_months
secToYears = sec_to_years
