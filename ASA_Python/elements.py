"""astexpy.orbits.elements

Keplerian <-> Cartesian state conversions.

Goal: keep these helpers *generic* and compatible with both:
  - NumPy float arrays
  - DACEyPy DA scalars / daceypy.array vectors

This module is NOT a dynamics model; it only changes state representation.
"""

from __future__ import annotations

from typing import Any, Tuple, overload

import numpy as np
from numpy.typing import NDArray

# DACEyPy is optional here: this module is usable for float-only projects.
try:
    import daceypy as dp
    from daceypy import op as dop  # sin/cos/sqrt/etc. that work on float + DA
except Exception:  # pragma: no cover
    dp = None  # type: ignore
    dop = None  # type: ignore

FloatVec3 = NDArray[np.float64]
FloatState6 = NDArray[np.float64]

# Keep typing lightweight: Pylance has a hard time with DACEyPy's dynamic types.
Scalar = Any
Vec3 = Any
State6 = Any

# ------------------------------------------------------------
# Math + array helpers
#
# If DACEyPy is installed, its `op` functions work on floats, NumPy arrays,
# DA scalars, and DACEyPy arrays. If it's not installed, we fall back to NumPy.
# ------------------------------------------------------------

if dop is not None:
    sin = dop.sin
    cos = dop.cos
    sqrt = dop.sqrt
else:
    sin = np.sin
    cos = np.cos
    sqrt = np.sqrt


def _uses_da(x: Any) -> bool:
    """Heuristic: does `x` contain DACEyPy DA types?

    We must NOT construct `dp.array(...)` unless DA is already initialized.
    So we only use DACEyPy containers when the inputs are already DA-like.
    """
    if dp is None:
        return False

    # Direct DA types
    if isinstance(x, (dp.DA, dp.array)):
        return True

    # Common container cases
    if isinstance(x, (list, tuple)):
        return len(x) > 0 and _uses_da(x[0])

    if isinstance(x, np.ndarray):
        if x.size == 0:
            return False
        # If it's an object array, peek at a couple entries
        if x.dtype == object:
            flat = x.ravel()
            return any(isinstance(flat[idx], dp.DA) for idx in (0, min(1, flat.size - 1)))
        return False

    return False


def op_array(x: Any): 
    """Return a vector/matrix as NumPy (float) OR DACEyPy array.

    - For pure-float inputs: return `np.asarray(..., dtype=float)` (no DA init needed).
    - For DA-like inputs: return `dp.array(...)` (assumes DA was initialized by the caller).
    """
    if _uses_da(x):
        # Caller is intentionally working in DA space and should have done DA.init(...)
        return dp.array(x)  # type: ignore[call-arg]
    return np.asarray(x, dtype=float)


def _as_vec3(v: Any):
    """Return a 3-vector as a DACEyPy array (if available) or NumPy array."""
    return op_array(v)


def _as_mat3(M: Any):
    """Return a 3x3 matrix as a DACEyPy array (if available) or NumPy array."""
    return op_array(M)


# Rotation about 3rd-axis
def _rot3(angle: Any):
    c = cos(angle)
    s = sin(angle)
    R = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    return _as_mat3(R)


# Rotation about 1st-axis
def _rot1(angle: Any):
    c = cos(angle)
    s = sin(angle)
    R = [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
    return _as_mat3(R)


@overload
def kep2cart(a: float, e: float, inc: float, raan: float, argp: float, nu: float, *, mu: float) -> Tuple[FloatVec3, FloatVec3]: ...


@overload
def kep2cart(a: Scalar, e: Scalar, inc: Scalar, raan: Scalar, argp: Scalar, nu: Scalar, *, mu: float) -> Tuple[Vec3, Vec3]: ...


def kep2cart(a: Scalar, e: Scalar, inc: Scalar, raan: Scalar, argp: Scalar, nu: Scalar, *, mu: float) -> Tuple[Vec3, Vec3]:
    """Convert Keplerian elements -> Cartesian position/velocity.

    Inputs are *osculating* Keplerian elements (a,e,i,Ω,ω,ν).

    Args:
        a: semi-major axis [length]
        e: eccentricity [-]
        inc: inclination i [rad]
        raan: right ascension of ascending node Ω [rad]
        argp: argument of periapsis ω [rad]
        nu: true anomaly ν [rad]
        mu: gravitational parameter GM [length^3 / time^2]

    Returns:
        (r_eci, v_eci): position and velocity in an inertial Cartesian frame.

    Notes:
        - Works with floats and with DA scalars (DACEyPy).
        - For DA inputs, `mu` must be a float constant (typical use).
    """
    # Semi-latus rectum p = a (1 - e^2)
    p = a * (1.0 - e * e)

    # Radius magnitude in perifocal frame: r = p / (1 + e cos(nu))
    cnu = cos(nu)
    snu = sin(nu)
    rmag = p / (1.0 + e * cnu)

    # Perifocal position/velocity
    r_pf = _as_vec3([rmag * cnu, rmag * snu, 0.0])
    vpref = sqrt(mu / p)
    v_pf = _as_vec3([-vpref * snu, vpref * (e + cnu), 0.0])

    # Rotation: perifocal -> inertial (ECI-like)
    # r_eci = R3(raan) * R1(inc) * R3(argp) * r_pf
    Q = _rot3(raan) @ _rot1(inc) @ _rot3(argp)
    r_eci = Q @ r_pf
    v_eci = Q @ v_pf

    return r_eci, v_eci


@overload
def kep2state(a: float, e: float, inc: float, raan: float, argp: float, nu: float, *, mu: float) -> FloatState6: ...


@overload
def kep2state(a: Scalar, e: Scalar, inc: Scalar, raan: Scalar, argp: Scalar, nu: Scalar, *, mu: float) -> Vec3: ...


def kep2state(a: Scalar, e: Scalar, inc: Scalar, raan: Scalar, argp: Scalar, nu: Scalar, *, mu: float):
    """Keplerian elements -> 6D Cartesian state [x,y,z,xdot,ydot,zdot]."""
    r, v = kep2cart(a, e, inc, raan, argp, nu, mu=mu)
    if dp is not None and isinstance(r, dp.array):
        return dp.array([r[0], r[1], r[2], v[0], v[1], v[2]])
    return np.array([r[0], r[1], r[2], v[0], v[1], v[2]], dtype=float)


def cart2kep(r: FloatVec3, v: FloatVec3, *, mu: float) -> Tuple[float, float, float, float, float, float]:
    """Cartesian -> Keplerian elements (float-only).

    This is intentionally float-only because inverse trig / branching is awkward for DA.
    Returns (a,e,i,raan,argp,nu) in SI-like consistent units.
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)

    rmag = float(np.linalg.norm(r))
    vmag = float(np.linalg.norm(v))

    h = np.cross(r, v)
    hmag = float(np.linalg.norm(h))

    k = np.array([0.0, 0.0, 1.0])
    n = np.cross(k, h)
    nmag = float(np.linalg.norm(n))

    e_vec = (np.cross(v, h) / mu) - (r / rmag)
    e = float(np.linalg.norm(e_vec))

    energy = 0.5 * vmag * vmag - mu / rmag
    a = -mu / (2.0 * energy) if abs(energy) > 0.0 else np.inf

    inc = float(np.arccos(np.clip(h[2] / hmag, -1.0, 1.0)))

    raan = 0.0
    if nmag > 0.0:
        raan = float(np.arctan2(n[1], n[0]))

    argp = 0.0
    if nmag > 0.0 and e > 0.0:
        argp = float(np.arctan2(np.dot(np.cross(n, e_vec), h) / (nmag * hmag), np.dot(n, e_vec) / (nmag * e)))

    nu = 0.0
    if e > 0.0:
        nu = float(np.arctan2(np.dot(np.cross(e_vec, r), h) / (e * hmag), np.dot(e_vec, r) / (e * rmag)))

    # Wrap angles to [0, 2pi)
    twopi = 2.0 * np.pi
    raan = raan % twopi
    argp = argp % twopi
    nu = nu % twopi

    return a, e, inc, raan, argp, nu