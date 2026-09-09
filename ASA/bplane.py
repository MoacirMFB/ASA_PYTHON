"""B-plane geometry for planetary close encounters, in Opik-Valsecchi coordinates.

The b-plane of an encounter is the plane through the planet's centre,
perpendicular to ``U``, the planetocentric velocity the small body would have if
the planet had no gravity. The impact parameter ``b`` is the vector from the
planet to the point where that undeflected straight line pierces the plane. It
is where the encounter geometry lives with gravitational focusing removed, which
is what makes it the natural currency for deflection: a velocity change years
earlier maps almost linearly onto a displacement here, while it does not map
linearly onto the miss distance itself.

Two b-plane conventions are in common use and they are NOT interchangeable:

* ``B.R`` / ``B.T``, used for spacecraft navigation, whose axes come from an
  external reference plane (usually the ecliptic). Most textbooks teach this one.
* ``xi`` / ``zeta`` (Opik-Valsecchi), used for impact monitoring and planetary
  defence, whose axes come from the encounter geometry itself. ``xi`` is the
  signed local MOID, so it measures how far the two orbits miss each other
  geometrically; ``zeta`` is a timing coordinate, measuring whether the body
  arrives early or late. Deflection almost always acts through ``zeta``.

This module implements the second. The axes follow Valsecchi et al. (2003),
A&A 408, 1179, via the reference frame in which ``X`` points radially outward
from the Sun to the planet, ``Y`` along the planet's heliocentric velocity and
``Z`` along the planet's orbit normal:

    xi_hat   = ( cos(phi),               0,          -sin(phi)             )
    eta_hat  = ( sin(phi) sin(theta),    cos(theta),  cos(phi) sin(theta)  )
    zeta_hat = ( sin(phi) cos(theta),   -sin(theta),  cos(phi) cos(theta)  )

with ``theta`` the angle between ``U`` and the planet's velocity and ``phi`` the
orientation of ``U`` about it. Rather than form ``theta`` and ``phi`` - which
would drag in Opik's assumption of a circular planetary orbit - the frame is
built directly from vectors. ``eta_hat`` is ``U`` itself, and expanding the
projection of the planet's velocity direction onto the b-plane shows it equals
``-sin(theta) zeta_hat``, so the two constructions are algebraically identical:

    eta_hat  = U / |U|
    zeta_hat = -normalize(v_planet_hat - (v_planet_hat . eta_hat) eta_hat)
    xi_hat   = eta_hat x zeta_hat

Where to evaluate
-----------------
The coordinates describe a two-body hyperbolic passage about the planet, so
they only mean anything for a state inside the planet's sphere of influence.
Sampled further out, the "hyperbola" fitted to a geocentric state is dominated
by the Sun and the resulting b is meaningless - for the 2024 PDC25 encounter it
comes out at 1.7 million km thirty days ahead of an impact. Inside the sphere
of influence the values settle quickly: that same encounter is stable to about
0.1% once within roughly 200,000 km. Use :func:`sphere_of_influence_radius` to
check, and prefer a consistent evaluation point when comparing encounters.

Distances are in kilometres, velocities in km/s, and ``mu`` in km^3/s^2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# Below this, U is close enough to parallel with the planet's velocity that the
# projection defining zeta_hat is numerically meaningless. sin(theta) = 1e-8
# leaves the frame axes accurate to ~1e-8 relative, already far looser than any
# encounter geometry of physical interest.
MINIMUM_SIN_THETA = 1e-8


@dataclass(frozen=True)
class HyperbolicEncounter:
    """Incoming-asymptote geometry of one planetocentric hyperbolic passage."""

    v_infinity_kmps: float
    """Speed on the incoming asymptote, sqrt(2 * specific energy)."""

    s_hat: FloatArray
    """Unit vector along the incoming asymptote, in the direction of motion."""

    b_vector_km: FloatArray
    """Impact parameter vector: planet centre to the asymptote, perpendicular to s_hat."""

    b_km: float
    """Impact parameter magnitude, ``|b_vector_km|``."""

    periapsis_radius_km: float
    """Osculating periapsis radius of the passage."""

    eccentricity: float
    """Osculating eccentricity, always > 1."""


@dataclass(frozen=True)
class BPlaneCoordinates:
    """Opik-Valsecchi b-plane coordinates of one encounter."""

    xi_km: float
    """Signed local MOID: how far the orbits miss each other geometrically."""

    zeta_km: float
    """Timing coordinate: whether the body arrives early or late."""

    b_km: float
    """Impact parameter magnitude, ``sqrt(xi^2 + zeta^2)``."""

    v_infinity_kmps: float
    """Speed on the incoming asymptote."""

    periapsis_radius_km: float
    """Osculating periapsis radius of the passage."""


def _as_vector(values: Iterable[float] | FloatArray, name: str) -> FloatArray:
    """Return a finite 3-vector, or raise."""

    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size != 3:
        raise ValueError(f"{name} must have 3 components, got {vector.size}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values.")
    return vector


def _unit(vector: FloatArray, name: str) -> FloatArray:
    """Normalize a vector that must not be degenerate."""

    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{name} must have non-zero magnitude.")
    return vector / norm


def bplane_frame(
    u_vec_kmps: Iterable[float] | FloatArray,
    planet_velocity_kmps: Iterable[float] | FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return the ``(xi_hat, eta_hat, zeta_hat)`` unit vectors of the b-plane frame.

    ``u_vec_kmps`` is the planetocentric velocity on the incoming asymptote and
    ``planet_velocity_kmps`` the planet's heliocentric velocity. Only their
    directions matter. The triad is right-handed with ``xi_hat x eta_hat = zeta_hat``.
    """

    eta_hat = _unit(_as_vector(u_vec_kmps, "u_vec_kmps"), "u_vec_kmps")
    planet_hat = _unit(
        _as_vector(planet_velocity_kmps, "planet_velocity_kmps"), "planet_velocity_kmps"
    )

    # The planet's velocity direction, with its component along U removed, is
    # -sin(theta) * zeta_hat. Its length is therefore sin(theta) itself.
    in_plane = planet_hat - float(np.dot(planet_hat, eta_hat)) * eta_hat
    sin_theta = float(np.linalg.norm(in_plane))
    if sin_theta < MINIMUM_SIN_THETA:
        raise ValueError(
            "The b-plane frame is undefined when U is parallel to the planet's velocity "
            f"(sin(theta) = {sin_theta:.3e})."
        )

    zeta_hat = -in_plane / sin_theta
    xi_hat = np.cross(eta_hat, zeta_hat)
    return xi_hat, eta_hat, zeta_hat


def hyperbolic_encounter(
    position_km: Iterable[float] | FloatArray,
    velocity_kmps: Iterable[float] | FloatArray,
    mu_km3_s2: float,
) -> HyperbolicEncounter:
    """Reduce a planetocentric state to its incoming-asymptote geometry.

    ``position_km`` and ``velocity_kmps`` are relative to the planet's centre.
    The state must be hyperbolic; a bound state has no asymptote and raises.

    The asymptote direction follows from the true anomaly at infinity,
    ``cos(nu_inf) = -1/e``, which gives ``s_hat = (e_hat + sqrt(e^2 - 1) q_hat) / e``
    with ``q_hat = h_hat x e_hat``. The impact parameter is then
    ``b_vec = (h / v_inf) (s_hat x h_hat)``.
    """

    position_km = _as_vector(position_km, "position_km")
    velocity_kmps = _as_vector(velocity_kmps, "velocity_kmps")
    mu_km3_s2 = float(mu_km3_s2)
    if not np.isfinite(mu_km3_s2) or mu_km3_s2 <= 0.0:
        raise ValueError("mu_km3_s2 must be finite and positive.")

    radius_km = float(np.linalg.norm(position_km))
    if radius_km <= 0.0:
        raise ValueError("position_km must have non-zero magnitude.")
    speed_kmps = float(np.linalg.norm(velocity_kmps))

    specific_energy = 0.5 * speed_kmps**2 - mu_km3_s2 / radius_km
    if specific_energy <= 0.0:
        raise ValueError(
            "The encounter must be hyperbolic to have an incoming asymptote "
            f"(specific energy {specific_energy:.6e} km^2/s^2 <= 0)."
        )
    v_infinity_kmps = float(np.sqrt(2.0 * specific_energy))

    angular_momentum = np.cross(position_km, velocity_kmps)
    angular_momentum_norm = float(np.linalg.norm(angular_momentum))
    if angular_momentum_norm <= 0.0:
        raise ValueError("A radial (zero angular momentum) encounter has no b-plane.")
    h_hat = angular_momentum / angular_momentum_norm

    eccentricity_vector = (
        (speed_kmps**2 - mu_km3_s2 / radius_km) * position_km
        - float(np.dot(position_km, velocity_kmps)) * velocity_kmps
    ) / mu_km3_s2
    eccentricity = float(np.linalg.norm(eccentricity_vector))
    if eccentricity <= 1.0:
        raise ValueError(f"The encounter must be hyperbolic, got eccentricity {eccentricity:.6f}.")
    e_hat = eccentricity_vector / eccentricity
    q_hat = np.cross(h_hat, e_hat)

    s_hat = (e_hat + np.sqrt(eccentricity**2 - 1.0) * q_hat) / eccentricity
    b_km = angular_momentum_norm / v_infinity_kmps
    b_vector_km = b_km * np.cross(s_hat, h_hat)
    periapsis_radius_km = angular_momentum_norm**2 / (mu_km3_s2 * (1.0 + eccentricity))

    return HyperbolicEncounter(
        v_infinity_kmps=v_infinity_kmps,
        s_hat=s_hat,
        b_vector_km=b_vector_km,
        b_km=b_km,
        periapsis_radius_km=periapsis_radius_km,
        eccentricity=eccentricity,
    )


def bplane_coordinates(
    position_km: Iterable[float] | FloatArray,
    velocity_kmps: Iterable[float] | FloatArray,
    planet_velocity_kmps: Iterable[float] | FloatArray,
    mu_km3_s2: float,
) -> BPlaneCoordinates:
    """Return the Opik-Valsecchi ``(xi, zeta)`` of one planetocentric encounter.

    ``position_km`` and ``velocity_kmps`` are the small body's state relative to
    the planet, at any epoch on the hyperbolic passage; ``planet_velocity_kmps``
    is the planet's heliocentric velocity, which sets the frame's orientation.
    """

    encounter = hyperbolic_encounter(position_km, velocity_kmps, mu_km3_s2)
    xi_hat, _, zeta_hat = bplane_frame(
        encounter.v_infinity_kmps * encounter.s_hat, planet_velocity_kmps
    )
    return BPlaneCoordinates(
        xi_km=float(np.dot(encounter.b_vector_km, xi_hat)),
        zeta_km=float(np.dot(encounter.b_vector_km, zeta_hat)),
        b_km=encounter.b_km,
        v_infinity_kmps=encounter.v_infinity_kmps,
        periapsis_radius_km=encounter.periapsis_radius_km,
    )


def impact_parameter_from_periapsis(
    periapsis_radius_km: float, v_infinity_kmps: float, mu_km3_s2: float
) -> float:
    """Impact parameter that produces a given periapsis radius.

    ``b = r_p sqrt(1 + 2 mu / (r_p v_inf^2))``. The factor above unity is
    gravitational focusing: the planet pulls in trajectories that would
    otherwise have missed, so ``b`` always exceeds the periapsis radius.
    """

    periapsis_radius_km = float(periapsis_radius_km)
    v_infinity_kmps = float(v_infinity_kmps)
    mu_km3_s2 = float(mu_km3_s2)
    if periapsis_radius_km <= 0.0:
        raise ValueError("periapsis_radius_km must be positive.")
    if v_infinity_kmps <= 0.0:
        raise ValueError("v_infinity_kmps must be positive.")
    if mu_km3_s2 <= 0.0:
        raise ValueError("mu_km3_s2 must be positive.")
    focusing = 1.0 + 2.0 * mu_km3_s2 / (periapsis_radius_km * v_infinity_kmps**2)
    return periapsis_radius_km * float(np.sqrt(focusing))


def periapsis_from_impact_parameter(
    b_km: float, v_infinity_kmps: float, mu_km3_s2: float
) -> float:
    """Periapsis radius produced by a given impact parameter.

    Inverse of :func:`impact_parameter_from_periapsis`, from the positive root of
    ``v_inf^2 r_p^2 + 2 mu r_p - b^2 v_inf^2 = 0``.
    """

    b_km = float(b_km)
    v_infinity_kmps = float(v_infinity_kmps)
    mu_km3_s2 = float(mu_km3_s2)
    if b_km < 0.0:
        raise ValueError("b_km must be non-negative.")
    if v_infinity_kmps <= 0.0:
        raise ValueError("v_infinity_kmps must be positive.")
    if mu_km3_s2 <= 0.0:
        raise ValueError("mu_km3_s2 must be positive.")
    discriminant = mu_km3_s2**2 + (b_km * v_infinity_kmps**2) ** 2
    return (np.sqrt(discriminant) - mu_km3_s2) / v_infinity_kmps**2


def capture_impact_parameter(
    body_radius_km: float, v_infinity_kmps: float, mu_km3_s2: float
) -> float:
    """Largest impact parameter that still strikes the body.

    This is :func:`impact_parameter_from_periapsis` evaluated at the body's own
    radius: the radius of the gravitational capture cross-section in the b-plane.
    An encounter impacts when ``b < capture_impact_parameter(...)``.
    """

    return impact_parameter_from_periapsis(body_radius_km, v_infinity_kmps, mu_km3_s2)


def sphere_of_influence_radius(
    mu_small_km3_s2: float, mu_large_km3_s2: float, separation_km: float
) -> float:
    """Radius of the smaller body's sphere of influence, ``r = a (m/M)^(2/5)``.

    The distance within which the smaller body, not the larger, dominates the
    motion of a third object. B-plane coordinates are only meaningful for states
    inside it: outside, no planetocentric hyperbola describes the trajectory.

    For Earth about the Sun at 1 au this is about 925,000 km.
    """

    mu_small_km3_s2 = float(mu_small_km3_s2)
    mu_large_km3_s2 = float(mu_large_km3_s2)
    separation_km = float(separation_km)
    if mu_small_km3_s2 <= 0.0 or mu_large_km3_s2 <= 0.0:
        raise ValueError("Gravitational parameters must be positive.")
    if separation_km <= 0.0:
        raise ValueError("separation_km must be positive.")
    if mu_small_km3_s2 >= mu_large_km3_s2:
        raise ValueError("mu_small_km3_s2 must be the smaller of the two bodies.")
    return separation_km * (mu_small_km3_s2 / mu_large_km3_s2) ** 0.4
