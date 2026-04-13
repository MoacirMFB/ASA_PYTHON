"""Celestial-body constants required by the virtual thrust workflow."""

from __future__ import annotations

from dataclasses import dataclass
import math

AU_KM = 149_597_870.7


@dataclass(frozen=True)
class GravitationalParameter:
    """Gravitational parameter with convenience accessors."""

    km: float

    @property
    def m(self) -> float:
        return self.km * 1e9

    @property
    def AU(self) -> float:
        return self.km / (AU_KM**3)


@dataclass(frozen=True)
class Length:
    """Length with convenience accessors."""

    km: float

    @property
    def m(self) -> float:
        return self.km * 1e3

    @property
    def AU(self) -> float:
        return self.km / AU_KM


@dataclass(frozen=True)
class OrbitElements:
    """Representative body orbit data used by this repository."""

    aphelion_km: float
    perihelion_km: float
    a_km: float
    e: float
    i_deg: float
    RAAN_deg: float
    M0_deg: float
    period_days: float
    speed_km_s: float
    arg_peri_deg: float = math.nan

    @property
    def aphelion_AU(self) -> float:
        return self.aphelion_km / AU_KM

    @property
    def perihelion_AU(self) -> float:
        return self.perihelion_km / AU_KM

    @property
    def a_AU(self) -> float:
        return self.a_km / AU_KM

    @property
    def period_sec(self) -> float:
        return self.period_days * 86400.0


@dataclass(frozen=True)
class _BodyData:
    """Resolved body data used to construct ``CelestialBody`` objects."""

    name: str
    mu: GravitationalParameter
    radius: Length
    J2: float
    omega: float
    soi: Length
    orbit: OrbitElements


class CelestialBody:
    """Minimal `CelestialBody` representation used by the Python workflow."""

    def __init__(self, body_name: str):
        body = _get_body_data(body_name)
        self.name = body.name
        self.mu = body.mu
        self.radius = body.radius
        self.J2 = body.J2
        self.omega = body.omega
        self.soi = body.soi
        self.orbit = body.orbit

    def __repr__(self) -> str:
        return f"CelestialBody({self.name!r})"


def _get_body_data(body_name: str) -> _BodyData:
    """Return resolved body data from the repository constants."""

    key = body_name.lower()
    if key == "earth":
        return _BodyData(
            name="Earth",
            mu=GravitationalParameter(398600.4418),
            radius=Length(6378.1363),
            J2=1.08262668e-3,
            omega=7.2921159e-5,
            soi=Length(145.0 * 6378.1363),
            orbit=OrbitElements(
                aphelion_km=152097597.0,
                perihelion_km=147098450.0,
                a_km=149598023.0,
                e=0.0167086,
                i_deg=0.00005,
                RAAN_deg=-11.26064,
                M0_deg=358.617,
                period_days=365.256363004,
                speed_km_s=29.7827,
                arg_peri_deg=102.94719,
            ),
        )
    if key == "moon":
        return _BodyData(
            name="Moon",
            mu=GravitationalParameter(4902.800066),
            radius=Length(1737.4),
            J2=2.03263e-4,
            omega=2.6617e-6,
            soi=Length(math.nan),
            orbit=OrbitElements(
                aphelion_km=405503.0,
                perihelion_km=363300.0,
                a_km=384400.0,
                e=0.0549,
                i_deg=5.145,
                RAAN_deg=125.08,
                M0_deg=115.3654,
                period_days=27.321661,
                speed_km_s=1.022,
            ),
        )
    if key == "sun":
        return _BodyData(
            name="Sun",
            mu=GravitationalParameter(1.32712440041279419e11),
            radius=Length(696340.0),
            J2=math.nan,
            omega=2.86533e-6,
            soi=Length(math.nan),
            orbit=OrbitElements(
                aphelion_km=math.nan,
                perihelion_km=math.nan,
                a_km=math.nan,
                e=math.nan,
                i_deg=math.nan,
                RAAN_deg=math.nan,
                M0_deg=math.nan,
                period_days=math.nan,
                speed_km_s=math.nan,
            ),
        )
    raise ValueError(f"Unsupported celestial body: {body_name}")


def get_celestial_body(body_name: str) -> CelestialBody:
    """Factory helper for callers that prefer a functional interface."""

    return CelestialBody(body_name)
