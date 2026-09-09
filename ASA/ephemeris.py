"""Ephemeris access through SPICE kernels.

A deliberately thin layer over ``spiceypy``. SPICE already solves kernel
management, time systems and state lookup correctly; this module exists to give
the rest of ASA a small, typed surface onto it, to keep kernel paths in one
place, and to convert SPICE's conventions into the ones ASA uses elsewhere
(kilometres, seconds, NumPy arrays).

What a planetary ephemeris actually is
--------------------------------------
DE441 and its relatives are not models you run. JPL integrated the solar system
once, fitted the result with Chebyshev polynomials, and published the
coefficients. Asking for Jupiter's position at some epoch evaluates a
polynomial; nothing is integrated at call time. An "n-body propagation" then
means integrating only the small body, reading each planet's position from the
ephemeris at every step and adding its pull. The planets move on rails.

Kernels are large binaries and are not stored in the repository. They are
fetched on demand into a cache directory, ``~/.asa/kernels`` by default,
overridable with the ``ASA_KERNEL_DIR`` environment variable.

Time
----
SPICE works in ephemeris time (ET, TDB seconds past the J2000 epoch). Any
conversion from a calendar string needs a leapseconds kernel loaded first;
:func:`ensure_leapseconds` fetches and loads one if it is missing.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import urlparse

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

try:  # pragma: no cover - exercised by the import, not by the tests
    import spiceypy
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ASA.ephemeris requires spiceypy. Install it with `pip install spiceypy`."
    ) from exc

DEFAULT_KERNEL_DIR = Path(
    os.environ.get("ASA_KERNEL_DIR", Path.home() / ".asa" / "kernels")
)

LEAPSECONDS_KERNEL_URL = (
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls"
)

# SPICE integer ids for the bodies ASA reaches for most often. Strings are also
# accepted everywhere a body is named, so this is a convenience, not a registry.
SOLAR_SYSTEM_BARYCENTER = "0"
SUN = "10"
EARTH = "399"
EARTH_MOON_BARYCENTER = "3"
MOON = "301"

_LOADED_KERNELS: list[Path] = []


@dataclass(frozen=True)
class Coverage:
    """Epochs for which a kernel can provide states.

    ``intervals`` is the full set, in order; coverage is not guaranteed to be
    contiguous, so ``start_et_s`` and ``end_et_s`` are the outer bounds rather
    than a promise about everything between them. Use :meth:`contains` to test a
    specific epoch.
    """

    body: str
    intervals: tuple[tuple[float, float], ...]

    @property
    def start_et_s(self) -> float:
        return self.intervals[0][0]

    @property
    def end_et_s(self) -> float:
        return self.intervals[-1][1]

    @property
    def duration_s(self) -> float:
        return sum(end - start for start, end in self.intervals)

    def contains(self, et_s: float) -> bool:
        """True when some interval covers this epoch."""

        return any(start <= et_s <= end for start, end in self.intervals)


def kernel_dir() -> Path:
    """Return the kernel cache directory, creating it if needed."""

    DEFAULT_KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_KERNEL_DIR


def fetch_kernel(url: str, *, filename: str | None = None, overwrite: bool = False) -> Path:
    """Download one kernel into the cache directory and return its path.

    Nothing is downloaded if the file is already present unless ``overwrite`` is
    set. Only ``https`` URLs are accepted: kernels determine every trajectory
    computed downstream, so they are not worth fetching over a channel that can
    be tampered with in transit.
    """

    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"Kernels must be fetched over https, got {parsed.scheme!r}.")
    name = filename or Path(parsed.path).name
    if not name:
        raise ValueError(f"Could not determine a filename from {url!r}.")

    destination = kernel_dir() / name
    if destination.exists() and not overwrite:
        return destination

    partial = destination.with_suffix(destination.suffix + ".partial")
    with urllib.request.urlopen(url) as response, partial.open("wb") as handle:  # noqa: S310
        while chunk := response.read(1 << 20):
            handle.write(chunk)
    partial.replace(destination)
    return destination


def load_kernels(paths: Iterable[str | Path]) -> list[Path]:
    """Load kernels into the SPICE pool, skipping any already loaded."""

    loaded: list[Path] = []
    for path in paths:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Kernel not found: {resolved}")
        if resolved in _LOADED_KERNELS:
            continue
        spiceypy.furnsh(str(resolved))
        _LOADED_KERNELS.append(resolved)
        loaded.append(resolved)
    return loaded


def loaded_kernels() -> tuple[Path, ...]:
    """Kernels currently loaded through this module, in load order."""

    return tuple(_LOADED_KERNELS)


def unload_kernels() -> None:
    """Clear the SPICE pool of everything this module loaded."""

    for path in reversed(_LOADED_KERNELS):
        spiceypy.unload(str(path))
    _LOADED_KERNELS.clear()


def ensure_leapseconds() -> Path:
    """Load a leapseconds kernel, fetching it first if the cache lacks one.

    Calendar-string conversion is undefined without one, so anything that calls
    :func:`utc_to_et` needs this to have run.
    """

    path = fetch_kernel(LEAPSECONDS_KERNEL_URL)
    load_kernels([path])
    return path


def utc_to_et(utc: str) -> float:
    """Convert a calendar string to ephemeris time (TDB seconds past J2000)."""

    return float(spiceypy.str2et(utc))


def et_to_utc(et_s: float, *, fmt: str = "C", decimals: int = 3) -> str:
    """Convert ephemeris time to a calendar string."""

    return str(spiceypy.et2utc(float(et_s), fmt, decimals))


def state(
    target: str,
    et_s: float | Sequence[float] | FloatArray,
    *,
    observer: str = SUN,
    frame: str = "ECLIPJ2000",
    aberration_correction: str = "NONE",
) -> FloatArray:
    """Return ``[x, y, z, vx, vy, vz]`` in km and km/s at one or many epochs.

    A scalar ``et_s`` gives shape ``(6,)``; a sequence gives ``(N, 6)``.
    ``aberration_correction`` defaults to ``"NONE"``, which is what dynamics
    wants: the body's true geometric state, not its apparent one.
    """

    epochs = np.atleast_1d(np.asarray(et_s, dtype=float))
    if epochs.ndim != 1:
        raise ValueError("et_s must be a scalar or a one-dimensional sequence.")
    if not np.all(np.isfinite(epochs)):
        raise ValueError("et_s contains non-finite values.")

    states = np.empty((epochs.size, 6), dtype=float)
    for index, epoch_s in enumerate(epochs):
        vector, _light_time = spiceypy.spkezr(
            str(target), float(epoch_s), frame, aberration_correction, str(observer)
        )
        states[index] = vector
    return states[0] if np.isscalar(et_s) or np.ndim(et_s) == 0 else states


def position(
    target: str,
    et_s: float | Sequence[float] | FloatArray,
    *,
    observer: str = SUN,
    frame: str = "ECLIPJ2000",
    aberration_correction: str = "NONE",
) -> FloatArray:
    """Position only, in km. Same shape rules as :func:`state`."""

    result = state(
        target,
        et_s,
        observer=observer,
        frame=frame,
        aberration_correction=aberration_correction,
    )
    return result[..., :3]


def relative_state(
    target: str,
    et_s: float | Sequence[float] | FloatArray,
    *,
    observer: str,
    frame: str = "ECLIPJ2000",
) -> FloatArray:
    """State of ``target`` relative to ``observer``. Alias for clarity at call sites."""

    return state(target, et_s, observer=observer, frame=frame)


def _coverage_window(kernel_path: str | Path, body: str):
    """SPICE window of the epochs a kernel covers for one body."""

    name = str(body)
    identifier = int(name) if name.lstrip("-").isdigit() else spiceypy.bods2c(name)
    window = spiceypy.stypes.SPICEDOUBLE_CELL(2000)
    spiceypy.spkcov(str(Path(kernel_path).expanduser().resolve()), identifier, window)
    if spiceypy.wncard(window) == 0:
        raise ValueError(f"Kernel {kernel_path} provides no coverage for body {body!r}.")
    return window


def coverage(kernel_path: str | Path, body: str) -> Coverage:
    """Return the epochs a kernel covers for one body.

    Useful for finding where a scenario kernel begins and ends: an object on an
    impact trajectory typically has its coverage stop at the impact itself.
    """

    window = _coverage_window(kernel_path, body)
    return Coverage(
        body=str(body),
        intervals=tuple(
            tuple(float(value) for value in spiceypy.wnfetd(window, index))
            for index in range(spiceypy.wncard(window))
        ),
    )


def common_coverage(kernel_path: str | Path, bodies: Sequence[str]) -> Coverage:
    """Epochs a kernel covers for *every* one of ``bodies``.

    A relative state needs both bodies at once, and their coverage need not
    agree: in JPL's 2024 PDC25 kernels the asteroid runs one second longer than
    Earth, so the asteroid's own end epoch raises SPKINSUFFDATA when differenced
    against Earth. Intersecting first is the difference between a usable epoch
    bound and one that fails at the boundary.
    """

    if not bodies:
        raise ValueError("bodies must not be empty.")
    window = _coverage_window(kernel_path, bodies[0])
    for body in bodies[1:]:
        window = spiceypy.wnintd(window, _coverage_window(kernel_path, body))
    if spiceypy.wncard(window) == 0:
        raise ValueError(f"Kernel {kernel_path} has no epochs covering all of {list(bodies)}.")
    return Coverage(
        body=", ".join(str(body) for body in bodies),
        intervals=tuple(
            tuple(float(value) for value in spiceypy.wnfetd(window, index))
            for index in range(spiceypy.wncard(window))
        ),
    )


def bodies_in_kernel(kernel_path: str | Path) -> tuple[int, ...]:
    """SPICE ids of every body an SPK file provides."""

    ids = spiceypy.stypes.SPICEINT_CELL(1000)
    spiceypy.spkobj(str(Path(kernel_path).expanduser().resolve()), ids)
    return tuple(int(identifier) for identifier in ids)


def osculating_elements(
    target: str,
    et_s: float,
    mu_km3_s2: float,
    *,
    observer: str = SUN,
    frame: str = "ECLIPJ2000",
) -> dict[str, float]:
    """Osculating elements at one epoch, with angles in radians.

    These are the two-body elements matching the true state instantaneously.
    They drift under perturbations, so they describe the orbit at ``et_s`` only.
    """

    elements = spiceypy.oscltx(
        state(target, et_s, observer=observer, frame=frame), float(et_s), float(mu_km3_s2)
    )
    return {
        "periapsis_radius_km": float(elements[0]),
        "eccentricity": float(elements[1]),
        "inclination_rad": float(elements[2]),
        "raan_rad": float(elements[3]),
        "argument_of_periapsis_rad": float(elements[4]),
        "mean_anomaly_rad": float(elements[5]),
        "epoch_et_s": float(elements[6]),
        "mu_km3_s2": float(elements[7]),
        "true_anomaly_rad": float(elements[8]),
        "semi_major_axis_km": float(elements[9]),
        "period_s": float(elements[10]),
    }
