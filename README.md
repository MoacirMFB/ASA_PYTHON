# ASA Python

`asa_python` is a small Python package for astrodynamics and interception-analysis utilities. The current codebase provides:

- celestial-body constants for the Sun, Earth, and Moon
- two-body propagation and orbital-element conversion helpers
- exact zero-order-hold discrete-time linearization utilities
- a minimal Earth/asteroid interception workflow used by a related virtual-thrust toolchain

The package is intentionally compact. It looks like a focused Python port of the subset of functionality needed by the repository's virtual-thrust workflow and tests.

## Repository Layout

```text
ASA/
  __init__.py
  bodies.py
  control.py
  interception.py
  keplerian.py
  tests/
pyproject.toml
```

## Requirements

- Python 3.10 or newer
- `numpy`
- `scipy`

Optional test and integration dependencies:

- `pytest`
- `plotly`
- `cvxpy`
- `virtual_thrust` (used only by optional integration-style tests in `ASA/tests/test_virtual_thrust_helpers.py` and `ASA/tests/test_virtual_thrust_scp.py`)

## Installation

Create and activate a virtual environment, then install the package and its core runtime dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy
python -m pip install -e .
```

For local testing:

```bash
python -m pip install pytest
```

If you also want to run the optional virtual-thrust-related tests:

```bash
python -m pip install plotly cvxpy
```

You will also need the external `virtual_thrust` package available in the same environment for those tests to run.

## Public API

The package exports the following top-level symbols from `ASA`:

### Bodies and constants

- `CelestialBody`
- `OrbitElements`
- `get_celestial_body`

### Keplerian and propagation helpers

- `coe_to_cartesian`
- `dynamics_2bp_cartesian`
- `jacobian_2bp_cartesian`
- `propagate_stm_2bp`
- `propagate_two_body`

### Control helper

- `linear_discrete_time_matrices`

### Interception workflow

- `AsteroidRecord`
- `ClosestApproach`
- `Environment`
- `MBIState`
- `get_asteroid`
- `get_ca_moid`
- `get_states_at_mbi`
- `make_bodies_for_plot`
- `make_env`
- `prepare_mbi`
- `propagate_asteroid`
- `propagate_earth`

## Quick Start

This example propagates Earth and one asteroid, computes the sampled minimum-distance geometry, and prepares states at a set of months-before-impact epochs.

```python
import numpy as np

from ASA import (
    get_asteroid,
    get_ca_moid,
    make_env,
    prepare_mbi,
    propagate_asteroid,
    propagate_earth,
)

env = make_env(
    years=0.15,
    step_min=24.0 * 60.0,
    rtol=1e-10,
    atol=1e-10,
)

t_earth, x_earth = propagate_earth(env)
asteroid = propagate_asteroid(get_asteroid("Apophis"), env)

ca, moid = get_ca_moid(x_earth, asteroid.X_hist, t_earth, asteroid.t_hist)
print(f"Closest sampled synchronized approach: {ca.d_km:.3f} km")
print(f"Sampled MOID proxy: {moid.d_km:.3f} km")

asteroid_out, mbi_states, state_at_moid = prepare_mbi(
    asteroid,
    env,
    months_back=np.array([0.0, 1.0, 3.0]),
    force_impact=True,
)

print(asteroid_out.name)
print(f"Prepared {len(mbi_states)} months-before-impact states")
print(f"Forced-impact state distance: {state_at_moid.d_km:.3f} km")
```

## Included Asteroid Catalog

`ASA.interception.get_asteroid()` currently resolves entries from a small in-memory catalog. The supported names are:

- `2007 DX40`
- `2004 VD17`
- `2007 FT3`
- `1979 XB`
- `1950 DA`
- `Bennu`
- `Apophis`
- `2011 AG5`
- `2022 AE1`
- `2000 SG344`
- `2023 DW`

## Testing

Run the core test suite with:

```bash
python -m pytest ASA/tests/test_keplerian.py ASA/tests/test_control.py ASA/tests/test_interception.py
```

Run the full suite, including optional virtual-thrust integration tests, with:

```bash
python -m pytest ASA/tests
```

The virtual-thrust tests are written with `pytest.importorskip(...)`, so they are skipped automatically when their optional dependencies are not installed.

## Notes and Limitations

- The package metadata currently specifies the package name and Python version, but it does not declare runtime dependencies in `pyproject.toml`. Install `numpy` and `scipy` explicitly before using the package.
- The interception workflow uses a simplified two-body heliocentric model.
- `get_ca_moid()` computes a sampled closest-approach pair and a nearest-neighbor sampled MOID-style proxy. It is not a full analytic MOID solver.
- The asteroid catalog is embedded directly in code rather than loaded from an external ephemeris or database.
