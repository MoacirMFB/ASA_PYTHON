# Examples

Standalone scripts that exercise the `ASA` package end-to-end. Examples are
not part of the importable package — run them directly from the repo root.

## Available scripts

- **`ps4_part_c_min_fuel.py`** — Earth-to-Mars minimum-fuel transfer
  reproducing PS4 Part C (AAE 590 ACA, Spring 2024). Drives the indirect-
  shooting solver through ρ-continuation, plots the smoothing function, the
  converged trajectory / state / costate / control, the Hamiltonian drift,
  and a 20-trial random-λ₀ convergence study.

## Running

```bash
cd /path/to/ASA_PYTHON
conda activate virtual_thrust
python examples/ps4_part_c_min_fuel.py
```

## Outputs

PNG figures land in `examples/figures/<script_name>/`. The directory is
created automatically.
