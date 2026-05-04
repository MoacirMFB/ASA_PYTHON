"""PS4 Part C reproduction: planar Earth-to-Mars minimum-fuel transfer.

Mirrors `PS4_MMFB.mlx` (AAE 590 ACA, Spring 2024) using the 3D Python port
in ``ASA.control``. The planar problem is embedded in 3D by zeroing the z
and v_z components of state and costate; the dynamics preserve this invariant
exactly, so in-plane numbers reproduce the MATLAB run to integrator tolerance.

Run from the repo root:

    conda activate virtual_thrust
    python examples/ps4_part_c_min_fuel.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ASA.control import (
    min_fuel_combined_dynamics,
    min_fuel_hamiltonian,
    min_fuel_optimal_control,
    solve_min_fuel_bvp,
)

FIG_DIR = Path(__file__).parent / "figures" / "ps4_part_c"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Setup -- non-dimensional units, mu = 1 (l_char = 1 AU, t_char from mu_sun)
# ---------------------------------------------------------------------------

MU = 1.0
A_EARTH = 1.0
A_MARS = 1.524
NU_EARTH_0 = 0.0
NU_MARS_0 = np.pi
N_MARS = np.sqrt(MU / A_MARS**3)

T0 = 0.0
TF = 8.0
N_POINTS = 200
UMAX = 0.1
RTOL = ATOL = 1e-12
ROOT_TOL = 1e-10


def planar_circular_state(a: float, mu: float, nu: float) -> np.ndarray:
    """3D state for a circular orbit in the xy-plane (z = v_z = 0)."""
    v_circ = np.sqrt(mu / a)
    return np.array(
        [a * np.cos(nu), a * np.sin(nu), 0.0,
         -v_circ * np.sin(nu), v_circ * np.cos(nu), 0.0]
    )


def lift_planar(lam_2d: np.ndarray) -> np.ndarray:
    """Lift planar costate [lam_r1, lam_r2, lam_v1, lam_v2] to 3D with z components zero."""
    return np.array([lam_2d[0], lam_2d[1], 0.0, lam_2d[2], lam_2d[3], 0.0])


x0 = planar_circular_state(A_EARTH, MU, NU_EARTH_0)
nu_mars_tf = NU_MARS_0 + TF * N_MARS
xf = planar_circular_state(A_MARS, MU, nu_mars_tf)

print("=" * 70)
print("PS4 Part C  --  Earth -> Mars Minimum-Fuel Transfer")
print("=" * 70)
print(f"x0 = {x0}")
print(f"xf = {xf}   (Mars at nu_M(tf) = {nu_mars_tf:.6f})")
print(f"tf = {TF}, umax = {UMAX}, mu = {MU}")
print()


# ---------------------------------------------------------------------------
# c.2  --  Smoothing function gamma(S, rho)
# ---------------------------------------------------------------------------

print("--- (c.2) Gamma vs S smoothing ---")
S_grid = np.linspace(-1.0, 1.0, 100)
rho_smoothing = [1.0, 0.5, 0.1, 0.05, 0.01, 0.001]

fig, ax = plt.subplots(figsize=(8, 5))
for rho in rho_smoothing:
    gamma_curve = (UMAX / 2.0) * (1.0 + np.tanh(S_grid / rho))
    ax.plot(S_grid, gamma_curve, lw=2, label=fr"$\rho = {rho}$")
ax.set_xlabel("Switching function $S$", fontsize=12)
ax.set_ylabel(r"Control magnitude $\Gamma^*$", fontsize=12)
ax.set_title(r"$\Gamma^* = (u_{max}/2)\,(1 + \tanh(S/\rho))$")
ax.grid(True)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "gamma_vs_S.png", dpi=150)
print(f"  saved {FIG_DIR / 'gamma_vs_S.png'}")
print()


# ---------------------------------------------------------------------------
# c.3  --  Indirect shooting with rho-continuation
# ---------------------------------------------------------------------------

print("--- (c.3) Rho-continuation indirect shooting ---")
lam0_guess = lift_planar(np.array([0.8, 0.1, 0.2, 1.1]))
rho_sequence = [1.0, 0.1, 0.01, 0.001]
result = None
print(f"  {'rho':>8}  {'lambda0_star (in-plane: r1, r2, v1, v2)':>52}  {'residual':>12}")
for rho in rho_sequence:
    result = solve_min_fuel_bvp(
        T0, TF, x0, xf, MU, UMAX, rho, lam0_guess,
        n_eval=N_POINTS, root_tol=ROOT_TOL, rtol=RTOL, atol=ATOL,
    )
    lam_in_plane = result.lam0[[0, 1, 3, 4]]
    print(f"  {rho:>8.4g}  {np.array2string(lam_in_plane, formatter={'float': '{: .4f}'.format}):>52}  {result.residual_norm:>12.3e}")
    lam0_guess = result.lam0  # warm start for next rho

assert result is not None
t_hist = result.t
x_hist = result.x_hist
lam_hist = result.lam_hist
u_hist = result.u_hist

# Planar invariant sanity
z_state = float(np.max(np.abs(x_hist[:, [2, 5]])))
z_costate = float(np.max(np.abs(lam_hist[:, [2, 5]])))
print(f"  Planar invariant: max|x_z, vz| = {z_state:.2e}, max|lam_z, lam_vz| = {z_costate:.2e}  -> {'OK' if max(z_state, z_costate) < 1e-9 else 'WARN'}")
print()


# Trajectory plot
print("--- Plots ---")
theta = np.linspace(0, 2 * np.pi, 200)
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(x_hist[:, 0], x_hist[:, 1], "g-", lw=1.8, label="Transfer trajectory")
ax.plot(A_EARTH * np.cos(theta), A_EARTH * np.sin(theta), "b--", label="Earth orbit")
ax.plot(A_MARS * np.cos(theta), A_MARS * np.sin(theta), "r--", label="Mars orbit")
ax.plot(x0[0], x0[1], "bo", ms=10, label="Start (Earth)")
ax.plot(xf[0], xf[1], "ro", ms=10, label="End (Mars at $t_f$)")
ax.set_xlabel("x [AU]")
ax.set_ylabel("y [AU]")
ax.set_aspect("equal")
ax.grid(True)
ax.legend(loc="lower left")
ax.set_title("Min-Fuel Earth -> Mars Transfer (Part C)")
fig.tight_layout()
fig.savefig(FIG_DIR / "trajectory.png", dpi=150)
print(f"  saved {FIG_DIR / 'trajectory.png'}")

# State + costate plot
fig, ax = plt.subplots(figsize=(10, 6))
state_labels = ["$x_1$", "$x_2$", "$v_1$", "$v_2$"]
state_idx = [0, 1, 3, 4]
costate_labels = [r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$", r"$\lambda_4$"]
for idx, lab in zip(state_idx, state_labels):
    ax.plot(t_hist, x_hist[:, idx], "--", lw=1.2, label=lab)
for idx, lab in zip(state_idx, costate_labels):
    ax.plot(t_hist, lam_hist[:, idx], "-", lw=1.2, label=lab)
ax.set_xlabel("Time [non-dim]")
ax.set_ylabel("State / costate")
ax.set_title("State and Costate History")
ax.grid(True)
ax.legend(loc="best", ncol=2, fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "state_costate.png", dpi=150)
print(f"  saved {FIG_DIR / 'state_costate.png'}")

# Control plot
gamma_star = np.linalg.norm(u_hist, axis=1)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t_hist, u_hist[:, 0], "b-", lw=1.5, label="$u_1$")
ax.plot(t_hist, u_hist[:, 1], "r-", lw=1.5, label="$u_2$")
ax.plot(t_hist, gamma_star, "k--", lw=1.5, label=r"$\Gamma^* = \|u\|$")
ax.axhline(UMAX, color="gray", ls=":", lw=1, label=f"$u_{{max}} = {UMAX}$")
ax.axhline(-UMAX, color="gray", ls=":", lw=1)
ax.set_xlabel("Time [non-dim]")
ax.set_ylabel("Control")
ax.set_title("Optimal Control and Magnitude History")
ax.grid(True)
ax.legend(loc="best")
fig.tight_layout()
fig.savefig(FIG_DIR / "control.png", dpi=150)
print(f"  saved {FIG_DIR / 'control.png'}")
print()


# ---------------------------------------------------------------------------
# c.4  --  Hamiltonian conservation
# ---------------------------------------------------------------------------

print("--- (c.4) Hamiltonian conservation ---")
H = min_fuel_hamiltonian(u_hist, lam_hist, x_hist, MU)
dH = H - H[0]
print(f"  H(0)        = {H[0]:.6e}")
print(f"  max|H - H0| = {np.max(np.abs(dH)):.3e}")
print(f"  H spread    = {(H.max() - H.min()):.3e}")

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(t_hist, dH, "b-", lw=1.8)
ax.set_xlabel("Time [non-dim]")
ax.set_ylabel(r"$\Delta H(t) = H(t) - H(0)$")
ax.set_title("Hamiltonian Drift (should be ~zero for an optimal trajectory)")
ax.grid(True)
fig.tight_layout()
fig.savefig(FIG_DIR / "delta_H.png", dpi=150)
print(f"  saved {FIG_DIR / 'delta_H.png'}")
print()


# ---------------------------------------------------------------------------
# c.5  --  Random lambda0 robustness experiment
# ---------------------------------------------------------------------------

print("--- (c.5) Random lambda0 convergence study (rho = 1.0) ---")
n_trials = 20
rng = np.random.default_rng(seed=42)
rho_random = 1.0
converged = 0

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(A_EARTH * np.cos(theta), A_EARTH * np.sin(theta), "b--", label="Earth orbit")
ax.plot(A_MARS * np.cos(theta), A_MARS * np.sin(theta), "r--", label="Mars orbit")

print(f"  {'#':>3}  {'lambda0_guess (r1, r2, v1, v2)':>40}  {'converged':>10}  {'residual':>12}")
for i in range(n_trials):
    lam_in_plane = rng.uniform(-10.0, 10.0, size=4)
    guess = lift_planar(lam_in_plane)
    res = solve_min_fuel_bvp(
        T0, TF, x0, xf, MU, UMAX, rho_random, guess,
        n_eval=N_POINTS, root_tol=ROOT_TOL, rtol=RTOL, atol=ATOL,
    )
    ok = res.residual_norm < 1e-6
    if ok:
        converged += 1
        ax.plot(res.x_hist[:, 0], res.x_hist[:, 1], lw=1.0, alpha=0.7)
    print(f"  {i:>3}  {np.array2string(lam_in_plane, formatter={'float': '{: .2f}'.format}):>40}  {str(ok):>10}  {res.residual_norm:>12.3e}")

ax.plot(x0[0], x0[1], "bo", ms=10, label="Start (Earth)")
ax.plot(xf[0], xf[1], "ro", ms=10, label="End (Mars at $t_f$)")
ax.set_xlabel("x [AU]")
ax.set_ylabel("y [AU]")
ax.set_aspect("equal")
ax.set_xlim(-1.7, 1.7)
ax.set_ylim(-1.7, 1.7)
ax.grid(True)
ax.legend(loc="lower left", fontsize=9)
ax.set_title(f"Random $\\lambda_0$ Trajectories ({converged}/{n_trials} converged)")
fig.tight_layout()
fig.savefig(FIG_DIR / "random_lambda0_trajectories.png", dpi=150)
print(f"  saved {FIG_DIR / 'random_lambda0_trajectories.png'}")
print(f"  Converged: {converged} / {n_trials}")
print()

print(f"All figures saved to: {FIG_DIR}")
plt.show()
