import pytest
import numpy as np
import plotly.graph_objects as go

cvxpy = pytest.importorskip("cvxpy")
pytest.importorskip("virtual_thrust")

from virtual_thrust import VirtualThrustConfig, _run_scp_optimization, _solve_scp_subproblem, run_virtual_thrust


def test_scp_subproblem_builds_and_solves_small_synthetic_case():
    assert cvxpy is not None

    n_intervals = 3
    Ak = np.repeat(np.eye(6)[None, :, :], n_intervals, axis=0)
    Bk = np.zeros((n_intervals, 6, 3))
    Bk[:, :3, :] = np.eye(3)  # direct synthetic position actuation for a tiny SOCP smoke test
    ck = np.zeros((n_intervals, 6))
    x0 = np.zeros(6)
    U_nom = np.zeros((n_intervals, 3))
    R_nom = np.array([1.0, 0.0, 0.0])
    rnom_tf = np.zeros(3)

    result = _solve_scp_subproblem(
        Ak,
        Bk,
        ck,
        x0,
        U_nom,
        R_nom,
        rnom_tf,
        dt_seg_s=1.0,
        tau_budget_s=10.0,
        delta_u=1.0,
    )

    assert result.x.shape == (n_intervals + 1, 6)
    assert result.u.shape == (n_intervals, 3)
    assert result.solver is not None
    assert result.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}


def test_reduced_size_virtual_thrust_runs_one_scp_iteration(monkeypatch):
    assert cvxpy is not None
    monkeypatch.setattr(go.Figure, "show", lambda self: None)

    config = VirtualThrustConfig(
        env_years=0.2,
        lead_time_years=1,
        t0_months=1,
        n_segments=4,
        kmax=1,
        run_scp=True,
    )
    result = run_virtual_thrust(config)

    assert result.scp is not None
    assert result.scp.success
    assert len(result.scp.iterations) == 1
    assert result.scp.U_opt is not None
    assert result.scp.X_opt is not None
