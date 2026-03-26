import numpy as np

from ..interception import (
    ast_catalog,
    get_ca_moid,
    make_env,
    prepare_mbi,
    propagate_asteroids,
    propagate_earth,
)


def test_minimal_interception_chain_propagates_and_prepares_mbi():
    env = make_env(years=0.15, step_min=24.0 * 60.0, rtol=1e-10, atol=1e-10)
    t_earth, x_earth = propagate_earth(env)
    asteroids = propagate_asteroids(ast_catalog("Apophis"), env)

    assert t_earth.shape[0] == x_earth.shape[0]
    assert x_earth.shape[1] == 6
    assert asteroids[0].X_hist is not None
    assert asteroids[0].t_hist is not None

    ca, moid = get_ca_moid(x_earth, asteroids[0].X_hist, t_earth, asteroids[0].t_hist)
    assert ca.d_km >= 0.0
    assert moid.d_km >= 0.0

    asteroids_out, mbi_sets, state_at_moid = prepare_mbi(
        asteroids,
        env,
        months_back=[0.0, 1.0],
        force_impact=True,
        rtol=1e-10,
        atol=1e-10,
    )
    assert asteroids_out[0].name.endswith("-forced")
    assert len(mbi_sets) == 1
    assert len(mbi_sets[0]) == 2
    np.testing.assert_allclose(mbi_sets[0][0].stateEarth[:3], mbi_sets[0][0].stateAst[:3], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(state_at_moid.stateEarth[:3], state_at_moid.stateAst[:3], rtol=1e-12, atol=1e-12)
