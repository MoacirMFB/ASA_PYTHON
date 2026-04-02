import numpy as np

from ..interception import (
    get_asteroid,
    get_ca_moid,
    make_env,
    prepare_mbi,
    propagate_asteroid,
    propagate_earth,
)


def test_minimal_interception_chain_propagates_and_prepares_mbi():
    env = make_env(years=0.15, step_min=24.0 * 60.0, rtol=1e-10, atol=1e-10)
    t_earth, x_earth = propagate_earth(env)
    asteroid = propagate_asteroid(get_asteroid("Apophis"), env)

    assert t_earth.shape[0] == x_earth.shape[0]
    assert x_earth.shape[1] == 6
    assert asteroid.X_hist is not None
    assert asteroid.t_hist is not None

    ca, moid = get_ca_moid(x_earth, asteroid.X_hist, t_earth, asteroid.t_hist)
    assert ca.d_km >= 0.0
    assert moid.d_km >= 0.0

    asteroid_out, mbi_states, state_at_moid = prepare_mbi(
        asteroid,
        env,
        months_back=[0.0, 1.0],
        force_impact=True,
        rtol=1e-10,
        atol=1e-10,
    )
    assert asteroid_out.name.endswith("-forced")
    assert len(mbi_states) == 2
    np.testing.assert_allclose(mbi_states[0].stateEarth[:3], mbi_states[0].stateAst[:3], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(state_at_moid.stateEarth[:3], state_at_moid.stateAst[:3], rtol=1e-12, atol=1e-12)
