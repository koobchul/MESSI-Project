import numpy as np
import jax.numpy as jnp
from project.envs.tactical_reward import assignment,contextual_terms,frozen_progress

def check_tactics(env,reset):
    s=reset.rollout.state
    s=s._replace(ball=s.ball._replace(live=jnp.bool_(True)),restart=s.restart._replace(kind=jnp.int32(0)))
    own,eligible,chaser,target,support=assignment(env,s)
    assert not bool(eligible[0]) and not bool(support[0])
    assert not np.asarray(eligible[11:]).any()
    edge=s._replace(ball=s.ball._replace(position=s.ball.position.at[0].set(-52.5*s.attack_direction[0])))
    assert np.max(np.abs(np.asarray(assignment(env,edge)[3])[:,0]))<=45.
    reflected=s._replace(attack_direction=-s.attack_direction,
        ball=s.ball._replace(position=s.ball.position.at[:2].multiply(-1)),
        players=s.players._replace(position=-s.players.position))
    np.testing.assert_allclose(assignment(env,reflected)[3],-target,atol=1e-5)
    a,b=contextual_terms(env,s,s,jnp.bool_(True))
    np.testing.assert_allclose([a,b],[0,0],atol=1e-8)
    # Moving the ball alone cannot pay frozen-target movement progress.
    n=s._replace(ball=s.ball._replace(position=s.ball.position+jnp.array([5.,3.,0.])))
    np.testing.assert_allclose(contextual_terms(env,s,n,jnp.bool_(True)),[0,0],atol=1e-8)
    direction=s.ball.position[:2]-s.players.position[chaser]
    direction=direction/jnp.maximum(jnp.linalg.norm(direction),1e-6)
    n=s._replace(players=s.players._replace(position=s.players.position.at[chaser].add(.1*direction)))
    assert float(contextual_terms(env,s,n,jnp.bool_(True))[0])>0
    n=s._replace(players=s.players._replace(position=s.players.position.at[chaser].add(-.1*direction)))
    assert float(contextual_terms(env,s,n,jnp.bool_(True))[0])<0
    np.testing.assert_allclose(contextual_terms(env,s,n,jnp.bool_(False)),[0,0],atol=1e-8)
    return dict(stationary_zero=True,ball_motion_alone_zero=True,toward_positive=True,
                away_negative=True,gk_and_opponent_excluded=True,invalid_transition_zero=True,
                halftime_target_reflection=True)
