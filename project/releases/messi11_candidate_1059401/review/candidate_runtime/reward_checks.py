"""Counterfactual event fixtures: routing/gates, not performance evidence."""
from types import SimpleNamespace as NS
import jax.numpy as jnp
import numpy as np
from footballworld.core.contact import INTENT_CONTROL,OUTCOME_CATCH,INTENT_SOURCE_POLICY,INTENT_SOURCE_ENVIRONMENT_FORCED
from footballworld.core.constants import RK_GK_HOLD
from project.envs.reward11 import reward11

def check_rewards(env,reset):
    s=reset.rollout.state
    s=s._replace(ball=s.ball._replace(live=jnp.bool_(True)),restart=s.restart._replace(kind=jnp.int32(0)))
    nxt=s._replace(restart=s.restart._replace(kind=jnp.int32(RK_GK_HOLD)))
    c=NS(actor=jnp.array([0]),intent=jnp.array([INTENT_CONTROL]),outcome=jnp.array([OUTCOME_CATCH]),intent_source=jnp.array([INTENT_SOURCE_POLICY]))
    def result(state,contact):
        return NS(rollout=reset.rollout._replace(state=state),events=NS(deliberate_contact=contact),reward=jnp.zeros(2),terminated=jnp.bool_(False))
    value=np.asarray(reward11(env,s,result(nxt,c)))
    assert np.isclose(value[9],.2),value
    forced=NS(**{**c.__dict__,'intent_source':jnp.array([INTENT_SOURCE_ENVIRONMENT_FORCED])})
    assert np.isclose(np.asarray(reward11(env,s,result(nxt,forced)))[9],0)
    absent=NS(**{**c.__dict__,'actor':jnp.array([-1])})
    empty=np.asarray(reward11(env,s,result(s,absent)))
    assert empty[4]==0 and empty[5]==0 and empty[9]==0
    costs=[]
    for depth in (20.,40.):
        st=s._replace(players=s.players._replace(position=s.players.position.at[0].set(jnp.array([-env.stadium.half_length+depth,0.])*s.attack_direction[0])))
        costs.append(float(reward11(env,st,result(st,absent))[7]))
    assert costs[1]<costs[0],costs
    return dict(policy_catch_reward_across_gk_hold=True,forced_catch_not_credited=True,
        absent_contact_no_shot_pass_catch_bonus=True,gk_distance_penalty_not_saturated_at_16m=True)
