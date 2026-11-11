"""Execution gates only; no football performance thresholds."""
import jax
import jax.numpy as jnp
import numpy as np
from project.models.role_actor import actor_forward
from project.models.intent_guard import apply_retouch_guard
from project.models.action_sampling import sample_action,action_log_prob
from project.models.gk_movement import limit_gk_move
from project.algorithms.update import actor_loss_with_params
from project.algorithms.rollout_jit import initialize_jit_rollout_carry
from project.envs.policy_obs import encode_policy_observation
from project.envs.curriculum import training_start
from project.envs.messi_11v11 import reset_env
from project.envs.reward11 import gk_local

def verify_invariants(e):
    from reward_checks import check_rewards
    reset=reset_env(e.env,12345)
    reward_checks=check_rewards(e.env,reset)
    obs=encode_policy_observation(e.env,reset.rollout,e.roster)[:11][None,...]
    output=apply_retouch_guard(actor_forward(e.ap,e.b,e.h,obs),obs)
    sampled=sample_action(output,jax.random.key(1));lp=action_log_prob(output,sampled.action.intent,sampled.pre_tanh_continuous)
    def grad_for(advantages):
        return jax.grad(lambda params:actor_loss_with_params(params,e.b,e.h,obs,sampled.action.intent,
          sampled.pre_tanh_continuous,lp,advantages,entropy_coef=0.,valid_mask=jnp.array([True]),retouch_guard_enabled=True)[0])(e.ap)
    print('GATE role gradients',flush=True)
    fg=grad_for(jnp.array([[1.,0.]]));gg=grad_for(jnp.array([[0.,1.]]))
    norm=lambda t:float(sum(jnp.sum(x*x) for x in jax.tree_util.tree_leaves(t)))
    assert norm(fg['gk'])==0 and norm(gg['field'])==0,'cross-role gradient leakage'
    assert norm(fg['field'])>0 and norm(gg['gk'])>0,'actor gradient missing'
    # Same exact mapping function is imported by both collector implementations.
    import trace as evaluation
    import project.algorithms.rollout_jit as training
    assert evaluation.limit_gk_move is training.limit_gk_move
    assert evaluation.field_movement is training.field_movement
    allobs=encode_policy_observation(e.env,reset.rollout,e.roster)
    a=sample_action(actor_forward(e.ap,e.b,e.h,allobs),jax.random.key(2)).action
    mapped,_,_=limit_gk_move(e.env,reset.rollout.state,a)
    np.testing.assert_array_equal(np.asarray(mapped.move)[1:],np.asarray(a.move)[1:])
    for key in ('intent','force_to_ball','launch','spin'):
        np.testing.assert_array_equal(np.asarray(getattr(mapped,key)),np.asarray(getattr(a,key)))
    assert np.max(np.abs(np.asarray(mapped.move)))<=1
    st=reset.rollout.state
    reflected=st._replace(attack_direction=-st.attack_direction,players=st.players._replace(position=-st.players.position,velocity=-st.players.velocity))
    reflected_action,_,_=limit_gk_move(e.env,reflected,a)
    # Engine performs world rotation: team-frame outputs must remain identical.
    np.testing.assert_allclose(np.asarray(reflected_action.move[0]),np.asarray(mapped.move[0]),atol=1e-6)
    # Distant midfield balls cannot yield GK local-approach reward.
    assert not bool(gk_local(e.env,st._replace(ball=st.ball._replace(position=jnp.array([0.,0.,.11]),live=jnp.bool_(True))))[0])
    scenarios={}
    for i,kind in enumerate(('acquire','advance','finish','gk')):
        r=training_start(e.env,reset_env(e.env,13000+i),13000+i,kind)
        assert int(r.rollout.state.possession.player)==-1
        c=initialize_jit_rollout_carry(e.env,r,jax.random.key(14000+i))
        batch,valid,after=e.collect(r.setup,c,e.roster,e.si,e.ap,e.cp,jax.random.key(15000+i))
        for x in jax.tree_util.tree_leaves(batch):
            assert np.isfinite(np.asarray(x)).all(),kind
        assert int(np.asarray(valid).sum())>0
        print('GATE training start',kind, 'valid',int(np.asarray(valid).sum()),flush=True)
        scenarios[kind]={'steps':int(np.asarray(valid).sum()),'lawful_ticks':float(np.asarray(batch.lawful_possession).sum()),
          'reward_sums':np.asarray(batch.reward_details).sum(0).tolist()}
    return dict(reward_checks=reward_checks,cross_role_gradient_zero=True,own_role_gradient_nonzero=True,
      training_evaluation_same_move_function=True,outfield_actions_unchanged=True,
      move_range=True,halftime_mapping_reflection=True,gk_far_ball_approach_disabled=True,
      training_start_no_fabricated_possession=True,scenarios=scenarios)
