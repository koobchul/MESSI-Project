import unittest
import numpy as np
import jax
import jax.numpy as jnp
from project.models.intent_guard import apply_retouch_guard, retouch_prohibited_from_mlp
from project.models.action_head import ActorOutput, HybridActionHead
from project.models.action_sampling import sample_action, action_log_prob, action_entropy
from project.models.mlp_actor import MLPActorBackbone
from project.algorithms.update import actor_loss_with_params
from project.envs.obs_encoder import encode_mlp_observation
from project.envs.messi_5v5 import make_5v5_env


def synthetic_obs(players=10):
    # Independent encoder-documented layout: player30 + context34 + categories80.
    x=np.zeros((2,players*30+114),dtype=np.float32)
    # Observer 0 is slot3; observer1 is slot2. Slot3 is the release taker.
    p=x[:,:players*30].reshape(2,players,30)
    p[0,3,27]=1;p[1,2,27]=1;p[:,3,18]=1
    c=players*30
    x[:,c+13]=1;x[:,c+18]=1;x[:,c+19]=1
    k=c+34
    x[:,k+44]=1  # current restart NONE
    x[:,k+56+7]=1  # restart release OFFSIDE
    return jnp.asarray(x)


class GuardTests(unittest.TestCase):
    def test_layout_and_release_conditions(self):
        for players in (10,22):
            obs=synthetic_obs(players)
            np.testing.assert_array_equal(retouch_prohibited_from_mlp(obs),[True,False])
            c=players*30
            for index in (c+13,c+18,c+19):
                np.testing.assert_array_equal(retouch_prohibited_from_mlp(obs.at[:,index].set(0)),[False,False])
            np.testing.assert_array_equal(retouch_prohibited_from_mlp(obs.at[:,c+34+44].set(0)),[False,False])
            held=obs.at[:,c+34+56+7].set(0).at[:,c+34+56+8].set(1)
            np.testing.assert_array_equal(retouch_prohibited_from_mlp(held),[False,False])

    def test_mask_sampling_entropy_and_finite_gradient(self):
        obs=synthetic_obs();logits=jnp.array([[0.,1.,2.,3.,4.,5.]]*2)
        raw=ActorOutput(logits,jnp.zeros((2,7)),jnp.full((2,7),-.5))
        guarded=jax.jit(apply_retouch_guard)(raw,obs)
        np.testing.assert_array_equal(guarded.intent_logits[1],raw.intent_logits[1])
        np.testing.assert_array_equal(apply_retouch_guard(raw,obs,False).intent_logits,raw.intent_logits)
        for k in range(20):self.assertEqual(int(sample_action(guarded,jax.random.key(k)).action.intent[0]),0)
        self.assertTrue(np.all(np.isfinite(action_entropy(guarded))))
        grad=jax.grad(lambda z:jnp.sum(action_entropy(apply_retouch_guard(raw._replace(intent_logits=z),obs))))(logits)
        self.assertTrue(np.all(np.isfinite(grad)))
        np.testing.assert_allclose(grad[0],0,atol=1e-6)

    def test_ppo_ratio_matches_rollout_including_mixed_guard_flags(self):
        obs=jnp.stack([synthetic_obs(),synthetic_obs()]);flags=jnp.array([True,False])
        backbone=MLPActorBackbone(hidden_dim=16);head=HybridActionHead()
        bv=backbone.init(jax.random.key(1),obs);latent=backbone.apply(bv,obs)
        hv=head.init(jax.random.key(2),latent)
        params={'backbone':bv['params'],'head':hv['params']}
        raw=head.apply(hv,latent);output=apply_retouch_guard(raw,obs,flags)
        sampled=sample_action(output,jax.random.key(3))
        old=action_log_prob(output,sampled.action.intent,sampled.pre_tanh_continuous)
        loss,metrics=actor_loss_with_params(params,backbone,head,obs,sampled.action.intent,sampled.pre_tanh_continuous,old,jnp.array([1.,-1.]),retouch_guard_enabled=flags)
        self.assertAlmostEqual(float(metrics.approx_kl),0.,places=6)
        self.assertAlmostEqual(float(metrics.clip_fraction),0.,places=6)
        gradients=jax.grad(lambda p:actor_loss_with_params(p,backbone,head,obs,sampled.action.intent,sampled.pre_tanh_continuous,old,jnp.array([1.,-1.]),retouch_guard_enabled=flags)[0])(params)
        self.assertTrue(all(np.all(np.isfinite(a)) for a in jax.tree_util.tree_leaves(gradients)))

    def test_matches_real_environment_encoder(self):
        env,reset=make_5v5_env(seed=1001);s=reset.rollout.state
        s=s._replace(ball=s.ball._replace(live=jnp.bool_(True)),restart=s.restart._replace(kind=jnp.int32(0)),restart_release=s.restart_release._replace(active=jnp.bool_(True),untouched=jnp.bool_(True),kind=jnp.int32(7),team=jnp.int32(0),taker=jnp.int32(3)))
        r=reset.rollout._replace(state=s);obs=env.observe_all(r);roster=env.roster_metadata(r)
        flat=encode_mlp_observation(obs,roster)
        expected=np.array([False,False,False,True,False,False,False,False,False,False])
        np.testing.assert_array_equal(retouch_prohibited_from_mlp(flat),expected)
        released=obs._replace(restart_release=obs.restart_release._replace(untouched=jnp.zeros_like(obs.restart_release.untouched)))
        self.assertFalse(bool(jnp.any(retouch_prohibited_from_mlp(encode_mlp_observation(released,roster)))))

    def test_evaluator_counts_and_guard_switch(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from project.algorithms.rollout_jit import JITRolloutCarry
        from footballworld.core.action import IntentAction
        from project.evaluate_retouch_guard import make_evaluator, METRICS
        env, reset = make_5v5_env(seed=1001)
        s = reset.rollout.state
        s = s._replace(ball=s.ball._replace(live=jnp.bool_(True)), restart=s.restart._replace(kind=jnp.int32(0)), restart_release=s.restart_release._replace(active=jnp.bool_(True),untouched=jnp.bool_(True),kind=jnp.int32(7),team=jnp.int32(0),taker=jnp.int32(3)))
        r=reset.rollout._replace(state=s)
        roster=env.roster_metadata(r);si_roster=env.roster_metadata_si(r)
        class FakeEnv:
            def observe_all(self,r): return env.observe_all(r)
            def observe_all_si(self,r): return env.observe_all_si(r)
            def step_with_events(self,r,setup,action,key):
                violation=action.intent[3]!=0
                actor=jnp.where(violation,3,-1).reshape(1)
                event=SimpleNamespace(actor=actor,intent=jnp.ones(1,dtype=jnp.int32),outcome=jnp.full(1,2,dtype=jnp.int32))
                retouch=SimpleNamespace(occurred=violation.reshape(1),team=jnp.ones(1,dtype=jnp.int32),offender=actor)
                return SimpleNamespace(rollout=r._replace(state=r.state._replace(control_tick=r.state.control_tick+1)),events=SimpleNamespace(retouch=retouch,deliberate_contact=event),terminated=jnp.bool_(False),truncated=jnp.bool_(False),done=jnp.bool_(False),restart_opened=violation)
        class Backbone:
            def apply(self,p,x): return x
        class Head:
            def apply(self,p,x):
                logits=jnp.zeros(x.shape[:-1]+(6,)).at[...,1].set(100.)
                return ActorOutput(logits,jnp.zeros(x.shape[:-1]+(7,)),jnp.zeros(x.shape[:-1]+(7,)))
        class Opponent:
            def step(self,obs,roster,state,key): return SimpleNamespace(action=IntentAction.from_array(jnp.zeros(10,dtype=jnp.int32),jnp.zeros((10,7))),state=state)
        carry=JITRolloutCarry(r,jnp.int32(0),jax.random.key(1),jnp.bool_(False),jnp.bool_(False),jnp.bool_(False))
        with patch('project.evaluate_retouch_guard.make_rule_based_policy',return_value=Opponent()):
            evaluator=make_evaluator(FakeEnv(),Backbone(),Head(),1)
        params={'backbone':{},'head':{}}
        _,off=evaluator(reset.setup,carry,roster,si_roster,params,jax.random.key(2),jnp.bool_(False))
        _,on=evaluator(reset.setup,carry,roster,si_roster,params,jax.random.key(2),jnp.bool_(True))
        off=dict(zip(METRICS,np.asarray(off)));on=dict(zip(METRICS,np.asarray(on)))
        self.assertEqual(off['retouch_team0'],1);self.assertEqual(on['retouch_team0'],0)
        self.assertEqual(off['nonmove_requests_in_guard_context'],1);self.assertEqual(on['nonmove_requests_in_guard_context'],0)
        self.assertEqual(off['control_traps_with_same_substep_retouch_team0'],1)
        self.assertEqual(on['control_traps_team0'],0)


if __name__=='__main__':unittest.main()
