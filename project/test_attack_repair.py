import unittest
import numpy as np
import jax,jax.numpy as jnp
from footballworld import FootballWorld,MatchConfig
from footballworld.core.action import IntentAction
from footballworld.dynamics.action import decode_physics_action
from project.envs.messi_5v5 import make_5v5_rosters
from project.envs.obs_encoder import encode_mlp_observation
from project.models.intent_guard import retouch_prohibited_from_mlp
from project.models.mlp_actor import MLPActorBackbone
from project.models.action_head import HybridActionHead
from project.algorithms.rollout_jit import initialize_jit_rollout_carry
from project.attack_repair_core import own_team_first,repair_loss,coverage_ok
from project.repair_attack_bc import make_collector


class RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env=FootballWorld(match=MatchConfig(halftime_seconds=1,fulltime_seconds=2,maximum_added_time_seconds_per_period=1,minimum_team_players=(5,5)))
        cls.r=cls.env.reset(*make_5v5_rosters(),key=jax.random.key(0))

    def test_move_uses_body_aim_in_actual_decoder(self):
        c=jnp.zeros((10,7)).at[:,2].set(1.)
        action=IntentAction.from_array(jnp.zeros(10,jnp.int32),c)
        physics=decode_physics_action(self.r.rollout.state,action)
        self.assertTrue(np.asarray(physics.body_target_valid).all())
        np.testing.assert_array_equal(physics.desired_velocity,np.zeros((10,2)))
        np.testing.assert_array_equal(physics.body_target[:,0],self.r.rollout.state.attack_direction[self.r.rollout.state.players.team_id])

    def test_move_body_force_gradient_is_supervised(self):
        b,h=MLPActorBackbone(hidden_dim=256),HybridActionHead()
        roster=self.env.roster_metadata(self.r.rollout)
        x=encode_mlp_observation(self.env.observe_all(self.r.rollout),roster)
        bp=b.init(jax.random.key(1),x)['params'];hp=h.init(jax.random.key(2),b.apply({'params':bp},x))['params']
        p={'backbone':bp,'head':hp};target=jnp.zeros((10,7)).at[:,2].set(1.)
        g=jax.grad(repair_loss)(p,x,jnp.zeros(10,jnp.int32),target,b,h)
        bias=np.asarray(g['head']['continuous_mean_head']['bias'])
        self.assertGreater(abs(bias[2]),1e-6)
        np.testing.assert_array_equal(bias[4:],np.zeros(3))

    def test_team_canonicalization_and_teacher_terminal(self):
        e,r=self.env,self.r;roster=e.roster_metadata(r.rollout);si=e.roster_metadata_si(r.rollout)
        raw=encode_mlp_observation(e.observe_all(r.rollout),roster)
        can=own_team_first(raw,roster.team_id)
        np.testing.assert_array_equal(can[:5],raw[:5])
        entities=np.asarray(can[:,:300]).reshape(10,10,30)
        self.assertTrue((entities[:,:5,29]==0).all())
        self.assertTrue((entities[:,5:,29]==1).all())
        np.testing.assert_array_equal(retouch_prohibited_from_mlp(raw),retouch_prohibited_from_mlp(can))
        c=initialize_jit_rollout_carry(e,r,jax.random.key(40))
        collect=make_collector(e)
        end,(x,y,a,v)=collect(r.setup,c,roster,si)
        self.assertTrue(bool(end.done));self.assertTrue(np.isfinite(x).all());self.assertGreater(int(v[:,5:].sum()),0)
        _,(_,_,_,v2)=collect(r.setup,end,roster,si);self.assertEqual(int(v2.sum()),0)

    def test_missing_shots_block_promotion(self):
        data=(np.zeros((1000,414)),np.ones(1000,dtype=int),np.zeros((1000,7)))
        self.assertFalse(coverage_ok(data,data))


if __name__=='__main__':unittest.main()
