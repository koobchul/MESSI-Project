"""Small integration tests; shortened matches are NOT fulltime performance evidence."""
import tempfile
import unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from footballworld import FootballWorld, MatchConfig
from project.envs.messi_5v5 import make_5v5_rosters
from project.train import initialize_models
from project.algorithms.rollout_jit import initialize_jit_rollout_carry
from project.evaluate_fulltime import make_evaluator, METRICS
from project.attack_common import completion, atomic_pickle


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = FootballWorld(match=MatchConfig(halftime_seconds=1., fulltime_seconds=2.,
                                  maximum_added_time_seconds_per_period=1., minimum_team_players=(5,5)))
        cls.reset = cls.env.reset(*make_5v5_rosters(), key=jax.random.key(123))

    def test_completion_labels(self):
        e, s = self.env, self.reset.rollout.state
        full, half = e.match.clock_ticks(e.timebase)
        self.assertEqual(completion(e,s,False)['terminal_reason'], 'external_cutoff')
        final = s._replace(control_tick=jnp.int32(full), dead_ball_control_ticks=jnp.int32(0),
                           restart=s.restart._replace(kind=jnp.int32(0)), first_half_wall_end_tick=jnp.int32(half))
        self.assertTrue(completion(e, final, True)['full_duration_complete'])
        limit = final._replace(control_tick=jnp.int32(full + e.match.maximum_added_time_ticks(e.timebase)),
                               dead_ball_control_ticks=jnp.int32(full))
        self.assertEqual(completion(e, limit, True)['terminal_reason'], 'added_time_limit_reached')
        self.assertFalse(completion(e, limit, True)['full_duration_complete'])
        abandoned = final._replace(players=final.players._replace(sent_off=final.players.sent_off.at[0].set(True)))
        self.assertFalse(completion(e, abandoned, True)['scheduled_match_finished'])

    def test_real_match_halftime_absorption_and_resume(self):
        e,r = self.env,self.reset
        backbone,head,_,params,_ = initialize_models(e,r)
        collect = make_evaluator(e,backbone,head,128,True)
        carry = initialize_jit_rollout_carry(e,r,jax.random.key(888))
        roster,si = e.roster_metadata(r.rollout),e.roster_metadata_si(r.rollout)
        end,counts = collect(r.setup,carry,roster,si,params,jax.random.key(777),jnp.bool_(True))
        self.assertTrue(bool(end.done))
        result=completion(e,end.rollout.state,True)
        self.assertTrue(result['halftime_reached'])
        self.assertTrue(result['scheduled_match_finished'])
        self.assertGreater(int(counts[0]),0)
        frozen,zeros=collect(r.setup,end,roster,si,params,jax.random.key(777),jnp.bool_(True))
        self.assertEqual(int(frozen.rollout.state.control_tick),int(end.rollout.state.control_tick))
        self.assertEqual(int(jnp.sum(zeros)),0)
        import pickle
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state.pkl'; atomic_pickle(p,jax.device_get(carry))
            restored=jax.device_put(pickle.loads(p.read_bytes()))
            again,c2=collect(r.setup,restored,roster,si,params,jax.random.key(777),jnp.bool_(True))
            np.testing.assert_array_equal(counts,c2)
            np.testing.assert_array_equal(end.rollout.state.score,again.rollout.state.score)


if __name__ == '__main__': unittest.main()
