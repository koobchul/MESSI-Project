import unittest
import jax
import jax.numpy as jnp
import numpy as np
from project.envs.messi_5v5 import make_5v5_env
from project.envs.reward_v3 import control_potentials, extra_control_shaping
from project.envs.reward import compute_shaped_reward


class RewardV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env, cls.reset = make_5v5_env(seed=0)

    def state(self, distance=10., carrier=-1):
        s = self.reset.rollout.state
        positions = jnp.full_like(s.players.position, 100.)
        positions = positions.at[1].set(jnp.array([distance, 0.]))
        return s._replace(
            ball=s.ball._replace(position=jnp.array([0., 0., .11]), live=jnp.bool_(True)),
            players=s.players._replace(position=positions),
            restart=s.restart._replace(kind=jnp.int32(0)),
            possession=s.possession._replace(team=jnp.int32(0 if carrier >= 0 else -1),
                player=jnp.int32(carrier), control_ticks=jnp.int32(2 if carrier >= 0 else 0)),
        )

    def test_si_distance_and_approach_direction(self):
        far, near = self.state(10.), self.state(5.)
        self.assertAlmostEqual(float(control_potentials(far).approach), np.exp(-1), places=6)
        approach, possession = jax.jit(extra_control_shaping)(far, near, False, 0.)
        self.assertGreater(float(approach), 0.)
        self.assertEqual(float(possession), 0.)
        self.assertLess(float(extra_control_shaping(near, far, False, 0.)[0]), 0.)

    def test_no_multiplier_for_extra_chasers_or_keeper(self):
        s = self.state(5.)
        same = s._replace(players=s.players._replace(position=s.players.position.at[2].set([5., 0.])))
        keeper_near = s._replace(players=s.players._replace(position=s.players.position.at[0].set([0., 0.])))
        self.assertEqual(float(control_potentials(s).approach), float(control_potentials(same).approach))
        self.assertEqual(float(control_potentials(s).approach), float(control_potentials(keeper_near).approach))

    def test_strict_possession_geometry_height_ticks_activity_team(self):
        s = self.state(.4, 1)
        self.assertEqual(float(control_potentials(s).possession), 1.)
        cases = [self.state(9., 1), s._replace(ball=s.ball._replace(position=jnp.array([0.,0.,5.]))),
                 s._replace(possession=s.possession._replace(control_ticks=jnp.int32(0))),
                 s._replace(players=s.players._replace(on_pitch=s.players.on_pitch.at[1].set(False))),
                 s._replace(possession=s.possession._replace(team=jnp.int32(1)))]
        for invalid in cases:
            self.assertEqual(float(control_potentials(invalid).possession), 0.)

    def test_retouch_taker_excluded_until_other_touch(self):
        s = self.state(.4, 1)
        s = s._replace(restart_release=s.restart_release._replace(active=jnp.bool_(True),
            untouched=jnp.bool_(True), kind=jnp.int32(7), taker=jnp.int32(1), team=jnp.int32(0)))
        self.assertEqual(float(control_potentials(s).possession), 0.)
        self.assertLess(float(control_potentials(s).approach), .001)
        lawful = s._replace(restart_release=s.restart_release._replace(untouched=jnp.bool_(False)))
        self.assertEqual(float(control_potentials(lawful).possession), 1.)

    def test_no_extra_reward_on_restart_teleport_or_goal(self):
        far, near = self.state(20.), self.state(.4, 1)
        dead = far._replace(ball=far.ball._replace(live=jnp.bool_(False)))
        pending = far._replace(restart=far.restart._replace(kind=jnp.int32(5)))
        for a,b,goal in [(dead,near,0.),(pending,near,0.),(near,dead,0.),(far,near,1.),(near,far,-1.)]:
            np.testing.assert_array_equal(extra_control_shaping(a,b,False,goal), (0.,0.))

    def test_acquisition_loss_no_per_tick_possession_bonus(self):
        loose, retained = self.state(.4), self.state(.4, 1)
        self.assertGreater(float(extra_control_shaping(loose, retained, False, 0.)[1]), 0.)
        self.assertLess(float(extra_control_shaping(retained, loose, False, 0.)[1]), 0.)
        self.assertLessEqual(float(extra_control_shaping(retained, retained, False, 0.)[1]), 0.)

    def test_terminal_zero_and_finite_gradient(self):
        s = self.state(.4, 1)
        a,p = extra_control_shaping(s,s,True,0.)
        self.assertAlmostEqual(float(p), -.05, places=6)
        grad = jax.grad(lambda x: control_potentials(self.state(x)).approach)(jnp.float32(5.))
        self.assertTrue(np.isfinite(grad))
        self.assertLess(float(grad), 0.)

    def test_zero_extra_matches_v2_and_v2_sparse_unchanged(self):
        a,b = self.state(10.), self.state(.4,1)
        np.testing.assert_array_equal(extra_control_shaping(a,b,False,0.,approach_coefficient=0.,possession_coefficient=0.), (0.,0.))
        ra,rb=self.reset.rollout._replace(state=a),self.reset.rollout._replace(state=b)
        components=compute_shaped_reward(0.,self.env.global_state_view(ra),self.env.global_state_view(rb),False,coefficient=0.)
        self.assertEqual(float(components.total_reward),0.)


if __name__ == '__main__':
    unittest.main()
