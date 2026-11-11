import jax,jax.numpy as jnp,numpy as np
from footballworld.core.action import IntentAction
from footballworld.dynamics.action import decode_physics_action
from project.models.field_movement import field_movement,MODE

def check_movement(env,reset):
 state=reset.rollout.state
 raw=IntentAction.from_array(jnp.zeros(22,dtype=jnp.int32),jnp.zeros((22,7)))
 rule=raw._replace(move=jnp.tile(jnp.array([.3,-.4]),(22,1)))
 mapped=field_movement(state,raw,rule)
 # HOME GK and all AWAY slots must bypass the field controller.
 np.testing.assert_array_equal(np.asarray(mapped.move)[[0,*range(11,22)]],np.asarray(raw.move)[[0,*range(11,22)]])
 for k in ('intent','force_to_ball','launch','spin'):np.testing.assert_array_equal(np.asarray(getattr(mapped,k)),np.asarray(getattr(raw,k)))
 if MODE!='original':np.testing.assert_array_equal(np.asarray(mapped.move)[1:11],np.asarray(rule.move)[1:11])
 reflected=state._replace(attack_direction=-state.attack_direction,players=state.players._replace(position=-state.players.position,velocity=-state.players.velocity))
 a=field_movement(reflected,raw,rule)
 np.testing.assert_allclose(np.asarray(decode_physics_action(reflected,a).desired_velocity),-np.asarray(decode_physics_action(state,mapped).desired_velocity),atol=1e-6)
 assert np.max(np.abs(np.asarray(mapped.move)))<=1
 return dict(mode=MODE,gk_and_away_bypass=True,contact_components_unchanged=True,zero_residual_recovers_rule_movement=MODE!='original',world_direction_rotated_once=True)
