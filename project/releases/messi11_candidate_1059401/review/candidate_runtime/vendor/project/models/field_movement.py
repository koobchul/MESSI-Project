"""Rule formation movement + bounded raw PPO residual, shared train/eval."""
import os
import jax.numpy as jnp
MODE=os.environ.get('MESSI_MOVEMENT_MODE','formation')
RESIDUAL=.15
CONTACT_MODE=os.environ.get('MESSI_CONTACT_MODE','raw')
if CONTACT_MODE not in ('raw','settle'):raise ValueError(CONTACT_MODE)
if MODE not in ('original','formation','rule_move'):raise ValueError(MODE)
def field_movement(state,action,rule_action):
    if MODE=='original':return action
    field=(state.players.team_id==0)&~state.players.is_goalkeeper&state.players.active
    residual=RESIDUAL*action.move if MODE=='formation' else jnp.zeros_like(action.move)
    residual=jnp.where(state.restart.kind==0,residual,0.)
    move=jnp.clip(rule_action.move+residual,-1.,1.)
    action=action._replace(move=jnp.where(field[:,None],move,action.move))
    if CONTACT_MODE=='raw':return action
    # Only where BOTH policies request CONTROL; never manufacture an intent.
    enabled=field&(action.intent==1)&(rule_action.intent==1)&state.ball.live&(state.restart.kind==0)
    def residual(base,raw):
        mapped=jnp.clip(base+.05*raw,-1.,1.)
        mask=enabled.reshape(enabled.shape+(1,)*(raw.ndim-1))
        return jnp.where(mask,mapped,raw)
    return action._replace(force_to_ball=residual(rule_action.force_to_ball,action.force_to_ball),
        launch=residual(rule_action.launch,action.launch),spin=residual(rule_action.spin,action.spin))
