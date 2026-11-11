"""Keep coefficient0.02; remove credit for loose-ball shot spam."""
import os
import jax.numpy as jnp

def scale_values(values,factor,shot_factor=1.):
 return values.at[1].set(values[1]*factor).at[4].set(values[4]*shot_factor)

def install(factor,shot_factor=1.):
 if factor!=.1 or shot_factor!=1.:raise ValueError('Predeclared reward only')
 import project.envs.reward11 as module
 from footballworld.dynamics.contact_predicates import verified_controlled_carrier
 from footballworld.core.contact import INTENT_SHOT,INTENT_SOURCE_POLICY,OUTCOME_RELEASE
 approach_scale=float(os.environ.get('MESSI_APPROACH_SCALE','1'))
 if approach_scale not in (1.,.1):raise ValueError('Unexpected approach scale')
 original=module.reward11
 def reward11(env,state,result,gamma=.995):
  values=scale_values(original(env,state,result,gamma),factor)
  values=values.at[0].set(values[0]*approach_scale)
  c=verified_controlled_carrier(state,env.ball,env.reach,env.action_scale)
  events=result.events.deliberate_contact
  from shot_credit import eligible
  gated=eligible(module.possession(env,state),state.possession.control_ticks,c,events.actor,events.intent,events.intent_source,events.outcome,INTENT_SHOT,INTENT_SOURCE_POLICY,OUTCOME_RELEASE)
  return values.at[4].set(jnp.where(gated,values[4],0.))
 module.reward11=reward11
 module.CONFIG.update(context_approach_output_scale=approach_scale,effective_context_approach=.1*approach_scale,effective_field_acquire=.02,effective_useful_shot=.02,
  shot_credit='Existing useful-shot conditions AND pre-step verified lawful field possession>=3control_ticks AND release by same carrier; no request credit',field_possession_potential_output_scale=.1)
