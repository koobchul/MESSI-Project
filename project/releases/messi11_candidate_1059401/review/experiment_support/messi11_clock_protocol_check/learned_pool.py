"""Frozen internal MLP opponents; no claim of student-policy compatibility."""
import os,pickle,json,hashlib
from pathlib import Path
import jax,jax.numpy as jnp
MODE=os.environ.get('MESSI_POOL_MODE','rule_only')
MANIFEST=Path(os.environ['MESSI_POOL_MANIFEST'])
POOL=None;EVAL_FIXED=None

def set_eval(index):
 global EVAL_FIXED
 EVAL_FIXED=index

def canonical_away(obs,xp=jnp):
 # MLP flattens absolute roster order. AWAY must see its own11 slots first.
 players=obs[...,:660].reshape(obs.shape[:-1]+(22,30))
 players=xp.concatenate([players[...,11:22,:],players[...,:11,:]],axis=-2)
 return xp.concatenate([players.reshape(obs.shape[:-1]+(660,)),obs[...,660:]],axis=-1)

def away_view(state):
 # Policy-side view only. The authoritative engine state is NEVER replaced.
 return state._replace(players=state.players._replace(team_id=1-state.players.team_id),attack_direction=state.attack_direction[::-1])

def load_pool():
 global POOL
 if POOL is None:
  from flax.core import unfreeze
  entries=json.loads(MANIFEST.read_text())['opponents'];params=[]
  for r in entries:
   p=Path(r['checkpoint']);raw=p.read_bytes()
   if hashlib.sha256(raw).hexdigest()!=r['file_sha256']:raise RuntimeError('Frozen opponent checkpoint changed')
   params.append(jax.device_put(unfreeze(pickle.loads(raw)['actor_params'])))
  shapes=[jax.tree_util.tree_map(lambda x:x.shape,p) for p in params]
  if not all(x==shapes[0] for x in shapes):raise RuntimeError('Opponent schemas differ; explicit adapter required')
  POOL=jax.tree_util.tree_map(lambda *xs:jnp.stack(xs),*params)
 return POOL

def make_controller(env,backbone,head):
 from project.models.role_actor import actor_forward
 from project.models.intent_guard import apply_retouch_guard
 from project.models.action_sampling import sample_action
 from project.envs.policy_obs import encode_policy_observation
 from project.models.field_movement import field_movement
 from project.models.gk_movement import limit_gk_move
 mode=MODE;fixed=EVAL_FIXED
 pool=load_pool()
 def action(rollout,roster,rule_action,match_key,tick):
  # IDs0=rule,1..3=frozen policies. Draw fixed per match, not per tick.
  idx=(jnp.int32(fixed) if fixed is not None else
       (jnp.int32(0) if mode=='rule_only' else jax.random.randint(jax.random.fold_in(match_key,43210),(),0,4)))
  def learned(_):
   params=jax.tree_util.tree_map(lambda x:jax.lax.dynamic_index_in_dim(x,jnp.maximum(idx-1,0),keepdims=False),pool)
   obs=canonical_away(encode_policy_observation(env,rollout,roster))
   output=apply_retouch_guard(actor_forward(params,backbone,head,obs),obs,enabled=True)
   a=sample_action(output,jax.random.fold_in(jax.random.fold_in(match_key,54321),tick)).action
   virtual=away_view(rollout.state)
   a=field_movement(virtual,a,rule_action)
   a,_,_=limit_gk_move(env,virtual,a)
   away=roster.team_id==1
   return jax.tree_util.tree_map(lambda x,y:jnp.where(away.reshape(away.shape+(1,)*(x.ndim-1)),x,y),a,rule_action)
  return jax.lax.cond(idx==0,lambda _:rule_action,learned,operand=None)
 return action
