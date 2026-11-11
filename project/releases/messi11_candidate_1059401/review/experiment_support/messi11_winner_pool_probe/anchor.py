"""Analytic KL(current || frozen initialization); no behavior cloning labels."""
def install_anchor(e,coefficient):
 import jax,jax.numpy as jnp
 import project.algorithms.update as update
 from project.models.role_actor import actor_forward
 from project.models.intent_guard import apply_retouch_guard
 reference=jax.tree_util.tree_map(jax.lax.stop_gradient,e.initial)
 original=update.actor_loss_with_params
 def wrapped(*args,**kw):
  loss,metrics=original(*args,**kw)
  if coefficient==0:return loss,metrics
  obs=kw['actor_obs']
  current=apply_retouch_guard(actor_forward(kw['actor_params'],kw['backbone'],kw['action_head'],obs),obs,enabled=kw['retouch_guard_enabled'])
  target=apply_retouch_guard(actor_forward(reference,kw['backbone'],kw['action_head'],obs),obs,enabled=kw['retouch_guard_enabled'])
  logp=jax.nn.log_softmax(current.intent_logits,-1);logq=jax.nn.log_softmax(target.intent_logits,-1)
  categorical=jnp.sum(jnp.exp(logp)*(logp-logq),axis=-1)
  l=current.continuous_log_std;r=target.continuous_log_std
  gaussian=jnp.sum(r-l+.5*(jnp.exp(2*(l-r))+(current.continuous_mean-target.continuous_mean)**2*jnp.exp(-2*r)-1),axis=-1)
  kl=categorical+gaussian
  players=obs[...,:660].reshape(obs.shape[:-1]+(22,30))
  active=jnp.any((players[...,27]>.5)&(players[...,13]>.5)&(players[...,14]<.5),axis=-1)
  valid=active if kw['valid_mask'] is None else active&kw['valid_mask'][...,None]
  gk=obs[...,-10]>.5
  def mean(mask):return jnp.sum(jnp.where(mask,kl,0.))/jnp.maximum(mask.sum(),1)
  penalty=.5*(mean(valid&gk)+mean(valid&~gk))
  total=loss+coefficient*penalty
  return total,metrics._replace(loss=total)
 update.actor_loss_with_params=wrapped
