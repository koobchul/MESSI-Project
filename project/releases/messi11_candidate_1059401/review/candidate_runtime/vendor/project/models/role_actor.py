"""Independent actor parameter trees; identical routing in rollout/update/eval."""
import os
import jax
CONTROL_PRIOR=float(os.environ.get("MESSI_CONTROL_PRIOR","0"))
import jax.numpy as jnp

def actor_forward(params, backbone, head, obs):
    def forward(p):
        return head.apply({'params':p['head']},backbone.apply({'params':p['backbone']},obs))
    field=forward(params['field']);gk=forward(params['gk'])
    field=field._replace(intent_logits=field.intent_logits.at[...,1].add(CONTROL_PRIOR*(obs[...,-5]>.5)))
    mask=obs[..., -10]>.5
    return jax.tree_util.tree_map(lambda f,g:jnp.where(mask[...,None],g,f),field,gk)
