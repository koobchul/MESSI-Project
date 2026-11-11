import jax
import jax.numpy as jnp
from footballworld.environment.action_availability import intent_availability_hint
from project.envs.obs_encoder import encode_mlp_observation
from project.envs.messi_11v11 import ROLES

def encode_policy_observation(env, rollout, roster):
    obs=encode_mlp_observation(env.observe_all(rollout),roster)
    if obs.shape[0] != 22: raise ValueError("11v11 requires 22 roster slots")
    roles=jax.nn.one_hot(jnp.asarray(ROLES+ROLES),4)
    hint=intent_availability_hint(env.observe_all_si(rollout),env.roster_metadata_si(rollout),
        ball=env.ball,stadium=env.stadium,reach=env.reach,scale=env.action_scale,body=env.body)
    return jnp.concatenate([obs,roles,hint.astype(jnp.float32)],axis=-1)
