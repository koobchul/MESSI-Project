"""A conservative restart-retouch guard derived only from player observations.

Only explicit non-MOVE requests by an untouched set-piece taker are masked.
Distances, speeds, ordinary open-play choices and continuous controls are not
changed. MOVE can still cause a passive contact; this is not a guarantee of
zero retouch events. The environment remains authoritative.
"""

import jax.numpy as jnp

from footballworld.core.constants import RESTART_COUNT, RK_NONE, RK_GK_HOLD
from footballworld.core.contact import (
    INTENT_MOVE, MECHANISM_COUNT, INTENT_COUNT, INTENT_SOURCE_COUNT,
    OUTCOME_COUNT, LAW11_EFFECT_COUNT,
)
from project.envs.obs_encoder import (
    PLAYER_FEATURE_DIM, CONTEXT_FEATURE_DIM, CATEGORICAL_FEATURE_DIM,
)

GUARD_SCHEMA = "restart-retouch/1"


def retouch_prohibited_from_mlp(actor_obs):
    """Return [...], using the encoder layout for either 5v5 or 11v11.

All inspected fields are boolean or one-hot, so SI conversion is unnecessary.
The slot marked is_self is used instead of assuming observer row == slot.
"""
    feature_count = actor_obs.shape[-1]
    player_flat = feature_count - CONTEXT_FEATURE_DIM - CATEGORICAL_FEATURE_DIM
    if player_flat <= 0 or player_flat % PLAYER_FEATURE_DIM:
        raise ValueError("Actor observation does not match the MLP encoder layout.")
    player_count = player_flat // PLAYER_FEATURE_DIM
    players = actor_obs[..., :player_flat].reshape(
        actor_obs.shape[:-1] + (player_count, PLAYER_FEATURE_DIM)
    )
    self_is_release_taker = jnp.any(
        (players[..., 27] > 0.5) & (players[..., 18] > 0.5), axis=-1
    )
    context = actor_obs[..., player_flat:player_flat + CONTEXT_FEATURE_DIM]
    categorical_start = player_flat + CONTEXT_FEATURE_DIM
    restart_offset = (
        3 + MECHANISM_COUNT + INTENT_COUNT + INTENT_SOURCE_COUNT
        + OUTCOME_COUNT + RESTART_COUNT + LAW11_EFFECT_COUNT
    )
    release_offset = restart_offset + RESTART_COUNT + 3
    restart_none = actor_obs[..., categorical_start + restart_offset + RK_NONE] > 0.5
    # These are the seven set pieces checked by detect_restart_retouch().
    set_piece = jnp.any(
        actor_obs[..., categorical_start + release_offset + 1:
                  categorical_start + release_offset + RK_GK_HOLD] > 0.5,
        axis=-1,
    )
    return (
        self_is_release_taker & set_piece & restart_none
        & (context[..., 13] > 0.5)  # ball.live
        & (context[..., 18] > 0.5)  # restart_release.active
        & (context[..., 19] > 0.5)  # restart_release.untouched
    )


def apply_retouch_guard(actor_output, actor_obs, enabled=True):
    """Use the same distribution for sampling, log-probability and entropy.

enabled may be a scalar or rollout flags [T] for observations [T, N, D].
Finite negative logits avoid 0 * -inf in the categorical entropy gradient.
"""
    if actor_output.intent_logits.shape[-1] != 6:
        raise ValueError("Expected the six MESSI action intents.")
    blocked = retouch_prohibited_from_mlp(actor_obs)
    enabled = jnp.asarray(enabled, dtype=jnp.bool_)
    while enabled.ndim < blocked.ndim:
        enabled = enabled[..., None]
    blocked = blocked & enabled
    forbidden = blocked[..., None] & (jnp.arange(6) != INTENT_MOVE)
    logits = jnp.where(
        forbidden, jnp.asarray(-1e9, dtype=actor_output.intent_logits.dtype),
        actor_output.intent_logits,
    )
    return actor_output._replace(intent_logits=logits)
