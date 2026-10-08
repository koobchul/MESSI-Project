from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.core.action import IntentAction

from project.models.action_head import ActorOutput


class SampledAction(NamedTuple):
    """
    Action sampled from the actor distribution.

    action:
        MESSI IntentAction passed to env.step().

    pre_tanh_continuous:
        Gaussian sample before tanh squashing.

        This will be needed later when computing
        the PPO continuous-action log probability.
    """

    action: IntentAction
    pre_tanh_continuous: jax.Array


def sample_action(
    actor_output: ActorOutput,
    key: jax.Array,
) -> SampledAction:
    """
    Sample one hybrid MESSI action for every agent.

    Discrete:
        categorical distribution over 6 intents.

    Continuous:
        Gaussian distribution over 7 controls,
        followed by tanh to enforce [-1, 1].
    """

    # ------------------------------------------------------------
    # 1. Split random keys
    # ------------------------------------------------------------
    intent_key, continuous_key = jax.random.split(key)

    # ------------------------------------------------------------
    # 2. Sample discrete intent
    # ------------------------------------------------------------
    intent = jax.random.categorical(
        intent_key,
        actor_output.intent_logits,
        axis=-1,
    ).astype(jnp.int32)

    # ------------------------------------------------------------
    # 3. Continuous Gaussian distribution
    # ------------------------------------------------------------
    std = jnp.exp(
        actor_output.continuous_log_std
    )

    noise = jax.random.normal(
        continuous_key,
        actor_output.continuous_mean.shape,
    )

    pre_tanh_continuous = (
        actor_output.continuous_mean
        + std * noise
    )

    # ------------------------------------------------------------
    # 4. Squash to MESSI action range [-1, 1]
    # ------------------------------------------------------------
    continuous = jnp.tanh(
        pre_tanh_continuous
    )

    # ------------------------------------------------------------
    # 5. Convert to MESSI IntentAction
    # ------------------------------------------------------------
    action = IntentAction.from_array(
        intent,
        continuous,
    )

    return SampledAction(
        action=action,
        pre_tanh_continuous=pre_tanh_continuous,
    )

def action_log_prob(
    actor_output: ActorOutput,
    intent: jax.Array,
    pre_tanh_continuous: jax.Array,
) -> jax.Array:
    """
    Compute joint log probability of the sampled hybrid action.

    Returns
    -------
    log_prob:
        Shape [batch_size]

        log pi(a | s)
        =
        categorical intent log-prob
        +
        tanh-squashed Gaussian continuous log-prob
    """

    # ============================================================
    # 1. Discrete categorical log probability
    # ============================================================
    log_probs_intent = jax.nn.log_softmax(
        actor_output.intent_logits,
        axis=-1,
    )

    intent_log_prob = jnp.take_along_axis(
        log_probs_intent,
        intent[..., None],
        axis=-1,
    ).squeeze(-1)

    # ============================================================
    # 2. Gaussian log probability before tanh
    # ============================================================
    mean = actor_output.continuous_mean
    log_std = actor_output.continuous_log_std

    std = jnp.exp(log_std)

    normalized = (
        pre_tanh_continuous - mean
    ) / std

    gaussian_log_prob = (
        -0.5 * normalized**2
        - log_std
        - 0.5 * jnp.log(2.0 * jnp.pi)
    )

    # ============================================================
    # 3. tanh change-of-variables correction
    # ============================================================
    squashed = jnp.tanh(
        pre_tanh_continuous
    )

    correction = jnp.log(
        1.0 - squashed**2 + 1e-6
    )

    continuous_log_prob = jnp.sum(
        gaussian_log_prob - correction,
        axis=-1,
    )

    # ============================================================
    # 4. Joint hybrid action log probability
    # ============================================================
    return (
        intent_log_prob
        + continuous_log_prob
    )

def action_entropy(
    actor_output: ActorOutput,
) -> jax.Array:
    """
    Compute entropy bonus for PPO.

    Returns
    -------
    entropy:
        Shape [batch_size]

    Notes
    -----
    Continuous entropy uses the underlying Gaussian entropy.
    This is a practical approximation for the tanh-squashed policy.
    """

    # ============================================================
    # 1. Categorical entropy
    # ============================================================
    log_probs = jax.nn.log_softmax(
        actor_output.intent_logits,
        axis=-1,
    )

    probs = jnp.exp(log_probs)

    intent_entropy = -jnp.sum(
        probs * log_probs,
        axis=-1,
    )

    # ============================================================
    # 2. Gaussian entropy
    # ============================================================
    log_std = actor_output.continuous_log_std

    gaussian_entropy = (
        log_std
        + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
    )

    continuous_entropy = jnp.sum(
        gaussian_entropy,
        axis=-1,
    )

    return (
        intent_entropy
        + continuous_entropy
    )