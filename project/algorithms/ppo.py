from typing import NamedTuple

import jax.numpy as jnp


class PPOActorMetrics(NamedTuple):
    loss: jnp.ndarray
    policy_loss: jnp.ndarray
    entropy: jnp.ndarray
    approx_kl: jnp.ndarray
    clip_fraction: jnp.ndarray


def ppo_actor_loss(
    old_log_prob,
    new_log_prob,
    team_advantages,
    entropy,
    clip_eps=0.2,
    entropy_coef=0.01,
):
    """
    PPO clipped actor objective for the shared multi-agent policy.

    Parameters
    ----------
    old_log_prob:
        [T, N_agents]

    new_log_prob:
        [T, N_agents]

    team_advantages:
        [T]

        One centralized team advantage for each timestep.
        The same advantage is shared by all controlled agents
        at that timestep.

    entropy:
        [T, N_agents]
    """

    # ============================================================
    # 1. Broadcast team advantage to agents
    # ============================================================
    advantages = jnp.broadcast_to(
        team_advantages[:, None],
        old_log_prob.shape,
    )

    # ============================================================
    # 2. PPO probability ratio
    #
    # ratio = pi_new(a|s) / pi_old(a|s)
    #       = exp(log pi_new - log pi_old)
    # ============================================================
    log_ratio = (
        new_log_prob
        - old_log_prob
    )

    ratio = jnp.exp(
        log_ratio
    )

    # ============================================================
    # 3. Unclipped objective
    # ============================================================
    surrogate_1 = (
        ratio
        * advantages
    )

    # ============================================================
    # 4. Clipped objective
    # ============================================================
    clipped_ratio = jnp.clip(
        ratio,
        1.0 - clip_eps,
        1.0 + clip_eps,
    )

    surrogate_2 = (
        clipped_ratio
        * advantages
    )

    # PPO maximizes the minimum surrogate.
    # Optimizers minimize losses, so apply minus sign.
    policy_loss = -jnp.mean(
        jnp.minimum(
            surrogate_1,
            surrogate_2,
        )
    )

    # ============================================================
    # 5. Entropy bonus
    # ============================================================
    mean_entropy = jnp.mean(
        entropy
    )

    loss = (
        policy_loss
        - entropy_coef
        * mean_entropy
    )

    # ============================================================
    # 6. Diagnostics
    # ============================================================
    approx_kl = jnp.mean(
        (ratio - 1.0)
        - log_ratio
    )

    clip_fraction = jnp.mean(
        (
            jnp.abs(
                ratio - 1.0
            )
            > clip_eps
        ).astype(
            jnp.float32
        )
    )

    return PPOActorMetrics(
        loss=loss,
        policy_loss=policy_loss,
        entropy=mean_entropy,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
    )


def ppo_critic_loss(
    values,
    returns,
):
    """
    Mean-squared centralized value loss.

    values:
        [T]

    returns:
        [T]
    """

    return 0.5 * jnp.mean(
        jnp.square(
            returns - values
        )
    )

def normalize_advantages(
    advantages,
    eps=1e-8,
):
    """
    Normalize team-level advantages over rollout time.
    """

    mean = jnp.mean(
        advantages
    )

    std = jnp.std(
        advantages
    )

    return (
        advantages - mean
    ) / (
        std + eps
    )