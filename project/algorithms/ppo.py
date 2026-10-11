#ppo.py

from typing import NamedTuple

import jax.numpy as jnp


class PPOActorMetrics(NamedTuple):
    loss: jnp.ndarray
    policy_loss: jnp.ndarray
    entropy: jnp.ndarray
    approx_kl: jnp.ndarray
    clip_fraction: jnp.ndarray


def _expand_mask(
    valid_mask,
    target,
):
    """
    Expand a timestep-level mask so that it can be broadcast
    to the shape of the target array.

    Examples
    --------
    valid_mask:
        [T]

    target:
        [T]
            -> mask [T]

        [T, N_agents]
            -> mask [T, N_agents]
    """

    mask = jnp.asarray(
        valid_mask,
        dtype=jnp.bool_,
    )

    # Add singleton dimensions until mask ndim matches target ndim.
    while mask.ndim < target.ndim:
        mask = mask[..., None]

    return jnp.broadcast_to(
        mask,
        target.shape,
    )


def masked_mean(
    values,
    valid_mask,
):
    """
    Compute the mean using only valid rollout entries.

    Parameters
    ----------
    values:
        [T] or [T, ...]

    valid_mask:
        [T]

        True:
            real environment transition

        False:
            padding row
    """

    values = jnp.asarray(
        values,
        dtype=jnp.float32,
    )

    mask = _expand_mask(
        valid_mask,
        values,
    )

    # IMPORTANT:
    #
    # Use jnp.where rather than values * mask.
    #
    # If an invalid padding row accidentally contains inf/nan,
    # expressions such as inf * 0 can still produce nan.
    #
    # jnp.where explicitly replaces invalid entries with zero.
    safe_values = jnp.where(
        mask,
        values,
        jnp.float32(0.0),
    )

    count = jnp.sum(
        mask.astype(
            jnp.float32
        )
    )

    count = jnp.maximum(
        count,
        jnp.float32(1.0),
    )

    return (
        jnp.sum(
            safe_values
        )
        / count
    )


def ppo_actor_loss(
    old_log_prob,
    new_log_prob,
    team_advantages,
    entropy,
    clip_eps=0.2,
    entropy_coef=0.01,
    valid_mask=None,
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

        The same team advantage is shared by all controlled
        agents at that timestep.

    entropy:
        [T, N_agents]

    valid_mask:
        [T] or None

        None:
            Original PPO behavior.
            Every timestep contributes to the loss.

        [T]:
            Only real rollout transitions contribute.

            Padding generated after a terminal/truncated
            transition is completely ignored.
    """

    # ============================================================
    # 1. Broadcast centralized team advantage to agents
    # ============================================================

    advantages = jnp.broadcast_to(
        team_advantages[:, None],
        old_log_prob.shape,
    )

    # ============================================================
    # 2. PPO probability ratio
    #
    # ratio = pi_new(a|s) / pi_old(a|s)
    #
    #       = exp(
    #           log pi_new(a|s)
    #           - log pi_old(a|s)
    #         )
    # ============================================================

    log_ratio = (
        new_log_prob
        - old_log_prob
    )

    ratio = jnp.exp(
        log_ratio
    )

    # ============================================================
    # 3. Unclipped PPO surrogate
    # ============================================================

    surrogate_1 = (
        ratio
        * advantages
    )

    # ============================================================
    # 4. Clipped PPO surrogate
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

    surrogate = jnp.minimum(
        surrogate_1,
        surrogate_2,
    )

    # ============================================================
    # 5. Diagnostics per sample
    # ============================================================

    approx_kl_per_sample = (
        (ratio - 1.0)
        - log_ratio
    )

    clipped = (
        jnp.abs(
            ratio - 1.0
        )
        > clip_eps
    ).astype(
        jnp.float32
    )

    # ============================================================
    # 6. Reduction
    #
    # Without valid_mask:
    #     original behavior
    #
    # With valid_mask:
    #     padding rows do not contribute to any PPO statistic
    # ============================================================

    if valid_mask is None:
        policy_loss = -jnp.mean(
            surrogate
        )

        mean_entropy = jnp.mean(
            entropy
        )

        approx_kl = jnp.mean(
            approx_kl_per_sample
        )

        clip_fraction = jnp.mean(
            clipped
        )

    else:
        policy_loss = -masked_mean(
            surrogate,
            valid_mask,
        )

        mean_entropy = masked_mean(
            entropy,
            valid_mask,
        )

        approx_kl = masked_mean(
            approx_kl_per_sample,
            valid_mask,
        )

        clip_fraction = masked_mean(
            clipped,
            valid_mask,
        )

    # ============================================================
    # 7. Total actor loss
    #
    # Minimize:
    #
    #     policy_loss
    #     - entropy_coef * entropy
    # ============================================================

    loss = (
        policy_loss
        - entropy_coef
        * mean_entropy
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
    valid_mask=None,
):
    """
    Mean-squared centralized critic loss.

    Parameters
    ----------
    values:
        [T]

    returns:
        [T]

    valid_mask:
        [T] or None

        When supplied, padding rows are ignored.
    """

    squared_error = jnp.square(
        returns - values
    )

    # ============================================================
    # Original behavior
    # ============================================================

    if valid_mask is None:
        return (
            0.5
            * jnp.mean(
                squared_error
            )
        )

    # ============================================================
    # Terminal-safe fixed-length rollout behavior
    # ============================================================

    return (
        0.5
        * masked_mean(
            squared_error,
            valid_mask,
        )
    )


def normalize_advantages(
    advantages,
    eps=1e-8,
    valid_mask=None,
):
    """
    Normalize team-level advantages over rollout time.

    Parameters
    ----------
    advantages:
        [T]

    eps:
        Numerical stability constant.

    valid_mask:
        [T] or None

        None:
            Original behavior.

        [T]:
            Mean and standard deviation are computed only
            from real environment transitions.

            Padding rows are returned as exactly zero.
    """

    advantages = jnp.asarray(
        advantages,
        dtype=jnp.float32,
    )

    # ============================================================
    # 1. Original behavior
    # ============================================================

    if valid_mask is None:
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

    # ============================================================
    # 2. Masked normalization
    # ============================================================

    valid_mask = jnp.asarray(
        valid_mask,
        dtype=jnp.bool_,
    )

    mask = valid_mask.astype(
        jnp.float32
    )

    count = jnp.sum(
        mask
    )

    count = jnp.maximum(
        count,
        jnp.float32(1.0),
    )

    # ------------------------------------------------------------
    # Mean over real rollout rows only
    # ------------------------------------------------------------

    mean = (
        jnp.sum(
            advantages
            * mask
        )
        / count
    )

    # ------------------------------------------------------------
    # Variance over real rollout rows only
    # ------------------------------------------------------------

    variance = (
        jnp.sum(
            jnp.square(
                advantages - mean
            )
            * mask
        )
        / count
    )

    std = jnp.sqrt(
        variance
    )

    normalized = (
        advantages - mean
    ) / (
        std + eps
    )

    # ------------------------------------------------------------
    # Padding must remain exactly zero.
    # ------------------------------------------------------------

    normalized = jnp.where(
        valid_mask,
        normalized,
        jnp.float32(0.0),
    )

    return normalized