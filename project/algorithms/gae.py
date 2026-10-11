#gae.py

import jax
import jax.numpy as jnp

def compute_gae(
    rewards,
    values,
    bootstrap_value,
    terminated,
    truncated,
    gamma=0.99,
    gae_lambda=0.95,
):
    """
    Compute Generalized Advantage Estimation (GAE).

    Parameters
    ----------
    rewards:
        Shape [T]

    values:
        Critic predictions V(s_t).
        Shape [T]

    bootstrap_value:
        V(s_T), the value of the state after the final
        collected transition.

    terminated:
        Shape [T].
        True when the episode reached a genuine terminal state.

    truncated:
        Shape [T].
        True when the episode was cut off without reaching
        a genuine terminal state.

    Returns
    -------
    advantages:
        Shape [T]

    returns:
        Shape [T]
        Critic regression target:
            returns = advantages + values
    """

    rewards = jnp.asarray(
        rewards,
        dtype=jnp.float32,
    )

    values = jnp.asarray(
        values,
        dtype=jnp.float32,
    )

    bootstrap_value = jnp.asarray(
        bootstrap_value,
        dtype=jnp.float32,
    )

    terminated = jnp.asarray(
        terminated,
        dtype=jnp.bool_,
    )

    truncated = jnp.asarray(
        truncated,
        dtype=jnp.bool_,
    )

    done = (
        terminated
        | truncated
    )

    advantages = jnp.zeros_like(
        rewards
    )

    gae = jnp.array(
        0.0,
        dtype=jnp.float32,
    )

    # ============================================================
    # Backward GAE recursion
    # ============================================================
    for t in reversed(
        range(rewards.shape[0])
    ):
        if t == rewards.shape[0] - 1:
            next_value = bootstrap_value
        else:
            next_value = values[t + 1]

        # --------------------------------------------------------
        # Genuine terminal:
        # no value bootstrap.
        #
        # Truncation:
        # bootstrap is still allowed.
        # --------------------------------------------------------
        bootstrap_mask = (
            1.0
            - terminated[t].astype(
                jnp.float32
            )
        )

        delta = (
            rewards[t]
            + gamma
            * bootstrap_mask
            * next_value
            - values[t]
        )

        # --------------------------------------------------------
        # Do not connect GAE across an episode boundary.
        # --------------------------------------------------------
        continuation_mask = (
            1.0
            - done[t].astype(
                jnp.float32
            )
        )

        gae = (
            delta
            + gamma
            * gae_lambda
            * continuation_mask
            * gae
        )

        advantages = advantages.at[
            t
        ].set(
            gae
        )

    returns = (
        advantages
        + values
    )

    return advantages, returns

def compute_masked_gae(
    rewards,
    values,
    bootstrap_value,
    terminated,
    truncated,
    valid_mask,
    gamma=0.99,
    gae_lambda=0.95,
):
    """
    Compute GAE for a fixed-length rollout containing padding rows.

    Parameters
    ----------
    rewards:
        [T]

    values:
        [T]

    bootstrap_value:
        scalar V(s_T) evaluated at the final real rollout state.

    terminated:
        [T] bool

        True terminal:
            do NOT bootstrap.

    truncated:
        [T] bool

        Time-limit truncation:
            bootstrap from the final state,
            but stop GAE recursion across the boundary.

    valid_mask:
        [T] bool

        True:
            real environment transition

        False:
            padding row generated after the match ended.

    Returns
    -------
    advantages:
        [T]

        Padding rows are exactly zero.

    returns:
        [T]

        Padding rows are exactly zero.
    """

    rewards = jnp.asarray(
        rewards,
        dtype=jnp.float32,
    )

    values = jnp.asarray(
        values,
        dtype=jnp.float32,
    )

    bootstrap_value = jnp.asarray(
        bootstrap_value,
        dtype=jnp.float32,
    )

    terminated = jnp.asarray(
        terminated,
        dtype=jnp.bool_,
    )

    truncated = jnp.asarray(
        truncated,
        dtype=jnp.bool_,
    )

    valid_mask = jnp.asarray(
        valid_mask,
        dtype=jnp.bool_,
    )

    # ============================================================
    # Reverse-time GAE scan
    #
    # Important:
    #
    # Invalid padding rows DO NOT modify next_value.
    #
    # Therefore, with:
    #
    # valid = [T, T, F, F]
    #
    # the second real transition sees bootstrap_value rather than
    # values[2] from the padding row.
    # ============================================================

    def scan_step(
        carry,
        inputs,
    ):
        next_advantage, next_value = carry

        (
            reward,
            value,
            is_terminated,
            is_truncated,
            is_valid,
        ) = inputs

        done = (
            is_terminated
            | is_truncated
        )

        # --------------------------------------------------------
        # Real transition
        # --------------------------------------------------------
        def valid_step(_):
            # True termination:
            # no bootstrap.
            #
            # Truncation:
            # bootstrap is allowed.
            bootstrap_mask = (
                1.0
                - is_terminated.astype(
                    jnp.float32
                )
            )

            # Do not propagate GAE recursively across either
            # termination or truncation.
            continuation_mask = (
                1.0
                - done.astype(
                    jnp.float32
                )
            )

            delta = (
                reward
                + gamma
                * bootstrap_mask
                * next_value
                - value
            )

            advantage = (
                delta
                + gamma
                * gae_lambda
                * continuation_mask
                * next_advantage
            )

            next_carry = (
                advantage,
                value,
            )

            return (
                next_carry,
                advantage,
            )

        # --------------------------------------------------------
        # Padding row
        #
        # Preserve next_value / next_advantage.
        #
        # This is what allows the last REAL row to use the actual
        # bootstrap_value even when padding follows it.
        # --------------------------------------------------------
        def padding_step(_):
            advantage = jnp.float32(
                0.0
            )

            return (
                (
                    next_advantage,
                    next_value,
                ),
                advantage,
            )

        return jax.lax.cond(
            is_valid,
            valid_step,
            padding_step,
            operand=None,
        )

    # ============================================================
    # Reverse scan
    # ============================================================

    (
        _,
        advantages_reversed,
    ) = jax.lax.scan(
        scan_step,
        (
            jnp.float32(0.0),
            bootstrap_value,
        ),
        (
            rewards[::-1],
            values[::-1],
            terminated[::-1],
            truncated[::-1],
            valid_mask[::-1],
        ),
    )

    advantages = (
        advantages_reversed[::-1]
    )

    returns = jnp.where(
        valid_mask,
        advantages + values,
        jnp.float32(0.0),
    )

    return (
        advantages,
        returns,
    )