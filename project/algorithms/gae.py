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