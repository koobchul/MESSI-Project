import jax.numpy as jnp

from project.algorithms.gae import (
    compute_gae,
)


def test_gae_without_terminal():
    rewards = jnp.array(
        [1.0, 1.0, 1.0],
        dtype=jnp.float32,
    )

    values = jnp.array(
        [0.5, 0.6, 0.7],
        dtype=jnp.float32,
    )

    bootstrap_value = jnp.array(
        0.8,
        dtype=jnp.float32,
    )

    terminated = jnp.array(
        [False, False, False]
    )

    truncated = jnp.array(
        [False, False, False]
    )

    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        terminated=terminated,
        truncated=truncated,
        gamma=0.99,
        gae_lambda=0.95,
    )

    print("\n=== GAE ===")
    print("advantages:", advantages)
    print("returns:", returns)

    expected_advantages = jnp.array(
        [
            3.0878844,
            2.1200260,
            1.0920000,
        ],
        dtype=jnp.float32,
    )

    assert advantages.shape == (3,)
    assert returns.shape == (3,)

    assert jnp.allclose(
        advantages,
        expected_advantages,
        atol=1e-5,
    )

    assert jnp.allclose(
        returns,
        advantages + values,
    )


def test_terminated_does_not_bootstrap():
    rewards = jnp.array(
        [1.0],
        dtype=jnp.float32,
    )

    values = jnp.array(
        [0.5],
        dtype=jnp.float32,
    )

    # Deliberately large.
    # It must be ignored for a true termination.
    bootstrap_value = jnp.array(
        100.0,
        dtype=jnp.float32,
    )

    terminated = jnp.array(
        [True]
    )

    truncated = jnp.array(
        [False]
    )

    advantages, returns = compute_gae(
        rewards,
        values,
        bootstrap_value,
        terminated,
        truncated,
    )

    print(
        "\nterminated advantage:",
        advantages,
    )

    assert jnp.allclose(
        advantages,
        jnp.array([0.5]),
    )

    assert jnp.allclose(
        returns,
        jnp.array([1.0]),
    )


def test_truncated_does_bootstrap():
    rewards = jnp.array(
        [1.0],
        dtype=jnp.float32,
    )

    values = jnp.array(
        [0.5],
        dtype=jnp.float32,
    )

    bootstrap_value = jnp.array(
        0.8,
        dtype=jnp.float32,
    )

    terminated = jnp.array(
        [False]
    )

    truncated = jnp.array(
        [True]
    )

    advantages, returns = compute_gae(
        rewards,
        values,
        bootstrap_value,
        terminated,
        truncated,
        gamma=0.99,
        gae_lambda=0.95,
    )

    print(
        "\ntruncated advantage:",
        advantages,
    )

    expected_advantage = (
        1.0
        + 0.99 * 0.8
        - 0.5
    )

    assert jnp.allclose(
        advantages,
        jnp.array(
            [expected_advantage]
        ),
    )

    assert jnp.allclose(
        returns,
        advantages + values,
    )