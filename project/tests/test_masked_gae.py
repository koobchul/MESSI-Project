import jax.numpy as jnp

from project.algorithms.gae import (
    compute_masked_gae,
)


def test_masked_gae_with_truncation_and_padding():
    # ============================================================
    # Fixed T = 4
    #
    # t=0 : real
    # t=1 : real + truncated
    # t=2 : padding
    # t=3 : padding
    # ============================================================

    rewards = jnp.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=jnp.float32,
    )

    values = jnp.array(
        [
            1.0,
            2.0,
            0.0,
            0.0,
        ],
        dtype=jnp.float32,
    )

    # V(s_2)
    #
    # Because t=1 is truncated rather than terminated,
    # this value MUST be used for t=1's TD residual.
    bootstrap_value = jnp.float32(
        3.0
    )

    terminated = jnp.array(
        [
            False,
            False,
            False,
            False,
        ]
    )

    truncated = jnp.array(
        [
            False,
            True,
            False,
            False,
        ]
    )

    valid_mask = jnp.array(
        [
            True,
            True,
            False,
            False,
        ]
    )

    advantages, returns = compute_masked_gae(
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        terminated=terminated,
        truncated=truncated,
        valid_mask=valid_mask,
        gamma=0.99,
        gae_lambda=0.95,
    )

    print(
        "\n=== Masked GAE ==="
    )

    print(
        "advantages:",
        advantages,
    )

    print(
        "returns:",
        returns,
    )

    # ============================================================
    # Expected t=1
    #
    # truncated=True:
    #
    # delta_1
    # = 0 + 0.99 * 3 - 2
    # = 0.97
    #
    # No recursive continuation across truncation.
    # ============================================================

    expected_a1 = jnp.float32(
        0.97
    )

    # ============================================================
    # Expected t=0
    #
    # delta_0
    # = 0 + 0.99 * 2 - 1
    # = 0.98
    #
    # A_0
    # = delta_0
    #   + 0.99 * 0.95 * A_1
    # ============================================================

    expected_a0 = (
        jnp.float32(0.98)
        + jnp.float32(0.99)
        * jnp.float32(0.95)
        * expected_a1
    )

    expected_advantages = jnp.array(
        [
            expected_a0,
            expected_a1,
            0.0,
            0.0,
        ],
        dtype=jnp.float32,
    )

    expected_returns = jnp.array(
        [
            expected_a0 + 1.0,
            expected_a1 + 2.0,
            0.0,
            0.0,
        ],
        dtype=jnp.float32,
    )

    assert jnp.allclose(
        advantages,
        expected_advantages,
        atol=1e-6,
    )

    assert jnp.allclose(
        returns,
        expected_returns,
        atol=1e-6,
    )


def test_masked_gae_true_terminal_does_not_bootstrap():
    # ============================================================
    # True termination must ignore bootstrap_value.
    # ============================================================

    rewards = jnp.array(
        [
            0.5,
            0.0,
        ],
        dtype=jnp.float32,
    )

    values = jnp.array(
        [
            0.2,
            0.0,
        ],
        dtype=jnp.float32,
    )

    bootstrap_value = jnp.float32(
        100.0
    )

    terminated = jnp.array(
        [
            True,
            False,
        ]
    )

    truncated = jnp.array(
        [
            False,
            False,
        ]
    )

    valid_mask = jnp.array(
        [
            True,
            False,
        ]
    )

    advantages, returns = compute_masked_gae(
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        terminated=terminated,
        truncated=truncated,
        valid_mask=valid_mask,
        gamma=0.99,
        gae_lambda=0.95,
    )

    print(
        "\n=== Masked GAE True Terminal ==="
    )

    print(
        "advantages:",
        advantages,
    )

    print(
        "returns:",
        returns,
    )

    # 0.5 - 0.2 = 0.3
    assert jnp.allclose(
        advantages,
        jnp.array(
            [
                0.3,
                0.0,
            ],
            dtype=jnp.float32,
        ),
        atol=1e-6,
    )

    assert jnp.allclose(
        returns,
        jnp.array(
            [
                0.5,
                0.0,
            ],
            dtype=jnp.float32,
        ),
        atol=1e-6,
    )