import jax.numpy as jnp

from project.algorithms.ppo import (
    normalize_advantages,
    ppo_actor_loss,
    ppo_critic_loss,
)


def test_masked_advantage_normalization():
    advantages = jnp.array(
        [
            1.0,
            3.0,
            1000.0,
            -1000.0,
        ],
        dtype=jnp.float32,
    )

    valid_mask = jnp.array(
        [
            True,
            True,
            False,
            False,
        ]
    )

    normalized = normalize_advantages(
        advantages,
        valid_mask=valid_mask,
    )

    print(
        "\n=== Masked Advantage Normalization ==="
    )

    print(
        "normalized:",
        normalized,
    )

    expected = jnp.array(
        [
            -1.0,
            1.0,
            0.0,
            0.0,
        ],
        dtype=jnp.float32,
    )

    assert jnp.allclose(
        normalized,
        expected,
        atol=1e-6,
    )


def test_masked_actor_loss_ignores_padding():
    # ============================================================
    # T = 4
    # N_agents = 2
    #
    # Only first two timesteps are real.
    #
    # Padding rows deliberately contain absurd values.
    # They must have zero effect on the loss.
    # ============================================================

    old_log_prob = jnp.zeros(
        (
            4,
            2,
        ),
        dtype=jnp.float32,
    )

    new_log_prob = jnp.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],

            # padding garbage
            [10.0, -10.0],
            [20.0, -20.0],
        ],
        dtype=jnp.float32,
    )

    team_advantages = jnp.array(
        [
            1.0,
            -1.0,

            # padding garbage
            1000.0,
            -1000.0,
        ],
        dtype=jnp.float32,
    )

    entropy = jnp.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],

            # padding garbage
            [100.0, 100.0],
            [100.0, 100.0],
        ],
        dtype=jnp.float32,
    )

    valid_mask = jnp.array(
        [
            True,
            True,
            False,
            False,
        ]
    )

    metrics = ppo_actor_loss(
        old_log_prob=old_log_prob,
        new_log_prob=new_log_prob,
        team_advantages=team_advantages,
        entropy=entropy,
        clip_eps=0.2,
        entropy_coef=0.01,
        valid_mask=valid_mask,
    )

    print(
        "\n=== Masked Actor PPO ==="
    )

    print(
        "loss:",
        metrics.loss,
    )

    print(
        "policy loss:",
        metrics.policy_loss,
    )

    print(
        "entropy:",
        metrics.entropy,
    )

    print(
        "KL:",
        metrics.approx_kl,
    )

    print(
        "clip fraction:",
        metrics.clip_fraction,
    )

    # ratio=1 on all REAL entries.
    #
    # advantages:
    # +1, +1, -1, -1 across two agents
    #
    # mean surrogate = 0
    assert jnp.allclose(
        metrics.policy_loss,
        0.0,
        atol=1e-6,
    )

    # Only real entropy=1 should count.
    assert jnp.allclose(
        metrics.entropy,
        1.0,
        atol=1e-6,
    )

    # Real log probabilities are unchanged.
    assert jnp.allclose(
        metrics.approx_kl,
        0.0,
        atol=1e-6,
    )

    assert jnp.allclose(
        metrics.clip_fraction,
        0.0,
        atol=1e-6,
    )

    # total = 0 - 0.01 * 1
    assert jnp.allclose(
        metrics.loss,
        -0.01,
        atol=1e-6,
    )


def test_masked_critic_loss_ignores_padding():
    returns = jnp.array(
        [
            1.0,
            2.0,

            # padding garbage
            1000.0,
            -1000.0,
        ],
        dtype=jnp.float32,
    )

    values = jnp.array(
        [
            1.5,
            1.5,

            # padding garbage
            -1000.0,
            1000.0,
        ],
        dtype=jnp.float32,
    )

    valid_mask = jnp.array(
        [
            True,
            True,
            False,
            False,
        ]
    )

    loss = ppo_critic_loss(
        returns=returns,
        values=values,
        valid_mask=valid_mask,
    )

    print(
        "\n=== Masked Critic PPO ==="
    )

    print(
        "critic loss:",
        loss,
    )

    # Real errors:
    #
    # (1 - 1.5)^2 = 0.25
    # (2 - 1.5)^2 = 0.25
    #
    # mean = 0.25
    # PPO critic factor = 0.5
    #
    # expected = 0.125
    assert jnp.allclose(
        loss,
        0.125,
        atol=1e-6,
    )