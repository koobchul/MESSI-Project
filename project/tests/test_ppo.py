import jax.numpy as jnp

from project.algorithms.ppo import (
    ppo_actor_loss,
    ppo_critic_loss,
)


def test_ppo_same_policy_ratio_is_one():
    # ============================================================
    # Same old/new policy
    # ============================================================
    old_log_prob = jnp.array(
        [
            [-1.0, -2.0],
            [-1.5, -2.5],
        ],
        dtype=jnp.float32,
    )

    new_log_prob = old_log_prob

    advantages = jnp.array(
        [
            1.0,
            -0.5,
        ],
        dtype=jnp.float32,
    )

    entropy = jnp.ones(
        (2, 2),
        dtype=jnp.float32,
    )

    metrics = ppo_actor_loss(
        old_log_prob=old_log_prob,
        new_log_prob=new_log_prob,
        team_advantages=advantages,
        entropy=entropy,
        clip_eps=0.2,
        entropy_coef=0.01,
    )

    print("\n=== Same Policy PPO ===")
    print("loss:", metrics.loss)
    print("policy loss:", metrics.policy_loss)
    print("entropy:", metrics.entropy)
    print("approx KL:", metrics.approx_kl)
    print("clip fraction:", metrics.clip_fraction)

    # Same policy:
    # ratio = exp(0) = 1
    assert jnp.allclose(
        metrics.approx_kl,
        0.0,
    )

    assert jnp.allclose(
        metrics.clip_fraction,
        0.0,
    )


def test_ppo_clipping():
    # ============================================================
    # Force ratio = 1.5
    # ============================================================
    old_log_prob = jnp.zeros(
        (1, 2),
        dtype=jnp.float32,
    )

    new_log_prob = jnp.full(
        (1, 2),
        jnp.log(1.5),
        dtype=jnp.float32,
    )

    advantages = jnp.array(
        [1.0],
        dtype=jnp.float32,
    )

    entropy = jnp.zeros(
        (1, 2),
        dtype=jnp.float32,
    )

    metrics = ppo_actor_loss(
        old_log_prob=old_log_prob,
        new_log_prob=new_log_prob,
        team_advantages=advantages,
        entropy=entropy,
        clip_eps=0.2,
        entropy_coef=0.0,
    )

    print("\n=== PPO Clipping ===")
    print("policy loss:", metrics.policy_loss)
    print("clip fraction:", metrics.clip_fraction)

    # ratio = 1.5
    # clipped ratio = 1.2
    #
    # positive advantage:
    # min(1.5, 1.2) = 1.2
    #
    # loss = -1.2
    assert jnp.allclose(
        metrics.policy_loss,
        -1.2,
        atol=1e-6,
    )

    assert jnp.allclose(
        metrics.clip_fraction,
        1.0,
    )


def test_ppo_critic_loss():
    values = jnp.array(
        [
            0.2,
            0.4,
            0.6,
        ],
        dtype=jnp.float32,
    )

    returns = jnp.array(
        [
            0.3,
            0.5,
            0.7,
        ],
        dtype=jnp.float32,
    )

    loss = ppo_critic_loss(
        values,
        returns,
    )

    print("\n=== Critic Loss ===")
    print("critic loss:", loss)

    expected = 0.5 * (
        0.1**2
        + 0.1**2
        + 0.1**2
    ) / 3.0

    assert jnp.allclose(
        loss,
        expected,
        atol=1e-6,
    )