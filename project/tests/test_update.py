import jax
import jax.numpy as jnp

from project.algorithms.buffer import Transition
from project.algorithms.optim import make_optimizer
from project.algorithms.ppo import normalize_advantages
from project.algorithms.update import ppo_update_step

from project.models.action_head import HybridActionHead
from project.models.action_sampling import (
    action_log_prob,
    sample_action,
)
from project.models.critic import CentralizedCritic
from project.models.mlp_actor import MLPActorBackbone


def tree_l2_distance(tree_a, tree_b):
    leaves_a = jax.tree_util.tree_leaves(tree_a)
    leaves_b = jax.tree_util.tree_leaves(tree_b)

    return jnp.sqrt(
        sum(
            jnp.sum(jnp.square(a - b))
            for a, b in zip(
                leaves_a,
                leaves_b,
            )
        )
    )


def tree_all_finite(tree):
    return all(
        bool(jnp.all(jnp.isfinite(x)))
        for x in jax.tree_util.tree_leaves(tree)
    )


def test_normalize_advantages():
    advantages = jnp.array(
        [1.0, -0.5, 0.75, -1.25],
        dtype=jnp.float32,
    )

    normalized = normalize_advantages(
        advantages
    )

    print("\n=== Advantage Normalization ===")
    print("raw       :", advantages)
    print("normalized:", normalized)
    print("mean      :", jnp.mean(normalized))
    print("std       :", jnp.std(normalized))

    assert jnp.all(
        jnp.isfinite(normalized)
    )

    assert jnp.allclose(
        jnp.mean(normalized),
        0.0,
        atol=1e-6,
    )

    assert jnp.allclose(
        jnp.std(normalized),
        1.0,
        atol=1e-5,
    )


def test_ppo_update_step():
    # ============================================================
    # 1. Dummy rollout dimensions
    # ============================================================
    T = 4
    N = 5

    # ============================================================
    # 2. Actor observations
    # ============================================================
    actor_obs = jax.random.normal(
        jax.random.key(0),
        (T, N, 414),
    )

    # ============================================================
    # 3. Actor initialization
    # ============================================================
    backbone = MLPActorBackbone(
        hidden_dim=256,
    )

    backbone_variables = backbone.init(
        jax.random.key(1),
        actor_obs,
    )

    latent = backbone.apply(
        backbone_variables,
        actor_obs,
    )

    action_head = HybridActionHead()

    head_variables = action_head.init(
        jax.random.key(2),
        latent,
    )

    actor_output = action_head.apply(
        head_variables,
        latent,
    )

    actor_params = {
        "backbone":
            backbone_variables["params"],

        "head":
            head_variables["params"],
    }

    # ============================================================
    # 4. Simulated rollout actions
    # ============================================================
    sampled = sample_action(
        actor_output,
        jax.random.key(3),
    )

    intent = jax.lax.stop_gradient(
        sampled.action.intent
    )

    pre_tanh_continuous = jax.lax.stop_gradient(
        sampled.pre_tanh_continuous
    )

    old_log_prob = action_log_prob(
        actor_output,
        intent,
        pre_tanh_continuous,
    )

    old_log_prob = jax.lax.stop_gradient(
        old_log_prob
    )

    # ============================================================
    # 5. Critic
    # ============================================================
    critic_obs = jax.random.normal(
        jax.random.key(4),
        (T, 411),
    )

    critic = CentralizedCritic(
        hidden_dim=256,
    )

    critic_variables = critic.init(
        jax.random.key(5),
        critic_obs,
    )

    critic_params = critic_variables[
        "params"
    ]

    # ============================================================
    # 6. Dummy Transition batch
    # ============================================================
    batch = Transition(
        actor_obs=actor_obs,

        critic_obs=critic_obs,

        intent=intent,

        pre_tanh_continuous=(
            pre_tanh_continuous
        ),

        log_prob=old_log_prob,

        value=jnp.zeros(
            (T,),
            dtype=jnp.float32,
        ),

        reward=jnp.zeros(
            (T,),
            dtype=jnp.float32,
        ),

        terminated=jnp.zeros(
            (T,),
            dtype=jnp.bool_,
        ),

        truncated=jnp.zeros(
            (T,),
            dtype=jnp.bool_,
        ),

        done=jnp.zeros(
            (T,),
            dtype=jnp.bool_,
        ),
    )

    # ============================================================
    # 7. GAE-like targets
    # ============================================================
    advantages = jnp.array(
        [
            1.0,
            -0.5,
            0.75,
            -1.25,
        ],
        dtype=jnp.float32,
    )

    returns = jnp.array(
        [
            0.5,
            -0.2,
            1.0,
            0.3,
        ],
        dtype=jnp.float32,
    )

    # ============================================================
    # 8. Optimizers
    # ============================================================
    actor_optimizer = make_optimizer(
        learning_rate=3e-4,
        max_grad_norm=0.5,
    )

    critic_optimizer = make_optimizer(
        learning_rate=3e-4,
        max_grad_norm=0.5,
    )

    actor_opt_state = actor_optimizer.init(
        actor_params
    )

    critic_opt_state = critic_optimizer.init(
        critic_params
    )

    # ============================================================
    # 9. One unified PPO update
    # ============================================================
    (
        new_actor_params,
        new_critic_params,
        new_actor_opt_state,
        new_critic_opt_state,
        metrics,
    ) = ppo_update_step(
        actor_params=actor_params,
        critic_params=critic_params,

        actor_opt_state=actor_opt_state,
        critic_opt_state=critic_opt_state,

        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,

        backbone=backbone,
        action_head=action_head,
        critic=critic,

        batch=batch,

        advantages=advantages,
        returns=returns,

        clip_eps=0.2,
        entropy_coef=0.01,
    )

    # ============================================================
    # 10. Parameter changes
    # ============================================================
    actor_change = tree_l2_distance(
        actor_params,
        new_actor_params,
    )

    critic_change = tree_l2_distance(
        critic_params,
        new_critic_params,
    )

    print("\n=== Unified PPO Update ===")

    print(
        "actor loss:",
        metrics.actor_loss,
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
        "critic loss:",
        metrics.critic_loss,
    )

    print(
        "post-update KL:",
        metrics.approx_kl,
    )

    print(
        "post-update clip fraction:",
        metrics.clip_fraction,
    )

    print(
        "actor parameter change:",
        actor_change,
    )

    print(
        "critic parameter change:",
        critic_change,
    )

    # ============================================================
    # 11. Assertions
    # ============================================================
    assert jnp.isfinite(
        metrics.actor_loss
    )

    assert jnp.isfinite(
        metrics.critic_loss
    )

    assert jnp.isfinite(
        metrics.approx_kl
    )

    assert jnp.isfinite(
        metrics.clip_fraction
    )

    assert (
        0.0
        <= metrics.clip_fraction
        <= 1.0
    )

    assert tree_all_finite(
        new_actor_params
    )

    assert tree_all_finite(
        new_critic_params
    )

    assert actor_change > 0.0
    assert critic_change > 0.0