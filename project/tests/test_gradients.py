#test_gradients.py

import jax
import jax.numpy as jnp

from project.algorithms.update import (
    actor_loss_with_params,
    critic_loss_with_params,
)

from project.models.action_head import (
    HybridActionHead,
)

from project.models.action_sampling import (
    action_log_prob,
    sample_action,
)

from project.models.critic import (
    CentralizedCritic,
)

from project.models.mlp_actor import (
    MLPActorBackbone,
)


def tree_l2_norm(tree):
    leaves = jax.tree_util.tree_leaves(
        tree
    )

    return jnp.sqrt(
        sum(
            jnp.sum(
                jnp.square(x)
            )
            for x in leaves
        )
    )


def tree_all_finite(tree):
    leaves = jax.tree_util.tree_leaves(
        tree
    )

    return all(
        bool(
            jnp.all(
                jnp.isfinite(x)
            )
        )
        for x in leaves
    )


def test_actor_and_critic_gradients():
    # ============================================================
    # 1. Dummy MAPPO batch dimensions
    # ============================================================
    T = 4
    N = 5

    # ============================================================
    # 2. Actor observations
    # ============================================================
    actor_obs = jax.random.normal(
        jax.random.key(0),
        (
            T,
            N,
            414,
        ),
    )

    # ============================================================
    # 3. Actor models
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

    # ============================================================
    # 4. Simulate stored rollout actions
    # ============================================================
    sampled = sample_action(
        actor_output,
        jax.random.key(3),
    )

    intent = sampled.action.intent

    pre_tanh_continuous = (
        sampled.pre_tanh_continuous
    )

    old_log_prob = action_log_prob(
        actor_output,
        intent,
        pre_tanh_continuous,
    )

    # Treat rollout data as fixed data.
    old_log_prob = jax.lax.stop_gradient(
        old_log_prob
    )

    intent = jax.lax.stop_gradient(
        intent
    )

    pre_tanh_continuous = (
        jax.lax.stop_gradient(
            pre_tanh_continuous
        )
    )

    # Team-level advantages
    advantages = jnp.array(
        [
            1.0,
            -0.5,
            0.75,
            -1.25,
        ],
        dtype=jnp.float32,
    )

    # ============================================================
    # 5. Combine actor parameters
    # ============================================================
    actor_params = {
        "backbone":
            backbone_variables["params"],

        "head":
            head_variables["params"],
    }

    # ============================================================
    # 6. Actor gradient
    # ============================================================
    def actor_loss_fn(params):
        return actor_loss_with_params(
            actor_params=params,

            backbone=backbone,
            action_head=action_head,

            actor_obs=actor_obs,

            intent=intent,

            pre_tanh_continuous=(
                pre_tanh_continuous
            ),

            old_log_prob=old_log_prob,

            advantages=advantages,

            clip_eps=0.2,
            entropy_coef=0.01,
        )

    (
        actor_loss,
        actor_metrics,
    ), actor_grads = jax.value_and_grad(
        actor_loss_fn,
        has_aux=True,
    )(
        actor_params
    )

    actor_grad_norm = tree_l2_norm(
        actor_grads
    )

    print("\n=== Actor Gradient ===")

    print(
        "actor loss:",
        actor_loss,
    )

    print(
        "policy loss:",
        actor_metrics.policy_loss,
    )

    print(
        "entropy:",
        actor_metrics.entropy,
    )

    print(
        "approx KL:",
        actor_metrics.approx_kl,
    )

    print(
        "clip fraction:",
        actor_metrics.clip_fraction,
    )

    print(
        "gradient norm:",
        actor_grad_norm,
    )

        # ============================================================
    # 6-B. Verify PPO policy gradient without entropy bonus
    # ============================================================

    def pure_policy_loss_fn(params):
        loss, metrics = actor_loss_with_params(
            actor_params=params,

            backbone=backbone,
            action_head=action_head,

            actor_obs=actor_obs,

            intent=intent,

            pre_tanh_continuous=(
                pre_tanh_continuous
            ),

            old_log_prob=old_log_prob,

            advantages=advantages,

            clip_eps=0.2,

            # Disable entropy bonus.
            entropy_coef=0.0,
        )

        return loss, metrics

    (
        pure_policy_loss,
        pure_policy_metrics,
    ), pure_policy_grads = jax.value_and_grad(
        pure_policy_loss_fn,
        has_aux=True,
    )(
        actor_params
    )

    pure_policy_grad_norm = tree_l2_norm(
        pure_policy_grads
    )

    print("\n=== Pure PPO Policy Gradient ===")

    print(
        "policy loss:",
        pure_policy_metrics.policy_loss,
    )

    print(
        "entropy coefficient:",
        0.0,
    )

    print(
        "gradient norm:",
        pure_policy_grad_norm,
    )

    assert tree_all_finite(
        pure_policy_grads
    )

    assert pure_policy_grad_norm > 0.0

    # ============================================================
    # 7. Critic
    # ============================================================
    critic_obs = jax.random.normal(
        jax.random.key(4),
        (
            T,
            411,
        ),
    )

    critic = CentralizedCritic(
        hidden_dim=256,
    )

    critic_variables = critic.init(
        jax.random.key(5),
        critic_obs,
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
    # 8. Critic gradient
    # ============================================================
    def critic_loss_fn(params):
        return critic_loss_with_params(
            critic_params=params,
            critic=critic,
            critic_obs=critic_obs,
            returns=returns,
        )

    critic_loss, critic_grads = (
        jax.value_and_grad(
            critic_loss_fn
        )(
            critic_variables[
                "params"
            ]
        )
    )

    critic_grad_norm = tree_l2_norm(
        critic_grads
    )

    print("\n=== Critic Gradient ===")

    print(
        "critic loss:",
        critic_loss,
    )

    print(
        "gradient norm:",
        critic_grad_norm,
    )

    # ============================================================
    # 9. Assertions
    # ============================================================
    assert jnp.isfinite(
        actor_loss
    )

    assert jnp.isfinite(
        critic_loss
    )

    assert tree_all_finite(
        actor_grads
    )

    assert tree_all_finite(
        critic_grads
    )

    assert actor_grad_norm > 0.0
    assert critic_grad_norm > 0.0