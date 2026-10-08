import jax
import jax.numpy as jnp
import optax

from project.algorithms.optim import (
    make_optimizer,
)

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


def tree_l2_distance(
    tree_a,
    tree_b,
):
    leaves_a = jax.tree_util.tree_leaves(
        tree_a
    )

    leaves_b = jax.tree_util.tree_leaves(
        tree_b
    )

    return jnp.sqrt(
        sum(
            jnp.sum(
                jnp.square(
                    a - b
                )
            )
            for a, b in zip(
                leaves_a,
                leaves_b,
            )
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


def test_optimizer_updates_actor_and_critic():
    # ============================================================
    # 1. Dummy MAPPO batch
    # ============================================================
    T = 4
    N = 5

    actor_obs = jax.random.normal(
        jax.random.key(0),
        (
            T,
            N,
            414,
        ),
    )

    # ============================================================
    # 2. Actor initialization
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
    # 3. Stored rollout action
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
    # 4. Actor loss + gradient
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
        actor_loss_before,
        actor_metrics_before,
    ), actor_grads = jax.value_and_grad(
        actor_loss_fn,
        has_aux=True,
    )(
        actor_params
    )

    # ============================================================
    # 5. Actor optimizer step
    # ============================================================
    actor_optimizer = make_optimizer(
        learning_rate=3e-4,
        max_grad_norm=0.5,
    )

    actor_opt_state = actor_optimizer.init(
        actor_params
    )

    actor_updates, actor_opt_state = (
        actor_optimizer.update(
            actor_grads,
            actor_opt_state,
            actor_params,
        )
    )

    new_actor_params = optax.apply_updates(
        actor_params,
        actor_updates,
    )

    actor_param_change = tree_l2_distance(
        actor_params,
        new_actor_params,
    )

    (
        actor_loss_after,
        actor_metrics_after,
    ) = actor_loss_fn(
        new_actor_params
    )

    print("\n=== Actor Optimizer ===")

    print(
        "loss before:",
        actor_loss_before,
    )

    print(
        "loss after :",
        actor_loss_after,
    )

    print(
        "parameter change:",
        actor_param_change,
    )

    print(
        "KL after update:",
        actor_metrics_after.approx_kl,
    )

    print(
        "clip fraction after update:",
        actor_metrics_after.clip_fraction,
    )

    # ============================================================
    # 6. Critic initialization
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

    critic_params = critic_variables[
        "params"
    ]

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
    # 7. Critic loss + gradient
    # ============================================================
    def critic_loss_fn(params):
        return critic_loss_with_params(
            critic_params=params,
            critic=critic,
            critic_obs=critic_obs,
            returns=returns,
        )

    (
        critic_loss_before,
        critic_grads,
    ) = jax.value_and_grad(
        critic_loss_fn
    )(
        critic_params
    )

    # ============================================================
    # 8. Critic optimizer step
    # ============================================================
    critic_optimizer = make_optimizer(
        learning_rate=3e-4,
        max_grad_norm=0.5,
    )

    critic_opt_state = critic_optimizer.init(
        critic_params
    )

    critic_updates, critic_opt_state = (
        critic_optimizer.update(
            critic_grads,
            critic_opt_state,
            critic_params,
        )
    )

    new_critic_params = optax.apply_updates(
        critic_params,
        critic_updates,
    )

    critic_param_change = tree_l2_distance(
        critic_params,
        new_critic_params,
    )

    critic_loss_after = critic_loss_fn(
        new_critic_params
    )

    print("\n=== Critic Optimizer ===")

    print(
        "loss before:",
        critic_loss_before,
    )

    print(
        "loss after :",
        critic_loss_after,
    )

    print(
        "parameter change:",
        critic_param_change,
    )

    # ============================================================
    # 9. Assertions
    # ============================================================
    assert jnp.isfinite(
        actor_loss_before
    )

    assert jnp.isfinite(
        actor_loss_after
    )

    assert jnp.isfinite(
        critic_loss_before
    )

    assert jnp.isfinite(
        critic_loss_after
    )

    assert tree_all_finite(
        new_actor_params
    )

    assert tree_all_finite(
        new_critic_params
    )

    assert actor_param_change > 0.0
    assert critic_param_change > 0.0