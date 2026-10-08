import jax
import jax.numpy as jnp

from project.algorithms.buffer import Transition
from project.algorithms.optim import make_optimizer
from project.algorithms.update import ppo_update_epochs

from project.models.action_head import HybridActionHead
from project.models.action_sampling import (
    action_log_prob,
    sample_action,
)
from project.models.critic import CentralizedCritic
from project.models.mlp_actor import MLPActorBackbone


def test_multiple_ppo_epochs_with_kl_stop():
    # ============================================================
    # 1. Dimensions
    # ============================================================
    T = 4
    N = 5

    # ============================================================
    # 2. Actor
    # ============================================================
    actor_obs = jax.random.normal(
        jax.random.key(0),
        (T, N, 414),
    )

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

    # ============================================================
    # 4. Critic
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
    # 5. Rollout batch
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
    # 6. Optimizers
    #
    # Learning-rate sensitivity test selected 3e-5.
    # ============================================================
    actor_optimizer = make_optimizer(
        learning_rate=3e-5,
        max_grad_norm=0.5,
    )

    critic_optimizer = make_optimizer(
        learning_rate=3e-5,
        max_grad_norm=0.5,
    )

    actor_opt_state = actor_optimizer.init(
        actor_params
    )

    critic_opt_state = critic_optimizer.init(
        critic_params
    )

    # ============================================================
    # 7. Multiple PPO epochs
    # ============================================================
    (
        new_actor_params,
        new_critic_params,
        new_actor_opt_state,
        new_critic_opt_state,
        metrics_history,
    ) = ppo_update_epochs(
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

        num_epochs=4,

        target_kl=0.02,

        clip_eps=0.2,
        entropy_coef=0.01,
    )

    # ============================================================
    # 8. Print epoch diagnostics
    # ============================================================
    print("\n=== PPO Epochs ===")

    for epoch, metrics in enumerate(
        metrics_history,
        start=1,
    ):
        print(
            f"epoch={epoch} | "
            f"actor_loss={float(metrics.actor_loss):.6f} | "
            f"critic_loss={float(metrics.critic_loss):.6f} | "
            f"KL={float(metrics.approx_kl):.6f} | "
            f"clip_fraction={float(metrics.clip_fraction):.3f}"
        )

    print(
        "epochs run:",
        len(metrics_history),
    )

    # ============================================================
    # 9. Assertions
    # ============================================================
    assert 1 <= len(
        metrics_history
    ) <= 4

    for metrics in metrics_history:
        assert jnp.isfinite(
            metrics.actor_loss
        )

        assert jnp.isfinite(
            metrics.critic_loss
        )

        assert jnp.isfinite(
            metrics.approx_kl
        )

        assert (
            0.0
            <= metrics.clip_fraction
            <= 1.0
        )

    # If early stopping happened, the final epoch should
    # have exceeded the KL threshold.
    if len(metrics_history) < 4:
        assert (
            metrics_history[-1].approx_kl
            > 0.02
        )