import jax
import jax.numpy as jnp

from project.algorithms.optim import make_optimizer
from project.algorithms.rollout import initialize_rollout_carry
from project.algorithms.train_step import train_iteration

from project.envs.critic_encoder import encode_critic_state
from project.envs.messi_5v5 import make_5v5_env
from project.envs.obs_encoder import encode_mlp_observation

from project.models.action_head import HybridActionHead
from project.models.critic import CentralizedCritic
from project.models.mlp_actor import MLPActorBackbone


def tree_l2_distance(tree_a, tree_b):
    leaves_a = jax.tree_util.tree_leaves(tree_a)
    leaves_b = jax.tree_util.tree_leaves(tree_b)

    return jnp.sqrt(
        sum(
            jnp.sum(
                jnp.square(a - b)
            )
            for a, b in zip(
                leaves_a,
                leaves_b,
            )
        )
    )


def test_real_messi_train_iteration():
    # ============================================================
    # 1. Environment
    # ============================================================
    env, reset = make_5v5_env(
        seed=0
    )

    # ============================================================
    # 2. Initialize actor
    # ============================================================
    obs = env.observe_all(
        reset.rollout
    )

    roster = env.roster_metadata(
        reset.rollout
    )

    actor_obs = encode_mlp_observation(
        obs,
        roster,
    )

    backbone = MLPActorBackbone(
        hidden_dim=256,
    )

    backbone_variables = backbone.init(
        jax.random.key(0),
        actor_obs,
    )

    latent = backbone.apply(
        backbone_variables,
        actor_obs,
    )

    action_head = HybridActionHead()

    head_variables = action_head.init(
        jax.random.key(1),
        latent,
    )

    actor_params = {
        "backbone":
            backbone_variables["params"],

        "head":
            head_variables["params"],
    }

    # ============================================================
    # 3. Initialize critic
    # ============================================================
    global_state = env.global_state_view(
        reset.rollout
    )

    critic_obs = encode_critic_state(
        global_state,
        controlled_team=0,
    )

    critic = CentralizedCritic(
        hidden_dim=256,
    )

    critic_variables = critic.init(
        jax.random.key(2),
        critic_obs[None, :],
    )

    critic_params = critic_variables[
        "params"
    ]

    # ============================================================
    # 4. Optimizers
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
    # 5. Persistent match carry
    # ============================================================
    carry = initialize_rollout_carry(
        env=env,
        reset=reset,
        opponent_key=jax.random.key(100),
    )

    # ============================================================
    # 6. One REAL training iteration
    # ============================================================
    (
        new_actor_params,
        new_critic_params,
        new_actor_opt_state,
        new_critic_opt_state,
        new_carry,
        batch,
        advantages,
        returns,
        metrics_history,
    ) = train_iteration(
        env=env,
        setup=reset.setup,
        carry=carry,

        backbone=backbone,
        action_head=action_head,
        critic=critic,

        actor_params=actor_params,
        critic_params=critic_params,

        actor_opt_state=actor_opt_state,
        critic_opt_state=critic_opt_state,

        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,

        rollout_length=4,

        key=jax.random.key(10),

        controlled_team=0,

        gamma=0.99,
        gae_lambda=0.95,

        num_epochs=4,
        target_kl=0.02,

        clip_eps=0.2,
        entropy_coef=0.01,
    )

    # ============================================================
    # 7. Parameter changes
    # ============================================================
    actor_change = tree_l2_distance(
        actor_params,
        new_actor_params,
    )

    critic_change = tree_l2_distance(
        critic_params,
        new_critic_params,
    )

    # ============================================================
    # 8. Print diagnostics
    # ============================================================
    print("\n=== Real MAPPO Train Iteration ===")

    print(
        "rollout length:",
        batch.reward.shape[0],
    )

    print(
        "rewards:",
        batch.reward,
    )

    print(
        "advantages:",
        advantages,
    )

    print(
        "returns:",
        returns,
    )

    print(
        "actor parameter change:",
        actor_change,
    )

    print(
        "critic parameter change:",
        critic_change,
    )

    print(
        "final control tick:",
        new_carry.rollout.state.control_tick,
    )

    print(
        "PPO epochs:",
        len(metrics_history),
    )

    print("\n=== PPO Metrics ===")

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

    # ============================================================
    # 9. Assertions
    # ============================================================
    assert batch.actor_obs.shape == (
        4,
        5,
        414,
    )

    assert batch.critic_obs.shape == (
        4,
        411,
    )

    assert advantages.shape == (
        4,
    )

    assert returns.shape == (
        4,
    )

    assert jnp.all(
        jnp.isfinite(
            advantages
        )
    )

    assert jnp.all(
        jnp.isfinite(
            returns
        )
    )

    assert actor_change > 0.0
    assert critic_change > 0.0

    assert (
        new_carry.rollout.state.control_tick
        == 4
    )

    assert (
        1
        <= len(metrics_history)
        <= 4
    )