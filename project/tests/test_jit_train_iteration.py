import jax
import jax.numpy as jnp

from project.algorithms.optim import (
    make_optimizer,
)

from project.algorithms.rollout_jit import (
    initialize_jit_rollout_carry,
    make_jitted_rollout_collector,
)

from project.algorithms.train_step import (
    train_iteration_jit,
)

from project.envs.critic_encoder import (
    encode_critic_state,
)

from project.envs.messi_5v5 import (
    make_5v5_env,
)

from project.envs.obs_encoder import (
    encode_mlp_observation,
)

from project.models.action_head import (
    HybridActionHead,
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
    """
    Compute L2 distance between two parameter pytrees.
    """

    leaves_a = jax.tree_util.tree_leaves(
        tree_a
    )

    leaves_b = jax.tree_util.tree_leaves(
        tree_b
    )

    squared_distance = sum(
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

    return jnp.sqrt(
        squared_distance
    )


def test_jit_train_iteration():
    # ============================================================
    # 1. Environment
    # ============================================================

    env, reset = make_5v5_env(
        seed=0
    )

    setup = reset.setup

    # ============================================================
    # 2. Roster metadata
    #
    # Normalized roster:
    #     learned policy / observation encoder
    #
    # SI roster:
    #     built-in MESSI rule-based opponent
    # ============================================================

    roster = env.roster_metadata(
        reset.rollout
    )

    si_roster = env.roster_metadata_si(
        reset.rollout
    )

    controlled_indices = jnp.array(
        [
            0,
            1,
            2,
            3,
            4,
        ],
        dtype=jnp.int32,
    )

    # ============================================================
    # 3. Initial actor observation
    # ============================================================

    obs = env.observe_all(
        reset.rollout
    )

    actor_obs = encode_mlp_observation(
        obs,
        roster,
    )

    print(
        "\n=== Initial Shapes ==="
    )

    print(
        "all-player actor obs:",
        actor_obs.shape,
    )

    assert actor_obs.shape == (
        10,
        414,
    )

    # ============================================================
    # 4. Actor backbone
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

    # ============================================================
    # 5. Hybrid action head
    # ============================================================

    action_head = HybridActionHead()

    head_variables = action_head.init(
        jax.random.key(2),
        latent,
    )

    actor_params = {
        "backbone":
            backbone_variables["params"],

        "head":
            head_variables["params"],
    }

    # ============================================================
    # 6. Initial centralized critic observation
    # ============================================================

    global_state = env.global_state_view(
        reset.rollout
    )

    critic_obs = encode_critic_state(
        global_state,
        controlled_team=0,
    )

    print(
        "critic obs:",
        critic_obs.shape,
    )

    assert critic_obs.shape == (
        411,
    )

    # ============================================================
    # 7. Centralized critic
    # ============================================================

    critic = CentralizedCritic(
        hidden_dim=256,
    )

    critic_variables = critic.init(
        jax.random.key(3),
        critic_obs[None, :],
    )

    critic_params = (
        critic_variables["params"]
    )

    # ============================================================
    # 8. Optimizers
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
    # 9. Persistent JIT rollout carry
    #
    # Rule-based opponent state persists across rollout chunks.
    # ============================================================

    carry = initialize_jit_rollout_carry(
        env=env,
        reset=reset,
        opponent_key=jax.random.key(100),
    )

    # ============================================================
    # 10. Build JIT collector ONCE
    #
    # rollout_length = 4 for this integration test.
    # ============================================================

    collector = make_jitted_rollout_collector(
        env=env,

        backbone=backbone,
        action_head=action_head,
        critic=critic,

        rollout_length=4,

        controlled_indices=controlled_indices,

        controlled_team=0,
    )

    # ============================================================
    # 11. Preserve initial parameters
    #
    # We will verify that one PPO iteration actually changes them.
    # ============================================================

    initial_actor_params = actor_params

    initial_critic_params = critic_params

    # ============================================================
    # 12. Match-level rollout key
    #
    # Keep this key fixed across chunks of the same match.
    # The collector folds absolute control_tick into it.
    # ============================================================

    rollout_key = jax.random.key(
        200
    )

    # ============================================================
    # 13. One complete JIT MAPPO iteration
    #
    # JIT rollout
    #     -> valid mask
    #     -> masked GAE
    #     -> masked PPO
    # ============================================================

    (
        new_actor_params,
        new_critic_params,

        new_actor_opt_state,
        new_critic_opt_state,

        new_carry,

        batch,
        valid_mask,

        advantages,
        returns,

        metrics_history,
    ) = train_iteration_jit(
        env=env,
        setup=setup,

        carry=carry,
        collector=collector,

        roster=roster,
        si_roster=si_roster,

        backbone=backbone,
        action_head=action_head,
        critic=critic,

        actor_params=actor_params,
        critic_params=critic_params,

        actor_opt_state=actor_opt_state,
        critic_opt_state=critic_opt_state,

        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,

        rollout_key=rollout_key,

        controlled_team=0,

        gamma=0.99,
        gae_lambda=0.95,

        num_epochs=4,
        target_kl=0.02,

        clip_eps=0.2,
        entropy_coef=0.01,
    )

    # ============================================================
    # 14. Synchronize before diagnostics
    # ============================================================

    jax.block_until_ready(
        new_carry.rollout.state.control_tick
    )

    # ============================================================
    # 15. Parameter changes
    # ============================================================

    actor_change = tree_l2_distance(
        initial_actor_params,
        new_actor_params,
    )

    critic_change = tree_l2_distance(
        initial_critic_params,
        new_critic_params,
    )

    # ============================================================
    # 16. Diagnostics
    # ============================================================

    print(
        "\n=== JIT MAPPO Train Iteration ==="
    )

    print(
        "final tick:",
        new_carry.rollout.state.control_tick,
    )

    print(
        "valid mask:",
        valid_mask,
    )

    print(
        "batch actor obs:",
        batch.actor_obs.shape,
    )

    print(
        "batch critic obs:",
        batch.critic_obs.shape,
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
        "PPO epochs:",
        len(
            metrics_history
        ),
    )

    for index, metrics in enumerate(
        metrics_history,
        start=1,
    ):
        print(
            f"epoch={index}"
            f" | actor_loss={float(metrics.actor_loss):.6f}"
            f" | critic_loss={float(metrics.critic_loss):.6f}"
            f" | KL={float(metrics.approx_kl):.6f}"
            f" | clip_fraction={float(metrics.clip_fraction):.3f}"
        )

    # ============================================================
    # 17. Rollout assertions
    # ============================================================

    assert int(
        new_carry.rollout.state.control_tick
    ) == 4

    assert batch.actor_obs.shape == (
        4,
        5,
        414,
    )

    assert batch.critic_obs.shape == (
        4,
        411,
    )

    assert batch.intent.shape == (
        4,
        5,
    )

    assert batch.pre_tanh_continuous.shape == (
        4,
        5,
        7,
    )

    assert batch.log_prob.shape == (
        4,
        5,
    )

    # ============================================================
    # 18. Default match should still be alive after 4 steps
    #
    # Therefore every row should be real.
    # ============================================================

    assert bool(
        jnp.all(
            valid_mask
        )
    )

    assert not bool(
        new_carry.done
    )

    # ============================================================
    # 19. GAE assertions
    # ============================================================

    assert advantages.shape == (
        4,
    )

    assert returns.shape == (
        4,
    )

    assert bool(
        jnp.all(
            jnp.isfinite(
                advantages
            )
        )
    )

    assert bool(
        jnp.all(
            jnp.isfinite(
                returns
            )
        )
    )

    # ============================================================
    # 20. PPO must actually update parameters
    # ============================================================

    assert float(
        actor_change
    ) > 0.0

    assert float(
        critic_change
    ) > 0.0

    # ============================================================
    # 21. At least one PPO epoch must run
    # ============================================================

    assert len(
        metrics_history
    ) >= 1

    assert len(
        metrics_history
    ) <= 4

    # ============================================================
    # 22. PPO metrics must be finite
    # ============================================================

    for metrics in metrics_history:
        assert bool(
            jnp.isfinite(
                metrics.actor_loss
            )
        )

        assert bool(
            jnp.isfinite(
                metrics.critic_loss
            )
        )

        assert bool(
            jnp.isfinite(
                metrics.approx_kl
            )
        )

        assert bool(
            jnp.isfinite(
                metrics.clip_fraction
            )
        )

    # Silence unused-variable concerns and make explicit that the
    # optimizer states are part of the train-iteration output.
    assert new_actor_opt_state is not None
    assert new_critic_opt_state is not None