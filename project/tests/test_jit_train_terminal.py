import jax
import jax.numpy as jnp

from footballworld import (
    FootballWorld,
    MatchConfig,
)

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
    make_player,
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


def make_short_5v5_env(
    seed=0,
):
    home = (
        make_player(
            1,
            -50.0,
            0.0,
            goalkeeper=True,
        ),
        make_player(
            2,
            -30.0,
            -15.0,
        ),
        make_player(
            3,
            -30.0,
            15.0,
        ),
        make_player(
            4,
            -10.0,
            -8.0,
        ),
        make_player(
            5,
            -10.0,
            8.0,
        ),
    )

    away = (
        make_player(
            101,
            -50.0,
            0.0,
            goalkeeper=True,
        ),
        make_player(
            102,
            -30.0,
            -15.0,
        ),
        make_player(
            103,
            -30.0,
            15.0,
        ),
        make_player(
            104,
            -10.0,
            -8.0,
        ),
        make_player(
            105,
            -10.0,
            8.0,
        ),
    )

    env = FootballWorld(
        match=MatchConfig(
            fulltime_seconds=0.2,
            halftime_enabled=False,
            maximum_added_time_seconds_per_period=0.0,
            minimum_team_players=(5, 5),
        )
    )

    reset = env.reset(
        home,
        away,
        key=jax.random.key(seed),
    )

    return env, reset


def test_jit_train_iteration_terminal_and_empty_chunk():
    # ============================================================
    # 1. Very short match
    # ============================================================

    env, reset = make_short_5v5_env(
        seed=0
    )

    setup = reset.setup

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
    # 2. Actor
    # ============================================================

    obs = env.observe_all(
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

    actor_params = {
        "backbone":
            backbone_variables["params"],

        "head":
            head_variables["params"],
    }

    # ============================================================
    # 3. Critic
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
        jax.random.key(3),
        critic_obs[None, :],
    )

    critic_params = (
        critic_variables["params"]
    )

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
    # 5. Carry + JIT collector
    # ============================================================

    carry = initialize_jit_rollout_carry(
        env=env,
        reset=reset,
        opponent_key=jax.random.key(100),
    )

    collector = make_jitted_rollout_collector(
        env=env,

        backbone=backbone,
        action_head=action_head,
        critic=critic,

        rollout_length=4,

        controlled_indices=controlled_indices,

        controlled_team=0,
    )

    rollout_key = jax.random.key(
        200
    )

    # ============================================================
    # 6. FIRST iteration
    #
    # Real match should end after two control steps:
    #
    # [True, True, False, False]
    # ============================================================

    (
        actor_params_1,
        critic_params_1,

        actor_opt_state_1,
        critic_opt_state_1,

        carry_1,

        batch_1,
        valid_mask_1,

        advantages_1,
        returns_1,

        metrics_1,
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
    # 7. First iteration diagnostics
    # ============================================================

    print(
        "\n=== Terminal Iteration 1 ==="
    )

    print(
        "tick:",
        carry_1.rollout.state.control_tick,
    )

    print(
        "valid mask:",
        valid_mask_1,
    )

    print(
        "terminated:",
        batch_1.terminated,
    )

    print(
        "truncated:",
        batch_1.truncated,
    )

    print(
        "done:",
        batch_1.done,
    )

    print(
        "advantages:",
        advantages_1,
    )

    print(
        "returns:",
        returns_1,
    )

    print(
        "PPO epochs:",
        len(
            metrics_1
        ),
    )

    expected_valid_1 = jnp.array(
        [
            True,
            True,
            False,
            False,
        ]
    )

    assert bool(
        jnp.all(
            valid_mask_1
            == expected_valid_1
        )
    )

    assert int(
        carry_1.rollout.state.control_tick
    ) == 2

    assert bool(
        carry_1.done
    )

    assert bool(
        carry_1.truncated
    )

    assert not bool(
        carry_1.terminated
    )

    # Padding rows must have zero GAE targets.
    assert bool(
        jnp.all(
            advantages_1[2:]
            == 0.0
        )
    )

    assert bool(
        jnp.all(
            returns_1[2:]
            == 0.0
        )
    )

    assert len(
        metrics_1
    ) >= 1

    # ============================================================
    # 8. SECOND iteration
    #
    # Match is already finished.
    #
    # Collector must produce only padding.
    # train_iteration_jit must SKIP PPO entirely.
    # ============================================================

    (
        actor_params_2,
        critic_params_2,

        actor_opt_state_2,
        critic_opt_state_2,

        carry_2,

        batch_2,
        valid_mask_2,

        advantages_2,
        returns_2,

        metrics_2,
    ) = train_iteration_jit(
        env=env,
        setup=setup,

        carry=carry_1,
        collector=collector,

        roster=roster,
        si_roster=si_roster,

        backbone=backbone,
        action_head=action_head,
        critic=critic,

        actor_params=actor_params_1,
        critic_params=critic_params_1,

        actor_opt_state=actor_opt_state_1,
        critic_opt_state=critic_opt_state_1,

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
    # 9. Second iteration diagnostics
    # ============================================================

    print(
        "\n=== Terminal Iteration 2 ==="
    )

    print(
        "tick:",
        carry_2.rollout.state.control_tick,
    )

    print(
        "valid mask:",
        valid_mask_2,
    )

    print(
        "rewards:",
        batch_2.reward,
    )

    print(
        "advantages:",
        advantages_2,
    )

    print(
        "returns:",
        returns_2,
    )

    print(
        "PPO epochs:",
        len(
            metrics_2
        ),
    )

    # ============================================================
    # 10. Already-done chunk must contain no real transitions
    # ============================================================

    assert not bool(
        jnp.any(
            valid_mask_2
        )
    )

    assert int(
        carry_2.rollout.state.control_tick
    ) == 2

    assert bool(
        carry_2.done
    )

    assert len(
        metrics_2
    ) == 0

    assert bool(
        jnp.all(
            advantages_2
            == 0.0
        )
    )

    assert bool(
        jnp.all(
            returns_2
            == 0.0
        )
    )

    # ============================================================
    # 11. NO PPO UPDATE must occur on the empty chunk
    # ============================================================

    actor_difference = tree_l2_distance(
        actor_params_1,
        actor_params_2,
    )

    critic_difference = tree_l2_distance(
        critic_params_1,
        critic_params_2,
    )

    print(
        "\n=== Empty Chunk Parameter Check ==="
    )

    print(
        "actor parameter difference:",
        actor_difference,
    )

    print(
        "critic parameter difference:",
        critic_difference,
    )

    assert float(
        actor_difference
    ) == 0.0

    assert float(
        critic_difference
    ) == 0.0

    # Returned optimizer states should also simply be the states
    # from the previous successful update.
    assert actor_opt_state_2 is not None
    assert critic_opt_state_2 is not None