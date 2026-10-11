import jax
import jax.numpy as jnp

from footballworld import (
    FootballWorld,
    MatchConfig,
)

from project.algorithms.rollout_jit import (
    initialize_jit_rollout_carry,
    make_jitted_rollout_collector,
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


def make_short_5v5_env(
    seed=0,
):
    """
    Build a deliberately short 5v5 match.

    fulltime_seconds=0.2

    MESSI policy/control frequency is 10 Hz, so the regulation
    boundary is two control ticks.

    maximum_added_time_seconds_per_period=0.0 ensures that even
    kickoff/dead-ball time cannot extend this test beyond the
    intended wall-clock boundary.

    halftime is disabled because this test only needs a single
    short period.
    """

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

    return (
        env,
        reset,
    )


def test_jit_rollout_mid_chunk_truncation():
    # ============================================================
    # 1. Short environment
    # ============================================================
    env, reset = make_short_5v5_env(
        seed=0
    )

    fulltime_tick, halftime_tick = (
        env.match.clock_ticks(
            env.timebase
        )
    )

    print(
        "\n=== Short Match Configuration ==="
    )

    print(
        "fulltime tick:",
        fulltime_tick,
    )

    print(
        "halftime tick:",
        halftime_tick,
    )

    # ============================================================
    # 2. Rosters
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
    # 3. Actor
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
    # 4. Critic
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

    critic_params = (
        critic_variables["params"]
    )

    # ============================================================
    # 5. Persistent JIT carry
    # ============================================================
    carry = initialize_jit_rollout_carry(
        env=env,
        reset=reset,
        opponent_key=jax.random.key(100),
    )

    # ============================================================
    # 6. Four-slot rollout
    #
    # Match should truncate after only two real transitions.
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

    rollout_key = jax.random.key(
        200
    )

    # ============================================================
    # 7. Collect
    # ============================================================
    batch, valid_mask, carry = collector(
        reset.setup,
        carry,

        roster,
        si_roster,

        actor_params,
        critic_params,

        rollout_key,
    )

    jax.block_until_ready(
        carry.rollout.state.control_tick
    )

    # ============================================================
    # 8. Diagnostics
    # ============================================================
    print(
        "\n=== Mid-chunk Terminal JIT Rollout ==="
    )

    print(
        "final control tick:",
        carry.rollout.state.control_tick,
    )

    print(
        "valid mask:",
        valid_mask,
    )

    print(
        "terminated:",
        batch.terminated,
    )

    print(
        "truncated:",
        batch.truncated,
    )

    print(
        "done:",
        batch.done,
    )

    print(
        "rewards:",
        batch.reward,
    )

    print(
        "carry terminated:",
        carry.terminated,
    )

    print(
        "carry truncated:",
        carry.truncated,
    )

    print(
        "carry done:",
        carry.done,
    )

    # ============================================================
    # 9. Environment boundary
    # ============================================================
    assert fulltime_tick == 2

    assert (
        carry.rollout.state.control_tick
        == 2
    )

    # ============================================================
    # 10. Fixed-shape rollout
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

    assert valid_mask.shape == (
        4,
    )

    # ============================================================
    # 11. Expected real/padding rows
    #
    # t=0 : real
    # t=1 : real + truncation occurs
    # t=2 : padding
    # t=3 : padding
    # ============================================================
    expected_valid = jnp.array(
        [
            True,
            True,
            False,
            False,
        ]
    )

    assert bool(
        jnp.all(
            valid_mask
            == expected_valid
        )
    )

    # ============================================================
    # 12. Termination semantics
    #
    # This short match ends because its time budget expires.
    #
    # Therefore:
    #
    # terminated = False
    # truncated  = True
    # ============================================================
    expected_terminated = jnp.array(
        [
            False,
            False,
            False,
            False,
        ]
    )

    expected_truncated = jnp.array(
        [
            False,
            True,
            False,
            False,
        ]
    )

    assert bool(
        jnp.all(
            batch.terminated
            == expected_terminated
        )
    )

    assert bool(
        jnp.all(
            batch.truncated
            == expected_truncated
        )
    )

    # ============================================================
    # 13. done field
    #
    # Padding rows deliberately retain done=True because the
    # environment carry is already terminal.
    #
    # valid_mask tells the learner that those rows are padding.
    # ============================================================
    expected_done = jnp.array(
        [
            False,
            True,
            True,
            True,
        ]
    )

    assert bool(
        jnp.all(
            batch.done
            == expected_done
        )
    )

    # ============================================================
    # 14. Final carry
    # ============================================================
    assert not bool(
        carry.terminated
    )

    assert bool(
        carry.truncated
    )

    assert bool(
        carry.done
    )

    # ============================================================
    # 15. Padding rows
    # ============================================================
    assert bool(
        jnp.all(
            batch.reward[
                ~valid_mask
            ]
            == 0.0
        )
    )