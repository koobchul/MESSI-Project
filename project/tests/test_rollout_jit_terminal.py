import jax
import jax.numpy as jnp

from project.algorithms.rollout_jit import (
    initialize_jit_rollout_carry,
    make_jitted_rollout_collector,
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


def test_jit_rollout_terminal_freeze():
    # ============================================================
    # 1. Environment
    # ============================================================
    env, reset = make_5v5_env(
        seed=0
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
        [0, 1, 2, 3, 4],
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
        "backbone": backbone_variables["params"],
        "head": head_variables["params"],
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

    critic_params = critic_variables[
        "params"
    ]

    # ============================================================
    # 5. Initial JIT carry
    # ============================================================
    carry = initialize_jit_rollout_carry(
        env=env,
        reset=reset,
        opponent_key=jax.random.key(100),
    )

    # ============================================================
    # 6. Build fixed 4-step collector
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
    # 7. Normal chunk
    #
    # Expected:
    # tick 0 -> 4
    # all rows valid
    # ============================================================
    batch_1, valid_1, carry = collector(
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

    tick_after_normal = (
        carry.rollout.state.control_tick
    )

    # ============================================================
    # 8. Artificially mark match as terminal
    #
    # This lets us test the terminal pause branch without waiting
    # for an entire football match to finish.
    # ============================================================
    terminal_carry = carry._replace(
        terminated=jnp.bool_(True),
        truncated=jnp.bool_(False),
        done=jnp.bool_(True),
    )

    tick_before_terminal_chunk = (
        terminal_carry.rollout.state.control_tick
    )

    # ============================================================
    # 9. Collector called after match already ended
    #
    # Expected:
    #
    # - tick stays 4
    # - no physics advances
    # - no opponent state advances
    # - all valid_mask entries False
    # ============================================================
    batch_2, valid_2, terminal_carry = collector(
        reset.setup,
        terminal_carry,

        roster,
        si_roster,

        actor_params,
        critic_params,

        rollout_key,
    )

    jax.block_until_ready(
        terminal_carry.rollout.state.control_tick
    )

    tick_after_terminal_chunk = (
        terminal_carry.rollout.state.control_tick
    )

    # ============================================================
    # 10. Diagnostics
    # ============================================================
    print(
        "\n=== Terminal-safe JIT Rollout ==="
    )

    print(
        "tick after normal chunk:",
        tick_after_normal,
    )

    print(
        "normal valid mask:",
        valid_1,
    )

    print(
        "tick before terminal chunk:",
        tick_before_terminal_chunk,
    )

    print(
        "tick after terminal chunk:",
        tick_after_terminal_chunk,
    )

    print(
        "terminal valid mask:",
        valid_2,
    )

    print(
        "terminal done:",
        terminal_carry.done,
    )

    print(
        "terminal terminated:",
        terminal_carry.terminated,
    )

    print(
        "terminal truncated:",
        terminal_carry.truncated,
    )

    print(
        "terminal rewards:",
        batch_2.reward,
    )

    # ============================================================
    # 11. Normal chunk checks
    # ============================================================
    assert (
        tick_after_normal
        == 4
    )

    assert batch_1.actor_obs.shape == (
        4,
        5,
        414,
    )

    assert valid_1.shape == (
        4,
    )

    assert bool(
        jnp.all(valid_1)
    )

    # ============================================================
    # 12. Terminal freeze checks
    # ============================================================
    assert (
        tick_before_terminal_chunk
        == 4
    )

    assert (
        tick_after_terminal_chunk
        == 4
    )

    assert batch_2.actor_obs.shape == (
        4,
        5,
        414,
    )

    assert valid_2.shape == (
        4,
    )

    assert not bool(
        jnp.any(valid_2)
    )

    assert bool(
        terminal_carry.done
    )

    assert bool(
        terminal_carry.terminated
    )

    assert not bool(
        terminal_carry.truncated
    )

    # Padding transitions should contain zero reward.
    assert bool(
        jnp.all(
            batch_2.reward == 0.0
        )
    )