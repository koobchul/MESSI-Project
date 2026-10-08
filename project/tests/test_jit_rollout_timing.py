import time

import jax
import jax.numpy as jnp

from project.algorithms.rollout import (
    initialize_rollout_carry,
)

from project.algorithms.rollout_jit_smoke import (
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


def test_jit_rollout_timing():
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

    # Current 5v5 fixture:
    #
    # Team 0 = slots 0~4
    # Team 1 = slots 5~9
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
    # 3. Actor initialization
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
    # 4. Critic initialization
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
    # 5. Persistent match carry
    # ============================================================
    carry = initialize_rollout_carry(
        env=env,
        reset=reset,
        opponent_key=jax.random.key(100),
    )

    # ============================================================
    # 6. Build ONE fixed-length compiled collector
    # ============================================================
    collector = make_jitted_rollout_collector(
        env=env,

        backbone=backbone,
        action_head=action_head,
        critic=critic,

        rollout_length=4,

        controlled_indices=(
            controlled_indices
        ),

        controlled_team=0,
    )

    # Same match-level rollout key is deliberately
    # reused across rollout chunks.
    rollout_key = jax.random.key(
        200
    )

    # ============================================================
    # 7. First call
    #
    # Compilation + execution
    # ============================================================
    start = time.perf_counter()

    batch_1, carry = collector(
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

    first_elapsed = (
        time.perf_counter()
        - start
    )

    # ============================================================
    # 8. Second call
    #
    # Compiled executable should be reused.
    # ============================================================
    start = time.perf_counter()

    batch_2, carry = collector(
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

    second_elapsed = (
        time.perf_counter()
        - start
    )

    # ============================================================
    # 9. Results
    # ============================================================
    print(
        "\n=== JIT Rollout Timing ==="
    )

    print(
        "final tick:",
        carry.rollout.state.control_tick,
    )

    print(
        "batch 1:",
        batch_1.actor_obs.shape,
    )

    print(
        "batch 2:",
        batch_2.actor_obs.shape,
    )

    print(
        "first call (compile + run):",
        first_elapsed,
        "sec",
    )

    print(
        "second call (compiled run):",
        second_elapsed,
        "sec",
    )

    print(
        "speed ratio:",
        second_elapsed
        / first_elapsed,
    )

    print(
        "done chunk 1:",
        batch_1.done,
    )

    print(
        "done chunk 2:",
        batch_2.done,
    )

    # ============================================================
    # 10. Smoke checks
    # ============================================================
    assert (
        carry.rollout.state.control_tick
        == 8
    )

    assert batch_1.actor_obs.shape == (
        4,
        5,
        414,
    )

    assert batch_2.actor_obs.shape == (
        4,
        5,
        414,
    )

    assert not bool(
        jnp.any(
            batch_1.done
        )
    )

    assert not bool(
        jnp.any(
            batch_2.done
        )
    )