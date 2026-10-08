import jax
import jax.numpy as jnp

from project.algorithms.rollout import (
    collect_rollout,
    initialize_rollout_carry,
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


def test_rollout_continuity():
    # ============================================================
    # 1. Environment
    # ============================================================
    env, reset = make_5v5_env(
        seed=0
    )

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

    # ============================================================
    # 2. Actor
    # ============================================================
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
        jax.random.key(2),
        critic_obs[None, :],
    )

    # ============================================================
    # 4. Initialize persistent match carry
    # ============================================================
    carry = initialize_rollout_carry(
        env=env,
        reset=reset,
        opponent_key=jax.random.key(100),
    )

    initial_tick = (
        carry.rollout.state.control_tick
    )

    # ============================================================
    # 5. First rollout chunk
    # ============================================================
    batch_1, carry = collect_rollout(
        env=env,
        setup=reset.setup,
        carry=carry,

        backbone=backbone,
        backbone_variables=backbone_variables,

        action_head=action_head,
        head_variables=head_variables,

        critic=critic,
        critic_variables=critic_variables,

        rollout_length=2,

        key=jax.random.key(10),

        controlled_team=0,
    )

    tick_after_chunk_1 = (
        carry.rollout.state.control_tick
    )

    # ============================================================
    # 6. Second rollout chunk
    # ============================================================
    batch_2, carry = collect_rollout(
        env=env,
        setup=reset.setup,
        carry=carry,

        backbone=backbone,
        backbone_variables=backbone_variables,

        action_head=action_head,
        head_variables=head_variables,

        critic=critic,
        critic_variables=critic_variables,

        rollout_length=2,

        key=jax.random.key(11),

        controlled_team=0,
    )

    tick_after_chunk_2 = (
        carry.rollout.state.control_tick
    )

    # ============================================================
    # 7. Inspect
    # ============================================================
    print("\n=== Rollout Continuity ===")

    print(
        "initial tick:",
        initial_tick,
    )

    print(
        "after chunk 1:",
        tick_after_chunk_1,
    )

    print(
        "after chunk 2:",
        tick_after_chunk_2,
    )

    print(
        "chunk 1 actor obs:",
        batch_1.actor_obs.shape,
    )

    print(
        "chunk 2 actor obs:",
        batch_2.actor_obs.shape,
    )

    # ============================================================
    # 8. Assertions
    # ============================================================
    assert batch_1.actor_obs.shape == (
        2,
        5,
        414,
    )

    assert batch_2.actor_obs.shape == (
        2,
        5,
        414,
    )

    assert (
        tick_after_chunk_1
        > initial_tick
    )

    assert (
        tick_after_chunk_2
        > tick_after_chunk_1
    )

    assert jnp.all(
        jnp.isfinite(
            batch_1.log_prob
        )
    )

    assert jnp.all(
        jnp.isfinite(
            batch_2.log_prob
        )
    )