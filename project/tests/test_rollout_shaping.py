#test_rollout_shaping.py

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


def test_jit_rollout_shaped_reward():
    # ============================================================
    # 1. Environment
    # ============================================================

    env, reset = make_5v5_env(
        seed=0
    )

    roster = env.roster_metadata(
        reset.rollout
    )

    si_roster = env.roster_metadata_si(
        reset.rollout
    )

    # ============================================================
    # 2. Actor input
    # ============================================================

    obs = env.observe_all(
        reset.rollout
    )

    actor_obs = encode_mlp_observation(
        obs,
        roster,
    )

    # ============================================================
    # 3. Actor
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
        jax.random.key(3),
        critic_obs[None, :],
    )

    critic_params = (
        critic_variables["params"]
    )

    # ============================================================
    # 5. Persistent rollout carry
    # ============================================================

    carry = initialize_jit_rollout_carry(
        env=env,
        reset=reset,
        opponent_key=jax.random.key(100),
    )

    # ============================================================
    # 6. Shaped rollout collector
    # ============================================================

    collector = make_jitted_rollout_collector(
        env=env,

        backbone=backbone,
        action_head=action_head,
        critic=critic,

        rollout_length=4,

        controlled_indices=jnp.array(
            [0, 1, 2, 3, 4],
            dtype=jnp.int32,
        ),

        controlled_team=0,

        reward_shaping_coefficient=0.05,
        reward_gamma=0.99,
    )

    # ============================================================
    # 7. Collect rollout
    # ============================================================

    batch, valid_mask, new_carry = collector(
        reset.setup,
        carry,

        roster,
        si_roster,

        actor_params,
        critic_params,

        jax.random.key(200),
    )

    jax.block_until_ready(
        new_carry.rollout.state.control_tick
    )

    # ============================================================
    # 8. Print diagnostics
    # ============================================================

    print(
        "\n=== Shaped JIT Rollout ==="
    )

    print(
        "base reward:",
        batch.base_reward,
    )

    print(
        "shaping reward:",
        batch.shaping_reward,
    )

    print(
        "total reward:",
        batch.reward,
    )

    print(
        "valid mask:",
        valid_mask,
    )

    print(
        "control tick:",
        new_carry.rollout.state.control_tick,
    )

    # ============================================================
    # 9. Shape checks
    # ============================================================

    assert batch.base_reward.shape == (
        4,
    )

    assert batch.shaping_reward.shape == (
        4,
    )

    assert batch.reward.shape == (
        4,
    )

    assert valid_mask.shape == (
        4,
    )

    # ============================================================
    # 10. Numerical checks
    # ============================================================

    assert jnp.all(
        jnp.isfinite(
            batch.base_reward
        )
    )

    assert jnp.all(
        jnp.isfinite(
            batch.shaping_reward
        )
    )

    assert jnp.all(
        jnp.isfinite(
            batch.reward
        )
    )

    # ============================================================
    # 11. Reward identity
    #
    # total
    # =
    # sparse goal reward
    # +
    # potential shaping reward
    # ============================================================

    assert jnp.allclose(
        batch.reward,
        (
            batch.base_reward
            + batch.shaping_reward
        ),
        atol=1e-6,
    )

    # ============================================================
    # 12. All four transitions should be real here
    # ============================================================

    assert jnp.all(
        valid_mask
    )

    assert int(
        new_carry.rollout.state.control_tick
    ) == 4