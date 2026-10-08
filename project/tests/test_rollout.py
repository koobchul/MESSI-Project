import jax
import jax.numpy as jnp

from project.algorithms.gae import compute_gae

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


def test_real_rollout_collection():
    # ============================================================
    # 1. Environment
    # ============================================================
    env, reset = make_5v5_env(
        seed=0
    )

    # ============================================================
    # 2. Example actor observation
    # ============================================================
    obs = env.observe_all(
        reset.rollout
    )

    roster = env.roster_metadata(
        reset.rollout
    )

    mlp_obs = encode_mlp_observation(
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
        jax.random.key(0),
        mlp_obs,
    )

    latent = backbone.apply(
        backbone_variables,
        mlp_obs,
    )

    action_head = HybridActionHead()

    head_variables = action_head.init(
        jax.random.key(1),
        latent,
    )

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

    # ============================================================
    # 5. Persistent rollout carry
    # ============================================================
    carry = initialize_rollout_carry(
        env=env,
        reset=reset,
        opponent_key=jax.random.key(100),
    )

    # ============================================================
    # 6. Collect four real MESSI steps
    # ============================================================
    batch, carry = collect_rollout(
        env=env,
        setup=reset.setup,
        carry=carry,

        backbone=backbone,
        backbone_variables=backbone_variables,

        action_head=action_head,
        head_variables=head_variables,

        critic=critic,
        critic_variables=critic_variables,

        rollout_length=4,

        key=jax.random.key(3),

        controlled_team=0,
    )

    # ============================================================
    # 7. Inspect rollout
    # ============================================================
    print("\n=== Real Rollout ===")

    print(
        "actor_obs:",
        batch.actor_obs.shape,
    )

    print(
        "critic_obs:",
        batch.critic_obs.shape,
    )

    print(
        "intent:",
        batch.intent.shape,
    )

    print(
        "pre_tanh_continuous:",
        batch.pre_tanh_continuous.shape,
    )

    print(
        "log_prob:",
        batch.log_prob.shape,
    )

    print(
        "value:",
        batch.value.shape,
    )

    print(
        "reward:",
        batch.reward.shape,
    )

    print(
        "rewards:",
        batch.reward,
    )

    print(
        "done:",
        batch.done,
    )

    # ============================================================
    # 8. Basic rollout checks
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

    assert batch.value.shape == (
        4,
    )

    assert batch.reward.shape == (
        4,
    )

    assert jnp.all(
        jnp.isfinite(
            batch.log_prob
        )
    )

    assert jnp.all(
        jnp.isfinite(
            batch.value
        )
    )

    # ============================================================
    # 9. Bootstrap value from final carried state
    # ============================================================
    final_global_state = env.global_state_view(
        carry.rollout
    )

    final_critic_obs = encode_critic_state(
        final_global_state,
        controlled_team=0,
    )

    bootstrap_value = critic.apply(
        critic_variables,
        final_critic_obs[None, :],
    )[0]

    # True terminal state:
    # no value bootstrap.
    bootstrap_value = jnp.where(
        batch.terminated[-1],
        jnp.float32(0.0),
        bootstrap_value,
    )

    print("\n=== Bootstrap ===")

    print(
        "bootstrap value:",
        bootstrap_value,
    )

    # ============================================================
    # 10. GAE
    # ============================================================
    advantages, returns = compute_gae(
        rewards=batch.reward,
        values=batch.value,

        bootstrap_value=bootstrap_value,

        terminated=batch.terminated,
        truncated=batch.truncated,

        gamma=0.99,
        gae_lambda=0.95,
    )

    print("\n=== Real Rollout GAE ===")

    print(
        "values:",
        batch.value,
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

    # ============================================================
    # 11. GAE checks
    # ============================================================
    assert advantages.shape == (
        batch.reward.shape
    )

    assert returns.shape == (
        batch.reward.shape
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