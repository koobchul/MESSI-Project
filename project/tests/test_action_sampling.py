import jax
import jax.numpy as jnp

from project.envs.messi_5v5 import make_5v5_env
from project.envs.obs_encoder import encode_mlp_observation
from project.models.action_head import HybridActionHead
from project.models.action_sampling import sample_action
from project.models.mlp_actor import MLPActorBackbone


def test_action_sampling_and_env_step():
    # ============================================================
    # 1. Environment
    # ============================================================
    env, reset = make_5v5_env(seed=0)

    obs = env.observe_all(reset.rollout)
    roster = env.roster_metadata(reset.rollout)

    # ============================================================
    # 2. Observation encoder
    # ============================================================
    mlp_obs = encode_mlp_observation(
        obs,
        roster,
    )

    assert mlp_obs.shape == (10, 414)

    # ============================================================
    # 3. MLP backbone
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

    # ============================================================
    # 4. Hybrid action head
    # ============================================================
    action_head = HybridActionHead()

    head_variables = action_head.init(
        jax.random.key(1),
        latent,
    )

    actor_output = action_head.apply(
        head_variables,
        latent,
    )

    # ============================================================
    # 5. Sample action
    # ============================================================
    sampled = sample_action(
        actor_output,
        jax.random.key(2),
    )

    action = sampled.action

    print("\n=== Sampled Action ===")
    print("intent shape:", action.intent.shape)
    print("intent:", action.intent)

    continuous = action.as_continuous_array()

    print("continuous shape:", continuous.shape)
    print("continuous min:", continuous.min())
    print("continuous max:", continuous.max())

    assert action.intent.shape == (10,)
    assert continuous.shape == (10, 7)

    # Valid discrete actions
    assert jnp.all(action.intent >= 0)
    assert jnp.all(action.intent < 6)

    # Valid normalized continuous controls
    assert jnp.all(continuous >= -1.0)
    assert jnp.all(continuous <= 1.0)

    assert jnp.all(
        jnp.isfinite(continuous)
    )

    # ============================================================
    # 6. Send learned-policy action to MESSI
    # ============================================================
    result = env.step(
        reset.rollout,
        reset.setup,
        action,
        jax.random.key(3),
    )

    print("\n=== Environment Step ===")
    print("reward:", result.reward)
    print("done:", result.done)

    assert result.reward.shape == (2,)