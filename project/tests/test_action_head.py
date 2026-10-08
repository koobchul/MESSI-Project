import jax

from project.envs.messi_5v5 import make_5v5_env
from project.envs.obs_encoder import encode_mlp_observation
from project.models.action_head import HybridActionHead
from project.models.mlp_actor import MLPActorBackbone


def test_hybrid_action_head():
    # ============================================================
    # 1. Environment
    # ============================================================
    env, reset = make_5v5_env(seed=0)

    obs = env.observe_all(reset.rollout)
    roster = env.roster_metadata(reset.rollout)

    mlp_obs = encode_mlp_observation(
        obs,
        roster,
    )

    assert mlp_obs.shape == (10, 414)

    # ============================================================
    # 2. MLP backbone
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

    assert latent.shape == (10, 256)

    # ============================================================
    # 3. Hybrid action head
    # ============================================================
    action_head = HybridActionHead()

    head_variables = action_head.init(
        jax.random.key(1),
        latent,
    )

    output = action_head.apply(
        head_variables,
        latent,
    )

    # ============================================================
    # 4. Check shapes
    # ============================================================
    print("\n=== Hybrid Action Head ===")

    print(
        "intent logits shape:",
        output.intent_logits.shape,
    )

    print(
        "continuous mean shape:",
        output.continuous_mean.shape,
    )

    print(
        "continuous log std shape:",
        output.continuous_log_std.shape,
    )

    assert output.intent_logits.shape == (
        10,
        6,
    )

    assert output.continuous_mean.shape == (
        10,
        7,
    )

    assert output.continuous_log_std.shape == (
        10,
        7,
    )