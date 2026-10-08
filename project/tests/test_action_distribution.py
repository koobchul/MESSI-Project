import jax
import jax.numpy as jnp

from project.envs.messi_5v5 import make_5v5_env
from project.envs.obs_encoder import encode_mlp_observation
from project.models.action_head import HybridActionHead
from project.models.action_sampling import (
    action_entropy,
    action_log_prob,
    sample_action,
)
from project.models.mlp_actor import MLPActorBackbone


def test_action_distribution():
    # ============================================================
    # 1. Environment
    # ============================================================
    env, reset = make_5v5_env(seed=0)

    obs = env.observe_all(reset.rollout)
    roster = env.roster_metadata(reset.rollout)

    # ============================================================
    # 2. Observation
    # ============================================================
    mlp_obs = encode_mlp_observation(
        obs,
        roster,
    )

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
    # 4. Action head
    # ============================================================
    head = HybridActionHead()

    head_variables = head.init(
        jax.random.key(1),
        latent,
    )

    output = head.apply(
        head_variables,
        latent,
    )

    # ============================================================
    # 5. Sample
    # ============================================================
    sampled = sample_action(
        output,
        jax.random.key(2),
    )

    # ============================================================
    # 6. Log probability
    # ============================================================
    log_prob = action_log_prob(
        output,
        sampled.action.intent,
        sampled.pre_tanh_continuous,
    )

    # ============================================================
    # 7. Entropy
    # ============================================================
    entropy = action_entropy(
        output,
    )

    print("\n=== Action Distribution ===")
    print("log_prob shape:", log_prob.shape)
    print("log_prob:", log_prob)

    print("entropy shape:", entropy.shape)
    print("entropy:", entropy)

    # One scalar log-prob / entropy per agent
    assert log_prob.shape == (10,)
    assert entropy.shape == (10,)

    assert jnp.all(
        jnp.isfinite(log_prob)
    )

    assert jnp.all(
        jnp.isfinite(entropy)
    )