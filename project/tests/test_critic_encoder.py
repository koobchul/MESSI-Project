import jax

from project.envs.critic_encoder import (
    build_global_categorical_features,
    build_global_context_features,
    build_global_player_features,
    encode_critic_state,
)
from project.envs.messi_5v5 import make_5v5_env


def test_critic_encoder():
    # ============================================================
    # 1. Environment
    # ============================================================
    env, reset = make_5v5_env(seed=0)

    # ============================================================
    # 2. Global state
    # ============================================================
    global_state = env.global_state_view(
        reset.rollout
    )

    # ============================================================
    # 3. Global player features
    # ============================================================
    player_features = build_global_player_features(
        global_state,
        controlled_team=0,
    )

    print("\n=== Critic Player Features ===")
    print("shape:", player_features.shape)
    print("dtype:", player_features.dtype)

    assert player_features.shape == (10, 28)
    assert player_features.dtype == jax.numpy.float32

    # ============================================================
    # 4. Global context
    # ============================================================
    context_features = build_global_context_features(
        global_state,
        controlled_team=0,
    )

    print("\n=== Critic Context Features ===")
    print("shape:", context_features.shape)
    print("dtype:", context_features.dtype)

    assert context_features.shape == (39,)
    assert context_features.dtype == jax.numpy.float32

    # ============================================================
    # 5. Global categorical
    # ============================================================
    categorical_features = build_global_categorical_features(
        global_state,
        controlled_team=0,
    )

    print("\n=== Critic Categorical Features ===")
    print("shape:", categorical_features.shape)
    print("dtype:", categorical_features.dtype)

    assert categorical_features.shape == (92,)
    assert categorical_features.dtype == jax.numpy.float32

    # ============================================================
    # 6. Final centralized critic input
    # ============================================================
    critic_input = encode_critic_state(
        global_state,
        controlled_team=0,
    )

    print("\n=== Critic Input ===")
    print("shape:", critic_input.shape)
    print("dtype:", critic_input.dtype)

    assert critic_input.shape == (411,)
    assert critic_input.dtype == jax.numpy.float32

    assert jax.numpy.all(
        jax.numpy.isfinite(critic_input)
    )