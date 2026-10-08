#test_5v5_env.py

import jax

from footballworld import neutral_action
from project.envs.messi_5v5 import make_5v5_env
from project.envs.obs_encoder import (
    build_categorical_features,
    build_context_features,
    build_player_features,
    describe_observation,
    encode_mlp_observation,
)


def test_5v5_environment():
    # ============================================================
    # 1. Create 5v5 MESSI environment
    # ============================================================
    env, reset = make_5v5_env(seed=0)

    # ============================================================
    # 2. Player Observation
    # ============================================================
    obs = env.observe_all(reset.rollout)

    # Print detailed observation structure
    summary = describe_observation(obs)

    print("\n=== Observation ===")
    print("valid shape:", obs.valid.shape)

    # 5v5 = 10 total players
    assert summary["num_observers"] == 10
    assert obs.valid.shape == (10,)

    # ============================================================
    # 3. Roster Metadata
    # ============================================================
    roster = env.roster_metadata(reset.rollout)

    print("\n=== Roster Metadata ===")
    print(
        jax.tree_util.tree_map(
            lambda x: (x.shape, x.dtype),
            roster,
        )
    )

    # ============================================================
    # 4. Player Feature Encoder
    # ============================================================
    player_features = build_player_features(
        obs,
        roster,
    )

    print("\n=== Player Features ===")
    print("shape:", player_features.shape)
    print("dtype:", player_features.dtype)

    # 5v5:
    # 10 observers
    # 10 players
    # 30 features per player
    assert player_features.shape == (10, 10, 30)
    assert player_features.dtype == jax.numpy.float32

    # ============================================================
    # Context Feature Encoder
    # ============================================================
    context_features = build_context_features(
        obs,
        roster,
    )

    print("\n=== Context Features ===")
    print("shape:", context_features.shape)
    print("dtype:", context_features.dtype)

    assert context_features.shape == (10, 34)
    assert context_features.dtype == jax.numpy.float32

    # ============================================================
    # Categorical Feature Encoder
    # ============================================================
    categorical_features = build_categorical_features(
        obs,
        roster,
    )

    print("\n=== Categorical Features ===")
    print("shape:", categorical_features.shape)
    print("dtype:", categorical_features.dtype)

    assert categorical_features.shape == (10, 80)
    assert categorical_features.dtype == jax.numpy.float32

    # ============================================================
    # Final MLP Observation
    # ============================================================
    mlp_obs = encode_mlp_observation(
        obs,
        roster,
    )

    print("\n=== MLP Observation ===")
    print("shape:", mlp_obs.shape)
    print("dtype:", mlp_obs.dtype)

    assert mlp_obs.shape == (10, 414)
    assert mlp_obs.dtype == jax.numpy.float32

    # ============================================================
    # 5. Global State
    #    Used later by the centralized critic in MAPPO
    # ============================================================
    global_state = env.global_state_view(reset.rollout)

    print("\n=== Global State ===")
    print(
        jax.tree_util.tree_map(
            lambda x: (x.shape, x.dtype),
            global_state,
        )
    )

    # ============================================================
    # 6. Neutral Action
    #    Temporary action before implementing our learned policy
    # ============================================================
    action = neutral_action(player_count=10)

    print("\n=== Action ===")
    print("intent:", action.intent.shape)
    print("move:", action.move.shape)
    print("force_to_ball:", action.force_to_ball.shape)
    print("launch:", action.launch.shape)
    print("spin:", action.spin.shape)

    assert action.intent.shape == (10,)
    assert action.move.shape == (10, 2)
    assert action.force_to_ball.shape == (10, 2)
    assert action.launch.shape == (10,)
    assert action.spin.shape == (10, 2)

    # ============================================================
    # 7. One Environment Step
    # ============================================================
    result = env.step(
        reset.rollout,
        reset.setup,
        action,
        jax.random.key(1),
    )

    print("\n=== Step Result ===")
    print("reward:", result.reward)
    print("done:", result.done)

    # Team 0 reward, Team 1 reward
    assert result.reward.shape == (2,)