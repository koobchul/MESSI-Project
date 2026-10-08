import jax

from project.envs.messi_5v5 import make_5v5_env
from project.envs.obs_encoder import encode_mlp_observation
from project.models.mlp_actor import MLPActorBackbone


def test_mlp_actor_backbone():
    # ============================================================
    # 1. Create 5v5 environment
    # ============================================================
    env, reset = make_5v5_env(seed=0)

    # ============================================================
    # 2. Build MLP observation
    # ============================================================
    obs = env.observe_all(reset.rollout)
    roster = env.roster_metadata(reset.rollout)

    mlp_obs = encode_mlp_observation(
        obs,
        roster,
    )

    print("\n=== MLP Input ===")
    print("shape:", mlp_obs.shape)
    print("dtype:", mlp_obs.dtype)

    assert mlp_obs.shape == (10, 414)
    assert mlp_obs.dtype == jax.numpy.float32

    # ============================================================
    # 3. Create shared MLP actor backbone
    # ============================================================
    model = MLPActorBackbone(
        hidden_dim=256,
    )

    # ============================================================
    # 4. Initialize network parameters
    # ============================================================
    key = jax.random.key(0)

    variables = model.init(
        key,
        mlp_obs,
    )

    # ============================================================
    # 5. Forward pass
    # ============================================================
    latent = model.apply(
        variables,
        mlp_obs,
    )

    print("\n=== MLP Actor Backbone ===")
    print("input shape :", mlp_obs.shape)
    print("output shape:", latent.shape)
    print("output dtype:", latent.dtype)

    assert latent.shape == (10, 256)
    assert latent.dtype == jax.numpy.float32

    # ============================================================
    # 6. Parameter count
    # ============================================================
    parameter_count = sum(
        x.size
        for x in jax.tree_util.tree_leaves(
            variables["params"]
        )
    )

    print("\n=== Parameter Count ===")
    print("MLP backbone parameters:", parameter_count)

    assert parameter_count > 0