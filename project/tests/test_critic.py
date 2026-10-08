import jax
import jax.numpy as jnp

from project.envs.critic_encoder import encode_critic_state
from project.envs.messi_5v5 import make_5v5_env
from project.models.critic import CentralizedCritic


def test_centralized_critic():
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
    # 3. Critic input
    # ============================================================
    critic_input = encode_critic_state(
        global_state,
        controlled_team=0,
    )

    print("\n=== Critic Input ===")
    print("shape:", critic_input.shape)
    print("dtype:", critic_input.dtype)

    assert critic_input.shape == (411,)
    assert critic_input.dtype == jnp.float32

    # Add batch dimension
    critic_batch = critic_input[None, :]

    # ============================================================
    # 4. Critic model
    # ============================================================
    critic = CentralizedCritic(
        hidden_dim=256,
    )

    variables = critic.init(
        jax.random.key(0),
        critic_batch,
    )

    value = critic.apply(
        variables,
        critic_batch,
    )

    # ============================================================
    # 5. Check output
    # ============================================================
    print("\n=== Centralized Critic ===")
    print("input shape :", critic_batch.shape)
    print("value shape :", value.shape)
    print("value       :", value)

    assert value.shape == (1,)
    assert jnp.all(jnp.isfinite(value))

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
    print("Critic parameters:", parameter_count)

    assert parameter_count > 0