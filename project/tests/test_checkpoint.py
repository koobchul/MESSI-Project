#test_checkpoint.py

import jax
import jax.numpy as jnp

from project.algorithms.optim import (
    make_optimizer,
)

from project.algorithms.rollout_jit import (
    initialize_jit_rollout_carry,
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

from project.utils.checkpoint import (
    CHECKPOINT_VERSION,
    load_checkpoint,
    save_checkpoint,
)


def tree_allclose(
    tree_a,
    tree_b,
    atol=1e-7,
):
    """
    Compare two pytrees leaf by leaf.

    Works for JAX arrays and NumPy arrays loaded from pickle.
    """

    leaves_a = jax.tree_util.tree_leaves(
        tree_a
    )

    leaves_b = jax.tree_util.tree_leaves(
        tree_b
    )

    if len(leaves_a) != len(leaves_b):
        return False

    return all(
        bool(
            jnp.allclose(
                jnp.asarray(a),
                jnp.asarray(b),
                atol=atol,
                rtol=0.0,
            )
        )
        for a, b in zip(
            leaves_a,
            leaves_b,
        )
    )


def test_checkpoint_round_trip_with_jit_carry(
    tmp_path,
):
    # ============================================================
    # 1. Real MESSI 5v5 environment
    # ============================================================

    env, reset = make_5v5_env(
        seed=0
    )

    roster = env.roster_metadata(
        reset.rollout
    )

    # ============================================================
    # 2. Initialize shared Actor
    # ============================================================

    obs = env.observe_all(
        reset.rollout
    )

    actor_obs = encode_mlp_observation(
        obs,
        roster,
    )

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
    # 3. Initialize centralized Critic
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
    # 4. Optimizer states
    # ============================================================

    actor_optimizer = make_optimizer(
        learning_rate=3e-5,
        max_grad_norm=0.5,
    )

    critic_optimizer = make_optimizer(
        learning_rate=3e-5,
        max_grad_norm=0.5,
    )

    actor_opt_state = actor_optimizer.init(
        actor_params
    )

    critic_opt_state = critic_optimizer.init(
        critic_params
    )

    # ============================================================
    # 5. REAL JIT rollout carry
    #
    # Contains:
    #
    # rollout
    # rule_state
    # opponent_key
    # terminated
    # truncated
    # done
    # ============================================================

    carry = initialize_jit_rollout_carry(
        env=env,
        reset=reset,
        opponent_key=jax.random.key(100),
    )

    rollout_key = jax.random.key(
        200
    )

    # ============================================================
    # 6. Training counters
    # ============================================================

    iteration = 17

    total_env_steps = 2176

    episode = 3

    episode_steps = 640

    episode_return = -1.0

    # ============================================================
    # 7. Save checkpoint
    # ============================================================

    checkpoint_path = (
        tmp_path
        / "test_checkpoint_v2.pkl"
    )

    save_checkpoint(
        path=str(
            checkpoint_path
        ),

        actor_params=actor_params,
        critic_params=critic_params,

        actor_opt_state=actor_opt_state,
        critic_opt_state=critic_opt_state,

        carry=carry,
        rollout_key=rollout_key,

        iteration=iteration,
        total_env_steps=total_env_steps,

        episode=episode,
        episode_steps=episode_steps,
        episode_return=episode_return,
    )

    # ============================================================
    # 8. File checks
    # ============================================================

    assert checkpoint_path.exists()

    assert not (
        tmp_path
        / "test_checkpoint_v2.pkl.tmp"
    ).exists()

    # ============================================================
    # 9. Load
    # ============================================================

    loaded = load_checkpoint(
        str(
            checkpoint_path
        )
    )

    # ============================================================
    # 10. Version / counters
    # ============================================================

    assert loaded["version"] == (
        CHECKPOINT_VERSION
    )

    assert loaded["iteration"] == (
        iteration
    )

    assert loaded["total_env_steps"] == (
        total_env_steps
    )

    assert loaded["episode"] == (
        episode
    )

    assert loaded["episode_steps"] == (
        episode_steps
    )

    assert loaded["episode_return"] == (
        episode_return
    )

    # ============================================================
    # 11. Neural-network state
    # ============================================================

    assert tree_allclose(
        actor_params,
        loaded["actor_params"],
    )

    assert tree_allclose(
        critic_params,
        loaded["critic_params"],
    )

    # ============================================================
    # 12. Optimizer state
    # ============================================================

    assert tree_allclose(
        actor_opt_state,
        loaded["actor_opt_state"],
    )

    assert tree_allclose(
        critic_opt_state,
        loaded["critic_opt_state"],
    )

    # ============================================================
    # 13. Match-level randomness
    # ============================================================

    assert tree_allclose(
        rollout_key,
        loaded["rollout_key"],
    )

    # ============================================================
    # 14. Full JIT rollout carry
    #
    # This is the key test for exact mid-match resume.
    # ============================================================

    assert type(
        loaded["carry"]
    ) is type(
        carry
    )

    assert tree_allclose(
        carry,
        loaded["carry"],
    )

    # ============================================================
    # 15. Explicit carry semantics
    # ============================================================

    loaded_carry = (
        loaded["carry"]
    )

    assert int(
        loaded_carry.rollout.state.control_tick
    ) == int(
        carry.rollout.state.control_tick
    )

    assert bool(
        loaded_carry.terminated
    ) == bool(
        carry.terminated
    )

    assert bool(
        loaded_carry.truncated
    ) == bool(
        carry.truncated
    )

    assert bool(
        loaded_carry.done
    ) == bool(
        carry.done
    )

    # ============================================================
    # 16. Diagnostics
    # ============================================================

    print(
        "\n=== Checkpoint V2 Round Trip ==="
    )

    print(
        "checkpoint version:",
        loaded["version"],
    )

    print(
        "iteration:",
        loaded["iteration"],
    )

    print(
        "total env steps:",
        loaded["total_env_steps"],
    )

    print(
        "episode:",
        loaded["episode"],
    )

    print(
        "episode steps:",
        loaded["episode_steps"],
    )

    print(
        "episode return:",
        loaded["episode_return"],
    )

    print(
        "carry type preserved:",
        type(
            loaded["carry"]
        ) is type(
            carry
        ),
    )

    print(
        "carry tree identical:",
        tree_allclose(
            carry,
            loaded["carry"],
        ),
    )

    print(
        "control tick:",
        loaded_carry.rollout.state.control_tick,
    )

    print(
        "terminated:",
        loaded_carry.terminated,
    )

    print(
        "truncated:",
        loaded_carry.truncated,
    )

    print(
        "done:",
        loaded_carry.done,
    )

    print(
        "actor params identical:",
        tree_allclose(
            actor_params,
            loaded["actor_params"],
        ),
    )

    print(
        "critic params identical:",
        tree_allclose(
            critic_params,
            loaded["critic_params"],
        ),
    )

    print(
        "actor optimizer identical:",
        tree_allclose(
            actor_opt_state,
            loaded["actor_opt_state"],
        ),
    )

    print(
        "critic optimizer identical:",
        tree_allclose(
            critic_opt_state,
            loaded["critic_opt_state"],
        ),
    )

    print(
        "rollout key identical:",
        tree_allclose(
            rollout_key,
            loaded["rollout_key"],
        ),
    )