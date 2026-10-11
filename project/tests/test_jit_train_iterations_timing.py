import time

import jax
import jax.numpy as jnp

from project.algorithms.optim import (
    make_optimizer,
)

from project.algorithms.rollout_jit import (
    initialize_jit_rollout_carry,
    make_jitted_rollout_collector,
)

from project.algorithms.train_step import (
    train_iteration_jit,
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


def test_jit_train_iterations_timing():
    # ============================================================
    # 1. Environment
    # ============================================================

    env, reset = make_5v5_env(
        seed=0
    )

    setup = reset.setup

    roster = env.roster_metadata(
        reset.rollout
    )

    si_roster = env.roster_metadata_si(
        reset.rollout
    )

    controlled_indices = jnp.array(
        [
            0,
            1,
            2,
            3,
            4,
        ],
        dtype=jnp.int32,
    )

    # ============================================================
    # 2. Actor initialization
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
    # 3. Critic initialization
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
    # 4. Optimizers
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
    # 5. Persistent match carry
    # ============================================================

    carry = initialize_jit_rollout_carry(
        env=env,
        reset=reset,
        opponent_key=jax.random.key(100),
    )

    # ============================================================
    # 6. Build collector ONCE
    #
    # This is important.
    #
    # We want iterations 2 and 3 to reuse the already compiled
    # rollout executable.
    # ============================================================

    collector = make_jitted_rollout_collector(
        env=env,

        backbone=backbone,
        action_head=action_head,
        critic=critic,

        rollout_length=128,

        controlled_indices=controlled_indices,

        controlled_team=0,
    )

    # ============================================================
    # 7. Match-level rollout key
    #
    # Keep it fixed throughout this SAME match.
    #
    # Absolute control_tick is folded into the key inside the
    # collector, so consecutive chunks still receive different
    # per-frame randomness.
    # ============================================================

    rollout_key = jax.random.key(
        200
    )

    # ============================================================
    # 8. Run three consecutive MAPPO iterations
    # ============================================================

    elapsed_times = []

    for iteration in range(
        1,
        4,
    ):
        start = time.perf_counter()

        (
            actor_params,
            critic_params,

            actor_opt_state,
            critic_opt_state,

            carry,

            batch,
            valid_mask,

            advantages,
            returns,

            metrics_history,
        ) = train_iteration_jit(
            env=env,
            setup=setup,

            carry=carry,
            collector=collector,

            roster=roster,
            si_roster=si_roster,

            backbone=backbone,
            action_head=action_head,
            critic=critic,

            actor_params=actor_params,
            critic_params=critic_params,

            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,

            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,

            rollout_key=rollout_key,

            controlled_team=0,

            gamma=0.99,
            gae_lambda=0.95,

            num_epochs=4,
            target_kl=0.02,

            clip_eps=0.2,
            entropy_coef=0.01,
        )

        # --------------------------------------------------------
        # Synchronize all work before stopping timer.
        #
        # Use a value produced by the final optimization result,
        # not only the rollout state.
        # --------------------------------------------------------

        final_metric = (
            metrics_history[-1]
        )

        jax.block_until_ready(
            final_metric.approx_kl
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        elapsed_times.append(
            elapsed
        )

        print(
            f"\n=== Iteration {iteration} ==="
        )

        print(
            "tick:",
            carry.rollout.state.control_tick,
        )

        print(
            "valid mask:",
            valid_mask,
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
            "PPO epochs:",
            len(
                metrics_history
            ),
        )

        print(
            "final KL:",
            final_metric.approx_kl,
        )

        print(
            "elapsed:",
            elapsed,
            "sec",
        )

    # ============================================================
    # 9. Summary
    # ============================================================

    print(
        "\n=== JIT MAPPO Timing Summary ==="
    )

    print(
        "iteration 1:",
        elapsed_times[0],
        "sec",
    )

    print(
        "iteration 2:",
        elapsed_times[1],
        "sec",
    )

    print(
        "iteration 3:",
        elapsed_times[2],
        "sec",
    )

    print(
        "steady-state average (iter 2-3):",
        (
            elapsed_times[1]
            + elapsed_times[2]
        )
        / 2.0,
        "sec",
    )

    # ============================================================
    # 10. Correctness assertions
    # ============================================================

    # 3 chunks x 4 control steps
    assert int(
        carry.rollout.state.control_tick
    ) == 384

    assert bool(
        jnp.all(
            valid_mask
        )
    )

    assert not bool(
        carry.done
    )

    assert len(
        elapsed_times
    ) == 3

    assert all(
        time_value > 0.0
        for time_value in elapsed_times
    )