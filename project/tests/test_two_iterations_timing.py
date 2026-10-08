import time

import jax

from project.algorithms.optim import make_optimizer
from project.algorithms.rollout import initialize_rollout_carry
from project.algorithms.train_step import train_iteration

from project.envs.critic_encoder import encode_critic_state
from project.envs.messi_5v5 import make_5v5_env
from project.envs.obs_encoder import encode_mlp_observation

from project.models.action_head import HybridActionHead
from project.models.critic import CentralizedCritic
from project.models.mlp_actor import MLPActorBackbone


def test_two_training_iterations_timing():
    # ============================================================
    # 1. Environment
    # ============================================================
    env, reset = make_5v5_env(
        seed=0
    )

    # ============================================================
    # 2. Actor initialization
    # ============================================================
    obs = env.observe_all(
        reset.rollout
    )

    roster = env.roster_metadata(
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
        jax.random.key(0),
        actor_obs,
    )

    latent = backbone.apply(
        backbone_variables,
        actor_obs,
    )

    action_head = HybridActionHead()

    head_variables = action_head.init(
        jax.random.key(1),
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
        jax.random.key(2),
        critic_obs[None, :],
    )

    critic_params = critic_variables[
        "params"
    ]

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
    carry = initialize_rollout_carry(
        env=env,
        reset=reset,
        opponent_key=jax.random.key(100),
    )

    print("\n=== Two MAPPO Iterations ===")
    print(
        "initial tick:",
        carry.rollout.state.control_tick,
    )

    times = []

    # ============================================================
    # 6. Two consecutive training iterations
    # ============================================================
    for iteration in range(2):
        start = time.perf_counter()

        (
            actor_params,
            critic_params,
            actor_opt_state,
            critic_opt_state,
            carry,
            batch,
            advantages,
            returns,
            metrics_history,
        ) = train_iteration(
            env=env,
            setup=reset.setup,
            carry=carry,

            backbone=backbone,
            action_head=action_head,
            critic=critic,

            actor_params=actor_params,
            critic_params=critic_params,

            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,

            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,

            rollout_length=4,

            key=jax.random.key(
                10 + iteration
            ),

            controlled_team=0,

            gamma=0.99,
            gae_lambda=0.95,

            num_epochs=4,
            target_kl=0.02,

            clip_eps=0.2,
            entropy_coef=0.01,
        )

        # JAX GPU operations can be asynchronous.
        # Force completion before measuring elapsed time.
        jax.block_until_ready(
            metrics_history[-1].approx_kl
        )

        end = time.perf_counter()

        elapsed = end - start
        times.append(elapsed)

        last_metrics = metrics_history[-1]

        print(
            f"\niteration={iteration + 1}"
        )

        print(
            "control tick:",
            carry.rollout.state.control_tick,
        )

        print(
            "rollout length:",
            batch.reward.shape[0],
        )

        print(
            "PPO epochs:",
            len(metrics_history),
        )

        print(
            "final KL:",
            last_metrics.approx_kl,
        )

        print(
            "clip fraction:",
            last_metrics.clip_fraction,
        )

        print(
            "elapsed seconds:",
            elapsed,
        )

    # ============================================================
    # 7. Timing summary
    # ============================================================
    print("\n=== Timing Summary ===")

    print(
        "iteration 1:",
        times[0],
        "sec",
    )

    print(
        "iteration 2:",
        times[1],
        "sec",
    )

    print(
        "speed ratio iter2 / iter1:",
        times[1] / times[0],
    )

    # ============================================================
    # 8. Continuity check
    # ============================================================
    assert (
        carry.rollout.state.control_tick
        == 8
    )

    assert len(times) == 2