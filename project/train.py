#train.py

import argparse
import csv
import os
import time

import jax
import jax.numpy as jnp

from project.algorithms.optim import make_optimizer

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
    reset_5v5_env,
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
    load_checkpoint,
    save_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train 5v5 MLP-MAPPO on MESSI."
    )

    parser.add_argument(
        "--total-steps",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--rollout-length",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--actor-lr",
        type=float,
        default=3e-5,
    )

    parser.add_argument(
        "--critic-lr",
        type=float,
        default=3e-5,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--clip-eps",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--reward-shaping-coef",
        type=float,
        default=0.0,
        help=(
            "Potential-based ball-progress reward coefficient. "
            "0.0 reproduces the original sparse goal reward."
        ),
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="project/logs",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="project/checkpoints",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=4096,
        help=(
            "Save checkpoint every N real environment steps. "
            "Use 0 to disable periodic checkpoints."
        ),
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint file to resume from.",
    )

    return parser.parse_args()


def initialize_models(
    env,
    reset,
):
    """
    Initialize shared Actor and centralized Critic modules.

    On resume, the newly initialized parameters are replaced by
    checkpoint parameters. Initialization is still required to
    construct the Flax module structures.

    Existing initialization keys are intentionally preserved here
    so the reward-shaping change does not simultaneously change
    the model initialization used by the current baseline.
    """

    roster = env.roster_metadata(
        reset.rollout
    )

    obs = env.observe_all(
        reset.rollout
    )

    actor_obs = encode_mlp_observation(
        obs,
        roster,
    )

    # ============================================================
    # Actor
    # ============================================================

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
    # Critic
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

    return (
        backbone,
        action_head,
        critic,
        actor_params,
        critic_params,
    )


def write_log_header(
    log_path,
):
    os.makedirs(
        os.path.dirname(log_path),
        exist_ok=True,
    )

    with open(
        log_path,
        "w",
        newline="",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "iteration",
                "episode",
                "total_env_steps",
                "valid_steps",
                "episode_steps",
                "episode_return",
                "iteration_total_reward",
                "iteration_base_reward",
                "iteration_shaping_reward",
                "mean_abs_shaping_reward",
                "max_abs_shaping_reward",
                "score_home",
                "score_away",
                "ppo_epochs",
                "actor_loss",
                "critic_loss",
                "approx_kl",
                "clip_fraction",
                "entropy",
                "iteration_seconds",
            ]
        )


def initialize_log(
    log_path,
    resume,
):
    """
    Fresh run:
        overwrite/create a new CSV.

    Resume:
        preserve the existing CSV and append to it.

        If the CSV does not exist, create a new one.
    """

    if (
        resume is not None
        and os.path.exists(
            log_path
        )
    ):
        return

    write_log_header(
        log_path
    )


def append_log(
    log_path,
    row,
):
    with open(
        log_path,
        "a",
        newline="",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow(
            row
        )


def reward_run_name(
    seed,
    reward_shaping_coef,
):
    """
    Keep the historical sparse filename unchanged.

    Shaped runs receive a separate name so they do not overwrite
    the existing sparse baseline logs/checkpoints.
    """

    if reward_shaping_coef == 0.0:
        return (
            f"mlp_mappo_seed_{seed}"
        )

    coefficient_tag = (
        f"{reward_shaping_coef:g}"
        .replace("-", "m")
        .replace(".", "p")
    )

    return (
    f"mlp_mappo_shaped_v2_c"
    f"{coefficient_tag}_seed_{seed}"
    )


def main():
    args = parse_args()

    controlled_team = 0

    if args.reward_shaping_coef < 0.0:
        raise ValueError(
            "--reward-shaping-coef must be >= 0."
        )

    # ============================================================
    # 1. Optional checkpoint load
    #
    # Load counters first because the current episode determines
    # which deterministic reset/setup must be reconstructed.
    # ============================================================

    checkpoint = None

    if args.resume is not None:
        checkpoint = load_checkpoint(
            args.resume
        )

        episode = int(
            checkpoint["episode"]
        )

        iteration = int(
            checkpoint["iteration"]
        )

        total_env_steps = int(
            checkpoint[
                "total_env_steps"
            ]
        )

        episode_steps = int(
            checkpoint[
                "episode_steps"
            ]
        )

        episode_return = float(
            checkpoint[
                "episode_return"
            ]
        )

        current_reset_seed = (
            args.seed
            + episode
        )

    else:
        episode = 0
        iteration = 0
        total_env_steps = 0
        episode_steps = 0
        episode_return = 0.0

        current_reset_seed = (
            args.seed
        )

    # ============================================================
    # 2. Environment
    #
    # On resume we reconstruct the static setup for the current
    # episode. The dynamic rollout state itself comes from carry.
    # ============================================================

    env, reset = make_5v5_env(
        seed=current_reset_seed
    )

    setup = reset.setup

    # ============================================================
    # 3. Model structures
    # ============================================================

    (
        backbone,
        action_head,
        critic,
        initial_actor_params,
        initial_critic_params,
    ) = initialize_models(
        env,
        reset,
    )

    # ============================================================
    # 4. Optimizers
    # ============================================================

    actor_optimizer = make_optimizer(
        learning_rate=args.actor_lr,
        max_grad_norm=args.max_grad_norm,
    )

    critic_optimizer = make_optimizer(
        learning_rate=args.critic_lr,
        max_grad_norm=args.max_grad_norm,
    )

    # ============================================================
    # 5. Deterministic match-level base keys
    # ============================================================

    opponent_base_key = jax.random.key(
        args.seed + 10000
    )

    rollout_base_key = jax.random.key(
        args.seed + 20000
    )

    # ============================================================
    # 6. Fresh run OR resume state
    # ============================================================

    if checkpoint is None:
        actor_params = (
            initial_actor_params
        )

        critic_params = (
            initial_critic_params
        )

        actor_opt_state = (
            actor_optimizer.init(
                actor_params
            )
        )

        critic_opt_state = (
            critic_optimizer.init(
                critic_params
            )
        )

        opponent_key = jax.random.fold_in(
            opponent_base_key,
            episode,
        )

        rollout_key = jax.random.fold_in(
            rollout_base_key,
            episode,
        )

        carry = initialize_jit_rollout_carry(
            env=env,
            reset=reset,
            opponent_key=opponent_key,
        )

    else:
        # --------------------------------------------------------
        # Move restored host arrays back to the active JAX device.
        # --------------------------------------------------------

        actor_params = jax.device_put(
            checkpoint[
                "actor_params"
            ]
        )

        critic_params = jax.device_put(
            checkpoint[
                "critic_params"
            ]
        )

        actor_opt_state = jax.device_put(
            checkpoint[
                "actor_opt_state"
            ]
        )

        critic_opt_state = jax.device_put(
            checkpoint[
                "critic_opt_state"
            ]
        )

        carry = jax.device_put(
            checkpoint["carry"]
        )

        rollout_key = jax.device_put(
            checkpoint[
                "rollout_key"
            ]
        )

    # ============================================================
    # 7. Roster metadata
    #
    # Use the actual current rollout state, including on resume.
    # ============================================================

    roster = env.roster_metadata(
        carry.rollout
    )

    si_roster = env.roster_metadata_si(
        carry.rollout
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
    # 8. Build JIT collector ONCE
    # ============================================================

    collector = make_jitted_rollout_collector(
        env=env,

        backbone=backbone,
        action_head=action_head,
        critic=critic,

        rollout_length=args.rollout_length,

        controlled_indices=controlled_indices,

        controlled_team=controlled_team,

        reward_shaping_coefficient=(
            args.reward_shaping_coef
        ),

        reward_gamma=args.gamma,
    )

    # ============================================================
    # 9. Paths
    #
    # Sparse:
    #     mlp_mappo_seed_0.*
    #
    # Shaping coefficient 0.05:
    #     mlp_mappo_shaped_c0p05_seed_0.*
    #
    # This prevents the shaped experiment from overwriting the
    # existing sparse baseline.
    # ============================================================

    run_name = reward_run_name(
        seed=args.seed,
        reward_shaping_coef=(
            args.reward_shaping_coef
        ),
    )

    log_path = os.path.join(
        args.log_dir,
        f"{run_name}.csv",
    )

    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        (
            f"{run_name}_latest.pkl"
        ),
    )

    os.makedirs(
        args.checkpoint_dir,
        exist_ok=True,
    )

    initialize_log(
        log_path,
        resume=args.resume,
    )

    # ============================================================
    # 10. Training information
    # ============================================================

    reward_mode = (
        "sparse"
        if args.reward_shaping_coef == 0.0
        else "potential-shaped"
    )

    print(
        "\n========================================"
    )

    print(
        "MESSI 5v5 MLP-MAPPO Training"
    )

    print(
        "========================================"
    )

    print(
        "seed:",
        args.seed,
    )

    print(
        "target total steps:",
        args.total_steps,
    )

    print(
        "rollout length:",
        args.rollout_length,
    )

    print(
        "actor LR:",
        args.actor_lr,
    )

    print(
        "critic LR:",
        args.critic_lr,
    )

    print(
        "gamma:",
        args.gamma,
    )

    print(
        "reward mode:",
        reward_mode,
    )

    print(
        "reward shaping coefficient:",
        args.reward_shaping_coef,
    )

    print(
        "run name:",
        run_name,
    )

    if checkpoint is not None:
        current_tick = int(
            jax.device_get(
                carry.rollout.state.control_tick
            )
        )

        current_score = jax.device_get(
            carry.rollout.state.score
        )

        print(
            "\nRESUMED FROM CHECKPOINT"
        )

        print(
            "checkpoint:",
            args.resume,
        )

        print(
            "iteration:",
            iteration,
        )

        print(
            "total env steps:",
            total_env_steps,
        )

        print(
            "episode:",
            episode,
        )

        print(
            "episode steps:",
            episode_steps,
        )

        print(
            "episode return:",
            episode_return,
        )

        print(
            "control tick:",
            current_tick,
        )

        print(
            "score:",
            (
                f"{int(current_score[0])}"
                f"-{int(current_score[1])}"
            ),
        )

    # ============================================================
    # 11. Determine next periodic checkpoint threshold
    # ============================================================

    if args.save_every > 0:
        next_save_step = (
            (
                total_env_steps
                // args.save_every
            )
            + 1
        ) * args.save_every

    else:
        next_save_step = None

    training_start = (
        time.perf_counter()
    )

    # ============================================================
    # 12. Main training loop
    # ============================================================

    while (
        total_env_steps
        < args.total_steps
    ):
        iteration += 1

        iteration_start = (
            time.perf_counter()
        )

        # --------------------------------------------------------
        # Complete MAPPO iteration
        # --------------------------------------------------------

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

            controlled_team=controlled_team,

            gamma=args.gamma,
            gae_lambda=args.gae_lambda,

            num_epochs=args.ppo_epochs,
            target_kl=args.target_kl,

            clip_eps=args.clip_eps,
            entropy_coef=args.entropy_coef,
        )

        # --------------------------------------------------------
        # Synchronize actual optimization work.
        # --------------------------------------------------------

        jax.block_until_ready(
            carry.rollout.state.control_tick
        )

        if metrics_history:
            jax.block_until_ready(
                metrics_history[
                    -1
                ].approx_kl
            )

        iteration_seconds = (
            time.perf_counter()
            - iteration_start
        )

        # ========================================================
        # 13. Count real transitions
        # ========================================================

        valid_steps = int(
            jax.device_get(
                jnp.sum(
                    valid_mask.astype(
                        jnp.int32
                    )
                )
            )
        )

        total_env_steps += (
            valid_steps
        )

        episode_steps += (
            valid_steps
        )

        # ========================================================
        # 14. Reward diagnostics
        #
        # batch.reward:
        #     reward actually used by GAE / PPO
        #
        # batch.base_reward:
        #     original sparse MESSI goal reward
        #
        # batch.shaping_reward:
        #     potential-based shaping component
        # ========================================================

        masked_total_reward = jnp.where(
            valid_mask,
            batch.reward,
            jnp.float32(0.0),
        )

        masked_base_reward = jnp.where(
            valid_mask,
            batch.base_reward,
            jnp.float32(0.0),
        )

        masked_shaping_reward = jnp.where(
            valid_mask,
            batch.shaping_reward,
            jnp.float32(0.0),
        )

        iteration_reward = float(
            jax.device_get(
                jnp.sum(
                    masked_total_reward
                )
            )
        )

        iteration_base_reward = float(
            jax.device_get(
                jnp.sum(
                    masked_base_reward
                )
            )
        )

        iteration_shaping_reward = float(
            jax.device_get(
                jnp.sum(
                    masked_shaping_reward
                )
            )
        )

        if valid_steps > 0:
            mean_abs_shaping_reward = float(
                jax.device_get(
                    jnp.sum(
                        jnp.abs(
                            masked_shaping_reward
                        )
                    )
                    / jnp.float32(
                        valid_steps
                    )
                )
            )

        else:
            mean_abs_shaping_reward = 0.0

        max_abs_shaping_reward = float(
            jax.device_get(
                jnp.max(
                    jnp.where(
                        valid_mask,
                        jnp.abs(
                            batch.shaping_reward
                        ),
                        jnp.float32(0.0),
                    )
                )
            )
        )

        episode_return += (
            iteration_reward
        )

        # ========================================================
        # 15. Score
        # ========================================================

        score = jax.device_get(
            carry.rollout.state.score
        )

        home_score = int(
            score[0]
        )

        away_score = int(
            score[1]
        )

        # ========================================================
        # 16. PPO metrics
        # ========================================================

        if metrics_history:
            final_metrics = (
                metrics_history[-1]
            )

            ppo_epochs = len(
                metrics_history
            )

            actor_loss = float(
                final_metrics.actor_loss
            )

            critic_loss = float(
                final_metrics.critic_loss
            )

            approx_kl = float(
                final_metrics.approx_kl
            )

            clip_fraction = float(
                final_metrics.clip_fraction
            )

            entropy = float(
                final_metrics.entropy
            )

        else:
            ppo_epochs = 0

            actor_loss = 0.0
            critic_loss = 0.0

            approx_kl = 0.0
            clip_fraction = 0.0
            entropy = 0.0

        # ========================================================
        # 17. Console
        # ========================================================

        print(
            f"iter={iteration}"
            f" | episode={episode}"
            f" | steps={total_env_steps}/{args.total_steps}"
            f" | valid={valid_steps}"
            f" | score={home_score}-{away_score}"
            f" | reward={iteration_reward:.5f}"
            f" | base={iteration_base_reward:.3f}"
            f" | shape={iteration_shaping_reward:.5f}"
            f" | |shape|mean={mean_abs_shaping_reward:.6f}"
            f" | ep_return={episode_return:.4f}"
            f" | epochs={ppo_epochs}"
            f" | KL={approx_kl:.5f}"
            f" | actor={actor_loss:.5f}"
            f" | critic={critic_loss:.5f}"
            f" | time={iteration_seconds:.3f}s"
        )

        # ========================================================
        # 18. CSV
        # ========================================================

        append_log(
            log_path,
            [
                iteration,
                episode,
                total_env_steps,
                valid_steps,
                episode_steps,
                episode_return,

                iteration_reward,
                iteration_base_reward,
                iteration_shaping_reward,

                mean_abs_shaping_reward,
                max_abs_shaping_reward,

                home_score,
                away_score,

                ppo_epochs,

                actor_loss,
                critic_loss,

                approx_kl,
                clip_fraction,
                entropy,

                iteration_seconds,
            ],
        )

        # ========================================================
        # 19. Episode boundary
        #
        # Reset BEFORE checkpoint saving.
        #
        # Therefore a checkpoint at an episode boundary always
        # contains the state ready for the next training iteration.
        # ========================================================

        if bool(
            jax.device_get(
                carry.done
            )
        ):
            print(
                "\n----------------------------------------"
            )

            print(
                f"Episode {episode} finished"
            )

            print(
                "episode steps:",
                episode_steps,
            )

            print(
                "episode return:",
                episode_return,
            )

            print(
                "final score:",
                f"{home_score}-{away_score}",
            )

            print(
                "terminated:",
                bool(
                    carry.terminated
                ),
            )

            print(
                "truncated:",
                bool(
                    carry.truncated
                ),
            )

            print(
                "----------------------------------------\n"
            )

            episode += 1

            reset = reset_5v5_env(
                env,
                seed=(
                    args.seed
                    + episode
                ),
            )

            setup = reset.setup

            opponent_key = (
                jax.random.fold_in(
                    opponent_base_key,
                    episode,
                )
            )

            rollout_key = (
                jax.random.fold_in(
                    rollout_base_key,
                    episode,
                )
            )

            carry = (
                initialize_jit_rollout_carry(
                    env=env,
                    reset=reset,
                    opponent_key=opponent_key,
                )
            )

            roster = env.roster_metadata(
                carry.rollout
            )

            si_roster = (
                env.roster_metadata_si(
                    carry.rollout
                )
            )

            episode_steps = 0
            episode_return = 0.0

        # ========================================================
        # 20. Periodic checkpoint
        # ========================================================

        if (
            next_save_step is not None
            and total_env_steps
            >= next_save_step
        ):
            save_checkpoint(
                path=checkpoint_path,

                actor_params=actor_params,
                critic_params=critic_params,

                actor_opt_state=(
                    actor_opt_state
                ),

                critic_opt_state=(
                    critic_opt_state
                ),

                carry=carry,

                rollout_key=rollout_key,

                iteration=iteration,

                total_env_steps=(
                    total_env_steps
                ),

                episode=episode,

                episode_steps=(
                    episode_steps
                ),

                episode_return=(
                    episode_return
                ),
            )

            print(
                "checkpoint saved:",
                checkpoint_path,
            )

            while (
                next_save_step
                <= total_env_steps
            ):
                next_save_step += (
                    args.save_every
                )

    # ============================================================
    # 21. Always save final state
    # ============================================================

    save_checkpoint(
        path=checkpoint_path,

        actor_params=actor_params,
        critic_params=critic_params,

        actor_opt_state=actor_opt_state,
        critic_opt_state=critic_opt_state,

        carry=carry,

        rollout_key=rollout_key,

        iteration=iteration,

        total_env_steps=(
            total_env_steps
        ),

        episode=episode,

        episode_steps=episode_steps,

        episode_return=episode_return,
    )

    training_seconds = (
        time.perf_counter()
        - training_start
    )

    # ============================================================
    # 22. Summary
    # ============================================================

    print(
        "\n========================================"
    )

    print(
        "Training complete"
    )

    print(
        "========================================"
    )

    print(
        "iterations:",
        iteration,
    )

    print(
        "environment steps:",
        total_env_steps,
    )

    print(
        "episodes completed:",
        episode,
    )

    print(
        "wall time:",
        training_seconds,
        "sec",
    )

    print(
        "log:",
        log_path,
    )

    print(
        "checkpoint:",
        checkpoint_path,
    )


if __name__ == "__main__":
    main()