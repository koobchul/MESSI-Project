import argparse
import csv
import os
import time

import jax
import jax.numpy as jnp

from project.algorithms.rollout_jit import (
    initialize_jit_rollout_carry,
    make_jitted_rollout_collector,
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
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a frozen 5v5 MLP policy against "
            "the MESSI rule-based opponent."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Training checkpoint. "
            "If omitted, evaluate the initial untrained MLP."
        ),
    )

    parser.add_argument(
        "--label",
        type=str,
        default="initial",
    )

    parser.add_argument(
        "--num-matches",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--rollout-length",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Base seed for evaluation matches.",
    )

    parser.add_argument(
        "--max-chunks",
        type=int,
        default=1000,
        help="Safety guard against an unexpectedly non-ending match.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="project/eval_logs",
    )

    return parser.parse_args()


def initialize_models(
    env,
    reset,
):
    """
    Construct exactly the same MLP Actor / Critic architecture
    used by train.py.

    The initialization keys intentionally match train.py.

    When a checkpoint is supplied, these initial parameters are
    replaced by checkpoint parameters.
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
    # Actor backbone
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

    # ============================================================
    # Hybrid action head
    # ============================================================

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
    # Centralized critic
    #
    # The evaluator itself does not use the value estimate as a
    # metric, but the already-tested JIT rollout collector expects
    # critic parameters.
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


def write_results(
    path,
    rows,
):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    with open(
        path,
        "w",
        newline="",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "match",
                "reset_seed",
                "steps",
                "return",
                "goals_for",
                "goals_against",
                "goal_difference",
                "result",
                "wall_seconds",
            ]
        )

        writer.writerows(
            rows
        )


def main():
    args = parse_args()

    controlled_team = 0

    # ============================================================
    # 1. Build environment once
    # ============================================================

    env, reset = make_5v5_env(
        seed=args.seed
    )

    # ============================================================
    # 2. Construct model architecture
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
    # 3. Select frozen policy
    # ============================================================

    if args.checkpoint is None:
        actor_params = (
            initial_actor_params
        )

        critic_params = (
            initial_critic_params
        )

        source_description = (
            "initial untrained policy"
        )

    else:
        checkpoint = load_checkpoint(
            args.checkpoint
        )

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

        source_description = (
            args.checkpoint
        )

    # ============================================================
    # 4. Controlled players
    # ============================================================

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
    # 5. Build JIT rollout collector ONCE
    #
    # IMPORTANT:
    # No PPO update is ever performed in this script.
    #
    # Actor parameters remain frozen for all evaluation matches.
    # ============================================================

    collector = make_jitted_rollout_collector(
        env=env,

        backbone=backbone,
        action_head=action_head,
        critic=critic,

        rollout_length=args.rollout_length,

        controlled_indices=controlled_indices,

        controlled_team=controlled_team,
    )

    # ============================================================
    # 6. Evaluation RNG bases
    #
    # Using the same evaluation seed for initial and trained
    # policies gives the same sequence of match-level seeds.
    # ============================================================

    opponent_base_key = jax.random.key(
        args.seed + 10000
    )

    rollout_base_key = jax.random.key(
        args.seed + 20000
    )

    # ============================================================
    # 7. Summary accumulators
    # ============================================================

    wins = 0
    draws = 0
    losses = 0

    total_goals_for = 0
    total_goals_against = 0

    total_return = 0.0
    total_steps = 0

    rows = []

    print(
        "\n========================================"
    )

    print(
        "MESSI 5v5 Frozen Policy Evaluation"
    )

    print(
        "========================================"
    )

    print(
        "label:",
        args.label,
    )

    print(
        "policy source:",
        source_description,
    )

    print(
        "matches:",
        args.num_matches,
    )

    print(
        "rollout length:",
        args.rollout_length,
    )

    print(
        "evaluation seed:",
        args.seed,
    )

    # ============================================================
    # 8. Evaluation matches
    # ============================================================

    for match_index in range(
        args.num_matches
    ):
        reset_seed = (
            args.seed
            + match_index
        )

        # --------------------------------------------------------
        # Fresh match.
        #
        # Never restore training carry for evaluation.
        # --------------------------------------------------------

        if match_index == 0:
            reset = reset_5v5_env(
                env,
                seed=reset_seed,
            )

        else:
            reset = reset_5v5_env(
                env,
                seed=reset_seed,
            )

        setup = reset.setup

        roster = env.roster_metadata(
            reset.rollout
        )

        si_roster = env.roster_metadata_si(
            reset.rollout
        )

        opponent_key = jax.random.fold_in(
            opponent_base_key,
            match_index,
        )

        rollout_key = jax.random.fold_in(
            rollout_base_key,
            match_index,
        )

        carry = initialize_jit_rollout_carry(
            env=env,
            reset=reset,
            opponent_key=opponent_key,
        )

        match_return = 0.0
        match_steps = 0

        chunk_count = 0

        match_start = (
            time.perf_counter()
        )

        # ========================================================
        # Run until this match actually terminates/truncates.
        # ========================================================

        while not bool(
            jax.device_get(
                carry.done
            )
        ):
            chunk_count += 1

            if (
                chunk_count
                > args.max_chunks
            ):
                raise RuntimeError(
                    "Evaluation exceeded max_chunks="
                    f"{args.max_chunks}. "
                    "The match may not be terminating correctly."
                )

            (
                batch,
                valid_mask,
                carry,
            ) = collector(
                setup,
                carry,

                roster,
                si_roster,

                actor_params,
                critic_params,

                rollout_key,
            )

            # ----------------------------------------------------
            # Synchronize the compiled rollout.
            # ----------------------------------------------------

            jax.block_until_ready(
                carry.rollout.state.control_tick
            )

            # ----------------------------------------------------
            # Count only real transitions.
            # ----------------------------------------------------

            valid_steps = int(
                jax.device_get(
                    jnp.sum(
                        valid_mask.astype(
                            jnp.int32
                        )
                    )
                )
            )

            match_steps += (
                valid_steps
            )

            # ----------------------------------------------------
            # Team-0 sparse return.
            # ----------------------------------------------------

            chunk_return = float(
                jax.device_get(
                    jnp.sum(
                        jnp.where(
                            valid_mask,
                            batch.reward,
                            jnp.float32(0.0),
                        )
                    )
                )
            )

            match_return += (
                chunk_return
            )

            # ----------------------------------------------------
            # Lightweight progress output for a long match.
            # ----------------------------------------------------

            if (
                chunk_count % 50
                == 0
            ):
                current_score = (
                    jax.device_get(
                        carry.rollout.state.score
                    )
                )

                print(
                    f"match={match_index + 1}"
                    f" | chunks={chunk_count}"
                    f" | steps={match_steps}"
                    f" | score="
                    f"{int(current_score[0])}"
                    f"-{int(current_score[1])}"
                )

        # ========================================================
        # 9. Final match result
        # ========================================================

        match_seconds = (
            time.perf_counter()
            - match_start
        )

        score = jax.device_get(
            carry.rollout.state.score
        )

        goals_for = int(
            score[controlled_team]
        )

        goals_against = int(
            score[
                1 - controlled_team
            ]
        )

        goal_difference = (
            goals_for
            - goals_against
        )

        if goals_for > goals_against:
            result = "W"
            wins += 1

        elif goals_for == goals_against:
            result = "D"
            draws += 1

        else:
            result = "L"
            losses += 1

        total_goals_for += (
            goals_for
        )

        total_goals_against += (
            goals_against
        )

        total_return += (
            match_return
        )

        total_steps += (
            match_steps
        )

        rows.append(
            [
                match_index + 1,
                reset_seed,
                match_steps,
                match_return,
                goals_for,
                goals_against,
                goal_difference,
                result,
                match_seconds,
            ]
        )

        print(
            "\n----------------------------------------"
        )

        print(
            f"Match {match_index + 1} finished"
        )

        print(
            "seed:",
            reset_seed,
        )

        print(
            "steps:",
            match_steps,
        )

        print(
            "score:",
            f"{goals_for}-{goals_against}",
        )

        print(
            "return:",
            match_return,
        )

        print(
            "result:",
            result,
        )

        print(
            "wall time:",
            f"{match_seconds:.3f}s",
        )

        print(
            "----------------------------------------"
        )

    # ============================================================
    # 10. Aggregate statistics
    # ============================================================

    num_matches = (
        args.num_matches
    )

    avg_goals_for = (
        total_goals_for
        / num_matches
    )

    avg_goals_against = (
        total_goals_against
        / num_matches
    )

    avg_goal_difference = (
        (
            total_goals_for
            - total_goals_against
        )
        / num_matches
    )

    avg_return = (
        total_return
        / num_matches
    )

    win_rate = (
        wins
        / num_matches
    )

    # ============================================================
    # 11. CSV
    # ============================================================

    output_path = os.path.join(
        args.output_dir,
        f"{args.label}.csv",
    )

    write_results(
        output_path,
        rows,
    )

    # ============================================================
    # 12. Final summary
    # ============================================================

    print(
        "\n========================================"
    )

    print(
        "Evaluation Summary"
    )

    print(
        "========================================"
    )

    print(
        "label:",
        args.label,
    )

    print(
        "matches:",
        num_matches,
    )

    print(
        "W / D / L:",
        f"{wins} / {draws} / {losses}",
    )

    print(
        "win rate:",
        f"{win_rate:.3f}",
    )

    print(
        "total goals for:",
        total_goals_for,
    )

    print(
        "total goals against:",
        total_goals_against,
    )

    print(
        "average goals for:",
        f"{avg_goals_for:.3f}",
    )

    print(
        "average goals against:",
        f"{avg_goals_against:.3f}",
    )

    print(
        "average goal difference:",
        f"{avg_goal_difference:.3f}",
    )

    print(
        "average return:",
        f"{avg_return:.3f}",
    )

    print(
        "total evaluation steps:",
        total_steps,
    )

    print(
        "results:",
        output_path,
    )


if __name__ == "__main__":
    main()