import argparse

import jax
import jax.numpy as jnp
import numpy as np

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


INTENT_NAMES = [
    "MOVE",
    "CONTROL",
    "PASS",
    "SHOT",
    "CLEAR",
    "CHALLENGE",
]


CONTINUOUS_NAMES = [
    "move_x",
    "move_y",
    "force_ball_x",
    "force_ball_y",
    "launch",
    "spin_x",
    "spin_y",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare initial and trained MLP policy behavior."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--num-steps",
        type=int,
        default=6400,
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
    )

    return parser.parse_args()


def initialize_models(
    env,
    reset,
):
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


def evaluate_policy_behavior(
    label,
    env,
    backbone,
    action_head,
    critic,
    collector,
    actor_params,
    critic_params,
    num_steps,
    rollout_length,
    seed,
):
    """
    Run a frozen policy for a fixed number of environment steps
    and collect action-distribution diagnostics.
    """

    reset = reset_5v5_env(
        env,
        seed=seed,
    )

    setup = reset.setup

    roster = env.roster_metadata(
        reset.rollout
    )

    si_roster = env.roster_metadata_si(
        reset.rollout
    )

    opponent_base_key = jax.random.key(
        seed + 10000
    )

    rollout_base_key = jax.random.key(
        seed + 20000
    )

    opponent_key = jax.random.fold_in(
        opponent_base_key,
        0,
    )

    rollout_key = jax.random.fold_in(
        rollout_base_key,
        0,
    )

    carry = initialize_jit_rollout_carry(
        env=env,
        reset=reset,
        opponent_key=opponent_key,
    )

    # ============================================================
    # JIT actor distribution evaluator
    # ============================================================

    @jax.jit
    def get_actor_output(
        params,
        actor_obs,
    ):
        latent = backbone.apply(
            {
                "params":
                    params["backbone"]
            },
            actor_obs,
        )

        return action_head.apply(
            {
                "params":
                    params["head"]
            },
            latent,
        )

    # ============================================================
    # Diagnostics storage
    # ============================================================

    sampled_intents = []

    policy_probs = []

    policy_means = []
    policy_stds = []

    sampled_pre_tanh = []
    sampled_actions = []

    categorical_entropies = []

    total_return = 0.0
    total_valid_steps = 0

    num_chunks = (
        num_steps
        // rollout_length
    )

    print(
        "\n========================================"
    )

    print(
        f"Policy diagnostics: {label}"
    )

    print(
        "========================================"
    )

    # ============================================================
    # Rollout
    # ============================================================

    for chunk_index in range(
        num_chunks
    ):
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

        jax.block_until_ready(
            carry.rollout.state.control_tick
        )

        actor_output = get_actor_output(
            actor_params,
            batch.actor_obs,
        )

        jax.block_until_ready(
            actor_output.intent_logits
        )

        # --------------------------------------------------------
        # Move data to CPU
        # --------------------------------------------------------

        valid = np.asarray(
            jax.device_get(
                valid_mask
            )
        ).astype(bool)

        intents = np.asarray(
            jax.device_get(
                batch.intent
            )
        )

        pre_tanh = np.asarray(
            jax.device_get(
                batch.pre_tanh_continuous
            )
        )

        logits = np.asarray(
            jax.device_get(
                actor_output.intent_logits
            )
        )

        means = np.asarray(
            jax.device_get(
                actor_output.continuous_mean
            )
        )

        log_stds = np.asarray(
            jax.device_get(
                actor_output.continuous_log_std
            )
        )

        # --------------------------------------------------------
        # Keep real transitions only
        #
        # valid shape:
        #     [T]
        #
        # actor quantities:
        #     [T, 5, ...]
        # --------------------------------------------------------

        intents = intents[valid]

        pre_tanh = pre_tanh[valid]

        logits = logits[valid]

        means = means[valid]

        log_stds = log_stds[valid]

        # --------------------------------------------------------
        # Policy intent probabilities
        # --------------------------------------------------------

        logits_shifted = (
            logits
            - np.max(
                logits,
                axis=-1,
                keepdims=True,
            )
        )

        exp_logits = np.exp(
            logits_shifted
        )

        probs = (
            exp_logits
            / np.sum(
                exp_logits,
                axis=-1,
                keepdims=True,
            )
        )

        categorical_entropy = -np.sum(
            probs
            * np.log(
                probs + 1e-12
            ),
            axis=-1,
        )

        # --------------------------------------------------------
        # Actual continuous actions
        # --------------------------------------------------------

        actual_action = np.tanh(
            pre_tanh
        )

        stds = np.exp(
            log_stds
        )

        # --------------------------------------------------------
        # Store
        # --------------------------------------------------------

        sampled_intents.append(
            intents.reshape(-1)
        )

        policy_probs.append(
            probs.reshape(
                -1,
                probs.shape[-1],
            )
        )

        policy_means.append(
            means.reshape(
                -1,
                means.shape[-1],
            )
        )

        policy_stds.append(
            stds.reshape(
                -1,
                stds.shape[-1],
            )
        )

        sampled_pre_tanh.append(
            pre_tanh.reshape(
                -1,
                pre_tanh.shape[-1],
            )
        )

        sampled_actions.append(
            actual_action.reshape(
                -1,
                actual_action.shape[-1],
            )
        )

        categorical_entropies.append(
            categorical_entropy.reshape(
                -1
            )
        )

        # --------------------------------------------------------
        # Reward
        # --------------------------------------------------------

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

        total_return += (
            chunk_return
        )

        total_valid_steps += int(
            np.sum(
                valid
            )
        )

        if (
            (chunk_index + 1) % 10
            == 0
        ):
            score = np.asarray(
                jax.device_get(
                    carry.rollout.state.score
                )
            )

            print(
                f"chunk={chunk_index + 1}"
                f"/{num_chunks}"
                f" | steps={total_valid_steps}"
                f" | score="
                f"{int(score[0])}"
                f"-{int(score[1])}"
            )

    # ============================================================
    # Aggregate
    # ============================================================

    sampled_intents = np.concatenate(
        sampled_intents,
        axis=0,
    )

    policy_probs = np.concatenate(
        policy_probs,
        axis=0,
    )

    policy_means = np.concatenate(
        policy_means,
        axis=0,
    )

    policy_stds = np.concatenate(
        policy_stds,
        axis=0,
    )

    sampled_pre_tanh = np.concatenate(
        sampled_pre_tanh,
        axis=0,
    )

    sampled_actions = np.concatenate(
        sampled_actions,
        axis=0,
    )

    categorical_entropies = np.concatenate(
        categorical_entropies,
        axis=0,
    )

    score = np.asarray(
        jax.device_get(
            carry.rollout.state.score
        )
    )

    # ============================================================
    # Intent statistics
    # ============================================================

    intent_counts = np.bincount(
        sampled_intents,
        minlength=len(
            INTENT_NAMES
        ),
    )

    intent_frequency = (
        intent_counts
        / np.sum(
            intent_counts
        )
    )

    mean_policy_prob = np.mean(
        policy_probs,
        axis=0,
    )

    # ============================================================
    # Continuous statistics
    # ============================================================

    mean_policy_mean = np.mean(
        policy_means,
        axis=0,
    )

    mean_abs_policy_mean = np.mean(
        np.abs(
            policy_means
        ),
        axis=0,
    )

    mean_policy_std = np.mean(
        policy_stds,
        axis=0,
    )

    actual_mean = np.mean(
        sampled_actions,
        axis=0,
    )

    actual_abs_mean = np.mean(
        np.abs(
            sampled_actions
        ),
        axis=0,
    )

    actual_std = np.std(
        sampled_actions,
        axis=0,
    )

    saturation_rate = np.mean(
        np.abs(
            sampled_actions
        ) > 0.95,
        axis=0,
    )

    policy_mean_saturation = np.mean(
        np.abs(
            np.tanh(
                policy_means
            )
        ) > 0.95,
        axis=0,
    )

    # ============================================================
    # Print result
    # ============================================================

    print(
        "\nFinal score:",
        f"{int(score[0])}-{int(score[1])}",
    )

    print(
        "Return:",
        total_return,
    )

    print(
        "Controlled-player action samples:",
        len(
            sampled_intents
        ),
    )

    print(
        "\n--- Intent Distribution ---"
    )

    print(
        f"{'Intent':<12}"
        f"{'Sampled %':>12}"
        f"{'Mean Prob %':>15}"
    )

    for index, name in enumerate(
        INTENT_NAMES
    ):
        print(
            f"{name:<12}"
            f"{100.0 * intent_frequency[index]:>11.2f}%"
            f"{100.0 * mean_policy_prob[index]:>14.2f}%"
        )

    print(
        "\nMean categorical entropy:",
        f"{np.mean(categorical_entropies):.4f}",
    )

    print(
        "\n--- Continuous Action Diagnostics ---"
    )

    header = (
        f"{'Dimension':<15}"
        f"{'Mean(mu)':>11}"
        f"{'|mu|':>10}"
        f"{'Std(pi)':>10}"
        f"{'ActMean':>10}"
        f"{'|Act|':>10}"
        f"{'ActStd':>10}"
        f"{'Sat>0.95':>11}"
        f"{'MuSat':>9}"
    )

    print(
        header
    )

    for index, name in enumerate(
        CONTINUOUS_NAMES
    ):
        print(
            f"{name:<15}"
            f"{mean_policy_mean[index]:>11.4f}"
            f"{mean_abs_policy_mean[index]:>10.4f}"
            f"{mean_policy_std[index]:>10.4f}"
            f"{actual_mean[index]:>10.4f}"
            f"{actual_abs_mean[index]:>10.4f}"
            f"{actual_std[index]:>10.4f}"
            f"{100.0 * saturation_rate[index]:>10.2f}%"
            f"{100.0 * policy_mean_saturation[index]:>8.2f}%"
        )

    print(
        "\nOverall actual-action saturation (>0.95):",
        f"{100.0 * np.mean(saturation_rate):.2f}%",
    )

    print(
        "Overall policy-mean saturation:",
        f"{100.0 * np.mean(policy_mean_saturation):.2f}%",
    )

    return {
        "score":
            score,

        "return":
            total_return,

        "intent_frequency":
            intent_frequency,

        "mean_policy_prob":
            mean_policy_prob,

        "categorical_entropy":
            np.mean(
                categorical_entropies
            ),

        "mean_policy_mean":
            mean_policy_mean,

        "mean_abs_policy_mean":
            mean_abs_policy_mean,

        "mean_policy_std":
            mean_policy_std,

        "actual_mean":
            actual_mean,

        "actual_abs_mean":
            actual_abs_mean,

        "actual_std":
            actual_std,

        "saturation_rate":
            saturation_rate,

        "policy_mean_saturation":
            policy_mean_saturation,
    }


def print_comparison(
    initial,
    trained,
):
    print(
        "\n\n========================================"
    )

    print(
        "INITIAL vs TRAINED COMPARISON"
    )

    print(
        "========================================"
    )

    print(
        "\n--- Intent frequency change ---"
    )

    print(
        f"{'Intent':<12}"
        f"{'Initial %':>12}"
        f"{'16K %':>12}"
        f"{'Change':>12}"
    )

    for index, name in enumerate(
        INTENT_NAMES
    ):
        initial_value = (
            100.0
            * initial[
                "intent_frequency"
            ][index]
        )

        trained_value = (
            100.0
            * trained[
                "intent_frequency"
            ][index]
        )

        change = (
            trained_value
            - initial_value
        )

        print(
            f"{name:<12}"
            f"{initial_value:>11.2f}%"
            f"{trained_value:>11.2f}%"
            f"{change:>+11.2f}"
        )

    print(
        "\nCategorical entropy:"
    )

    print(
        "  initial:",
        f"{initial['categorical_entropy']:.4f}",
    )

    print(
        "  16K:",
        f"{trained['categorical_entropy']:.4f}",
    )

    print(
        "\nOverall action saturation:"
    )

    print(
        "  initial:",
        f"{100.0 * np.mean(initial['saturation_rate']):.2f}%",
    )

    print(
        "  16K:",
        f"{100.0 * np.mean(trained['saturation_rate']):.2f}%",
    )

    print(
        "\nOverall policy-mean saturation:"
    )

    print(
        "  initial:",
        (
            f"{100.0 * np.mean(initial['policy_mean_saturation']):.2f}%"
        ),
    )

    print(
        "  16K:",
        (
            f"{100.0 * np.mean(trained['policy_mean_saturation']):.2f}%"
        ),
    )


def main():
    args = parse_args()

    if (
        args.num_steps
        % args.rollout_length
        != 0
    ):
        raise ValueError(
            "--num-steps must be divisible by "
            "--rollout-length."
        )

    # ============================================================
    # Environment + initial models
    # ============================================================

    env, reset = make_5v5_env(
        seed=args.seed
    )

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
    # Load trained checkpoint
    # ============================================================

    checkpoint = load_checkpoint(
        args.checkpoint
    )

    trained_actor_params = jax.device_put(
        checkpoint[
            "actor_params"
        ]
    )

    trained_critic_params = jax.device_put(
        checkpoint[
            "critic_params"
        ]
    )

    # ============================================================
    # Same collector for both policies
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

    collector = make_jitted_rollout_collector(
        env=env,

        backbone=backbone,
        action_head=action_head,
        critic=critic,

        rollout_length=args.rollout_length,

        controlled_indices=controlled_indices,

        controlled_team=0,
    )

    # ============================================================
    # Initial policy
    # ============================================================

    initial_result = evaluate_policy_behavior(
        label="INITIAL MLP",

        env=env,

        backbone=backbone,
        action_head=action_head,
        critic=critic,

        collector=collector,

        actor_params=initial_actor_params,
        critic_params=initial_critic_params,

        num_steps=args.num_steps,
        rollout_length=args.rollout_length,

        seed=args.seed,
    )

    # ============================================================
    # Trained policy
    # ============================================================

    trained_result = evaluate_policy_behavior(
        label="MLP @ 16K",

        env=env,

        backbone=backbone,
        action_head=action_head,
        critic=critic,

        collector=collector,

        actor_params=trained_actor_params,
        critic_params=trained_critic_params,

        num_steps=args.num_steps,
        rollout_length=args.rollout_length,

        seed=args.seed,
    )

    # ============================================================
    # Direct comparison
    # ============================================================

    print_comparison(
        initial_result,
        trained_result,
    )


if __name__ == "__main__":
    main()