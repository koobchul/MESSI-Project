


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


SHOT_INDEX = INTENT_NAMES.index("SHOT")

# 5v5 MLP observation layout:
#   10 players * 30 features = 300 player features
#   context starts at index 300
#   self position:     300:302
#   self velocity:     302:304
#   ball relative xy:  304:306
# MESSI uses ball.relative_state[:, :2] as observer-relative ball XY.
MLP_5V5_PLAYER_FEATURE_DIM = 30
MLP_5V5_NUM_PLAYERS = 10
MLP_5V5_CONTEXT_START = (
    MLP_5V5_PLAYER_FEATURE_DIM
    * MLP_5V5_NUM_PLAYERS
)
MLP_5V5_BALL_XY_START = MLP_5V5_CONTEXT_START + 4
MLP_5V5_BALL_XY_END = MLP_5V5_BALL_XY_START + 2

# Per-player feature offsets inside each 30D player block.
# build_player_features() layout:
#   continuous features: 0..11
#   status features:
#       yellow_cards                    12
#       on_pitch                        13
#       sent_off                        14
#       offside                         15
#       possessor                       16
#       restart_taker                   17
#       release_taker                   18
#       last_actor                      19
#       contact_may_occur_this_frame    20
#   roster features:                   21..26
#   relationship:
#       is_self                         27
#       is_teammate                     28
#       is_opponent                     29
MLP_PLAYER_POSSESSOR_OFFSET = 16
MLP_PLAYER_CONTACT_MAY_OCCUR_OFFSET = 20
MLP_PLAYER_IS_SELF_OFFSET = 27

SHOT_DISTANCE_THRESHOLDS_M = (
    0.10,
    0.20,
    0.50,
)





# ================================================================

# Continuous dimensions that are most behaviorally relevant

# for each sampled intent.

#

# MOVE:

#     locomotion vector

#

# Non-MOVE intents:

#     contact-related continuous controls

#

# This mapping is used ONLY for readable diagnostics.

# It does not change the policy, action, or environment.

# ================================================================



RELEVANT_CONTINUOUS_BY_INTENT = {

    "MOVE": [

        0,  # move_x

        1,  # move_y

    ],



    "CONTROL": [

        2,  # force_ball_x

        3,  # force_ball_y

        4,  # launch

        5,  # spin_x

        6,  # spin_y

    ],



    "PASS": [

        2,

        3,

        4,

        5,

        6,

    ],



    "SHOT": [

        2,

        3,

        4,

        5,

        6,

    ],



    "CLEAR": [

        2,

        3,

        4,

        5,

        6,

    ],



    "CHALLENGE": [

        2,

        3,

        4,

        5,

        6,

    ],

}





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





def compute_continuous_statistics(

    policy_means,

    policy_stds,

    sampled_actions,

):

    """

    Compute continuous-action statistics for a selected collection

    of actor samples.



    Inputs

    ------

    policy_means:

        [N, 7]



    policy_stds:

        [N, 7]



    sampled_actions:

        [N, 7]

        Post-tanh actions in [-1, 1].

    """



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



    return {

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





def compute_intent_conditioned_statistics(

    sampled_intents,

    policy_means,

    policy_stds,

    sampled_actions,

):

    """

    Compute continuous-action statistics conditioned on the

    sampled discrete intent.



    Example

    -------

    PASS statistics contain only samples where:



        sampled_intent == PASS



    This prevents continuous outputs from unrelated intents from

    being mixed into one global average.

    """



    result = {}



    total_samples = len(

        sampled_intents

    )



    for intent_index, intent_name in enumerate(

        INTENT_NAMES

    ):

        mask = (

            sampled_intents

            == intent_index

        )



        count = int(

            np.sum(

                mask

            )

        )



        if count == 0:

            result[intent_name] = {

                "count": 0,

                "frequency": 0.0,

                "stats": None,

            }



            continue



        stats = compute_continuous_statistics(

            policy_means=policy_means[

                mask

            ],



            policy_stds=policy_stds[

                mask

            ],



            sampled_actions=sampled_actions[

                mask

            ],

        )



        result[intent_name] = {

            "count":

                count,



            "frequency":

                count / total_samples,



            "stats":

                stats,

        }



    return result





def print_continuous_table(

    stats,

    dimensions=None,

):

    """

    Print one continuous-action statistics table.



    dimensions:

        Optional list of dimension indices.

        If omitted, print all seven dimensions.

    """



    if dimensions is None:

        dimensions = list(

            range(

                len(

                    CONTINUOUS_NAMES

                )

            )

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



    for index in dimensions:

        name = CONTINUOUS_NAMES[

            index

        ]



        print(

            f"{name:<15}"

            f"{stats['mean_policy_mean'][index]:>11.4f}"

            f"{stats['mean_abs_policy_mean'][index]:>10.4f}"

            f"{stats['mean_policy_std'][index]:>10.4f}"

            f"{stats['actual_mean'][index]:>10.4f}"

            f"{stats['actual_abs_mean'][index]:>10.4f}"

            f"{stats['actual_std'][index]:>10.4f}"

            f"{100.0 * stats['saturation_rate'][index]:>10.2f}%"

            f"{100.0 * stats['policy_mean_saturation'][index]:>8.2f}%"

        )





def print_intent_conditioned_diagnostics(

    intent_conditioned,

):

    """

    Print continuous statistics separately for each sampled intent.

    """



    print(

        "\n--- Intent-Conditioned Continuous Diagnostics ---"

    )



    for intent_name in INTENT_NAMES:

        item = intent_conditioned[

            intent_name

        ]



        print(

            "\n"

            f"[{intent_name}]"

            f" samples={item['count']}"

            f""

            f" ({100.0 * item['frequency']:.2f}%)"

        )



        if item[

            "stats"

        ] is None:

            print(

                "No samples."

            )



            continue



        relevant_dimensions = (

            RELEVANT_CONTINUOUS_BY_INTENT[

                intent_name

            ]

        )



        print_continuous_table(

            item["stats"],

            dimensions=relevant_dimensions,

        )





def compute_shot_context_statistics(
    sampled_intents,
    policy_probs,
    actor_observations,
    policy_means,
    sampled_actions,
):
    """
    Diagnose whether SHOT is selected in physically meaningful ball contexts.

    The first two entries of ball.relative_state are the observer-relative
    horizontal ball position. In the 5v5 flat MLP observation they occupy
    indices 304:306. Their Euclidean norm is therefore the player's
    horizontal distance to the ball in the environment's distance units
    (meters in the MESSI contact geometry).
    """

    if actor_observations.ndim != 2:
        raise ValueError(
            "actor_observations must have shape [N, obs_dim]."
        )

    if actor_observations.shape[1] < MLP_5V5_BALL_XY_END:
        raise ValueError(
            "actor observation is too small for the expected 5v5 MLP layout: "
            f"got obs_dim={actor_observations.shape[1]}, "
            f"need at least {MLP_5V5_BALL_XY_END}."
        )

    if len(sampled_intents) != len(actor_observations):
        raise ValueError(
            "sampled_intents and actor_observations must have the same "
            "number of samples."
        )

    ball_xy = actor_observations[
        :,
        MLP_5V5_BALL_XY_START:MLP_5V5_BALL_XY_END,
    ]

    ball_distance = np.linalg.norm(
        ball_xy,
        axis=-1,
    )

    shot_mask = sampled_intents == SHOT_INDEX
    shot_probability = policy_probs[:, SHOT_INDEX]

    threshold_stats = {}

    for threshold in SHOT_DISTANCE_THRESHOLDS_M:
        near_mask = ball_distance <= threshold
        far_mask = ~near_mask

        near_count = int(np.sum(near_mask))
        far_count = int(np.sum(far_mask))
        near_shot_count = int(np.sum(shot_mask & near_mask))
        far_shot_count = int(np.sum(shot_mask & far_mask))

        threshold_stats[threshold] = {
            "near_count": near_count,
            "far_count": far_count,
            "near_sampled_shot_rate": (
                float(np.mean(shot_mask[near_mask]))
                if near_count > 0
                else np.nan
            ),
            "far_sampled_shot_rate": (
                float(np.mean(shot_mask[far_mask]))
                if far_count > 0
                else np.nan
            ),
            "near_mean_shot_probability": (
                float(np.mean(shot_probability[near_mask]))
                if near_count > 0
                else np.nan
            ),
            "far_mean_shot_probability": (
                float(np.mean(shot_probability[far_mask]))
                if far_count > 0
                else np.nan
            ),
            "shot_samples_near": near_shot_count,
            "shot_samples_far": far_shot_count,
            "fraction_of_shots_near": (
                near_shot_count / int(np.sum(shot_mask))
                if np.any(shot_mask)
                else np.nan
            ),
        }

    shot_continuous = None

    if np.any(shot_mask):
        shot_continuous = {
            "mean_policy_mean": np.mean(
                policy_means[shot_mask],
                axis=0,
            ),
            "mean_actual_action": np.mean(
                sampled_actions[shot_mask],
                axis=0,
            ),
        }

    return {
        "sample_count": int(len(sampled_intents)),
        "shot_count": int(np.sum(shot_mask)),
        "sampled_shot_rate": float(np.mean(shot_mask)),
        "mean_shot_probability": float(np.mean(shot_probability)),
        "mean_ball_distance": float(np.mean(ball_distance)),
        "median_ball_distance": float(np.median(ball_distance)),
        "mean_ball_distance_when_shot": (
            float(np.mean(ball_distance[shot_mask]))
            if np.any(shot_mask)
            else np.nan
        ),
        "median_ball_distance_when_shot": (
            float(np.median(ball_distance[shot_mask]))
            if np.any(shot_mask)
            else np.nan
        ),
        "thresholds": threshold_stats,
        "shot_continuous": shot_continuous,
    }



def compute_shot_contact_statistics(
    sampled_intents,
    policy_probs,
    actor_observations,
):
    """
    Diagnose whether SHOT is selected when the acting player can actually
    interact with the ball.

    The first 300 entries of the 5v5 MLP observation contain
    10 player blocks * 30 features. We locate the observer's own block using
    the explicit is_self feature, then read:
        possessor                     -> feature 16
        contact_may_occur_this_frame  -> feature 20

    This avoids assuming that observer row k always corresponds to player
    block k.
    """

    if actor_observations.ndim != 2:
        raise ValueError(
            "actor_observations must have shape [N, obs_dim]."
        )

    player_flat_dim = (
        MLP_5V5_NUM_PLAYERS
        * MLP_5V5_PLAYER_FEATURE_DIM
    )

    if actor_observations.shape[1] < player_flat_dim:
        raise ValueError(
            "actor observation is too small for 5v5 player features: "
            f"got obs_dim={actor_observations.shape[1]}, "
            f"need at least {player_flat_dim}."
        )

    player_features = actor_observations[
        :,
        :player_flat_dim,
    ].reshape(
        -1,
        MLP_5V5_NUM_PLAYERS,
        MLP_5V5_PLAYER_FEATURE_DIM,
    )

    self_scores = player_features[
        :,
        :,
        MLP_PLAYER_IS_SELF_OFFSET,
    ]

    self_index = np.argmax(
        self_scores,
        axis=1,
    )

    rows = np.arange(
        player_features.shape[0]
    )

    self_flags = self_scores[
        rows,
        self_index,
    ]

    if not np.all(self_flags > 0.5):
        bad = int(
            np.sum(
                self_flags <= 0.5
            )
        )
        raise ValueError(
            "Could not reliably locate self player block for "
            f"{bad} samples."
        )

    self_possessor = (
        player_features[
            rows,
            self_index,
            MLP_PLAYER_POSSESSOR_OFFSET,
        ]
        > 0.5
    )

    contact_possible = (
        player_features[
            rows,
            self_index,
            MLP_PLAYER_CONTACT_MAY_OCCUR_OFFSET,
        ]
        > 0.5
    )

    shot_mask = (
        sampled_intents
        == SHOT_INDEX
    )

    shot_probability = policy_probs[
        :,
        SHOT_INDEX,
    ]

    context_masks = {
        "contact possible":
            contact_possible,

        "no contact possible":
            ~contact_possible,

        "self is possessor":
            self_possessor,

        "self not possessor":
            ~self_possessor,

        "contact & possessor":
            contact_possible
            & self_possessor,

        "contact & not possessor":
            contact_possible
            & (~self_possessor),
    }

    contexts = {}

    total_shots = int(
        np.sum(
            shot_mask
        )
    )

    for name, mask in context_masks.items():
        count = int(
            np.sum(
                mask
            )
        )

        shot_count = int(
            np.sum(
                shot_mask
                & mask
            )
        )

        contexts[name] = {
            "count":
                count,

            "shot_count":
                shot_count,

            "sampled_shot_rate":
                (
                    float(
                        np.mean(
                            shot_mask[
                                mask
                            ]
                        )
                    )
                    if count > 0
                    else np.nan
                ),

            "mean_shot_probability":
                (
                    float(
                        np.mean(
                            shot_probability[
                                mask
                            ]
                        )
                    )
                    if count > 0
                    else np.nan
                ),

            "fraction_of_all_shots":
                (
                    shot_count
                    / total_shots
                    if total_shots > 0
                    else np.nan
                ),
        }

    return {
        "sample_count":
            int(
                len(
                    sampled_intents
                )
            ),

        "contact_possible_rate":
            float(
                np.mean(
                    contact_possible
                )
            ),

        "possessor_rate":
            float(
                np.mean(
                    self_possessor
                )
            ),

        "contexts":
            contexts,
    }


def print_shot_contact_diagnostics(shot_contact):
    """Print SHOT behavior conditioned on contact eligibility and possession."""

    print(
        "\n--- SHOT Contact / Possession Diagnostics ---"
    )

    print(
        "Contact possible rate:",
        f"{100.0 * shot_contact['contact_possible_rate']:.2f}%"
    )

    print(
        "Self possessor rate:",
        f"{100.0 * shot_contact['possessor_rate']:.2f}%"
    )

    print(
        "\n"
        f"{'Context':<26}"
        f"{'N':>8}"
        f"{'SHOT N':>9}"
        f"{'SHOT %':>11}"
        f"{'Mean P(SHOT) %':>17}"
        f"{'% of all SHOTs':>17}"
    )

    ordered_contexts = [
        "contact possible",
        "no contact possible",
        "self is possessor",
        "self not possessor",
        "contact & possessor",
        "contact & not possessor",
    ]

    for name in ordered_contexts:
        stats = shot_contact[
            "contexts"
        ][
            name
        ]

        print(
            f"{name:<26}"
            f"{stats['count']:>8}"
            f"{stats['shot_count']:>9}"
            f"{100.0 * stats['sampled_shot_rate']:>10.2f}%"
            f"{100.0 * stats['mean_shot_probability']:>16.2f}%"
            f"{100.0 * stats['fraction_of_all_shots']:>16.2f}%"
        )


def print_shot_contact_comparison(
    initial,
    trained,
):
    """Compare SHOT contact sensitivity between initial and trained policies."""

    print(
        "\n--- SHOT Contact / Possession Change ---"
    )

    print(
        f"{'Context':<26}"
        f"{'Initial SHOT %':>16}"
        f"{'Trained SHOT %':>16}"
        f"{'Change':>12}"
    )

    ordered_contexts = [
        "contact possible",
        "no contact possible",
        "self is possessor",
        "self not possessor",
        "contact & possessor",
        "contact & not possessor",
    ]

    for name in ordered_contexts:
        initial_rate = (
            100.0
            * initial[
                "contexts"
            ][
                name
            ][
                "sampled_shot_rate"
            ]
        )

        trained_rate = (
            100.0
            * trained[
                "contexts"
            ][
                name
            ][
                "sampled_shot_rate"
            ]
        )

        print(
            f"{name:<26}"
            f"{initial_rate:>16.2f}"
            f"{trained_rate:>16.2f}"
            f"{trained_rate - initial_rate:>+12.2f}"
        )


def print_shot_context_diagnostics(shot_context):
    """Print SHOT frequency as a function of player-to-ball distance."""

    print("\n--- SHOT Context Diagnostics ---")
    print(
        "Sampled SHOT rate:",
        f"{100.0 * shot_context['sampled_shot_rate']:.2f}%",
        f"({shot_context['shot_count']}/{shot_context['sample_count']})",
    )
    print(
        "Mean policy SHOT probability:",
        f"{100.0 * shot_context['mean_shot_probability']:.2f}%",
    )
    print(
        "Ball distance overall (mean / median):",
        f"{shot_context['mean_ball_distance']:.4f} / "
        f"{shot_context['median_ball_distance']:.4f} m",
    )
    print(
        "Ball distance when SHOT (mean / median):",
        f"{shot_context['mean_ball_distance_when_shot']:.4f} / "
        f"{shot_context['median_ball_distance_when_shot']:.4f} m",
    )

    print(
        "\n"
        f"{'Context':<18}"
        f"{'N':>8}"
        f"{'Sampled SHOT %':>17}"
        f"{'Mean SHOT Prob %':>19}"
        f"{'% of SHOTs here':>18}"
    )

    for threshold in SHOT_DISTANCE_THRESHOLDS_M:
        stats = shot_context["thresholds"][threshold]

        print(
            f"dist <= {threshold:.2f} m".ljust(18)
            + f"{stats['near_count']:>8}"
            + f"{100.0 * stats['near_sampled_shot_rate']:>16.2f}%"
            + f"{100.0 * stats['near_mean_shot_probability']:>18.2f}%"
            + f"{100.0 * stats['fraction_of_shots_near']:>17.2f}%"
        )

    far_threshold = SHOT_DISTANCE_THRESHOLDS_M[-1]
    far_stats = shot_context["thresholds"][far_threshold]

    print(
        f"dist >  {far_threshold:.2f} m".ljust(18)
        + f"{far_stats['far_count']:>8}"
        + f"{100.0 * far_stats['far_sampled_shot_rate']:>16.2f}%"
        + f"{100.0 * far_stats['far_mean_shot_probability']:>18.2f}%"
        + f"{100.0 * (1.0 - far_stats['fraction_of_shots_near']):>17.2f}%"
    )

    shot_continuous = shot_context["shot_continuous"]

    if shot_continuous is None:
        print("\nNo sampled SHOT actions; continuous SHOT diagnostics unavailable.")
        return

    print("\nSHOT continuous controls (all sampled SHOT actions)")
    print(
        f"{'Dimension':<15}"
        f"{'Mean(mu)':>12}"
        f"{'ActMean':>12}"
    )

    for index in RELEVANT_CONTINUOUS_BY_INTENT["SHOT"]:
        print(
            f"{CONTINUOUS_NAMES[index]:<15}"
            f"{shot_continuous['mean_policy_mean'][index]:>12.4f}"
            f"{shot_continuous['mean_actual_action'][index]:>12.4f}"
        )


def print_shot_context_comparison(initial, trained):
    """Compare SHOT context sensitivity between initial and trained policies."""

    print("\n--- SHOT Context Change ---")
    print(
        f"{'Metric':<34}"
        f"{'Initial':>12}"
        f"{'Trained':>12}"
        f"{'Change':>12}"
    )

    scalar_metrics = [
        (
            "Sampled SHOT %",
            100.0 * initial["sampled_shot_rate"],
            100.0 * trained["sampled_shot_rate"],
        ),
        (
            "Mean SHOT probability %",
            100.0 * initial["mean_shot_probability"],
            100.0 * trained["mean_shot_probability"],
        ),
        (
            "Mean ball distance when SHOT (m)",
            initial["mean_ball_distance_when_shot"],
            trained["mean_ball_distance_when_shot"],
        ),
    ]

    for name, initial_value, trained_value in scalar_metrics:
        print(
            f"{name:<34}"
            f"{initial_value:>12.4f}"
            f"{trained_value:>12.4f}"
            f"{trained_value - initial_value:>+12.4f}"
        )

    print("\nSHOT rate by ball-distance context")
    print(
        f"{'Context':<18}"
        f"{'Initial %':>12}"
        f"{'Trained %':>12}"
        f"{'Change':>12}"
    )

    for threshold in SHOT_DISTANCE_THRESHOLDS_M:
        initial_rate = (
            100.0
            * initial["thresholds"][threshold]["near_sampled_shot_rate"]
        )
        trained_rate = (
            100.0
            * trained["thresholds"][threshold]["near_sampled_shot_rate"]
        )

        print(
            f"dist <= {threshold:.2f} m".ljust(18)
            + f"{initial_rate:>12.2f}"
            + f"{trained_rate:>12.2f}"
            + f"{trained_rate - initial_rate:>+12.2f}"
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

    actor_observations = []

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


        actor_obs = np.asarray(

            jax.device_get(

                batch.actor_obs

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

        # valid:

        #     [T]

        #

        # actor quantities:

        #     [T, 5, ...]

        # --------------------------------------------------------



        intents = intents[

            valid

        ]


        actor_obs = actor_obs[

            valid

        ]



        pre_tanh = pre_tanh[

            valid

        ]



        logits = logits[

            valid

        ]



        means = means[

            valid

        ]



        log_stds = log_stds[

            valid

        ]



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

            intents.reshape(

                -1

            )

        )


        actor_observations.append(

            actor_obs.reshape(

                -1,

                actor_obs.shape[-1],

            )

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

        #

        # The diagnostic collector uses its default shaping

        # coefficient of 0.0. Therefore this return remains the

        # original sparse MESSI evaluation return.

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


    actor_observations = np.concatenate(

        actor_observations,

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

    # Overall continuous statistics

    # ============================================================



    continuous_stats = (

        compute_continuous_statistics(

            policy_means=policy_means,

            policy_stds=policy_stds,

            sampled_actions=sampled_actions,

        )

    )



    # ============================================================

    # Intent-conditioned continuous statistics

    # ============================================================



    intent_conditioned = (

        compute_intent_conditioned_statistics(

            sampled_intents=sampled_intents,

            policy_means=policy_means,

            policy_stds=policy_stds,

            sampled_actions=sampled_actions,

        )

    )


    shot_context = compute_shot_context_statistics(

        sampled_intents=sampled_intents,

        policy_probs=policy_probs,

        actor_observations=actor_observations,

        policy_means=policy_means,

        sampled_actions=sampled_actions,

    )


    shot_contact = compute_shot_contact_statistics(

        sampled_intents=sampled_intents,

        policy_probs=policy_probs,

        actor_observations=actor_observations,

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



    # ============================================================

    # Intent distribution

    # ============================================================



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



    # ============================================================

    # Overall continuous diagnostics

    # ============================================================



    print(

        "\n--- Overall Continuous Action Diagnostics ---"

    )



    print_continuous_table(

        continuous_stats

    )



    print(

        "\nOverall actual-action saturation (>0.95):",

        (

            f"{100.0 * np.mean(continuous_stats['saturation_rate']):.2f}%"

        ),

    )



    print(

        "Overall policy-mean saturation:",

        (

            f"{100.0 * np.mean(continuous_stats['policy_mean_saturation']):.2f}%"

        ),

    )



    # ============================================================

    # Intent-conditioned diagnostics

    # ============================================================



    print_intent_conditioned_diagnostics(

        intent_conditioned

    )


    print_shot_context_diagnostics(

        shot_context

    )


    print_shot_contact_diagnostics(

        shot_contact

    )



    # ============================================================

    # Return diagnostics

    # ============================================================



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

            continuous_stats[

                "mean_policy_mean"

            ],



        "mean_abs_policy_mean":

            continuous_stats[

                "mean_abs_policy_mean"

            ],



        "mean_policy_std":

            continuous_stats[

                "mean_policy_std"

            ],



        "actual_mean":

            continuous_stats[

                "actual_mean"

            ],



        "actual_abs_mean":

            continuous_stats[

                "actual_abs_mean"

            ],



        "actual_std":

            continuous_stats[

                "actual_std"

            ],



        "saturation_rate":

            continuous_stats[

                "saturation_rate"

            ],



        "policy_mean_saturation":

            continuous_stats[

                "policy_mean_saturation"

            ],



        "shot_context":

            shot_context,


        "shot_contact":

            shot_contact,



        "intent_conditioned":

            intent_conditioned,

    }





def print_intent_conditioned_comparison(

    initial,

    trained,

):

    """

    Directly compare continuous behavior within each sampled intent.



    This is the key diagnostic for determining whether large global

    continuous-action drift is occurring inside behaviorally relevant

    intent classes or merely in unrelated output dimensions.

    """



    print(

        "\n--- Intent-Conditioned Continuous Change ---"

    )



    for intent_name in INTENT_NAMES:

        initial_item = initial[

            "intent_conditioned"

        ][

            intent_name

        ]



        trained_item = trained[

            "intent_conditioned"

        ][

            intent_name

        ]



        print(

            "\n"

            f"[{intent_name}]"

            f" initial_n={initial_item['count']}"

            f""

            f" | trained_n={trained_item['count']}"

        )



        if (

            initial_item["stats"] is None

            or trained_item["stats"] is None

        ):

            print(

                "Insufficient samples."

            )



            continue



        initial_stats = initial_item[

            "stats"

        ]



        trained_stats = trained_item[

            "stats"

        ]



        print(

            f"{'Dimension':<15}"

            f"{'InitMu':>9}"

            f"{'TrainMu':>9}"

            f"{'dMu':>9}"

            f"{'InitAct':>10}"

            f"{'TrainAct':>10}"

            f"{'dAct':>9}"

            f"{'InitSat':>10}"

            f"{'TrainSat':>10}"

        )



        relevant_dimensions = (

            RELEVANT_CONTINUOUS_BY_INTENT[

                intent_name

            ]

        )



        for dimension_index in relevant_dimensions:

            dimension_name = (

                CONTINUOUS_NAMES[

                    dimension_index

                ]

            )



            initial_mu = (

                initial_stats[

                    "mean_policy_mean"

                ][dimension_index]

            )



            trained_mu = (

                trained_stats[

                    "mean_policy_mean"

                ][dimension_index]

            )



            delta_mu = (

                trained_mu

                - initial_mu

            )



            initial_action = (

                initial_stats[

                    "actual_mean"

                ][dimension_index]

            )



            trained_action = (

                trained_stats[

                    "actual_mean"

                ][dimension_index]

            )



            delta_action = (

                trained_action

                - initial_action

            )



            initial_saturation = (

                initial_stats[

                    "saturation_rate"

                ][dimension_index]

            )



            trained_saturation = (

                trained_stats[

                    "saturation_rate"

                ][dimension_index]

            )



            print(

                f"{dimension_name:<15}"

                f"{initial_mu:>9.4f}"

                f"{trained_mu:>9.4f}"

                f"{delta_mu:>+9.4f}"

                f"{initial_action:>10.4f}"

                f"{trained_action:>10.4f}"

                f"{delta_action:>+9.4f}"

                f"{100.0 * initial_saturation:>9.2f}%"

                f"{100.0 * trained_saturation:>9.2f}%"

            )





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



    # ============================================================

    # Intent frequency comparison

    # ============================================================



    print(

        "\n--- Intent Frequency Change ---"

    )



    print(

        f"{'Intent':<12}"

        f"{'Initial %':>12}"

        f"{'Trained %':>12}"

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



    # ============================================================

    # Entropy

    # ============================================================



    print(

        "\nCategorical entropy:"

    )



    print(

        "  initial:",

        f"{initial['categorical_entropy']:.4f}",

    )



    print(

        "  trained:",

        f"{trained['categorical_entropy']:.4f}",

    )



    # ============================================================

    # Overall saturation

    # ============================================================



    print(

        "\nOverall action saturation:"

    )



    print(

        "  initial:",

        (

            f"{100.0 * np.mean(initial['saturation_rate']):.2f}%"

        ),

    )



    print(

        "  trained:",

        (

            f"{100.0 * np.mean(trained['saturation_rate']):.2f}%"

        ),

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

        "  trained:",

        (

            f"{100.0 * np.mean(trained['policy_mean_saturation']):.2f}%"

        ),

    )



    # ============================================================

    # Intent-conditioned comparison

    # ============================================================



    print_intent_conditioned_comparison(

        initial,

        trained,

    )


    print_shot_context_comparison(

        initial["shot_context"],

        trained["shot_context"],

    )


    print_shot_contact_comparison(

        initial["shot_contact"],

        trained["shot_contact"],

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



    checkpoint_steps = checkpoint.get(

        "total_env_steps",

        None,

    )



    if checkpoint_steps is None:

        trained_label = (

            "TRAINED MLP"

        )



    else:

        trained_label = (

            f"MLP @ {int(checkpoint_steps)} steps"

        )



    # ============================================================

    # Same collector for both policies

    #

    # No shaping coefficient is supplied here.

    # Therefore the collector uses its default coefficient=0.0.

    # The diagnostic score/return is consequently evaluated using

    # the original sparse MESSI goal reward for both policies.

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

        label=trained_label,



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