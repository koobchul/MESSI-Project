# Diagnostic unit fix: restore physical fields on a separate CPU copy.
# Policy inference and rollout continue to use the original normalized input.
# Geometry is a proxy using default Ball/Reach/ActionScale values, not an
# exhaustive implementation of environment possession/release predicates.
import argparse
import jax
import jax.numpy as jnp
import numpy as np
from project.algorithms.rollout_jit import initialize_jit_rollout_carry, make_jitted_rollout_collector
from project.envs.critic_encoder import encode_critic_state
from project.envs.messi_5v5 import make_5v5_env, reset_5v5_env
from project.envs.obs_encoder import encode_mlp_observation
from project.models.action_head import HybridActionHead
from project.models.critic import CentralizedCritic
from project.models.mlp_actor import MLPActorBackbone
from project.utils.checkpoint import load_checkpoint
INTENT_NAMES = ['MOVE', 'CONTROL', 'PASS', 'SHOT', 'CLEAR', 'CHALLENGE']
CONTINUOUS_NAMES = ['move_x', 'move_y', 'force_ball_x', 'force_ball_y', 'launch', 'spin_x', 'spin_y']
SHOT_INDEX = INTENT_NAMES.index('SHOT')
CONTROL_INDEX = INTENT_NAMES.index('CONTROL')
MLP_5V5_PLAYER_FEATURE_DIM = 30
MLP_5V5_NUM_PLAYERS = 10
MLP_5V5_CONTEXT_START = MLP_5V5_PLAYER_FEATURE_DIM * MLP_5V5_NUM_PLAYERS
MLP_5V5_BALL_XY_START = MLP_5V5_CONTEXT_START + 4
MLP_5V5_BALL_XY_END = MLP_5V5_BALL_XY_START + 2
MLP_PLAYER_POSSESSOR_OFFSET = 16
MLP_PLAYER_LAST_ACTOR_OFFSET = 19
MLP_PLAYER_CONTACT_MAY_OCCUR_OFFSET = 20
MLP_PLAYER_IS_SELF_OFFSET = 27
MLP_PLAYER_IS_TEAMMATE_OFFSET = 28
MLP_PLAYER_IS_OPPONENT_OFFSET = 29
MLP_PLAYER_REACH_HEIGHT_OFFSET = 23
MLP_PLAYER_HEIGHT_OFFSET = 24
MLP_5V5_BALL_Z_INDEX = MLP_5V5_CONTEXT_START + 6
MLP_5V5_CONTROL_TICKS_INDEX = MLP_5V5_CONTEXT_START + 14
MLP_5V5_LAST_CONTACT_KICK_APPLIED_INDEX = MLP_5V5_CONTEXT_START + 15
MLP_5V5_CATEGORICAL_START = MLP_5V5_CONTEXT_START + 34
MLP_5V5_LAST_CONTACT_MECHANISM_START = MLP_5V5_CATEGORICAL_START + 3
MLP_5V5_LAST_CONTACT_MECHANISM_END = MLP_5V5_LAST_CONTACT_MECHANISM_START + 7
MLP_5V5_LAST_CONTACT_INTENT_START = MLP_5V5_CATEGORICAL_START + 3 + 7
MLP_5V5_LAST_CONTACT_INTENT_END = MLP_5V5_LAST_CONTACT_INTENT_START + 7
MLP_5V5_LAST_CONTACT_OUTCOME_START = MLP_5V5_LAST_CONTACT_INTENT_END + 3
MLP_5V5_LAST_CONTACT_OUTCOME_END = MLP_5V5_LAST_CONTACT_OUTCOME_START + 10
CONTACT_OUTCOME_NAMES = ('NONE', 'RELEASE', 'TRAP', 'INTERCEPTION', 'TACKLE_WON', 'DEFLECTION', 'CATCH', 'PARRY', 'FOUL', 'MISCONTROL')
OUTCOME_TRAP_INDEX = 2
OUTCOME_INTERCEPTION_INDEX = 3
OUTCOME_MISCONTROL_INDEX = 9
MECHANISM_CHEST_INDEX = 3
SHOT_DISTANCE_THRESHOLDS_M = (0.1, 0.2, 0.5)
RELEVANT_CONTINUOUS_BY_INTENT = {'MOVE': [0, 1], 'CONTROL': [2, 3, 4, 5, 6], 'PASS': [2, 3, 4, 5, 6], 'SHOT': [2, 3, 4, 5, 6], 'CLEAR': [2, 3, 4, 5, 6], 'CHALLENGE': [2, 3, 4, 5, 6]}

def restore_physical_diagnostic_fields(actor_observations, context):
    """Copy model observations and restore only fields used by diagnostics.

    This is NOT a full SI observation or a policy input. Boolean and
    categorical fields are unchanged. Scale each XY component before norm.
    The environment's context supplies scales; no pitch dimensions are assumed.
    """
    result = np.array(actor_observations, dtype=np.float64, copy=True)
    if result.ndim != 2 or result.shape[1] != 414:
        raise ValueError('Expected 5v5 actor observations with shape [N, 414].')
    result[:, MLP_5V5_BALL_XY_START] *= float(context.ball_relative_position_scale_x_m)
    result[:, MLP_5V5_BALL_XY_START + 1] *= float(context.ball_relative_position_scale_y_m)
    result[:, MLP_5V5_BALL_Z_INDEX] *= float(context.ball_height_scale_m)
    result[:, MLP_5V5_CONTROL_TICKS_INDEX] = np.rint(result[:, MLP_5V5_CONTROL_TICKS_INDEX] * float(context.counter_scale_ticks))
    players = result[:, :MLP_5V5_CONTEXT_START].reshape(-1, 10, 30)
    for offset, lower, upper in ((MLP_PLAYER_REACH_HEIGHT_OFFSET, context.min_reach_height_m, context.max_reach_height_m), (MLP_PLAYER_HEIGHT_OFFSET, context.min_height_m, context.max_height_m)):
        players[:, :, offset] = float(lower) + players[:, :, offset] * (float(upper) - float(lower))
    return result

def parse_args():
    parser = argparse.ArgumentParser(description='Compare initial and trained MLP policy behavior.')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num-steps', type=int, default=6400)
    parser.add_argument('--rollout-length', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1000)
    return parser.parse_args()

def initialize_models(env, reset):
    roster = env.roster_metadata(reset.rollout)
    obs = env.observe_all(reset.rollout)
    actor_obs = encode_mlp_observation(obs, roster)
    backbone = MLPActorBackbone(hidden_dim=256)
    backbone_variables = backbone.init(jax.random.key(1), actor_obs)
    latent = backbone.apply(backbone_variables, actor_obs)
    action_head = HybridActionHead()
    head_variables = action_head.init(jax.random.key(2), latent)
    actor_params = {'backbone': backbone_variables['params'], 'head': head_variables['params']}
    global_state = env.global_state_view(reset.rollout)
    critic_obs = encode_critic_state(global_state, controlled_team=0)
    critic = CentralizedCritic(hidden_dim=256)
    critic_variables = critic.init(jax.random.key(3), critic_obs[None, :])
    critic_params = critic_variables['params']
    return (backbone, action_head, critic, actor_params, critic_params)

def compute_continuous_statistics(policy_means, policy_stds, sampled_actions):
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
    mean_policy_mean = np.mean(policy_means, axis=0)
    mean_abs_policy_mean = np.mean(np.abs(policy_means), axis=0)
    mean_policy_std = np.mean(policy_stds, axis=0)
    actual_mean = np.mean(sampled_actions, axis=0)
    actual_abs_mean = np.mean(np.abs(sampled_actions), axis=0)
    actual_std = np.std(sampled_actions, axis=0)
    saturation_rate = np.mean(np.abs(sampled_actions) > 0.95, axis=0)
    policy_mean_saturation = np.mean(np.abs(np.tanh(policy_means)) > 0.95, axis=0)
    return {'mean_policy_mean': mean_policy_mean, 'mean_abs_policy_mean': mean_abs_policy_mean, 'mean_policy_std': mean_policy_std, 'actual_mean': actual_mean, 'actual_abs_mean': actual_abs_mean, 'actual_std': actual_std, 'saturation_rate': saturation_rate, 'policy_mean_saturation': policy_mean_saturation}

def compute_intent_conditioned_statistics(sampled_intents, policy_means, policy_stds, sampled_actions):
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
    total_samples = len(sampled_intents)
    for intent_index, intent_name in enumerate(INTENT_NAMES):
        mask = sampled_intents == intent_index
        count = int(np.sum(mask))
        if count == 0:
            result[intent_name] = {'count': 0, 'frequency': 0.0, 'stats': None}
            continue
        stats = compute_continuous_statistics(policy_means=policy_means[mask], policy_stds=policy_stds[mask], sampled_actions=sampled_actions[mask])
        result[intent_name] = {'count': count, 'frequency': count / total_samples, 'stats': stats}
    return result

def print_continuous_table(stats, dimensions=None):
    """
    Print one continuous-action statistics table.
    dimensions:
        Optional list of dimension indices.
        If omitted, print all seven dimensions.
    """
    if dimensions is None:
        dimensions = list(range(len(CONTINUOUS_NAMES)))
    header = f"{'Dimension':<15}{'Mean(mu)':>11}{'|mu|':>10}{'Std(pi)':>10}{'ActMean':>10}{'|Act|':>10}{'ActStd':>10}{'Sat>0.95':>11}{'MuSat':>9}"
    print(header)
    for index in dimensions:
        name = CONTINUOUS_NAMES[index]
        print(f"{name:<15}{stats['mean_policy_mean'][index]:>11.4f}{stats['mean_abs_policy_mean'][index]:>10.4f}{stats['mean_policy_std'][index]:>10.4f}{stats['actual_mean'][index]:>10.4f}{stats['actual_abs_mean'][index]:>10.4f}{stats['actual_std'][index]:>10.4f}{100.0 * stats['saturation_rate'][index]:>10.2f}%{100.0 * stats['policy_mean_saturation'][index]:>8.2f}%")

def print_intent_conditioned_diagnostics(intent_conditioned):
    """
    Print continuous statistics separately for each sampled intent.
    """
    print('\n--- Intent-Conditioned Continuous Diagnostics ---')
    for intent_name in INTENT_NAMES:
        item = intent_conditioned[intent_name]
        print(f"\n[{intent_name}] samples={item['count']} ({100.0 * item['frequency']:.2f}%)")
        if item['stats'] is None:
            print('No samples.')
            continue
        relevant_dimensions = RELEVANT_CONTINUOUS_BY_INTENT[intent_name]
        print_continuous_table(item['stats'], dimensions=relevant_dimensions)

def compute_shot_context_statistics(sampled_intents, policy_probs, actor_observations, policy_means, sampled_actions):
    """
    Diagnose whether SHOT is selected in physically meaningful ball contexts.

    The first two entries of ball.relative_state are the observer-relative
    horizontal ball position. In the 5v5 flat MLP observation they occupy
    indices 304:306. Their Euclidean norm is therefore the player's
    horizontal distance to the ball in the environment's distance units
    (meters in the MESSI contact geometry).
    """
    if actor_observations.ndim != 2:
        raise ValueError('actor_observations must have shape [N, obs_dim].')
    if actor_observations.shape[1] < MLP_5V5_BALL_XY_END:
        raise ValueError(f'actor observation is too small for the expected 5v5 MLP layout: got obs_dim={actor_observations.shape[1]}, need at least {MLP_5V5_BALL_XY_END}.')
    if len(sampled_intents) != len(actor_observations):
        raise ValueError('sampled_intents and actor_observations must have the same number of samples.')
    ball_xy = actor_observations[:, MLP_5V5_BALL_XY_START:MLP_5V5_BALL_XY_END]
    ball_distance = np.linalg.norm(ball_xy, axis=-1)
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
        threshold_stats[threshold] = {'near_count': near_count, 'far_count': far_count, 'near_sampled_shot_rate': float(np.mean(shot_mask[near_mask])) if near_count > 0 else np.nan, 'far_sampled_shot_rate': float(np.mean(shot_mask[far_mask])) if far_count > 0 else np.nan, 'near_mean_shot_probability': float(np.mean(shot_probability[near_mask])) if near_count > 0 else np.nan, 'far_mean_shot_probability': float(np.mean(shot_probability[far_mask])) if far_count > 0 else np.nan, 'shot_samples_near': near_shot_count, 'shot_samples_far': far_shot_count, 'fraction_of_shots_near': near_shot_count / int(np.sum(shot_mask)) if np.any(shot_mask) else np.nan}
    shot_continuous = None
    if np.any(shot_mask):
        shot_continuous = {'mean_policy_mean': np.mean(policy_means[shot_mask], axis=0), 'mean_actual_action': np.mean(sampled_actions[shot_mask], axis=0)}
    return {'sample_count': int(len(sampled_intents)), 'shot_count': int(np.sum(shot_mask)), 'sampled_shot_rate': float(np.mean(shot_mask)), 'mean_shot_probability': float(np.mean(shot_probability)), 'mean_ball_distance': float(np.mean(ball_distance)), 'median_ball_distance': float(np.median(ball_distance)), 'mean_ball_distance_when_shot': float(np.mean(ball_distance[shot_mask])) if np.any(shot_mask) else np.nan, 'median_ball_distance_when_shot': float(np.median(ball_distance[shot_mask])) if np.any(shot_mask) else np.nan, 'thresholds': threshold_stats, 'shot_continuous': shot_continuous}

def compute_shot_contact_statistics(sampled_intents, policy_probs, actor_observations):
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
        raise ValueError('actor_observations must have shape [N, obs_dim].')
    player_flat_dim = MLP_5V5_NUM_PLAYERS * MLP_5V5_PLAYER_FEATURE_DIM
    if actor_observations.shape[1] < player_flat_dim:
        raise ValueError(f'actor observation is too small for 5v5 player features: got obs_dim={actor_observations.shape[1]}, need at least {player_flat_dim}.')
    player_features = actor_observations[:, :player_flat_dim].reshape(-1, MLP_5V5_NUM_PLAYERS, MLP_5V5_PLAYER_FEATURE_DIM)
    self_scores = player_features[:, :, MLP_PLAYER_IS_SELF_OFFSET]
    self_index = np.argmax(self_scores, axis=1)
    rows = np.arange(player_features.shape[0])
    self_flags = self_scores[rows, self_index]
    if not np.all(self_flags > 0.5):
        bad = int(np.sum(self_flags <= 0.5))
        raise ValueError(f'Could not reliably locate self player block for {bad} samples.')
    self_possessor = player_features[rows, self_index, MLP_PLAYER_POSSESSOR_OFFSET] > 0.5
    contact_possible = player_features[rows, self_index, MLP_PLAYER_CONTACT_MAY_OCCUR_OFFSET] > 0.5
    shot_mask = sampled_intents == SHOT_INDEX
    shot_probability = policy_probs[:, SHOT_INDEX]
    context_masks = {'contact possible': contact_possible, 'no contact possible': ~contact_possible, 'self is possessor': self_possessor, 'self not possessor': ~self_possessor, 'contact & possessor': contact_possible & self_possessor, 'contact & not possessor': contact_possible & ~self_possessor}
    contexts = {}
    total_shots = int(np.sum(shot_mask))
    for name, mask in context_masks.items():
        count = int(np.sum(mask))
        shot_count = int(np.sum(shot_mask & mask))
        contexts[name] = {'count': count, 'shot_count': shot_count, 'sampled_shot_rate': float(np.mean(shot_mask[mask])) if count > 0 else np.nan, 'mean_shot_probability': float(np.mean(shot_probability[mask])) if count > 0 else np.nan, 'fraction_of_all_shots': shot_count / total_shots if total_shots > 0 else np.nan}
    return {'sample_count': int(len(sampled_intents)), 'contact_possible_rate': float(np.mean(contact_possible)), 'possessor_rate': float(np.mean(self_possessor)), 'contexts': contexts}

def print_shot_contact_diagnostics(shot_contact):
    """Print SHOT behavior conditioned on contact eligibility and possession."""
    print('\n--- SHOT Contact / Possession Diagnostics ---')
    print('Contact possible rate:', f"{100.0 * shot_contact['contact_possible_rate']:.2f}%")
    print('Self possessor rate:', f"{100.0 * shot_contact['possessor_rate']:.2f}%")
    print(f"\n{'Context':<26}{'N':>8}{'SHOT N':>9}{'SHOT %':>11}{'Mean P(SHOT) %':>17}{'% of all SHOTs':>17}")
    ordered_contexts = ['contact possible', 'no contact possible', 'self is possessor', 'self not possessor', 'contact & possessor', 'contact & not possessor']
    for name in ordered_contexts:
        stats = shot_contact['contexts'][name]
        print(f"{name:<26}{stats['count']:>8}{stats['shot_count']:>9}{100.0 * stats['sampled_shot_rate']:>10.2f}%{100.0 * stats['mean_shot_probability']:>16.2f}%{100.0 * stats['fraction_of_all_shots']:>16.2f}%")

def print_shot_contact_comparison(initial, trained):
    """Compare SHOT contact sensitivity between initial and trained policies."""
    print('\n--- SHOT Contact / Possession Change ---')
    print(f"{'Context':<26}{'Initial SHOT %':>16}{'Trained SHOT %':>16}{'Change':>12}")
    ordered_contexts = ['contact possible', 'no contact possible', 'self is possessor', 'self not possessor', 'contact & possessor', 'contact & not possessor']
    for name in ordered_contexts:
        initial_rate = 100.0 * initial['contexts'][name]['sampled_shot_rate']
        trained_rate = 100.0 * trained['contexts'][name]['sampled_shot_rate']
        print(f'{name:<26}{initial_rate:>16.2f}{trained_rate:>16.2f}{trained_rate - initial_rate:>+12.2f}')

def compute_possession_statistics(sampled_intents, policy_probs, actor_observations):
    """
    Diagnose possession at the environment-tick level.

    The rollout stores five controlled-player action rows per environment tick.
    Each observer row contains possessor flags for all ten players plus explicit
    self / teammate / opponent relationship features.

    This lets us recover:
      - controlled-team possession,
      - opponent possession,
      - loose/no-possessor states,
      - the action chosen by the actual controlled-team possessor.
    """
    controlled_players = 5
    sample_count = len(sampled_intents)
    if sample_count % controlled_players != 0:
        raise ValueError(f'Controlled-player sample count must be divisible by 5; got {sample_count}.')
    tick_count = sample_count // controlled_players
    obs_by_tick = actor_observations.reshape(tick_count, controlled_players, actor_observations.shape[-1])
    intents_by_tick = sampled_intents.reshape(tick_count, controlled_players)
    probs_by_tick = policy_probs.reshape(tick_count, controlled_players, policy_probs.shape[-1])
    player_flat_dim = MLP_5V5_NUM_PLAYERS * MLP_5V5_PLAYER_FEATURE_DIM
    player_features = obs_by_tick[:, :, :player_flat_dim].reshape(tick_count, controlled_players, MLP_5V5_NUM_PLAYERS, MLP_5V5_PLAYER_FEATURE_DIM)
    possessor = player_features[:, :, :, MLP_PLAYER_POSSESSOR_OFFSET] > 0.5
    last_actor = player_features[:, :, :, MLP_PLAYER_LAST_ACTOR_OFFSET] > 0.5
    is_self = player_features[:, :, :, MLP_PLAYER_IS_SELF_OFFSET] > 0.5
    is_teammate = player_features[:, :, :, MLP_PLAYER_IS_TEAMMATE_OFFSET] > 0.5
    is_opponent = player_features[:, :, :, MLP_PLAYER_IS_OPPONENT_OFFSET] > 0.5
    structural_contact = player_features[:, :, :, MLP_PLAYER_CONTACT_MAY_OCCUR_OFFSET] > 0.5
    self_possessor = np.any(possessor & is_self, axis=-1)
    teammate_possessor = np.any(possessor & is_teammate, axis=-1)
    opponent_possessor = np.any(possessor & is_opponent, axis=-1)
    team_possession_by_observer = self_possessor | teammate_possessor
    team_possession = np.any(team_possession_by_observer, axis=1)
    opponent_possession = np.any(opponent_possessor, axis=1)
    ambiguous = team_possession & opponent_possession
    loose = ~team_possession & ~opponent_possession
    team_disagreement = np.any(team_possession_by_observer != team_possession_by_observer[:, :1], axis=1)
    opponent_disagreement = np.any(opponent_possessor != opponent_possessor[:, :1], axis=1)
    self_possessor_count = np.sum(self_possessor, axis=1)
    controlled_possessor_mask = self_possessor
    controlled_possessor_intents = intents_by_tick[controlled_possessor_mask]
    controlled_possessor_shot_prob = probs_by_tick[:, :, SHOT_INDEX][controlled_possessor_mask]
    ball_xy = obs_by_tick[:, :, MLP_5V5_BALL_XY_START:MLP_5V5_BALL_XY_END]
    ball_distance = np.linalg.norm(ball_xy, axis=-1)
    controlled_possessor_ball_distance = ball_distance[controlled_possessor_mask]
    structural_self = np.any(structural_contact & is_self, axis=-1)
    controlled_possessor_structural_contact = structural_self[controlled_possessor_mask]
    possessor_intent_counts = np.bincount(controlled_possessor_intents, minlength=len(INTENT_NAMES))
    if controlled_possessor_intents.size > 0:
        possessor_intent_frequency = possessor_intent_counts / controlled_possessor_intents.size
    else:
        possessor_intent_frequency = np.full(len(INTENT_NAMES), np.nan, dtype=np.float64)
    last_contact_kick_applied = obs_by_tick[:, 0, MLP_5V5_LAST_CONTACT_KICK_APPLIED_INDEX] > 0.5
    first_row_last_actor = last_actor[:, 0, :]
    first_row_team_player = is_self[:, 0, :] | is_teammate[:, 0, :]
    first_row_opponent = is_opponent[:, 0, :]
    last_actor_team0 = np.any(first_row_last_actor & first_row_team_player, axis=-1)
    last_actor_opponent = np.any(first_row_last_actor & first_row_opponent, axis=-1)
    return {'tick_count': tick_count, 'team_possession_rate': float(np.mean(team_possession)), 'opponent_possession_rate': float(np.mean(opponent_possession)), 'loose_rate': float(np.mean(loose)), 'ambiguous_rate': float(np.mean(ambiguous)), 'team_possession_ticks': int(np.sum(team_possession)), 'opponent_possession_ticks': int(np.sum(opponent_possession)), 'loose_ticks': int(np.sum(loose)), 'team_disagreement_ticks': int(np.sum(team_disagreement)), 'opponent_disagreement_ticks': int(np.sum(opponent_disagreement)), 'controlled_possessor_samples': int(controlled_possessor_intents.size), 'controlled_possessor_intent_frequency': possessor_intent_frequency, 'controlled_possessor_shot_rate': float(np.mean(controlled_possessor_intents == SHOT_INDEX)) if controlled_possessor_intents.size > 0 else np.nan, 'controlled_possessor_mean_shot_probability': float(np.mean(controlled_possessor_shot_prob)) if controlled_possessor_shot_prob.size > 0 else np.nan, 'controlled_possessor_ball_distance_mean': float(np.mean(controlled_possessor_ball_distance)) if controlled_possessor_ball_distance.size > 0 else np.nan, 'controlled_possessor_structural_contact_rate': float(np.mean(controlled_possessor_structural_contact)) if controlled_possessor_structural_contact.size > 0 else np.nan, 'team_possession_ticks_with_one_self_possessor': int(np.sum(team_possession & (self_possessor_count == 1))), 'last_contact_kick_applied_rate': float(np.mean(last_contact_kick_applied)), 'last_contact_kick_applied_team0_rate': float(np.mean(last_contact_kick_applied & last_actor_team0)), 'last_contact_kick_applied_opponent_rate': float(np.mean(last_contact_kick_applied & last_actor_opponent))}

def compute_control_outcome_statistics(sampled_intents, actor_observations):
    """
    Diagnose what physically realized outcomes follow Team-0 CONTROL contacts.

    Important:
    actor_obs contains the *last-contact state*, which persists until another
    physical ball contact replaces it. Therefore two summaries are reported:

      1) state occupancy:
         how often each last-contact CONTROL outcome is visible.

      2) signature transitions:
         a conservative lower-bound proxy for distinct contact events, counted
         only when (actor team, intent, outcome, kick_applied) changes.

    The transition count can under-count repeated identical contacts, so it is
    diagnostic evidence rather than an exact physics-event counter.
    """
    controlled_players = 5
    sample_count = int(len(sampled_intents))
    if sample_count % controlled_players != 0:
        raise ValueError(f'Controlled-player sample count must be divisible by 5; got {sample_count}.')
    tick_count = sample_count // controlled_players
    obs_by_tick = actor_observations.reshape(tick_count, controlled_players, actor_observations.shape[-1])
    intents_by_tick = sampled_intents.reshape(tick_count, controlled_players)
    first_obs = obs_by_tick[:, 0, :]
    intent_one_hot = first_obs[:, MLP_5V5_LAST_CONTACT_INTENT_START:MLP_5V5_LAST_CONTACT_INTENT_END]
    outcome_one_hot = first_obs[:, MLP_5V5_LAST_CONTACT_OUTCOME_START:MLP_5V5_LAST_CONTACT_OUTCOME_END]
    last_contact_intent = np.argmax(intent_one_hot, axis=-1).astype(np.int32)
    last_contact_outcome = np.argmax(outcome_one_hot, axis=-1).astype(np.int32)
    kick_applied = first_obs[:, MLP_5V5_LAST_CONTACT_KICK_APPLIED_INDEX] > 0.5
    player_flat_dim = MLP_5V5_NUM_PLAYERS * MLP_5V5_PLAYER_FEATURE_DIM
    first_player_features = first_obs[:, :player_flat_dim].reshape(tick_count, MLP_5V5_NUM_PLAYERS, MLP_5V5_PLAYER_FEATURE_DIM)
    last_actor = first_player_features[:, :, MLP_PLAYER_LAST_ACTOR_OFFSET] > 0.5
    is_self = first_player_features[:, :, MLP_PLAYER_IS_SELF_OFFSET] > 0.5
    is_teammate = first_player_features[:, :, MLP_PLAYER_IS_TEAMMATE_OFFSET] > 0.5
    is_opponent = first_player_features[:, :, MLP_PLAYER_IS_OPPONENT_OFFSET] > 0.5
    actor_team0 = np.any(last_actor & (is_self | is_teammate), axis=-1)
    actor_opponent = np.any(last_actor & is_opponent, axis=-1)
    actor_known = actor_team0 | actor_opponent
    team0_control_state = actor_team0 & (last_contact_intent == CONTROL_INDEX)
    opponent_control_state = actor_opponent & (last_contact_intent == CONTROL_INDEX)

    def outcome_counts(mask):
        counts = np.zeros(len(CONTACT_OUTCOME_NAMES), dtype=np.int64)
        if np.any(mask):
            counts = np.bincount(last_contact_outcome[mask], minlength=len(CONTACT_OUTCOME_NAMES)).astype(np.int64)
        return counts
    team0_control_occupancy = outcome_counts(team0_control_state)
    opponent_control_occupancy = outcome_counts(opponent_control_state)
    team_code = np.where(actor_team0, 0, np.where(actor_opponent, 1, -1)).astype(np.int32)
    signature = np.stack([team_code, last_contact_intent, last_contact_outcome, kick_applied.astype(np.int32)], axis=-1)
    changed = np.ones(tick_count, dtype=bool)
    if tick_count > 1:
        changed[1:] = np.any(signature[1:] != signature[:-1], axis=-1)
    changed &= actor_known
    team0_control_transition = changed & team0_control_state
    opponent_control_transition = changed & opponent_control_state
    team0_control_transition_counts = outcome_counts(team0_control_transition)
    opponent_control_transition_counts = outcome_counts(opponent_control_transition)
    team0_control_requests = int(np.sum(intents_by_tick == CONTROL_INDEX))
    total_policy_actions = int(intents_by_tick.size)
    team0_control_transition_total = int(np.sum(team0_control_transition_counts))
    success_like_transition_count = int(team0_control_transition_counts[OUTCOME_TRAP_INDEX] + team0_control_transition_counts[OUTCOME_INTERCEPTION_INDEX])
    miscontrol_transition_count = int(team0_control_transition_counts[OUTCOME_MISCONTROL_INDEX])
    classified_transition_count = success_like_transition_count + miscontrol_transition_count
    return {'tick_count': tick_count, 'team0_control_requests': team0_control_requests, 'total_policy_actions': total_policy_actions, 'team0_control_request_rate': team0_control_requests / total_policy_actions if total_policy_actions > 0 else np.nan, 'team0_control_state_ticks': int(np.sum(team0_control_state)), 'opponent_control_state_ticks': int(np.sum(opponent_control_state)), 'team0_control_occupancy': team0_control_occupancy, 'opponent_control_occupancy': opponent_control_occupancy, 'team0_control_transition_counts': team0_control_transition_counts, 'opponent_control_transition_counts': opponent_control_transition_counts, 'team0_control_transition_total': team0_control_transition_total, 'team0_success_like_transition_count': success_like_transition_count, 'team0_miscontrol_transition_count': miscontrol_transition_count, 'team0_classified_transition_count': classified_transition_count, 'team0_success_like_fraction': success_like_transition_count / classified_transition_count if classified_transition_count > 0 else np.nan, 'team0_miscontrol_fraction': miscontrol_transition_count / classified_transition_count if classified_transition_count > 0 else np.nan}

def print_control_outcome_diagnostics(control_outcomes):
    """Print Team-0 CONTROL request/contact/outcome diagnostics."""
    print('\n--- CONTROL Outcome Diagnostics ---')
    print('Team0 policy CONTROL requests:', control_outcomes['team0_control_requests'], '/', control_outcomes['total_policy_actions'], f"({100.0 * control_outcomes['team0_control_request_rate']:.2f}%)")
    print('Ticks whose persisted last contact is Team0 CONTROL:', control_outcomes['team0_control_state_ticks'], '/', control_outcomes['tick_count'])
    print('\nTeam0 CONTROL last-contact state occupancy')
    occupancy = control_outcomes['team0_control_occupancy']
    occupancy_total = int(np.sum(occupancy))
    for index, name in enumerate(CONTACT_OUTCOME_NAMES):
        count = int(occupancy[index])
        if count == 0:
            continue
        fraction = count / occupancy_total if occupancy_total > 0 else np.nan
        print(f'  {name:<14}{count:>7}  ({100.0 * fraction:>6.2f}%)')
    print('\nConservative new-contact-signature transitions')
    transition_counts = control_outcomes['team0_control_transition_counts']
    transition_total = int(np.sum(transition_counts))
    for index, name in enumerate(CONTACT_OUTCOME_NAMES):
        count = int(transition_counts[index])
        if count == 0:
            continue
        fraction = count / transition_total if transition_total > 0 else np.nan
        print(f'  {name:<14}{count:>7}  ({100.0 * fraction:>6.2f}%)')
    print('  transition total:', transition_total)
    print('\nAmong Team0 CONTROL transitions classified as TRAP / INTERCEPTION / MISCONTROL:')
    print('  success-like (TRAP + INTERCEPTION):', control_outcomes['team0_success_like_transition_count'], f"({100.0 * control_outcomes['team0_success_like_fraction']:.2f}%)" if np.isfinite(control_outcomes['team0_success_like_fraction']) else '(nan%)')
    print('  MISCONTROL:', control_outcomes['team0_miscontrol_transition_count'], f"({100.0 * control_outcomes['team0_miscontrol_fraction']:.2f}%)" if np.isfinite(control_outcomes['team0_miscontrol_fraction']) else '(nan%)')
    print('\nNote: transition counts are a conservative lower bound because last_contact persists and identical consecutive contacts can be indistinguishable in actor_obs.')

def print_control_outcome_comparison(initial, trained):
    """Compare CONTROL outcome diagnostics between initial and trained policy."""
    print('\n--- CONTROL Outcome Change ---')
    print(f"{'Metric':<38}{'Initial':>12}{'Trained':>12}{'Change':>12}")
    rows = [('Policy CONTROL request %', 100.0 * initial['team0_control_request_rate'], 100.0 * trained['team0_control_request_rate']), ('CONTROL signature transitions', float(initial['team0_control_transition_total']), float(trained['team0_control_transition_total'])), ('Success-like CONTROL %', 100.0 * initial['team0_success_like_fraction'], 100.0 * trained['team0_success_like_fraction']), ('MISCONTROL %', 100.0 * initial['team0_miscontrol_fraction'], 100.0 * trained['team0_miscontrol_fraction'])]
    for name, initial_value, trained_value in rows:
        print(f'{name:<38}{initial_value:>12.4f}{trained_value:>12.4f}{trained_value - initial_value:>+12.4f}')

def compute_trap_retention_windows(sampled_intents, actor_observations):
    """
    Inspect Team-0 CONTROL->TRAP events at policy-decision boundaries.

    This diagnostic intentionally uses the actor observation that the learned
    policy actually receives. It prints a short window around each conservative
    new Team-0 TRAP signature so we can verify the environment's documented
    one-decision trap grace.

    It answers:
      - Was the trapping player visible as the current possessor?
      - Was possession.control_ticks == 1?
      - Did possession survive to the next decision?
      - Was the ball still inside the strict physical carrier envelope?
      - What action did the trapping player choose immediately afterward?
    """
    from footballworld.dynamics.contact_predicates import ActionScale, Ball, Reach
    controlled_players = 5
    sample_count = int(len(sampled_intents))
    if sample_count % controlled_players != 0:
        raise ValueError(f'Controlled-player sample count must be divisible by 5; got {sample_count}.')
    tick_count = sample_count // controlled_players
    obs_by_tick = actor_observations.reshape(tick_count, controlled_players, actor_observations.shape[-1])
    intents_by_tick = sampled_intents.reshape(tick_count, controlled_players)
    player_flat_dim = MLP_5V5_NUM_PLAYERS * MLP_5V5_PLAYER_FEATURE_DIM
    player_features = obs_by_tick[:, :, :player_flat_dim].reshape(tick_count, controlled_players, MLP_5V5_NUM_PLAYERS, MLP_5V5_PLAYER_FEATURE_DIM)
    last_actor = player_features[:, :, :, MLP_PLAYER_LAST_ACTOR_OFFSET] > 0.5
    possessor = player_features[:, :, :, MLP_PLAYER_POSSESSOR_OFFSET] > 0.5
    is_self = player_features[:, :, :, MLP_PLAYER_IS_SELF_OFFSET] > 0.5
    is_teammate = player_features[:, :, :, MLP_PLAYER_IS_TEAMMATE_OFFSET] > 0.5
    self_is_last_actor = np.any(last_actor & is_self, axis=-1)
    actor_row_known = np.any(self_is_last_actor, axis=-1)
    actor_row = np.argmax(self_is_last_actor, axis=-1).astype(np.int32)
    first_row = player_features[:, 0]
    actor_team0 = np.any((first_row[:, :, MLP_PLAYER_LAST_ACTOR_OFFSET] > 0.5) & ((first_row[:, :, MLP_PLAYER_IS_SELF_OFFSET] > 0.5) | (first_row[:, :, MLP_PLAYER_IS_TEAMMATE_OFFSET] > 0.5)), axis=-1)
    first_obs = obs_by_tick[:, 0, :]
    mechanism_one_hot = first_obs[:, MLP_5V5_LAST_CONTACT_MECHANISM_START:MLP_5V5_LAST_CONTACT_MECHANISM_END]
    intent_one_hot = first_obs[:, MLP_5V5_LAST_CONTACT_INTENT_START:MLP_5V5_LAST_CONTACT_INTENT_END]
    outcome_one_hot = first_obs[:, MLP_5V5_LAST_CONTACT_OUTCOME_START:MLP_5V5_LAST_CONTACT_OUTCOME_END]
    last_mechanism = np.argmax(mechanism_one_hot, axis=-1).astype(np.int32)
    last_intent = np.argmax(intent_one_hot, axis=-1).astype(np.int32)
    last_outcome = np.argmax(outcome_one_hot, axis=-1).astype(np.int32)
    kick_applied = first_obs[:, MLP_5V5_LAST_CONTACT_KICK_APPLIED_INDEX] > 0.5
    team0_trap = actor_team0 & actor_row_known & (last_intent == CONTROL_INDEX) & (last_outcome == OUTCOME_TRAP_INDEX)
    signature = np.stack([actor_row.astype(np.int32), last_mechanism, last_intent, last_outcome, kick_applied.astype(np.int32)], axis=-1)
    signature_changed = np.ones(tick_count, dtype=bool)
    if tick_count > 1:
        signature_changed[1:] = np.any(signature[1:] != signature[:-1], axis=-1)
    event_ticks = np.flatnonzero(team0_trap & signature_changed)
    ball = Ball()
    reach = Reach()
    scale = ActionScale()
    carry_radius_limit = float(reach.carry_radius_m + ball.radius)
    events = []
    for event_index, tick in enumerate(event_ticks):
        row = int(actor_row[tick])
        mechanism = int(last_mechanism[tick])
        is_chest = mechanism == MECHANISM_CHEST_INDEX
        event_rows = []
        for dt in (-1, 0, 1, 2, 3):
            t = int(tick + dt)
            if t < 0 or t >= tick_count:
                continue
            obs = obs_by_tick[t, row]
            pf = player_features[t, row]
            self_mask = pf[:, MLP_PLAYER_IS_SELF_OFFSET] > 0.5
            if not np.any(self_mask):
                continue
            self_slot = int(np.argmax(self_mask))
            self_block = pf[self_slot]
            self_possessor = bool(np.any((pf[:, MLP_PLAYER_POSSESSOR_OFFSET] > 0.5) & self_mask))
            control_ticks = float(obs[MLP_5V5_CONTROL_TICKS_INDEX])
            ball_xy = obs[MLP_5V5_BALL_XY_START:MLP_5V5_BALL_XY_END]
            distance_xy = float(np.linalg.norm(ball_xy))
            ball_z = float(obs[MLP_5V5_BALL_Z_INDEX])
            reach_height = float(self_block[MLP_PLAYER_REACH_HEIGHT_OFFSET])
            player_height = float(self_block[MLP_PLAYER_HEIGHT_OFFSET])
            carrier_control_height = reach_height if is_chest else player_height * float(scale.pelvis_height_factor)
            height_limit = float(carrier_control_height + ball.radius)
            distance_ok = distance_xy <= carry_radius_limit
            height_ok = ball_z <= height_limit
            event_rows.append({'dt': dt, 'tick': t, 'action': int(intents_by_tick[t, row]), 'possessor': self_possessor, 'control_ticks': control_ticks, 'distance_xy': distance_xy, 'distance_ok': bool(distance_ok), 'ball_z': ball_z, 'height_limit': height_limit, 'height_ok': bool(height_ok), 'geometry_ok': bool(distance_ok and height_ok), 'last_mechanism': int(last_mechanism[t]), 'last_intent': int(last_intent[t]), 'last_outcome': int(last_outcome[t])})
        events.append({'event_index': event_index, 'event_tick': int(tick), 'actor_row': row, 'mechanism': mechanism, 'is_chest': is_chest, 'rows': event_rows})
    return {'event_count': len(events), 'carry_radius_limit': carry_radius_limit, 'events': events}

def print_trap_retention_windows(retention):
    print('\n--- Team0 TRAP Retention Windows ---')
    print('Strict horizontal carrier limit:', f"{retention['carry_radius_limit']:.4f} m")
    print('Conservative Team0 CONTROL->TRAP events:', retention['event_count'])
    if retention['event_count'] == 0:
        print('No conservative Team0 TRAP event was observed.')
        return
    for event in retention['events']:
        print('\nEvent', event['event_index'], '| tick=', event['event_tick'], '| controlled_row=', event['actor_row'], '| mechanism=', event['mechanism'], '| chest=', event['is_chest'])
        print(f"{'dt':>3} {'tick':>6} {'action':>10} {'poss':>5} {'ctrl':>6} {'dist':>8} {'DOK':>4} {'ball_z':>8} {'z_lim':>8} {'HOK':>4} {'geom':>5} {'lc_i':>5} {'lc_o':>5}")
        for row in event['rows']:
            action_name = INTENT_NAMES[row['action']] if 0 <= row['action'] < len(INTENT_NAMES) else str(row['action'])
            print(f"{row['dt']:>3} {row['tick']:>6} {action_name:>10} {str(row['possessor']):>5} {row['control_ticks']:>6.1f} {row['distance_xy']:>8.4f} {str(row['distance_ok']):>4} {row['ball_z']:>8.4f} {row['height_limit']:>8.4f} {str(row['height_ok']):>4} {str(row['geometry_ok']):>5} {row['last_intent']:>5} {row['last_outcome']:>5}")
    print('\nInterpretation:')
    print('  dt=0 is the first policy-boundary observation where the new Team0 TRAP signature is visible.')
    print("  poss and ctrl describe the policy-boundary state; verify the environment's trap-grace semantics before requiring ctrl=1.")
    print('  If poss disappears at dt=1 while distance/height fail, carrier geometry is a candidate cause, not proof; check release/contact events.')
    print('  If dt=0 already has poss=False, first validate event/observation alignment before changing reward or policy design.')

def print_possession_diagnostics(possession):
    """Print environment-tick possession diagnostics."""
    print('\n--- Possession Diagnostics ---')
    print('Environment ticks:', possession['tick_count'])
    print(f"\n{'State':<22}{'Ticks':>10}{'Rate':>12}")
    for name, ticks_key, rate_key in [('Controlled team', 'team_possession_ticks', 'team_possession_rate'), ('Opponent', 'opponent_possession_ticks', 'opponent_possession_rate'), ('Loose / no possessor', 'loose_ticks', 'loose_rate')]:
        print(f'{name:<22}{possession[ticks_key]:>10}{100.0 * possession[rate_key]:>11.3f}%')
    print('Ambiguous possession rate:', f"{100.0 * possession['ambiguous_rate']:.4f}%")
    print('Observer-label disagreements (team / opponent):', possession['team_disagreement_ticks'], '/', possession['opponent_disagreement_ticks'])
    print('\nControlled-team possessor samples:', possession['controlled_possessor_samples'])
    print('Team-possession ticks with exactly one controlled self-possessor:', possession['team_possession_ticks_with_one_self_possessor'])
    print('Actual possessor SHOT rate:', f"{100.0 * possession['controlled_possessor_shot_rate']:.2f}%")
    print('Actual possessor mean P(SHOT):', f"{100.0 * possession['controlled_possessor_mean_shot_probability']:.2f}%")
    print('Actual possessor mean ball distance:', f"{possession['controlled_possessor_ball_distance_mean']:.4f} m")
    print('Actual possessor structural-contact gate rate:', f"{100.0 * possession['controlled_possessor_structural_contact_rate']:.2f}%")
    print('\nActual controlled possessor intent distribution')
    for index, name in enumerate(INTENT_NAMES):
        frequency = possession['controlled_possessor_intent_frequency'][index]
        print(f'  {name:<12}{100.0 * frequency:>8.2f}%')
    print('\nPrevious-contact state flags (descriptive only; not event counts)')
    print('  last_contact.kick_applied:', f"{100.0 * possession['last_contact_kick_applied_rate']:.2f}%")
    print('  kick_applied & last_actor team0:', f"{100.0 * possession['last_contact_kick_applied_team0_rate']:.2f}%")
    print('  kick_applied & last_actor opponent:', f"{100.0 * possession['last_contact_kick_applied_opponent_rate']:.2f}%")

def print_possession_comparison(initial, trained):
    """Compare possession behavior between initial and trained policies."""
    print('\n--- Possession Change ---')
    print(f"{'Metric':<34}{'Initial':>12}{'Trained':>12}{'Change':>12}")
    rows = [('Controlled-team possession %', 100.0 * initial['team_possession_rate'], 100.0 * trained['team_possession_rate']), ('Opponent possession %', 100.0 * initial['opponent_possession_rate'], 100.0 * trained['opponent_possession_rate']), ('Loose/no possessor %', 100.0 * initial['loose_rate'], 100.0 * trained['loose_rate']), ('Actual possessor SHOT %', 100.0 * initial['controlled_possessor_shot_rate'], 100.0 * trained['controlled_possessor_shot_rate']), ('Actual possessor mean P(SHOT) %', 100.0 * initial['controlled_possessor_mean_shot_probability'], 100.0 * trained['controlled_possessor_mean_shot_probability'])]
    for name, initial_value, trained_value in rows:
        print(f'{name:<34}{initial_value:>12.4f}{trained_value:>12.4f}{trained_value - initial_value:>+12.4f}')

def print_shot_context_diagnostics(shot_context):
    """Print SHOT frequency as a function of player-to-ball distance."""
    print('\n--- SHOT Context Diagnostics ---')
    print('Sampled SHOT rate:', f"{100.0 * shot_context['sampled_shot_rate']:.2f}%", f"({shot_context['shot_count']}/{shot_context['sample_count']})")
    print('Mean policy SHOT probability:', f"{100.0 * shot_context['mean_shot_probability']:.2f}%")
    print('Ball distance overall (mean / median):', f"{shot_context['mean_ball_distance']:.4f} / {shot_context['median_ball_distance']:.4f} m")
    print('Ball distance when SHOT (mean / median):', f"{shot_context['mean_ball_distance_when_shot']:.4f} / {shot_context['median_ball_distance_when_shot']:.4f} m")
    print(f"\n{'Context':<18}{'N':>8}{'Sampled SHOT %':>17}{'Mean SHOT Prob %':>19}{'% of SHOTs here':>18}")
    for threshold in SHOT_DISTANCE_THRESHOLDS_M:
        stats = shot_context['thresholds'][threshold]
        print(f'dist <= {threshold:.2f} m'.ljust(18) + f"{stats['near_count']:>8}" + f"{100.0 * stats['near_sampled_shot_rate']:>16.2f}%" + f"{100.0 * stats['near_mean_shot_probability']:>18.2f}%" + f"{100.0 * stats['fraction_of_shots_near']:>17.2f}%")
    far_threshold = SHOT_DISTANCE_THRESHOLDS_M[-1]
    far_stats = shot_context['thresholds'][far_threshold]
    print(f'dist >  {far_threshold:.2f} m'.ljust(18) + f"{far_stats['far_count']:>8}" + f"{100.0 * far_stats['far_sampled_shot_rate']:>16.2f}%" + f"{100.0 * far_stats['far_mean_shot_probability']:>18.2f}%" + f"{100.0 * (1.0 - far_stats['fraction_of_shots_near']):>17.2f}%")
    shot_continuous = shot_context['shot_continuous']
    if shot_continuous is None:
        print('\nNo sampled SHOT actions; continuous SHOT diagnostics unavailable.')
        return
    print('\nSHOT continuous controls (all sampled SHOT actions)')
    print(f"{'Dimension':<15}{'Mean(mu)':>12}{'ActMean':>12}")
    for index in RELEVANT_CONTINUOUS_BY_INTENT['SHOT']:
        print(f"{CONTINUOUS_NAMES[index]:<15}{shot_continuous['mean_policy_mean'][index]:>12.4f}{shot_continuous['mean_actual_action'][index]:>12.4f}")

def print_shot_context_comparison(initial, trained):
    """Compare SHOT context sensitivity between initial and trained policies."""
    print('\n--- SHOT Context Change ---')
    print(f"{'Metric':<34}{'Initial':>12}{'Trained':>12}{'Change':>12}")
    scalar_metrics = [('Sampled SHOT %', 100.0 * initial['sampled_shot_rate'], 100.0 * trained['sampled_shot_rate']), ('Mean SHOT probability %', 100.0 * initial['mean_shot_probability'], 100.0 * trained['mean_shot_probability']), ('Mean ball distance when SHOT (m)', initial['mean_ball_distance_when_shot'], trained['mean_ball_distance_when_shot'])]
    for name, initial_value, trained_value in scalar_metrics:
        print(f'{name:<34}{initial_value:>12.4f}{trained_value:>12.4f}{trained_value - initial_value:>+12.4f}')
    print('\nSHOT rate by ball-distance context')
    print(f"{'Context':<18}{'Initial %':>12}{'Trained %':>12}{'Change':>12}")
    for threshold in SHOT_DISTANCE_THRESHOLDS_M:
        initial_rate = 100.0 * initial['thresholds'][threshold]['near_sampled_shot_rate']
        trained_rate = 100.0 * trained['thresholds'][threshold]['near_sampled_shot_rate']
        print(f'dist <= {threshold:.2f} m'.ljust(18) + f'{initial_rate:>12.2f}' + f'{trained_rate:>12.2f}' + f'{trained_rate - initial_rate:>+12.2f}')

def evaluate_policy_behavior(label, env, backbone, action_head, critic, collector, actor_params, critic_params, num_steps, rollout_length, seed):
    """

    Run a frozen policy for a fixed number of environment steps

    and collect action-distribution diagnostics.

    """
    reset = reset_5v5_env(env, seed=seed)
    setup = reset.setup
    roster = env.roster_metadata(reset.rollout)
    si_roster = env.roster_metadata_si(reset.rollout)
    opponent_base_key = jax.random.key(seed + 10000)
    rollout_base_key = jax.random.key(seed + 20000)
    opponent_key = jax.random.fold_in(opponent_base_key, 0)
    rollout_key = jax.random.fold_in(rollout_base_key, 0)
    carry = initialize_jit_rollout_carry(env=env, reset=reset, opponent_key=opponent_key)

    @jax.jit
    def get_actor_output(params, actor_obs):
        latent = backbone.apply({'params': params['backbone']}, actor_obs)
        return action_head.apply({'params': params['head']}, latent)
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
    num_chunks = num_steps // rollout_length
    print('\n========================================')
    print(f'Policy diagnostics: {label}')
    print('========================================')
    for chunk_index in range(num_chunks):
        batch, valid_mask, carry = collector(setup, carry, roster, si_roster, actor_params, critic_params, rollout_key)
        jax.block_until_ready(carry.rollout.state.control_tick)
        actor_output = get_actor_output(actor_params, batch.actor_obs)
        jax.block_until_ready(actor_output.intent_logits)
        valid = np.asarray(jax.device_get(valid_mask)).astype(bool)
        intents = np.asarray(jax.device_get(batch.intent))
        actor_obs = np.asarray(jax.device_get(batch.actor_obs))
        pre_tanh = np.asarray(jax.device_get(batch.pre_tanh_continuous))
        logits = np.asarray(jax.device_get(actor_output.intent_logits))
        means = np.asarray(jax.device_get(actor_output.continuous_mean))
        log_stds = np.asarray(jax.device_get(actor_output.continuous_log_std))
        intents = intents[valid]
        actor_obs = actor_obs[valid]
        pre_tanh = pre_tanh[valid]
        logits = logits[valid]
        means = means[valid]
        log_stds = log_stds[valid]
        logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        categorical_entropy = -np.sum(probs * np.log(probs + 1e-12), axis=-1)
        actual_action = np.tanh(pre_tanh)
        stds = np.exp(log_stds)
        sampled_intents.append(intents.reshape(-1))
        actor_observations.append(actor_obs.reshape(-1, actor_obs.shape[-1]))
        policy_probs.append(probs.reshape(-1, probs.shape[-1]))
        policy_means.append(means.reshape(-1, means.shape[-1]))
        policy_stds.append(stds.reshape(-1, stds.shape[-1]))
        sampled_pre_tanh.append(pre_tanh.reshape(-1, pre_tanh.shape[-1]))
        sampled_actions.append(actual_action.reshape(-1, actual_action.shape[-1]))
        categorical_entropies.append(categorical_entropy.reshape(-1))
        chunk_return = float(jax.device_get(jnp.sum(jnp.where(valid_mask, batch.reward, jnp.float32(0.0)))))
        total_return += chunk_return
        total_valid_steps += int(np.sum(valid))
        if (chunk_index + 1) % 10 == 0:
            score = np.asarray(jax.device_get(carry.rollout.state.score))
            print(f'chunk={chunk_index + 1}/{num_chunks} | steps={total_valid_steps} | score={int(score[0])}-{int(score[1])}')
    sampled_intents = np.concatenate(sampled_intents, axis=0)
    actor_observations = np.concatenate(actor_observations, axis=0)
    policy_probs = np.concatenate(policy_probs, axis=0)
    policy_means = np.concatenate(policy_means, axis=0)
    policy_stds = np.concatenate(policy_stds, axis=0)
    sampled_pre_tanh = np.concatenate(sampled_pre_tanh, axis=0)
    sampled_actions = np.concatenate(sampled_actions, axis=0)
    categorical_entropies = np.concatenate(categorical_entropies, axis=0)
    score = np.asarray(jax.device_get(carry.rollout.state.score))
    intent_counts = np.bincount(sampled_intents, minlength=len(INTENT_NAMES))
    intent_frequency = intent_counts / np.sum(intent_counts)
    mean_policy_prob = np.mean(policy_probs, axis=0)
    continuous_stats = compute_continuous_statistics(policy_means=policy_means, policy_stds=policy_stds, sampled_actions=sampled_actions)
    intent_conditioned = compute_intent_conditioned_statistics(sampled_intents=sampled_intents, policy_means=policy_means, policy_stds=policy_stds, sampled_actions=sampled_actions)
    diagnostic_observations = restore_physical_diagnostic_fields(actor_observations, env.normalization_context())
    shot_context = compute_shot_context_statistics(sampled_intents=sampled_intents, policy_probs=policy_probs, actor_observations=diagnostic_observations, policy_means=policy_means, sampled_actions=sampled_actions)
    shot_contact = compute_shot_contact_statistics(sampled_intents=sampled_intents, policy_probs=policy_probs, actor_observations=diagnostic_observations)
    possession = compute_possession_statistics(sampled_intents=sampled_intents, policy_probs=policy_probs, actor_observations=diagnostic_observations)
    control_outcomes = compute_control_outcome_statistics(sampled_intents=sampled_intents, actor_observations=diagnostic_observations)
    trap_retention = compute_trap_retention_windows(sampled_intents=sampled_intents, actor_observations=diagnostic_observations)
    print('\nFinal score:', f'{int(score[0])}-{int(score[1])}')
    print('Return:', total_return)
    print('Controlled-player action samples:', len(sampled_intents))
    print('\n--- Intent Distribution ---')
    print(f"{'Intent':<12}{'Sampled %':>12}{'Mean Prob %':>15}")
    for index, name in enumerate(INTENT_NAMES):
        print(f'{name:<12}{100.0 * intent_frequency[index]:>11.2f}%{100.0 * mean_policy_prob[index]:>14.2f}%')
    print('\nMean categorical entropy:', f'{np.mean(categorical_entropies):.4f}')
    print('\n--- Overall Continuous Action Diagnostics ---')
    print_continuous_table(continuous_stats)
    print('\nOverall actual-action saturation (>0.95):', f"{100.0 * np.mean(continuous_stats['saturation_rate']):.2f}%")
    print('Overall policy-mean saturation:', f"{100.0 * np.mean(continuous_stats['policy_mean_saturation']):.2f}%")
    print_intent_conditioned_diagnostics(intent_conditioned)
    print_shot_context_diagnostics(shot_context)
    print_shot_contact_diagnostics(shot_contact)
    print_possession_diagnostics(possession)
    print_control_outcome_diagnostics(control_outcomes)
    print_trap_retention_windows(trap_retention)
    return {'score': score, 'return': total_return, 'intent_frequency': intent_frequency, 'mean_policy_prob': mean_policy_prob, 'categorical_entropy': np.mean(categorical_entropies), 'mean_policy_mean': continuous_stats['mean_policy_mean'], 'mean_abs_policy_mean': continuous_stats['mean_abs_policy_mean'], 'mean_policy_std': continuous_stats['mean_policy_std'], 'actual_mean': continuous_stats['actual_mean'], 'actual_abs_mean': continuous_stats['actual_abs_mean'], 'actual_std': continuous_stats['actual_std'], 'saturation_rate': continuous_stats['saturation_rate'], 'policy_mean_saturation': continuous_stats['policy_mean_saturation'], 'shot_context': shot_context, 'shot_contact': shot_contact, 'possession': possession, 'control_outcomes': control_outcomes, 'trap_retention': trap_retention, 'intent_conditioned': intent_conditioned}

def print_intent_conditioned_comparison(initial, trained):
    """

    Directly compare continuous behavior within each sampled intent.



    This is the key diagnostic for determining whether large global

    continuous-action drift is occurring inside behaviorally relevant

    intent classes or merely in unrelated output dimensions.

    """
    print('\n--- Intent-Conditioned Continuous Change ---')
    for intent_name in INTENT_NAMES:
        initial_item = initial['intent_conditioned'][intent_name]
        trained_item = trained['intent_conditioned'][intent_name]
        print(f"\n[{intent_name}] initial_n={initial_item['count']} | trained_n={trained_item['count']}")
        if initial_item['stats'] is None or trained_item['stats'] is None:
            print('Insufficient samples.')
            continue
        initial_stats = initial_item['stats']
        trained_stats = trained_item['stats']
        print(f"{'Dimension':<15}{'InitMu':>9}{'TrainMu':>9}{'dMu':>9}{'InitAct':>10}{'TrainAct':>10}{'dAct':>9}{'InitSat':>10}{'TrainSat':>10}")
        relevant_dimensions = RELEVANT_CONTINUOUS_BY_INTENT[intent_name]
        for dimension_index in relevant_dimensions:
            dimension_name = CONTINUOUS_NAMES[dimension_index]
            initial_mu = initial_stats['mean_policy_mean'][dimension_index]
            trained_mu = trained_stats['mean_policy_mean'][dimension_index]
            delta_mu = trained_mu - initial_mu
            initial_action = initial_stats['actual_mean'][dimension_index]
            trained_action = trained_stats['actual_mean'][dimension_index]
            delta_action = trained_action - initial_action
            initial_saturation = initial_stats['saturation_rate'][dimension_index]
            trained_saturation = trained_stats['saturation_rate'][dimension_index]
            print(f'{dimension_name:<15}{initial_mu:>9.4f}{trained_mu:>9.4f}{delta_mu:>+9.4f}{initial_action:>10.4f}{trained_action:>10.4f}{delta_action:>+9.4f}{100.0 * initial_saturation:>9.2f}%{100.0 * trained_saturation:>9.2f}%')

def print_comparison(initial, trained):
    print('\n\n========================================')
    print('INITIAL vs TRAINED COMPARISON')
    print('========================================')
    print('\n--- Intent Frequency Change ---')
    print(f"{'Intent':<12}{'Initial %':>12}{'Trained %':>12}{'Change':>12}")
    for index, name in enumerate(INTENT_NAMES):
        initial_value = 100.0 * initial['intent_frequency'][index]
        trained_value = 100.0 * trained['intent_frequency'][index]
        change = trained_value - initial_value
        print(f'{name:<12}{initial_value:>11.2f}%{trained_value:>11.2f}%{change:>+11.2f}')
    print('\nCategorical entropy:')
    print('  initial:', f"{initial['categorical_entropy']:.4f}")
    print('  trained:', f"{trained['categorical_entropy']:.4f}")
    print('\nOverall action saturation:')
    print('  initial:', f"{100.0 * np.mean(initial['saturation_rate']):.2f}%")
    print('  trained:', f"{100.0 * np.mean(trained['saturation_rate']):.2f}%")
    print('\nOverall policy-mean saturation:')
    print('  initial:', f"{100.0 * np.mean(initial['policy_mean_saturation']):.2f}%")
    print('  trained:', f"{100.0 * np.mean(trained['policy_mean_saturation']):.2f}%")
    print_intent_conditioned_comparison(initial, trained)
    print_shot_context_comparison(initial['shot_context'], trained['shot_context'])
    print_shot_contact_comparison(initial['shot_contact'], trained['shot_contact'])
    print_possession_comparison(initial['possession'], trained['possession'])
    print_control_outcome_comparison(initial['control_outcomes'], trained['control_outcomes'])

def main():
    args = parse_args()
    if args.num_steps % args.rollout_length != 0:
        raise ValueError('--num-steps must be divisible by --rollout-length.')
    env, reset = make_5v5_env(seed=args.seed)
    backbone, action_head, critic, initial_actor_params, initial_critic_params = initialize_models(env, reset)
    checkpoint = load_checkpoint(args.checkpoint)
    trained_actor_params = jax.device_put(checkpoint['actor_params'])
    trained_critic_params = jax.device_put(checkpoint['critic_params'])
    checkpoint_steps = checkpoint.get('total_env_steps', None)
    if checkpoint_steps is None:
        trained_label = 'TRAINED MLP'
    else:
        trained_label = f'MLP @ {int(checkpoint_steps)} steps'
    controlled_indices = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)
    collector = make_jitted_rollout_collector(env=env, backbone=backbone, action_head=action_head, critic=critic, rollout_length=args.rollout_length, controlled_indices=controlled_indices, controlled_team=0, retouch_guard_enabled=False)
    initial_result = evaluate_policy_behavior(label='INITIAL MLP', env=env, backbone=backbone, action_head=action_head, critic=critic, collector=collector, actor_params=initial_actor_params, critic_params=initial_critic_params, num_steps=args.num_steps, rollout_length=args.rollout_length, seed=args.seed)
    trained_result = evaluate_policy_behavior(label=trained_label, env=env, backbone=backbone, action_head=action_head, critic=critic, collector=collector, actor_params=trained_actor_params, critic_params=trained_critic_params, num_steps=args.num_steps, rollout_length=args.rollout_length, seed=args.seed)
    print_comparison(initial_result, trained_result)
if __name__ == '__main__':
    main()
