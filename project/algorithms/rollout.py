from typing import NamedTuple

import jax

from footballworld.policies.rule_based import make_rule_based_policy

from project.algorithms.buffer import (
    Transition,
    stack_transitions,
)

from project.envs.critic_encoder import (
    encode_critic_state,
)

from project.envs.obs_encoder import (
    encode_mlp_observation,
)

from project.models.action_sampling import (
    action_log_prob,
    sample_action,
)

from project.utils.team_action import (
    combine_team_actions,
)


class RolloutCarry(NamedTuple):
    """
    State that must persist across rollout chunks
    within the same MESSI match.

    rollout:
        Current MESSI environment rollout state.

    rule_state:
        Recurrent state of the fixed rule-based opponent.

    opponent_key:
        Immutable match-level PRNG key for the rule-based opponent.
    """

    rollout: object
    rule_state: object
    opponent_key: jax.Array


def initialize_rollout_carry(
    env,
    reset,
    opponent_key,
):
    """
    Initialize persistent rollout state once at the beginning
    of a MESSI match.

    The rule-based opponent state is initialized here and then
    carried across rollout chunks instead of being reset every time.
    """

    # ============================================================
    # 1. Build fixed rule-based opponent
    # ============================================================
    rule_policy = make_rule_based_policy(
        env
    )

    # ============================================================
    # 2. SI observation / roster required by RuleBasedPolicy
    # ============================================================
    si_obs = env.observe_all_si(
        reset.rollout
    )

    si_roster = env.roster_metadata_si(
        reset.rollout
    )

    # ============================================================
    # 3. Initialize opponent recurrent state once
    # ============================================================
    rule_state = rule_policy.initialize(
        si_obs,
        si_roster,
    )

    # ============================================================
    # 4. Persistent match carry
    # ============================================================
    return RolloutCarry(
        rollout=reset.rollout,
        rule_state=rule_state,
        opponent_key=opponent_key,
    )


def collect_rollout(
    env,
    setup,
    carry,
    backbone,
    backbone_variables,
    action_head,
    head_variables,
    critic,
    critic_variables,
    rollout_length,
    key,
    controlled_team=0,
):
    """
    Collect one MAPPO rollout chunk from MESSI.

    Current setup
    -------------
    Controlled team:
        learned shared actor

    Opponent team:
        fixed MESSI rule-based policy

    Important
    ---------
    Environment state, rule-based opponent recurrent state,
    and opponent match key are carried across rollout chunks.
    """

    transitions = []

    # ============================================================
    # 1. Restore persistent rollout state
    # ============================================================
    rollout = carry.rollout
    rule_state = carry.rule_state
    opponent_key = carry.opponent_key

    # ============================================================
    # 2. Current roster metadata
    #
    # No substitutions are currently used, so this remains fixed
    # during one chunk. We still obtain it from the current rollout.
    # ============================================================
    roster = env.roster_metadata(
        rollout
    )

    controlled_mask = (
        roster.team_id == controlled_team
    )

    # ============================================================
    # 3. Rule-based opponent
    #
    # The policy object itself may be reconstructed.
    # Its recurrent rule_state is preserved in RolloutCarry.
    # ============================================================
    rule_policy = make_rule_based_policy(
        env
    )

    si_roster = env.roster_metadata_si(
        rollout
    )

    # ============================================================
    # 4. Separate random streams
    #
    # opponent_key is NOT regenerated here.
    # It stays fixed for the entire match.
    # ============================================================
    actor_key, env_key = jax.random.split(
        key,
        2,
    )

    # ============================================================
    # 5. Rollout loop
    # ============================================================
    for t in range(rollout_length):

        # --------------------------------------------------------
        # 5-1. Normalized player observations
        # --------------------------------------------------------
        obs = env.observe_all(
            rollout
        )

        mlp_obs = encode_mlp_observation(
            obs,
            roster,
        )

        # --------------------------------------------------------
        # 5-2. Learned shared actor
        #
        # Current implementation evaluates all player rows.
        # Only controlled-team rows are stored for PPO training.
        # --------------------------------------------------------
        latent = backbone.apply(
            backbone_variables,
            mlp_obs,
        )

        actor_output = action_head.apply(
            head_variables,
            latent,
        )

        step_actor_key = jax.random.fold_in(
            actor_key,
            t,
        )

        sampled = sample_action(
            actor_output,
            step_actor_key,
        )

        learned_action = sampled.action

        # --------------------------------------------------------
        # 5-3. Log probability under rollout policy
        # --------------------------------------------------------
        log_prob = action_log_prob(
            actor_output,
            learned_action.intent,
            sampled.pre_tanh_continuous,
        )

        # --------------------------------------------------------
        # 5-4. Centralized critic
        # --------------------------------------------------------
        global_state = env.global_state_view(
            rollout
        )

        critic_obs = encode_critic_state(
            global_state,
            controlled_team=controlled_team,
        )

        value = critic.apply(
            critic_variables,
            critic_obs[None, :],
        )[0]

        # --------------------------------------------------------
        # 5-5. Rule-based opponent
        #
        # RuleBasedPolicy consumes SI observations.
        # rule_state is updated every control frame.
        # opponent_key stays fixed across the whole match.
        # --------------------------------------------------------
        si_obs = env.observe_all_si(
            rollout
        )

        rule_step = rule_policy.step(
            si_obs,
            si_roster,
            rule_state,
            opponent_key,
        )

        opponent_action = rule_step.action

        rule_state = rule_step.state

        # --------------------------------------------------------
        # 5-6. Combine learned team + fixed opponent
        # --------------------------------------------------------
        combined_action = combine_team_actions(
            learned_action,
            opponent_action,
            roster.team_id,
            controlled_team=controlled_team,
        )

        # --------------------------------------------------------
        # 5-7. Environment transition
        # --------------------------------------------------------
        step_env_key = jax.random.fold_in(
            env_key,
            t,
        )

        result = env.step(
            rollout,
            setup,
            combined_action,
            step_env_key,
        )

        # --------------------------------------------------------
        # 5-8. Store MAPPO transition
        #
        # Actor data:
        # controlled team only
        #
        # Critic/value/reward:
        # centralized team-level quantities
        # --------------------------------------------------------
        transition = Transition(
            actor_obs=mlp_obs[
                controlled_mask
            ],

            critic_obs=critic_obs,

            intent=learned_action.intent[
                controlled_mask
            ],

            pre_tanh_continuous=(
                sampled.pre_tanh_continuous[
                    controlled_mask
                ]
            ),

            log_prob=log_prob[
                controlled_mask
            ],

            value=value,

            reward=result.reward[
                controlled_team
            ],

            terminated=result.terminated,
            truncated=result.truncated,
            done=result.done,
        )

        transitions.append(
            transition
        )

        # --------------------------------------------------------
        # 5-9. Advance environment state
        # --------------------------------------------------------
        rollout = result.rollout

        # --------------------------------------------------------
        # 5-10. Stop if episode ends
        # --------------------------------------------------------
        if bool(result.done):
            break

    # ============================================================
    # 6. Stack transitions along time axis
    # ============================================================
    batch = stack_transitions(
        transitions
    )

    # ============================================================
    # 7. Preserve state for the next rollout chunk
    # ============================================================
    new_carry = RolloutCarry(
        rollout=rollout,
        rule_state=rule_state,
        opponent_key=opponent_key,
    )

    return batch, new_carry