from project.models.field_movement import field_movement
from project.models.gk_movement import limit_gk_move
from project.models.role_actor import actor_forward
from project.envs.policy_obs import encode_policy_observation
from project.envs.reward11 import reward11
from project.envs.reward_v3 import extra_control_shaping, control_potentials
from project.models.intent_guard import apply_retouch_guard

#rollout_jit.py

from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.policies.rule_based import make_rule_based_policy

from project.algorithms.buffer import Transition

from project.algorithms.rollout import (
    initialize_rollout_carry,
)

from project.envs.critic_encoder import (
    encode_critic_state,
)

from project.envs.obs_encoder import (
    encode_mlp_observation,
)

from project.envs.reward import (
    compute_shaped_reward,
)

from project.models.action_sampling import (
    action_log_prob,
    sample_action,
)

from project.utils.team_action import (
    combine_team_actions,
)


class JITRolloutCarry(NamedTuple):
    """
    Persistent state across fixed-length JIT rollout chunks.

    rollout
        Current MESSI rollout state.

    rule_state
        Recurrent state of the fixed rule-based opponent.

    opponent_key
        Immutable match-level key used by the rule-based opponent.

    terminated
        Whether the current match ended via true termination.

    truncated
        Whether the current match ended via truncation.

    done
        terminated OR truncated.
    """

    rollout: object
    rule_state: object
    opponent_key: jax.Array

    terminated: jax.Array
    truncated: jax.Array
    done: jax.Array


def initialize_jit_rollout_carry(
    env,
    reset,
    opponent_key,
):
    """
    Initialize persistent JIT rollout state once per match.
    """

    base_carry = initialize_rollout_carry(
        env=env,
        reset=reset,
        opponent_key=opponent_key,
    )

    return JITRolloutCarry(
        rollout=base_carry.rollout,
        rule_state=base_carry.rule_state,
        opponent_key=base_carry.opponent_key,

        terminated=jnp.bool_(False),
        truncated=jnp.bool_(False),
        done=jnp.bool_(False),
    )


def make_jitted_rollout_collector(
    env,
    backbone,
    action_head,
    critic,
    rollout_length,
    controlled_indices,
    controlled_team=0,
    reward_shaping_coefficient=0.0,
    reward_gamma=0.99,
    retouch_guard_enabled=True,
    reward_version="v2",
    approach_coefficient=.05,
    possession_coefficient=.05,
    approach_distance_scale_m=10.,
):
    """
    Build a fixed-length terminal-safe MAPPO rollout collector.

    The entire rollout chunk is executed inside:

        jax.jit
            +
        jax.lax.scan

    Reward
    ------
    The reward stored in Transition.reward is:

        total_reward
        =
        original sparse MESSI goal reward
        +
        potential-based shaping reward

    reward_shaping_coefficient=0.0 reproduces the original
    sparse-reward behavior.

    If the match ends before rollout_length:

        - the terminal transition itself is retained
        - later scan rows are padded
        - environment state no longer advances
        - rule-policy state no longer advances
        - valid_mask=False for padded rows

    Returns
    -------
    batch:
        Time-major Transition pytree.

    valid_mask:
        [T] boolean mask.

        True:
            real environment transition

        False:
            padding after match termination

    new_carry:
        Persistent state for the next rollout chunk.
    """
    if reward_version not in ("v2", "v3"):
        raise ValueError("reward_version must be v2 or v3")
    if approach_coefficient < 0 or possession_coefficient < 0 or approach_distance_scale_m <= 0:
        raise ValueError("Control coefficients must be nonnegative and distance scale positive")


    # ============================================================
    # 1. Static objects
    # ============================================================

    rule_policy = make_rule_based_policy(
        env
    )
    from learned_pool import make_controller
    pool_action=make_controller(env,backbone,action_head)

    controlled_indices = jnp.asarray(
        controlled_indices,
        dtype=jnp.int32,
    )

    # Number of learning-team agents.
    controlled_count = (
        controlled_indices.shape[0]
    )

    # ============================================================
    # 2. Compiled rollout function
    # ============================================================

    def collect(
        setup,
        carry,
        roster,
        si_roster,
        actor_params,
        critic_params,
        rollout_key,
    ):
        # ========================================================
        # 2-1. Match-level random streams
        #
        # rollout_key is reused across chunks.
        # Absolute control_tick is folded in below.
        # ========================================================

        actor_key, env_key = jax.random.split(
            rollout_key,
            2,
        )

        # ========================================================
        # 2-2. One scan timestep
        # ========================================================

        def scan_step(
            scan_carry,
            _,
        ):
            (
                rollout,
                rule_state,
                terminated,
                truncated,
                done,
            ) = scan_carry

            # ----------------------------------------------------
            # Branch A:
            # match already ended before this scan row
            # ----------------------------------------------------

            def pause(_):
                """
                Do not advance physics or opponent state.

                Produce a fixed-shape padding transition and mark
                this row invalid.
                """

                # We only derive observation shapes here.
                # No policy sampling or environment transition.

                obs = env.observe_all(
                    rollout
                )

                mlp_obs = encode_policy_observation(env, rollout, roster)

                global_state = env.global_state_view(
                    rollout
                )

                critic_obs = encode_critic_state(
                    global_state,
                    controlled_team=controlled_team,
                )

                controlled_actor_obs = jnp.take(
                    mlp_obs,
                    controlled_indices,
                    axis=0,
                )

                zero_transition = Transition(
                    reward_details=jnp.zeros(12,jnp.float32),
                    retouch_guard_enabled=jnp.bool_(False),
                    approach_shaping_reward=jnp.float32(0.),
                    possession_shaping_reward=jnp.float32(0.),
                    lawful_possession=jnp.float32(0.),
                    actor_obs=jnp.zeros_like(
                        controlled_actor_obs
                    ),

                    critic_obs=jnp.zeros_like(
                        critic_obs
                    ),

                    intent=jnp.zeros(
                        (controlled_count,),
                        dtype=jnp.int32,
                    ),

                    pre_tanh_continuous=jnp.zeros(
                        (
                            controlled_count,
                            7,
                        ),
                        dtype=mlp_obs.dtype,
                    ),

                    log_prob=jnp.zeros(
                        (controlled_count,),
                        dtype=mlp_obs.dtype,
                    ),

                    value=jnp.zeros(
                        (2,),
                        dtype=critic_obs.dtype,
                    ),

                    # Reward used by GAE / PPO.
                    reward=jnp.zeros(
                        (2,),
                        dtype=critic_obs.dtype,
                    ),

                    # Padding rows are not new terminal events.
                    terminated=jnp.bool_(False),
                    truncated=jnp.bool_(False),

                    # State is already done.
                    done=jnp.bool_(True),

                    # Diagnostic reward components.
                    base_reward=jnp.zeros(
                        (),
                        dtype=critic_obs.dtype,
                    ),

                    shaping_reward=jnp.zeros(
                        (),
                        dtype=critic_obs.dtype,
                    ),
                )

                next_scan_carry = (
                    rollout,
                    rule_state,
                    terminated,
                    truncated,
                    done,
                )

                valid = jnp.bool_(False)

                return (
                    next_scan_carry,
                    (
                        zero_transition,
                        valid,
                    ),
                )

            # ----------------------------------------------------
            # Branch B:
            # execute one real environment transition
            # ----------------------------------------------------

            def advance(_):
                # =================================================
                # Absolute control tick
                # =================================================

                absolute_tick = (
                    rollout.state.control_tick.astype(
                        jnp.uint32
                    )
                )

                step_actor_key = jax.random.fold_in(
                    actor_key,
                    absolute_tick,
                )

                step_env_key = jax.random.fold_in(
                    env_key,
                    absolute_tick,
                )

                # =================================================
                # Player observations
                # =================================================

                obs = env.observe_all(
                    rollout
                )

                mlp_obs = encode_policy_observation(env, rollout, roster)

                # =================================================
                # Shared actor
                # =================================================

                actor_output = actor_forward(actor_params, backbone, action_head, mlp_obs)

                actor_output = apply_retouch_guard(
                    actor_output, mlp_obs, enabled=retouch_guard_enabled,
                )

                sampled = sample_action(
                    actor_output,
                    step_actor_key,
                )

                learned_action = (
                    sampled.action
                )

                log_prob = action_log_prob(
                    actor_output,
                    learned_action.intent,
                    sampled.pre_tanh_continuous,
                )

                # =================================================
                # Centralized critic
                #
                # global_state is the normalized state BEFORE
                # env.step(), so it is also s_t for reward shaping.
                # =================================================

                global_state = env.global_state_view(
                    rollout
                )

                critic_obs = encode_critic_state(
                    global_state,
                    controlled_team=controlled_team,
                )

                value = critic.apply(
                    {
                        "params":
                            critic_params
                    },
                    critic_obs[None, :],
                )[0]

                # =================================================
                # Rule-based opponent
                # =================================================

                si_obs = env.observe_all_si(
                    rollout
                )

                rule_step = rule_policy.step(
                    si_obs,
                    si_roster,
                    rule_state,
                    carry.opponent_key,
                )

                opponent_action = (
                    rule_step.action
                )
                opponent_action=pool_action(rollout,roster,opponent_action,carry.opponent_key,rollout.state.control_tick.astype(jnp.uint32))

                # =================================================
                # Combine both teams
                # =================================================

                combined_action = combine_team_actions(
                    learned_action,
                    opponent_action,
                    roster.team_id,
                    controlled_team=controlled_team,
                )

                # =================================================
                # Environment transition
                # =================================================

                combined_action = field_movement(rollout.state, combined_action, rule_step.action)
                combined_action, _, _ = limit_gk_move(env, rollout.state, combined_action)
                result = env.step_with_events(
                    rollout,
                    setup,
                    combined_action,
                    step_env_key,
                )

                # =================================================
                # Reward shaping
                #
                # global_state:
                #     s_t
                #
                # next_global_state:
                #     s_{t+1}
                #
                # total reward:
                #
                #     r_goal
                #       +
                #     alpha * [
                #         gamma * Phi(s_{t+1})
                #         - Phi(s_t)
                #     ]
                #
                # reward_shaping_coefficient=0.0 gives exactly the
                # original sparse-reward setup.
                # =================================================

                next_global_state = env.global_state_view(
                    result.rollout
                )

                base_reward = result.reward[
                    controlled_team
                ]

                values = reward11(env, rollout.state, result, reward_gamma)
                from project.envs.reward import RewardComponents
                reward_components = RewardComponents(base_reward=base_reward, shaping_reward=values.sum(), total_reward=jnp.stack([base_reward+values[:7].sum()+values[11], base_reward+values[7:11].sum()]))
                approach_shaping, possession_shaping = values[0], values[1]+values[2]
                lawful_possession = control_potentials(rollout.state, controlled_team).possession

                # =================================================
                # Learning-team actor rows
                # =================================================

                controlled_actor_obs = jnp.take(
                    mlp_obs,
                    controlled_indices,
                    axis=0,
                )

                controlled_intent = jnp.take(
                    learned_action.intent,
                    controlled_indices,
                    axis=0,
                )

                controlled_pre_tanh = jnp.take(
                    sampled.pre_tanh_continuous,
                    controlled_indices,
                    axis=0,
                )

                controlled_log_prob = jnp.take(
                    log_prob,
                    controlled_indices,
                    axis=0,
                )

                # =================================================
                # MAPPO transition
                #
                # Important:
                #
                # if result.done=True here, THIS transition is
                # still real and therefore valid=True.
                #
                # batch.reward is the reward actually used by
                # GAE / PPO.
                #
                # batch.base_reward and batch.shaping_reward are
                # stored separately for diagnostics.
                # =================================================

                transition = Transition(
                    reward_details=values,
                    retouch_guard_enabled=jnp.bool_(retouch_guard_enabled),
                    approach_shaping_reward=approach_shaping,
                    possession_shaping_reward=possession_shaping,
                    lawful_possession=lawful_possession,
                    actor_obs=(
                        controlled_actor_obs
                    ),

                    critic_obs=(
                        critic_obs
                    ),

                    intent=(
                        controlled_intent
                    ),

                    pre_tanh_continuous=(
                        controlled_pre_tanh
                    ),

                    log_prob=(
                        controlled_log_prob
                    ),

                    value=(
                        value
                    ),

                    reward=(
                        reward_components.total_reward
                    ),

                    terminated=(
                        result.terminated
                    ),

                    truncated=(
                        result.truncated
                    ),

                    done=(
                        result.done
                    ),

                    base_reward=(
                        reward_components.base_reward
                    ),

                    shaping_reward=(
                        reward_components.shaping_reward
                    ),
                )

                # =================================================
                # Persistent scan state
                # =================================================

                next_scan_carry = (
                    result.rollout,
                    rule_step.state,
                    result.terminated,
                    result.truncated,
                    result.done,
                )

                valid = jnp.bool_(True)

                return (
                    next_scan_carry,
                    (
                        transition,
                        valid,
                    ),
                )

            # ====================================================
            # Terminal-safe conditional
            #
            # done=True:
            #     freeze match and output padding
            #
            # done=False:
            #     execute one real control step
            # ====================================================

            return jax.lax.cond(
                done,
                pause,
                advance,
                operand=None,
            )

        # ========================================================
        # 3. Fixed-length scan
        # ========================================================

        (
            final_scan_carry,
            (
                batch,
                valid_mask,
            ),
        ) = jax.lax.scan(
            scan_step,
            (
                carry.rollout,
                carry.rule_state,
                carry.terminated,
                carry.truncated,
                carry.done,
            ),
            xs=None,
            length=rollout_length,
        )

        (
            final_rollout,
            final_rule_state,
            final_terminated,
            final_truncated,
            final_done,
        ) = final_scan_carry

        # ========================================================
        # 4. Persistent state for next chunk
        # ========================================================

        new_carry = JITRolloutCarry(
            rollout=final_rollout,
            rule_state=final_rule_state,
            opponent_key=carry.opponent_key,

            terminated=final_terminated,
            truncated=final_truncated,
            done=final_done,
        )

        return (
            batch,
            valid_mask,
            new_carry,
        )

    # ============================================================
    # 5. Compile whole fixed-length rollout
    # ============================================================

    return jax.jit(
        collect
    )