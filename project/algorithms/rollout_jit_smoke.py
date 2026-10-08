import jax
import jax.numpy as jnp

from footballworld.policies.rule_based import make_rule_based_policy

from project.algorithms.buffer import (
    Transition,
)

from project.algorithms.rollout import (
    RolloutCarry,
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


def make_jitted_rollout_collector(
    env,
    backbone,
    action_head,
    critic,
    rollout_length,
    controlled_indices,
    controlled_team=0,
):
    """
    Build a fixed-length JIT + lax.scan rollout collector.

    Current purpose
    ---------------
    Performance smoke test.

    The collector runs an entire rollout chunk inside one
    JAX/XLA computation instead of returning to Python after
    every environment step.

    Current limitation
    ------------------
    Terminal masking is not handled yet.

    Therefore this version is intended only for short rollout
    chunks that are known not to terminate during the test.
    """

    # ============================================================
    # 1. Static objects
    # ============================================================

    # MESSI built-in rule-based opponent.
    #
    # The policy object itself is static and can be captured by
    # the compiled function.
    rule_policy = make_rule_based_policy(
        env
    )

    # Team 0 player slots for the current 5v5 fixture.
    #
    # We use integer indexing instead of boolean dynamic indexing
    # because integer gather is JIT-safe and shape-static.
    controlled_indices = jnp.asarray(
        controlled_indices,
        dtype=jnp.int32,
    )

    # ============================================================
    # 2. JIT target function
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
        """
        Collect one fixed-length rollout chunk.

        Parameters
        ----------
        setup:
            MESSI MatchSetup.

        carry:
            RolloutCarry containing:
                current environment rollout
                rule-based opponent state
                immutable opponent match key

        roster:
            Normalized roster metadata used by learned policy.

        si_roster:
            SI-unit roster metadata used by rule-based policy.

        actor_params:
            {
                "backbone": ...,
                "head": ...
            }

        critic_params:
            Centralized critic parameters.

        rollout_key:
            Match-level random key for learned actor/environment.

            The SAME rollout_key should be reused across chunks
            of the same match. Absolute control_tick is folded
            into the key inside the scan.
        """

        # ========================================================
        # 3. Separate learned-policy and environment streams
        # ========================================================

        actor_key, env_key = jax.random.split(
            rollout_key,
            2,
        )

        # ========================================================
        # 4. One control-frame transition
        # ========================================================

        def scan_step(
            scan_carry,
            _,
        ):
            # ----------------------------------------------------
            # Current recurrent state
            # ----------------------------------------------------
            (
                rollout,
                rule_state,
            ) = scan_carry

            # ----------------------------------------------------
            # Absolute control tick
            #
            # We derive frame randomness from the absolute tick
            # rather than scan index.
            #
            # Therefore:
            #
            # chunk 1: ticks 0,1,2,3
            # chunk 2: ticks 4,5,6,7
            #
            # remains reproducible regardless of chunk boundary.
            # ----------------------------------------------------
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

            # ====================================================
            # 4-1. Learned actor observation
            # ====================================================

            obs = env.observe_all(
                rollout
            )

            mlp_obs = encode_mlp_observation(
                obs,
                roster,
            )

            # ====================================================
            # 4-2. Shared MLP actor
            # ====================================================

            latent = backbone.apply(
                {
                    "params":
                        actor_params["backbone"]
                },
                mlp_obs,
            )

            actor_output = action_head.apply(
                {
                    "params":
                        actor_params["head"]
                },
                latent,
            )

            # ====================================================
            # 4-3. Sample learned actions
            # ====================================================

            sampled = sample_action(
                actor_output,
                step_actor_key,
            )

            learned_action = (
                sampled.action
            )

            # ====================================================
            # 4-4. Rollout-policy log probability
            # ====================================================

            log_prob = action_log_prob(
                actor_output,
                learned_action.intent,
                sampled.pre_tanh_continuous,
            )

            # ====================================================
            # 4-5. Centralized critic
            # ====================================================

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

            # ====================================================
            # 4-6. Rule-based opponent
            #
            # RuleBasedPolicy requires SI observations.
            # ====================================================

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

            # ====================================================
            # 4-7. Combine both teams
            #
            # controlled_team:
            #     learned actor
            #
            # other team:
            #     MESSI rule-based policy
            # ====================================================

            combined_action = combine_team_actions(
                learned_action,
                opponent_action,
                roster.team_id,
                controlled_team=controlled_team,
            )

            # ====================================================
            # 4-8. MESSI environment transition
            # ====================================================

            result = env.step(
                rollout,
                setup,
                combined_action,
                step_env_key,
            )

            # ====================================================
            # 4-9. Extract learning-team actor data
            #
            # IMPORTANT:
            #
            # Do not use:
            #
            #     mlp_obs[roster.team_id == 0]
            #
            # inside JIT because boolean indexing may create a
            # dynamically sized result.
            #
            # Integer gather keeps the shape static.
            # ====================================================

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

            # ====================================================
            # 4-10. MAPPO transition
            # ====================================================

            transition = Transition(
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
                    result.reward[
                        controlled_team
                    ]
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
            )

            # ====================================================
            # 4-11. Next recurrent scan state
            # ====================================================

            next_scan_carry = (
                result.rollout,
                rule_step.state,
            )

            return (
                next_scan_carry,
                transition,
            )

        # ========================================================
        # 5. Fixed-length rollout entirely inside XLA
        # ========================================================

        (
            final_scan_carry,
            batch,
        ) = jax.lax.scan(
            scan_step,
            (
                carry.rollout,
                carry.rule_state,
            ),
            xs=None,
            length=rollout_length,
        )

        (
            final_rollout,
            final_rule_state,
        ) = final_scan_carry

        # ========================================================
        # 6. Preserve state across rollout chunks
        # ========================================================

        new_carry = RolloutCarry(
            rollout=final_rollout,
            rule_state=final_rule_state,
            opponent_key=carry.opponent_key,
        )

        return (
            batch,
            new_carry,
        )

    # ============================================================
    # 7. Compile whole rollout chunk
    # ============================================================

    return jax.jit(
        collect
    )