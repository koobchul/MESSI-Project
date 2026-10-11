import jax
import jax.numpy as jnp

from project.algorithms.gae import (
    compute_gae,
    compute_masked_gae,
)

from project.algorithms.rollout import (
    collect_rollout,
)

from project.algorithms.update import (
    ppo_update_epochs,
)

from project.envs.critic_encoder import (
    encode_critic_state,
)


def train_iteration(
    env,
    setup,
    carry,
    backbone,
    action_head,
    critic,
    actor_params,
    critic_params,
    actor_opt_state,
    critic_opt_state,
    actor_optimizer,
    critic_optimizer,
    rollout_length,
    key,
    controlled_team=0,
    gamma=0.99,
    gae_lambda=0.95,
    num_epochs=4,
    target_kl=0.02,
    clip_eps=0.2,
    entropy_coef=0.01,
):
    """
    Original Python-rollout MAPPO training iteration.

    This function is preserved as the correctness/reference
    implementation.

    Flow
    ----
    current policy
        -> Python MESSI rollout
        -> bootstrap value
        -> GAE
        -> PPO update epochs
        -> updated actor / critic
    """

    # ============================================================
    # 1. Convert parameter trees to Flax variable dictionaries
    # ============================================================

    backbone_variables = {
        "params":
            actor_params["backbone"]
    }

    head_variables = {
        "params":
            actor_params["head"]
    }

    critic_variables = {
        "params":
            critic_params
    }

    # ============================================================
    # 2. Collect rollout using CURRENT parameters
    # ============================================================

    batch, new_carry = collect_rollout(
        env=env,
        setup=setup,
        carry=carry,

        backbone=backbone,
        backbone_variables=backbone_variables,

        action_head=action_head,
        head_variables=head_variables,

        critic=critic,
        critic_variables=critic_variables,

        rollout_length=rollout_length,

        key=key,

        controlled_team=controlled_team,
    )

    # ============================================================
    # 3. Bootstrap V(s_T)
    # ============================================================

    final_global_state = env.global_state_view(
        new_carry.rollout
    )

    final_critic_obs = encode_critic_state(
        final_global_state,
        controlled_team=controlled_team,
    )

    bootstrap_value = critic.apply(
        critic_variables,
        final_critic_obs[None, :],
    )[0]

    # True terminal state:
    # no value bootstrap.
    bootstrap_value = jnp.where(
        batch.terminated[-1],
        jnp.float32(0.0),
        bootstrap_value,
    )

    # ============================================================
    # 4. GAE
    # ============================================================

    advantages, returns = compute_gae(
        rewards=batch.reward,
        values=batch.value,

        bootstrap_value=bootstrap_value,

        terminated=batch.terminated,
        truncated=batch.truncated,

        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    # PPO targets are treated as fixed rollout data.
    advantages = jax.lax.stop_gradient(
        advantages
    )

    returns = jax.lax.stop_gradient(
        returns
    )

    # ============================================================
    # 5. PPO optimization epochs
    # ============================================================

    (
        new_actor_params,
        new_critic_params,
        new_actor_opt_state,
        new_critic_opt_state,
        metrics_history,
    ) = ppo_update_epochs(
        actor_params=actor_params,
        critic_params=critic_params,

        actor_opt_state=actor_opt_state,
        critic_opt_state=critic_opt_state,

        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,

        backbone=backbone,
        action_head=action_head,
        critic=critic,

        batch=batch,

        advantages=advantages,
        returns=returns,

        num_epochs=num_epochs,
        target_kl=target_kl,

        clip_eps=clip_eps,
        entropy_coef=entropy_coef,
    )

    return (
        new_actor_params,
        new_critic_params,

        new_actor_opt_state,
        new_critic_opt_state,

        new_carry,

        batch,

        advantages,
        returns,

        metrics_history,
    )


def train_iteration_jit(
    env,
    setup,
    carry,
    collector,
    roster,
    si_roster,
    backbone,
    action_head,
    critic,
    actor_params,
    critic_params,
    actor_opt_state,
    critic_opt_state,
    actor_optimizer,
    critic_optimizer,
    rollout_key,
    controlled_team=0,
    gamma=0.99,
    gae_lambda=0.95,
    num_epochs=4,
    target_kl=0.02,
    clip_eps=0.2,
    entropy_coef=0.01,
):
    """
    Terminal-safe fixed-shape JIT MAPPO training iteration.

    Flow
    ----
    current policy
        -> jitted lax.scan MESSI rollout
        -> valid_mask
        -> bootstrap value
        -> masked GAE
        -> masked PPO update epochs
        -> updated actor / critic

    Important
    ---------
    collector must be created ONCE outside the training loop with:

        make_jitted_rollout_collector(...)

    Do not rebuild the collector every iteration, otherwise the
    compiled executable may not be reused efficiently.

    rollout_key is a match-level key.

    The same rollout_key should be reused across chunks of the
    SAME match because the collector folds the absolute
    control_tick into the key.

    A new match should receive a new rollout_key.
    """

    # ============================================================
    # 1. Collect fixed-length JIT rollout
    #
    # batch:
    #     fixed [T, ...] shapes
    #
    # valid_mask:
    #     [T]
    #
    # new_carry:
    #     persistent MESSI + opponent + terminal state
    # ============================================================

    (
        batch,
        valid_mask,
        new_carry,
    ) = collector(
        setup,
        carry,

        roster,
        si_roster,

        actor_params,
        critic_params,

        rollout_key,
    )

    # Make sure rollout execution has completed before the host
    # potentially inspects valid_mask below.
    jax.block_until_ready(
        new_carry.rollout.state.control_tick
    )

    # ============================================================
    # 2. Number of real environment transitions
    # ============================================================

    num_valid = int(
        jax.device_get(
            jnp.sum(
                valid_mask.astype(
                    jnp.int32
                )
            )
        )
    )

    # ============================================================
    # 3. Already-terminal match
    #
    # Example:
    #
    # valid_mask =
    # [False False False False]
    #
    # Do NOT call Adam with zero gradients here.
    #
    # Adam carries momentum, so a nominal zero-gradient update can
    # still move parameters due to optimizer state.
    # ============================================================

    if num_valid == 0:
        zero_advantages = jnp.zeros_like(
            batch.reward,
            dtype=jnp.float32,
        )

        zero_returns = jnp.zeros_like(
            batch.reward,
            dtype=jnp.float32,
        )

        return (
            actor_params,
            critic_params,

            actor_opt_state,
            critic_opt_state,

            new_carry,

            batch,
            valid_mask,

            zero_advantages,
            zero_returns,

            [],
        )

    # ============================================================
    # 4. Current critic variables
    # ============================================================

    critic_variables = {
        "params":
            critic_params
    }

    # ============================================================
    # 5. Bootstrap V(s_T)
    #
    # new_carry.rollout is the FINAL REAL environment state.
    #
    # This remains true even if the remainder of the fixed scan
    # was padding.
    # ============================================================

    final_global_state = env.global_state_view(
        new_carry.rollout
    )

    final_critic_obs = encode_critic_state(
        final_global_state,
        controlled_team=controlled_team,
    )

    bootstrap_value = critic.apply(
        critic_variables,
        final_critic_obs[None, :],
    )[0]

    # ============================================================
    # 6. Terminal bootstrap semantics
    #
    # terminated=True:
    #     no bootstrap
    #
    # truncated=True:
    #     bootstrap IS allowed
    #
    # no terminal:
    #     bootstrap normally
    #
    # IMPORTANT:
    #
    # Do NOT inspect batch.terminated[-1].
    #
    # With fixed-shape padding, the final array row may be padding
    # and therefore batch.terminated[-1] can be False even though
    # the real trajectory ended earlier.
    #
    # new_carry.terminated is authoritative.
    # ============================================================

    bootstrap_value = jnp.where(
        new_carry.terminated,
        jnp.float32(0.0),
        bootstrap_value,
    )

    # ============================================================
    # 7. Masked GAE
    #
    # Padding rows:
    #     advantage = 0
    #     return    = 0
    #
    # Truncation:
    #     uses bootstrap_value
    #     stops GAE recursion across episode boundary
    #
    # Termination:
    #     does not bootstrap
    # ============================================================

    advantages, returns = compute_masked_gae(
        rewards=batch.reward,
        values=batch.value,

        bootstrap_value=bootstrap_value,

        terminated=batch.terminated,
        truncated=batch.truncated,

        valid_mask=valid_mask,

        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    # ============================================================
    # 8. Freeze rollout targets
    # ============================================================

    advantages = jax.lax.stop_gradient(
        advantages
    )

    returns = jax.lax.stop_gradient(
        returns
    )

    # ============================================================
    # 9. Masked PPO optimization
    #
    # valid_mask is propagated through:
    #
    # advantage normalization
    # actor objective
    # entropy
    # KL
    # clip fraction
    # critic loss
    # ============================================================

    (
        new_actor_params,
        new_critic_params,

        new_actor_opt_state,
        new_critic_opt_state,

        metrics_history,
    ) = ppo_update_epochs(
        actor_params=actor_params,
        critic_params=critic_params,

        actor_opt_state=actor_opt_state,
        critic_opt_state=critic_opt_state,

        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,

        backbone=backbone,
        action_head=action_head,
        critic=critic,

        batch=batch,

        advantages=advantages,
        returns=returns,

        num_epochs=num_epochs,
        target_kl=target_kl,

        clip_eps=clip_eps,
        entropy_coef=entropy_coef,

        valid_mask=valid_mask,
    )

    return (
        new_actor_params,
        new_critic_params,

        new_actor_opt_state,
        new_critic_opt_state,

        new_carry,

        batch,
        valid_mask,

        advantages,
        returns,

        metrics_history,
    )