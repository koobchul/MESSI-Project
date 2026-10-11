from typing import NamedTuple

import jax
import optax

from project.algorithms.ppo import (
    normalize_advantages,
    ppo_actor_loss,
    ppo_critic_loss,
)

from project.models.action_sampling import (
    action_entropy,
    action_log_prob,
)


class PPOUpdateMetrics(NamedTuple):
    actor_loss: jax.Array
    policy_loss: jax.Array
    entropy: jax.Array

    critic_loss: jax.Array

    approx_kl: jax.Array
    clip_fraction: jax.Array


def actor_loss_with_params(
    actor_params,
    backbone,
    action_head,
    actor_obs,
    intent,
    pre_tanh_continuous,
    old_log_prob,
    advantages,
    clip_eps=0.2,
    entropy_coef=0.01,
    valid_mask=None,
):
    """
    Re-evaluate stored rollout actions using the current actor.

    Parameters
    ----------
    actor_obs:
        [T, N, 414]

    intent:
        [T, N]

    pre_tanh_continuous:
        [T, N, 7]

    old_log_prob:
        [T, N]

    advantages:
        [T]

    valid_mask:
        [T] or None

        True:
            real environment transition

        False:
            padding row after terminal/truncated state

        None:
            original behavior; all rows are used
    """

    # ============================================================
    # 1. Re-evaluate actor backbone
    # ============================================================
    latent = backbone.apply(
        {
            "params":
                actor_params["backbone"]
        },
        actor_obs,
    )

    # ============================================================
    # 2. Re-evaluate action distribution
    # ============================================================
    actor_output = action_head.apply(
        {
            "params":
                actor_params["head"]
        },
        latent,
    )

    # ============================================================
    # 3. Log probability of STORED rollout actions
    #
    # PPO must evaluate the same actions that were sampled
    # during rollout collection.
    # ============================================================
    new_log_prob = action_log_prob(
        actor_output,
        intent,
        pre_tanh_continuous,
    )

    # ============================================================
    # 4. Policy entropy
    # ============================================================
    entropy = action_entropy(
        actor_output
    )

    # ============================================================
    # 5. PPO actor objective
    #
    # valid_mask removes fixed-shape padding rows when supplied.
    # ============================================================
    metrics = ppo_actor_loss(
        old_log_prob=old_log_prob,
        new_log_prob=new_log_prob,

        team_advantages=advantages,

        entropy=entropy,

        clip_eps=clip_eps,
        entropy_coef=entropy_coef,

        valid_mask=valid_mask,
    )

    return (
        metrics.loss,
        metrics,
    )


def critic_loss_with_params(
    critic_params,
    critic,
    critic_obs,
    returns,
    valid_mask=None,
):
    """
    Re-evaluate centralized critic values.

    Parameters
    ----------
    critic_obs:
        [T, 411]

    returns:
        [T]

    valid_mask:
        [T] or None

        Padding rows are ignored when supplied.
    """

    # ============================================================
    # 1. Current centralized value estimates
    # ============================================================
    values = critic.apply(
        {
            "params":
                critic_params
        },
        critic_obs,
    )

    # ============================================================
    # 2. Critic loss
    # ============================================================
    loss = ppo_critic_loss(
        values=values,
        returns=returns,
        valid_mask=valid_mask,
    )

    return loss


def ppo_update_step(
    actor_params,
    critic_params,
    actor_opt_state,
    critic_opt_state,
    actor_optimizer,
    critic_optimizer,
    backbone,
    action_head,
    critic,
    batch,
    advantages,
    returns,
    clip_eps=0.2,
    entropy_coef=0.01,
    valid_mask=None,
):
    """
    Perform one full-batch PPO optimization step.

    This is one PPO optimizer update, not the complete
    multi-epoch PPO loop.

    Parameters
    ----------
    valid_mask:
        [T] or None

        When supplied, padded rollout rows are excluded from:

            - advantage normalization
            - actor objective
            - entropy
            - approximate KL
            - clip fraction
            - critic objective

        When None, behavior is identical to the original
        non-padded rollout implementation.
    """

    # ============================================================
    # 1. Normalize centralized team advantages
    # ============================================================
    normalized_advantages = normalize_advantages(
        advantages,
        valid_mask=valid_mask,
    )

    # ============================================================
    # 2. Actor loss
    # ============================================================
    def actor_loss_fn(params):
        return actor_loss_with_params(
            actor_params=params,

            backbone=backbone,
            action_head=action_head,

            actor_obs=batch.actor_obs,
            intent=batch.intent,

            pre_tanh_continuous=(
                batch.pre_tanh_continuous
            ),

            old_log_prob=batch.log_prob,

            advantages=(
                normalized_advantages
            ),

            clip_eps=clip_eps,
            entropy_coef=entropy_coef,

            valid_mask=valid_mask,
        )

    (
        (
            actor_loss,
            actor_metrics,
        ),
        actor_grads,
    ) = jax.value_and_grad(
        actor_loss_fn,
        has_aux=True,
    )(
        actor_params
    )

    # ============================================================
    # 3. Actor optimizer
    # ============================================================
    (
        actor_updates,
        new_actor_opt_state,
    ) = actor_optimizer.update(
        actor_grads,
        actor_opt_state,
        actor_params,
    )

    new_actor_params = optax.apply_updates(
        actor_params,
        actor_updates,
    )

    # ============================================================
    # 4. Critic loss
    # ============================================================
    def critic_loss_fn(params):
        return critic_loss_with_params(
            critic_params=params,

            critic=critic,

            critic_obs=batch.critic_obs,
            returns=returns,

            valid_mask=valid_mask,
        )

    (
        critic_loss,
        critic_grads,
    ) = jax.value_and_grad(
        critic_loss_fn
    )(
        critic_params
    )

    # ============================================================
    # 5. Critic optimizer
    # ============================================================
    (
        critic_updates,
        new_critic_opt_state,
    ) = critic_optimizer.update(
        critic_grads,
        critic_opt_state,
        critic_params,
    )

    new_critic_params = optax.apply_updates(
        critic_params,
        critic_updates,
    )

    # ============================================================
    # 6. Re-evaluate policy AFTER actor optimizer update
    #
    # actor_metrics above describe the policy before the optimizer
    # step.
    #
    # PPO KL early stopping must compare the updated policy against
    # the fixed rollout policy.
    #
    # Padding rows must also be excluded from these diagnostics.
    # ============================================================
    (
        _,
        post_actor_metrics,
    ) = actor_loss_with_params(
        actor_params=new_actor_params,

        backbone=backbone,
        action_head=action_head,

        actor_obs=batch.actor_obs,
        intent=batch.intent,

        pre_tanh_continuous=(
            batch.pre_tanh_continuous
        ),

        old_log_prob=batch.log_prob,

        advantages=(
            normalized_advantages
        ),

        clip_eps=clip_eps,
        entropy_coef=entropy_coef,

        valid_mask=valid_mask,
    )

    # ============================================================
    # 7. Update metrics
    # ============================================================
    metrics = PPOUpdateMetrics(
        actor_loss=actor_loss,

        policy_loss=(
            actor_metrics.policy_loss
        ),

        entropy=(
            actor_metrics.entropy
        ),

        critic_loss=critic_loss,

        approx_kl=(
            post_actor_metrics.approx_kl
        ),

        clip_fraction=(
            post_actor_metrics.clip_fraction
        ),
    )

    return (
        new_actor_params,
        new_critic_params,

        new_actor_opt_state,
        new_critic_opt_state,

        metrics,
    )


def ppo_update_epochs(
    actor_params,
    critic_params,
    actor_opt_state,
    critic_opt_state,
    actor_optimizer,
    critic_optimizer,
    backbone,
    action_head,
    critic,
    batch,
    advantages,
    returns,
    num_epochs=4,
    target_kl=0.02,
    clip_eps=0.2,
    entropy_coef=0.01,
    valid_mask=None,
):
    """
    Reuse one rollout batch for multiple PPO optimization epochs.

    Stops early when the post-update approximate KL exceeds
    target_kl.

    Notes
    -----
    This is currently a full-batch PPO update.

    old_log_prob remains fixed to the probabilities recorded
    during rollout collection throughout all PPO epochs.

    valid_mask remains fixed with the rollout batch and prevents
    terminal padding rows from participating in optimization.

    Minibatching can be added after the complete JIT rollout
    -> GAE -> PPO -> rollout training loop is verified.
    """

    metrics_history = []

    # ============================================================
    # PPO optimization epochs
    # ============================================================
    for epoch in range(
        num_epochs
    ):
        (
            actor_params,
            critic_params,

            actor_opt_state,
            critic_opt_state,

            metrics,
        ) = ppo_update_step(
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

            clip_eps=clip_eps,
            entropy_coef=entropy_coef,

            valid_mask=valid_mask,
        )

        metrics_history.append(
            metrics
        )

        # --------------------------------------------------------
        # KL early stopping
        #
        # The current update has already been applied.
        #
        # If the new policy has moved farther than target_kl
        # from the rollout policy, stop before another epoch.
        #
        # approx_kl itself is masked, so padding transitions do
        # not affect this stopping decision.
        # --------------------------------------------------------
        if (
            float(
                metrics.approx_kl
            )
            > target_kl
        ):
            break

    return (
        actor_params,
        critic_params,

        actor_opt_state,
        critic_opt_state,

        metrics_history,
    )