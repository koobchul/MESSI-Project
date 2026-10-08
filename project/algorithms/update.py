import jax

from project.algorithms.ppo import (
    ppo_actor_loss,
    ppo_critic_loss,
)

from project.models.action_sampling import (
    action_entropy,
    action_log_prob,
)


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
):
    """
    Re-evaluate stored rollout actions using the current actor.

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
    """

    latent = backbone.apply(
        {
            "params": actor_params[
                "backbone"
            ]
        },
        actor_obs,
    )

    actor_output = action_head.apply(
        {
            "params": actor_params[
                "head"
            ]
        },
        latent,
    )

    new_log_prob = action_log_prob(
        actor_output,
        intent,
        pre_tanh_continuous,
    )

    entropy = action_entropy(
        actor_output
    )

    metrics = ppo_actor_loss(
        old_log_prob=old_log_prob,
        new_log_prob=new_log_prob,
        team_advantages=advantages,
        entropy=entropy,
        clip_eps=clip_eps,
        entropy_coef=entropy_coef,
    )

    return metrics.loss, metrics


def critic_loss_with_params(
    critic_params,
    critic,
    critic_obs,
    returns,
):
    """
    Re-evaluate centralized values.

    critic_obs:
        [T, 411]

    returns:
        [T]
    """

    values = critic.apply(
        {
            "params": critic_params
        },
        critic_obs,
    )

    loss = ppo_critic_loss(
        values,
        returns,
    )

    return loss

from typing import NamedTuple

import optax

from project.algorithms.ppo import (
    normalize_advantages,
)


class PPOUpdateMetrics(NamedTuple):
    actor_loss: jax.Array
    policy_loss: jax.Array
    entropy: jax.Array

    critic_loss: jax.Array

    approx_kl: jax.Array
    clip_fraction: jax.Array


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
):
    """
    Perform one full-batch PPO update.

    This is one PPO optimization step, not an entire PPO epoch loop.
    """

    # ============================================================
    # 1. Normalize team advantages
    # ============================================================
    normalized_advantages = normalize_advantages(
        advantages
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
        )

    (
        actor_loss,
        actor_metrics,
    ), actor_grads = jax.value_and_grad(
        actor_loss_fn,
        has_aux=True,
    )(
        actor_params
    )

    # ============================================================
    # 3. Actor optimizer
    # ============================================================
    actor_updates, new_actor_opt_state = (
        actor_optimizer.update(
            actor_grads,
            actor_opt_state,
            actor_params,
        )
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
        )

    critic_loss, critic_grads = (
        jax.value_and_grad(
            critic_loss_fn
        )(
            critic_params
        )
    )

    # ============================================================
    # 5. Critic optimizer
    # ============================================================
    critic_updates, new_critic_opt_state = (
        critic_optimizer.update(
            critic_grads,
            critic_opt_state,
            critic_params,
        )
    )

    new_critic_params = optax.apply_updates(
        critic_params,
        critic_updates,
    )

    # ============================================================
    # 6. Measure policy AFTER the optimizer update
    #
    # Important:
    # actor_metrics above were measured before updating parameters.
    # For KL early stopping we need the post-update policy.
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
    )

    metrics = PPOUpdateMetrics(
        actor_loss=actor_loss,
        policy_loss=actor_metrics.policy_loss,
        entropy=actor_metrics.entropy,

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
):
    """
    Reuse one rollout batch for multiple PPO optimization epochs.

    Stops early when the post-update approximate KL exceeds target_kl.

    Notes
    -----
    This is currently a full-batch PPO update.

    Minibatching will be added later after the complete
    rollout -> update -> rollout training loop is verified.
    """

    metrics_history = []

    for epoch in range(num_epochs):
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
        )

        metrics_history.append(
            metrics
        )

        # --------------------------------------------------------
        # KL early stopping
        #
        # PPO has already applied this update.
        # If it moved too far from the rollout policy,
        # do not perform another PPO epoch.
        # --------------------------------------------------------
        if float(metrics.approx_kl) > target_kl:
            break

    return (
        actor_params,
        critic_params,
        actor_opt_state,
        critic_opt_state,
        metrics_history,
    )