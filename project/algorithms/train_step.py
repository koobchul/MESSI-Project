import jax
import jax.numpy as jnp

from project.algorithms.gae import compute_gae
from project.algorithms.rollout import collect_rollout
from project.algorithms.update import ppo_update_epochs
from project.envs.critic_encoder import encode_critic_state


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
    Perform one complete MAPPO training iteration.

    Flow
    ----
    current policy
        -> MESSI rollout
        -> bootstrap value
        -> GAE
        -> PPO update epochs
        -> updated actor / critic
    """

    # ============================================================
    # 1. Convert parameter trees to Flax variable dictionaries
    # ============================================================
    backbone_variables = {
        "params": actor_params["backbone"]
    }

    head_variables = {
        "params": actor_params["head"]
    }

    critic_variables = {
        "params": critic_params
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

    # True terminal states do not bootstrap.
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

    # Treat rollout targets as fixed PPO data.
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