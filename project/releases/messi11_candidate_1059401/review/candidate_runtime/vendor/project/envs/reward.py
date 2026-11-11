#reward.py

from typing import NamedTuple

import jax
import jax.numpy as jnp


class RewardComponents(NamedTuple):
    """
    Components of the shaped team reward.

    base_reward:
        Original MESSI sparse goal reward.

    shaping_reward:
        Potential-based shaping term.

    total_reward:
        base_reward + shaping_reward.
    """

    base_reward: jax.Array
    shaping_reward: jax.Array
    total_reward: jax.Array


def ball_progress_potential(
    global_rollout,
    controlled_team: int = 0,
) -> jax.Array:
    """
    Ball-progress potential from the controlled team's perspective.

    env.global_state_view() returns NormalizedGlobalRollout, whose
    actual normalized environment state is stored in `.state`.
    """

    state = global_rollout.state

    ball_x = jnp.asarray(
        state.ball.position[0],
        dtype=jnp.float32,
    )

    ball_x = jnp.clip(
        ball_x,
        -1.0,
        1.0,
    )

    attack_direction = jnp.asarray(
        state.attack_direction[
            controlled_team
        ],
        dtype=jnp.float32,
    )

    return (
        attack_direction
        * ball_x
    )

def potential_shaping_reward(
    state,
    next_state,
    terminated,
    controlled_team: int = 0,
    gamma: float = 0.99,
    coefficient: float = 0.05,
) -> jax.Array:
    """
    Compute potential-based shaping:

        F(s, s')
        =
        alpha * [gamma * Phi(s') - Phi(s)]

    True environment termination uses Phi(terminal) = 0.

    Truncation is intentionally not handled here as termination because
    the current MAPPO implementation bootstraps through truncation.
    """

    phi = ball_progress_potential(
        state,
        controlled_team=controlled_team,
    )

    next_phi = ball_progress_potential(
        next_state,
        controlled_team=controlled_team,
    )

    # A true terminal absorbing state has zero potential.
    next_phi = jnp.where(
        jnp.asarray(
            terminated,
            dtype=jnp.bool_,
        ),
        jnp.float32(0.0),
        next_phi,
    )

    return (
        jnp.float32(coefficient)
        * (
            jnp.float32(gamma)
            * next_phi
            - phi
        )
    )


def compute_shaped_reward(
    base_reward,
    state,
    next_state,
    terminated,
    controlled_team: int = 0,
    gamma: float = 0.99,
    coefficient: float = 0.05,
) -> RewardComponents:
    """
    Add ball-progress shaping to the original MESSI reward.

    Goal transitions receive no shaping reward because MESSI
    resets the ball toward the centre after a goal. Using the
    post-goal reset state directly would otherwise create an
    artificial shaping bonus for conceding and an artificial
    penalty for scoring.
    """

    base_reward = jnp.asarray(
        base_reward,
        dtype=jnp.float32,
    )

    shaping_reward = potential_shaping_reward(
        state=state,
        next_state=next_state,
        terminated=terminated,
        controlled_team=controlled_team,
        gamma=gamma,
        coefficient=coefficient,
    )

    goal_occurred = (
        jnp.abs(base_reward)
        > jnp.float32(0.0)
    )

    shaping_reward = jnp.where(
        goal_occurred,
        jnp.float32(0.0),
        shaping_reward,
    )

    total_reward = (
        base_reward
        + shaping_reward
    )

    return RewardComponents(
        base_reward=base_reward,
        shaping_reward=shaping_reward,
        total_reward=total_reward,
    )