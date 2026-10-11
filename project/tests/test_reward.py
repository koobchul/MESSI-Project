#test_reward.py

import jax
import jax.numpy as jnp

from project.envs.messi_5v5 import (
    make_5v5_env,
)

from project.envs.reward import (
    ball_progress_potential,
    compute_shaped_reward,
    potential_shaping_reward,
)


def set_ball_x(
    global_rollout,
    x,
):
    """
    Return a copy of NormalizedGlobalRollout
    with only normalized ball x changed.
    """

    state = global_rollout.state

    new_position = (
        state.ball.position
        .at[0]
        .set(
            jnp.float32(x)
        )
    )

    new_ball = (
        state.ball._replace(
            position=new_position
        )
    )

    new_state = (
        state._replace(
            ball=new_ball
        )
    )

    return global_rollout._replace(
        state=new_state
    )

def test_ball_progress_potential_team_direction():
    env, reset = make_5v5_env(
        seed=0
    )

    state = env.global_state_view(
        reset.rollout
    )

    state = set_ball_x(
        state,
        0.4,
    )

    # At kickoff:
    #
    # Team 0 attacks +x
    # Team 1 attacks -x

    phi_team0 = ball_progress_potential(
        state,
        controlled_team=0,
    )

    phi_team1 = ball_progress_potential(
        state,
        controlled_team=1,
    )

    assert jnp.allclose(
        phi_team0,
        0.4,
        atol=1e-6,
    )

    assert jnp.allclose(
        phi_team1,
        -0.4,
        atol=1e-6,
    )


def test_forward_ball_progress_positive_for_team0():
    env, reset = make_5v5_env(
        seed=0
    )

    base_state = env.global_state_view(
        reset.rollout
    )

    state = set_ball_x(
        base_state,
        -0.4,
    )

    next_state = set_ball_x(
        base_state,
        0.2,
    )

    shaping = potential_shaping_reward(
        state=state,
        next_state=next_state,
        terminated=False,
        controlled_team=0,
        gamma=0.99,
        coefficient=0.05,
    )

    expected = (
        0.05
        * (
            0.99 * 0.2
            - (-0.4)
        )
    )

    assert jnp.allclose(
        shaping,
        expected,
        atol=1e-6,
    )

    assert shaping > 0.0


def test_same_transition_is_negative_for_team1():
    env, reset = make_5v5_env(
        seed=0
    )

    base_state = env.global_state_view(
        reset.rollout
    )

    state = set_ball_x(
        base_state,
        -0.4,
    )

    next_state = set_ball_x(
        base_state,
        0.2,
    )

    team0_reward = (
        potential_shaping_reward(
            state=state,
            next_state=next_state,
            terminated=False,
            controlled_team=0,
            gamma=0.99,
            coefficient=0.05,
        )
    )

    team1_reward = (
        potential_shaping_reward(
            state=state,
            next_state=next_state,
            terminated=False,
            controlled_team=1,
            gamma=0.99,
            coefficient=0.05,
        )
    )

    assert team0_reward > 0.0
    assert team1_reward < 0.0

    assert jnp.allclose(
        team0_reward,
        -team1_reward,
        atol=1e-6,
    )


def test_goal_reward_is_preserved_without_shaping():
    env, reset = make_5v5_env(
        seed=0
    )

    base_state = env.global_state_view(
        reset.rollout
    )

    # Simulate a scoring-side transition whose ball position
    # would otherwise produce a misleading reset shaping term.
    state = set_ball_x(
        base_state,
        0.9,
    )

    next_state = set_ball_x(
        base_state,
        0.0,
    )

    result = compute_shaped_reward(
        base_reward=jnp.float32(1.0),
        state=state,
        next_state=next_state,
        terminated=False,
        controlled_team=0,
        gamma=0.99,
        coefficient=0.05,
    )

    # Goal reward must remain exactly +1.
    assert jnp.allclose(
        result.base_reward,
        1.0,
        atol=1e-6,
    )

    # Goal/reset transition must not receive shaping.
    assert jnp.allclose(
        result.shaping_reward,
        0.0,
        atol=1e-6,
    )

    assert jnp.allclose(
        result.total_reward,
        1.0,
        atol=1e-6,
    )

def test_conceding_goal_does_not_receive_reset_bonus():
    env, reset = make_5v5_env(
        seed=0
    )

    base_state = env.global_state_view(
        reset.rollout
    )

    # Team 0 concedes near its own goal.
    #
    # Without the goal-reset guard:
    #
    #     ball x: -0.9 -> 0.0
    #
    # would incorrectly look like positive ball progress for Team 0.
    state = set_ball_x(
        base_state,
        -0.9,
    )

    next_state = set_ball_x(
        base_state,
        0.0,
    )

    result = compute_shaped_reward(
        base_reward=jnp.float32(-1.0),
        state=state,
        next_state=next_state,
        terminated=False,
        controlled_team=0,
        gamma=0.99,
        coefficient=0.05,
    )

    assert jnp.allclose(
        result.base_reward,
        -1.0,
        atol=1e-6,
    )

    # No artificial positive reward from the kickoff reset.
    assert jnp.allclose(
        result.shaping_reward,
        0.0,
        atol=1e-6,
    )

    assert jnp.allclose(
        result.total_reward,
        -1.0,
        atol=1e-6,
    )

def test_true_terminal_uses_zero_next_potential():
    env, reset = make_5v5_env(
        seed=0
    )

    base_state = env.global_state_view(
        reset.rollout
    )

    state = set_ball_x(
        base_state,
        0.4,
    )

    next_state = set_ball_x(
        base_state,
        0.9,
    )

    shaping = potential_shaping_reward(
        state=state,
        next_state=next_state,
        terminated=True,
        controlled_team=0,
        gamma=0.99,
        coefficient=0.05,
    )

    expected = (
        0.05
        * (
            0.0
            - 0.4
        )
    )

    assert jnp.allclose(
        shaping,
        expected,
        atol=1e-6,
    )


def test_reward_function_is_jittable():
    env, reset = make_5v5_env(
        seed=0
    )

    base_state = env.global_state_view(
        reset.rollout
    )

    state = set_ball_x(
        base_state,
        -0.3,
    )

    next_state = set_ball_x(
        base_state,
        0.1,
    )

    reward_fn = jax.jit(
        lambda s, ns: (
            compute_shaped_reward(
                base_reward=jnp.float32(0.0),
                state=s,
                next_state=ns,
                terminated=False,
                controlled_team=0,
                gamma=0.99,
                coefficient=0.05,
            ).total_reward
        )
    )

    reward = reward_fn(
        state,
        next_state,
    )

    assert jnp.isfinite(
        reward
    )

    assert reward > 0.0