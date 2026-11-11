"""Experimental reward v3. Inputs are authoritative SI states, never observations.

Keeps v2 progress unchanged and adds bounded potential differences for approach
and physically verified possession. Extra signals apply only within live open
play, suppressing restart/goal placement transitions. This gated heuristic is
not claimed to preserve the optimal policy of the original sparse reward.
"""
from typing import NamedTuple

import jax.numpy as jnp
from footballworld.config.action import ActionScale
from footballworld.config.geometry import Ball
from footballworld.config.reach import Reach
from footballworld.dynamics.contact_predicates import verified_controlled_carrier


class ControlPotentials(NamedTuple):
    approach: object
    possession: object


def open_play(state):
    return state.ball.live & (state.restart.kind == 0)


def prohibited_taker_mask(state):
    release = state.restart_release
    pending = release.active & release.untouched & (release.kind >= 1) & (release.kind <= 7)
    return pending & (jnp.arange(state.players.position.shape[0]) == release.taker)


def control_potentials(state, controlled_team=0, distance_scale_m=10.0,
                       ball_geometry=Ball(), reach=Reach(), scale=ActionScale()):
    """Nearest eligible outfielder, plus strict lawful controlled possession.

    A 10 m approach scale provides gradient at starting positions without a
    clipped-distance plateau. Only one nearest teammate contributes, so adding
    more chasers does not multiply reward. Goalkeepers can earn possession but
    are excluded from the approach incentive to avoid pulling them out of goal.
    """
    players = state.players
    prohibited = prohibited_taker_mask(state)
    eligible = players.active & (players.team_id == controlled_team) & ~players.is_goalkeeper & ~prohibited
    distance = jnp.sqrt(jnp.sum((players.position - state.ball.position[:2]) ** 2, axis=-1) + 1e-8)
    nearest = jnp.min(jnp.where(eligible, distance, jnp.float32(jnp.inf)))
    approach = jnp.exp(-nearest / jnp.float32(distance_scale_m))
    carrier = verified_controlled_carrier(state, ball_geometry, reach, scale)
    safe = jnp.clip(carrier, 0, players.position.shape[0] - 1)
    possession = (carrier >= 0) & (players.team_id[safe] == controlled_team) & ~prohibited[safe]
    live = open_play(state)
    return ControlPotentials(jnp.where(live, approach, 0.).astype(jnp.float32),
                             (live & possession).astype(jnp.float32))


def extra_control_shaping(state, next_state, terminated, base_reward, gamma=.99,
                          approach_coefficient=.05, possession_coefficient=.05,
                          controlled_team=0, distance_scale_m=10.,
                          ball_geometry=Ball(), reach=Reach(), scale=ActionScale()):
    current = control_potentials(state, controlled_team, distance_scale_m, ball_geometry, reach, scale)
    following = control_potentials(next_state, controlled_team, distance_scale_m, ball_geometry, reach, scale)
    next_a = jnp.where(terminated, 0., following.approach)
    next_p = jnp.where(terminated, 0., following.possession)
    # Restart/goal/half-time teleports do not yield approach or acquisition reward.
    # A genuine terminal uses zero next potential; a truncation still bootstraps.
    eligible = open_play(state) & (open_play(next_state) | terminated) & (jnp.abs(base_reward) == 0)
    approach = approach_coefficient * (gamma * next_a - current.approach)
    possession = possession_coefficient * (gamma * next_p - current.possession)
    return jnp.where(eligible, approach, 0.).astype(jnp.float32), jnp.where(eligible, possession, 0.).astype(jnp.float32)
