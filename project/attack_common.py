"""Atomic checkpoints and explicit completion semantics for attack experiments."""
import os
import pickle
from pathlib import Path
import numpy as np


def atomic_pickle(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def completion(env, state, done):
    from footballworld.core.constants import RK_NONE, RK_GK_HOLD, RK_PENALTY, RESTART_COUNT
    from footballworld.environment.episode import MAX_WALL_CONTROL_TICKS
    active = np.asarray(state.players.active)
    teams = np.asarray(state.players.team_id)
    counts = [int(np.sum(active & (teams == t))) for t in (0, 1)]
    kind, team = int(state.restart.kind), int(state.restart.team)
    invalid = RK_NONE < kind < RESTART_COUNT and kind != RK_GK_HOLD and (team not in (0, 1) or counts[team] == 0)
    control = int(state.control_tick)
    regulation = max(control - int(state.dead_ball_control_ticks) - int(state.first_half_live_extension_ticks), 0)
    full, half = env.match.clock_ticks(env.timebase)
    pending = kind == RK_PENALTY or bool(state.penalty_completion_active)
    half_end = int(state.first_half_wall_end_tick)
    added = env.match.maximum_added_time_ticks(env.timebase)
    limit = ((half_end >= 0 and control - half_end >= full - half + added)
             if env.match.halftime_enabled else control >= full + added)
    if not done: reason = 'external_cutoff'
    elif any(c < m for c, m in zip(counts, env.match.minimum_team_players)): reason = 'below_minimum'
    elif invalid: reason = 'invalid_restart_team'
    elif control >= MAX_WALL_CONTROL_TICKS: reason = 'wall_clock_exhausted'
    elif regulation >= full and not pending: reason = 'regulation_complete'
    elif limit and not pending: reason = 'added_time_limit_reached'
    else: reason = 'unknown_environment_done'
    return dict(terminal_reason=reason,
                scheduled_match_finished=reason in ('regulation_complete', 'added_time_limit_reached'),
                full_duration_complete=reason == 'regulation_complete',
                halftime_reached=half_end >= 0,
                regulation_elapsed_seconds=regulation * env.timebase.control_dt,
                control_tick=control)
