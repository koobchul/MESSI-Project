"""Replay selected control frames with exact event telemetry.

Keep the ordinary collector and frozen policy. At selected pre-step ticks,
also run step_with_events from the identical state/action/key. Always advance
using the ordinary step result. Compare both endpoints before interpreting
event telemetry. Requires project/diagnose_trap_retention.py.
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from project.diagnose_trap_retention import (
    initialize_models,
    evaluate_policy_behavior,
    make_5v5_env,
    make_jitted_rollout_collector,
    load_checkpoint,
)


RESTART_NAMES = (
    "NONE", "KICKOFF", "THROWIN", "GOALKICK", "CORNER",
    "FREEKICK", "PENALTY", "OFFSIDE", "GK_HOLD",
)


def restart_name(value):
    i = int(value)
    return RESTART_NAMES[i] if 0 <= i < len(RESTART_NAMES) else str(i)


def print_frame_trace(tick, pre, post, events, equivalent):
    """Host callback: all coordinates are authoritative raw SI state."""
    print(f"\n=== EXACT FRAME TRACE: pre_tick={int(tick)}, post_tick={int(post.control_tick)} ===", flush=True)
    print(f"Lean/eventful endpoints equivalent: {bool(equivalent)}", flush=True)
    for name, state in (("PRE", pre), ("POST", post)):
        p = state.possession
        print(
            f"{name}: live={bool(state.ball.live)} "
            f"restart={restart_name(state.restart.kind)} team={int(state.restart.team)} "
            f"indirect={bool(state.restart.indirect)} taker={int(state.restart.taker)} "
            f"layout_ready={bool(state.restart_layout_ready)} "
            f"possession_team={int(p.team)} player={int(p.player)} ctrl={int(p.control_ticks)}",
            flush=True,
        )
        print(f"{name}: ball_xyz={np.asarray(state.ball.position)} ball_vel={np.asarray(state.ball.velocity)}", flush=True)
        print(f"{name}: last_contact={p.last_contact}", flush=True)
    displacement = np.linalg.norm(np.asarray(post.players.position) - np.asarray(pre.players.position), axis=-1)
    print(f"Player endpoint displacement (m): {displacement}", flush=True)
    # Last-contact actor at the endpoint may be the trapper even after stoppage.
    actor = int(post.possession.last_contact.actor)
    if 0 <= actor < len(displacement):
        for name, state in (("PRE", pre), ("POST", post)):
            pos = np.asarray(state.players.position)[actor]
            distance = np.linalg.norm(np.asarray(state.ball.position)[:2] - pos)
            print(f"{name}: last_actor={actor} player_xy={pos} ball_distance={distance:.6f} m", flush=True)
    for group in ("deliberate_contact", "contacts", "foul", "retouch", "offside", "boundary", "goalkeeper_holding"):
        event = getattr(events, group)
        print(f"EVENT {group} (leading axis = physics substep):", flush=True)
        for field in event._fields:
            print(f"  {field}: {np.asarray(getattr(event, field))}", flush=True)
    print("=== END FRAME TRACE ===\n", flush=True)


class TraceEnvironment:
    """Delegate the existing environment; retain ordinary step endpoints."""

    def __init__(self, env, ticks):
        self.env = env
        self.ticks = tuple(ticks)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, rollout, setup, action, key):
        ordinary = self.env.step(rollout, setup, action, key)
        selected = jnp.any(rollout.state.control_tick == jnp.asarray(self.ticks, dtype=jnp.int32))

        def inspect_frame(_):
            detailed = self.env.step_with_events(rollout, setup, action, key)
            left = jax.tree_util.tree_leaves(ordinary)
            right = jax.tree_util.tree_leaves(tuple(getattr(detailed, f) for f in ordinary._fields))
            # StepResult is a NamedTuple; its leaves have the same field order
            # as the selected ordinary fields of StepWithEventsResult.
            matches = []
            for a, b in zip(left, right):
                if jnp.issubdtype(a.dtype, jnp.inexact):
                    matches.append(jnp.allclose(a, b, rtol=1e-5, atol=1e-6))
                else:
                    matches.append(jnp.all(a == b))
            equivalent = jnp.all(jnp.stack(matches))
            jax.debug.callback(
                print_frame_trace, rollout.state.control_tick, rollout.state,
                ordinary.rollout.state, detailed.events, equivalent, ordered=True,
            )
            return jnp.int32(0)

        jax.lax.cond(selected, inspect_frame, lambda _: jnp.int32(0), operand=None)
        return ordinary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-steps", type=int, default=3072)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--trace-ticks", type=int, nargs="+", default=[1342, 2992])
    args = parser.parse_args()
    if args.num_steps <= 0 or args.rollout_length <= 0 or args.num_steps % args.rollout_length:
        parser.error("num-steps must be positive and divisible by rollout-length")
    if any(t < 0 or t >= args.num_steps for t in args.trace_ticks):
        parser.error("trace-ticks must lie in [0, num-steps)")

    env, reset = make_5v5_env(seed=args.seed)
    backbone, action_head, critic, _, _ = initialize_models(env, reset)
    checkpoint = load_checkpoint(args.checkpoint)
    trace_env = TraceEnvironment(env, args.trace_ticks)
    collector = make_jitted_rollout_collector(
        env=trace_env, backbone=backbone, action_head=action_head, critic=critic,
        rollout_length=args.rollout_length,
        controlled_indices=jnp.arange(5, dtype=jnp.int32), controlled_team=0,
        retouch_guard_enabled=False,
    )
    print(f"Trained policy only; seed={args.seed}; trace pre-step ticks={args.trace_ticks}", flush=True)
    print("Event traces are valid only if lean/eventful endpoints equivalent=True.", flush=True)
    evaluate_policy_behavior(
        label="TRAINED MLP: EXACT EVENT REPLAY", env=trace_env,
        backbone=backbone, action_head=action_head, critic=critic, collector=collector,
        actor_params=jax.device_put(checkpoint["actor_params"]),
        critic_params=jax.device_put(checkpoint["critic_params"]),
        num_steps=args.num_steps, rollout_length=args.rollout_length, seed=args.seed,
    )
    jax.effects_barrier()


if __name__ == "__main__":
    main()
