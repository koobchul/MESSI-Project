from project.envs.reward_v3 import control_potentials
"""Compare one frozen checkpoint with the retouch guard OFF versus ON.

Uses exact event telemetry, identical match seeds and the same RNG addressing
as rollout_jit.py. This is bounded evaluation unless the environment ends.
No weights, rules or rewards are updated.
"""

import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from footballworld.core.contact import INTENT_CONTROL, OUTCOME_TRAP
from footballworld.policies.rule_based import make_rule_based_policy
from project.algorithms.rollout_jit import initialize_jit_rollout_carry
from project.envs.messi_5v5 import make_5v5_env, reset_5v5_env
from project.envs.obs_encoder import encode_mlp_observation
from project.models.action_head import HybridActionHead
from project.models.mlp_actor import MLPActorBackbone
from project.models.action_sampling import sample_action
from project.models.intent_guard import apply_retouch_guard, retouch_prohibited_from_mlp
from project.utils.checkpoint import load_checkpoint
from project.utils.team_action import combine_team_actions

METRICS = (
    "valid_steps", "team_possession_ticks", "opponent_possession_ticks",
    "retouch_team0", "retouch_team1", "control_traps_team0",
    "control_traps_with_same_substep_retouch_team0", "guard_context_agent_ticks",
    "nonmove_requests_in_guard_context", "dead_ball_ticks", "restart_opened_frames",
    "lawful_team_possession_ticks", "lawful_opponent_possession_ticks",
)


def make_evaluator(env, backbone, head, rollout_length):
    opponent = make_rule_based_policy(env)

    @jax.jit
    def collect(setup, carry, roster, si_roster, params, rollout_key, guard_enabled):
        actor_key, env_key = jax.random.split(rollout_key, 2)

        def scan_step(carry, _):
            def pause(_):
                return carry, jnp.zeros(len(METRICS), dtype=jnp.int32)

            def advance(_):
                state = carry.rollout.state
                tick = state.control_tick.astype(jnp.uint32)
                obs = encode_mlp_observation(env.observe_all(carry.rollout), roster)
                latent = backbone.apply({"params": params["backbone"]}, obs)
                output = head.apply({"params": params["head"]}, latent)
                output = apply_retouch_guard(output, obs, enabled=guard_enabled)
                sampled = sample_action(output, jax.random.fold_in(actor_key, tick))
                si_obs = env.observe_all_si(carry.rollout)
                rule_step = opponent.step(si_obs, si_roster, carry.rule_state, carry.opponent_key)
                action = combine_team_actions(
                    sampled.action, rule_step.action, roster.team_id, controlled_team=0,
                )
                result = env.step_with_events(
                    carry.rollout, setup, action, jax.random.fold_in(env_key, tick),
                )
                retouch = result.events.retouch
                event_actor = result.events.deliberate_contact.actor
                safe_actor = jnp.clip(event_actor, 0, roster.team_id.shape[0] - 1)
                trap = (
                    (event_actor >= 0) & (roster.team_id[safe_actor] == 0)
                    & (result.events.deliberate_contact.intent == INTENT_CONTROL)
                    & (result.events.deliberate_contact.outcome == OUTCOME_TRAP)
                )
                context = retouch_prohibited_from_mlp(obs) & (roster.team_id == 0)
                counts = jnp.asarray([
                    1,
                    state.possession.team == 0,
                    state.possession.team == 1,
                    jnp.sum(retouch.occurred & (retouch.team == 1)),
                    jnp.sum(retouch.occurred & (retouch.team == 0)),
                    jnp.sum(trap),
                    jnp.sum(trap & retouch.occurred & (retouch.offender == event_actor)),
                    jnp.sum(context),
                    jnp.sum(context & (sampled.action.intent != 0)),
                    ~state.ball.live,
                    result.restart_opened,
                    control_potentials(state, controlled_team=0).possession > 0,
                    control_potentials(state, controlled_team=1).possession > 0,
                ], dtype=jnp.int32)
                next_carry = carry._replace(
                    rollout=result.rollout, rule_state=rule_step.state,
                    terminated=result.terminated, truncated=result.truncated, done=result.done,
                )
                return next_carry, counts

            return jax.lax.cond(carry.done, pause, advance, operand=None)

        carry, counts = jax.lax.scan(scan_step, carry, None, length=rollout_length)
        return carry, jnp.sum(counts, axis=0)

    return collect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-steps", type=int, default=6400)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1001])
    parser.add_argument("--csv", default="project/eval_logs/retouch_guard_8k_seed1001.csv")
    parser.add_argument("--guard-modes", nargs="+", choices=("OFF", "ON"), default=["OFF", "ON"])
    args = parser.parse_args()
    if args.num_steps <= 0 or args.rollout_length <= 0 or args.num_steps % args.rollout_length:
        parser.error("num-steps must be positive and divisible by rollout-length")
    env, _ = make_5v5_env(seed=args.seeds[0])
    backbone, head = MLPActorBackbone(hidden_dim=256), HybridActionHead()
    params = jax.device_put(load_checkpoint(args.checkpoint)["actor_params"])
    collect = make_evaluator(env, backbone, head, args.rollout_length)
    rows = []
    for seed in args.seeds:
        for mode in args.guard_modes:
            enabled = mode == "ON"
            reset = reset_5v5_env(env, seed=seed)
            roster = env.roster_metadata(reset.rollout)
            si_roster = env.roster_metadata_si(reset.rollout)
            opponent_key = jax.random.fold_in(jax.random.key(seed + 10000), 0)
            rollout_key = jax.random.fold_in(jax.random.key(seed + 20000), 0)
            carry = initialize_jit_rollout_carry(env, reset, opponent_key)
            totals = np.zeros(len(METRICS), dtype=np.int64)
            chunks = args.num_steps // args.rollout_length
            print(f"\nseed={seed}, guard={'ON' if enabled else 'OFF'}, max_steps={args.num_steps}", flush=True)
            for i in range(chunks):
                carry, counts = collect(
                    reset.setup, carry, roster, si_roster, params,
                    rollout_key, jnp.bool_(enabled),
                )
                totals += np.asarray(jax.device_get(counts), dtype=np.int64)
                if (i + 1) % 10 == 0 or i + 1 == chunks:
                    score = np.asarray(jax.device_get(carry.rollout.state.score))
                    print(f"chunk={i+1}/{chunks}, steps={totals[0]}, score={score.tolist()}, team0_retouch={totals[3]}", flush=True)
                if bool(jax.device_get(carry.done)):
                    break
            score = np.asarray(jax.device_get(carry.rollout.state.score))
            row = dict(seed=seed, guard="ON" if enabled else "OFF", checkpoint=args.checkpoint)
            row.update({name: int(value) for name, value in zip(METRICS, totals)})
            row.update(
                goals_for=int(score[0]), goals_against=int(score[1]),
                team_possession_pct=100.0 * totals[1] / max(1, totals[0]),
                terminated=bool(jax.device_get(carry.terminated)),
                truncated=bool(jax.device_get(carry.truncated)),
                external_cutoff=not bool(jax.device_get(carry.done)),
            )
            rows.append(row)
            print("RESULT:", row, flush=True)
    target = Path(args.csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved: {target.resolve()}", flush=True)
    print("TRAP counts are physical outcomes; the same-substep-retouch subset is not lawful retained possession.")
    print("A guard can reduce explicit illegal requests, but MOVE may still cause passive retouches.")


if __name__ == "__main__":
    main()
