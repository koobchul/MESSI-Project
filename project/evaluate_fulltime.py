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

from footballworld.core.contact import INTENT_CONTROL, OUTCOME_TRAP, INTENT_SHOT, INTENT_PASS, OUTCOME_RELEASE
from footballworld.core.action import IntentAction
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
    "shot_requests", "shot_releases", "pass_requests", "pass_releases",
)


def make_evaluator(env, backbone, head, rollout_length, deterministic=False):
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
                if deterministic:
                    action_mean = IntentAction.from_array(jnp.argmax(output.intent_logits, axis=-1), jnp.tanh(output.continuous_mean))
                    sampled = sampled._replace(action=action_mean)
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
                    jnp.sum((sampled.action.intent == INTENT_SHOT) & (roster.team_id == 0) & state.players.active),
                    jnp.sum((event_actor >= 0) & (roster.team_id[safe_actor] == 0) & (result.events.deliberate_contact.intent == INTENT_SHOT) & (result.events.deliberate_contact.outcome == OUTCOME_RELEASE)),
                    jnp.sum((sampled.action.intent == INTENT_PASS) & (roster.team_id == 0) & state.players.active),
                    jnp.sum((event_actor >= 0) & (roster.team_id[safe_actor] == 0) & (result.events.deliberate_contact.intent == INTENT_PASS) & (result.events.deliberate_contact.outcome == OUTCOME_RELEASE)),
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
    import pickle
    import hashlib
    from project.attack_common import atomic_pickle, completion
    parser = argparse.ArgumentParser(description="Frozen full-match evaluator, resumable at chunk boundaries")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1000])
    parser.add_argument("--output", required=True)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0, help="0: no external cutoff; nonzero is diagnostic only")
    args = parser.parse_args()
    if args.rollout_length <= 0 or args.max_steps < 0:
        parser.error("Invalid step budget")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    checkpoint_bytes = Path(args.checkpoint).read_bytes()
    ck = pickle.loads(checkpoint_bytes)
    fingerprint = dict(checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
                       deterministic=args.deterministic, max_steps=args.max_steps,
                       rollout_length=args.rollout_length)
    env, _ = make_5v5_env(seed=args.seeds[0])
    backbone, head = MLPActorBackbone(hidden_dim=256), HybridActionHead()
    params = jax.device_put(ck["actor_params"])
    collect = make_evaluator(env, backbone, head, args.rollout_length, args.deterministic)
    rows = []
    for seed in args.seeds:
        reset = reset_5v5_env(env, seed)
        roster, si_roster = env.roster_metadata(reset.rollout), env.roster_metadata_si(reset.rollout)
        rollout_key = jax.random.key(seed + 20000)
        carry = initialize_jit_rollout_carry(env, reset, jax.random.key(seed + 10000))
        totals = np.zeros(len(METRICS), np.int64)
        saved = out / f"seed_{seed}_state.pkl"
        if saved.exists():
            with saved.open("rb") as f: prior = pickle.load(f)
            if prior["fingerprint"] != fingerprint:
                raise ValueError(f"Evaluation changed; use a new output directory: {saved}")
            carry, totals = jax.device_put(prior["carry"]), prior["totals"]
        chunk = 0
        print(f"seed={seed} fulltime_seconds={env.match.fulltime_seconds} resumed_steps={totals[0]}", flush=True)
        while not bool(carry.done) and (not args.max_steps or totals[0] < args.max_steps):
            carry, counts = collect(reset.setup, carry, roster, si_roster, params, rollout_key, jnp.bool_(True))
            totals += np.asarray(jax.device_get(counts), dtype=np.int64)
            chunk += 1
            atomic_pickle(saved, dict(fingerprint=fingerprint, carry=jax.device_get(carry), totals=totals))
            if chunk % 10 == 0 or bool(carry.done):
                print(f"steps={totals[0]} score={np.asarray(carry.rollout.state.score).tolist()} lawful={totals[METRICS.index('lawful_team_possession_ticks')]} shots={totals[METRICS.index('shot_releases')]}", flush=True)
        row = dict(seed=seed, checkpoint=str(Path(args.checkpoint).resolve()), **fingerprint)
        row.update({name: int(v) for name, v in zip(METRICS, totals)})
        state = carry.rollout.state
        row.update(goals_for=int(state.score[0]), goals_against=int(state.score[1]),
                   terminated=bool(carry.terminated), truncated=bool(carry.truncated),
                   external_cutoff=not bool(carry.done), **completion(env, state, bool(carry.done)))
        row["lawful_possession_pct"] = 100 * totals[METRICS.index('lawful_team_possession_ticks')] / max(1, totals[0])
        rows.append(row)
        with (out / "results.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerows(rows)
        print("RESULT", row, flush=True)
    if not all(r["scheduled_match_finished"] for r in rows) and not args.max_steps:
        raise SystemExit("Match abandoned or watchdog reached; see results.csv. This is NOT fulltime success.")


if __name__ == "__main__":
    main()
