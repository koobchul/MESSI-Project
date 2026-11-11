"""Imitate the bundled rule policy, then export weights for independent MAPPO.

Teacher acts only during demonstration collection. Student uses the existing
414-dimensional per-player observation; no teacher or hidden state at inference.
This is imitation learning, not a pure-from-scratch RL baseline.
"""
import argparse
import json
import pickle
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import optax
from footballworld.policies.rule_based import make_rule_based_policy
from project.envs.messi_5v5 import make_5v5_env, reset_5v5_env
from project.envs.obs_encoder import encode_mlp_observation
from project.models.intent_guard import apply_retouch_guard, retouch_prohibited_from_mlp
from project.train import initialize_models
from project.algorithms.rollout_jit import initialize_jit_rollout_carry
from project.attack_common import atomic_pickle


def make_teacher_collector(env, length):
    teacher = make_rule_based_policy(env)
    @jax.jit
    def collect(setup, carry, roster, si_roster):
        obs_shape = encode_mlp_observation(env.observe_all(carry.rollout), roster).shape
        def step(c, _):
            def pause(_):
                return c, (jnp.zeros(obs_shape, jnp.float32), jnp.zeros(obs_shape[0], jnp.int32),
                           jnp.zeros((obs_shape[0], 7), jnp.float32), jnp.zeros(obs_shape[0], jnp.bool_))
            def advance(_):
                obs = encode_mlp_observation(env.observe_all(c.rollout), roster)
                t = teacher.step(env.observe_all_si(c.rollout), si_roster, c.rule_state, c.opponent_key)
                # Use precisely the same retouch constraint in demos and student.
                intent = jnp.where(retouch_prohibited_from_mlp(obs), 0, t.action.intent)
                action = t.action._replace(intent=intent)
                result = env.step(c.rollout, setup, action,
                                  jax.random.fold_in(c.opponent_key, c.rollout.state.control_tick))
                valid = c.rollout.state.players.active & (roster.team_id == 0)
                nxt = c._replace(rollout=result.rollout, rule_state=t.state,
                                 terminated=result.terminated, truncated=result.truncated, done=result.done)
                return nxt, (obs, intent, action.as_continuous_array(), valid)
            return jax.lax.cond(c.done, pause, advance, None)
        return jax.lax.scan(step, carry, None, length=length)
    return collect


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--demo-steps', type=int, default=8192)
    p.add_argument('--updates', type=int, default=3000)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--seeds', type=int, nargs='+', default=[10, 11, 12, 13, 14])
    a = p.parse_args()
    if min(a.demo_steps, a.updates, a.batch_size) <= 0 or len(a.seeds) < 2 or len(set(a.seeds)) != len(a.seeds):
        p.error('Positive sizes and at least two distinct seeds required')
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    config = vars(a).copy(); config.pop('output')
    config_path = out / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('BC configuration changed; use a new output folder')
    config_path.write_text(json.dumps(config, indent=2))
    env, reset = make_5v5_env(seed=a.seeds[0])
    backbone, head, _, params, _ = initialize_models(env, reset)
    collect = make_teacher_collector(env, 128)
    arrays = []
    for seed in a.seeds:
        dest = out / f'demo_seed{seed}.pkl'
        if dest.exists():
            with dest.open('rb') as f: data = pickle.load(f)
        else:
            reset = reset_5v5_env(env, seed)
            roster, si_roster = env.roster_metadata(reset.rollout), env.roster_metadata_si(reset.rollout)
            carry = initialize_jit_rollout_carry(env, reset, jax.random.key(seed + 30000))
            chunks = []; steps = 0
            progress = out / f'demo_seed{seed}_partial.pkl'
            if progress.exists():
                with progress.open('rb') as f: old = pickle.load(f)
                carry, chunks, steps = jax.device_put(old['carry']), old['chunks'], old['steps']
            while steps < a.demo_steps and not bool(carry.done):
                carry, batch = collect(reset.setup, carry, roster, si_roster)
                obs, intent, continuous, valid = jax.device_get(batch)
                # Trim the final chunk to the requested collection budget.
                take = min(128, a.demo_steps - steps)
                valid = np.array(valid, copy=True)
                valid[take:] = False
                chunks.append((obs[valid], intent[valid], continuous[valid]))
                steps += take
                atomic_pickle(progress, dict(carry=jax.device_get(carry), chunks=chunks, steps=steps))
                if steps % 1280 == 0: print(f'demo seed={seed} steps={steps} score={np.asarray(carry.rollout.state.score).tolist()}', flush=True)
            data = tuple(np.concatenate([c[i] for c in chunks]) for i in range(3))
            atomic_pickle(dest, data)
            progress.unlink(missing_ok=True)
        arrays.append(data)
        print('demo ready', seed, 'intent histogram', np.bincount(data[1], minlength=6).tolist(), flush=True)
    train = tuple(np.concatenate([d[i] for d in arrays[:-1]]) for i in range(3))
    valid = arrays[-1]  # Entire held-out seed, never random rows from the same trajectory.
    counts = np.bincount(train[1], minlength=6)
    weights = np.sqrt(max(1, len(train[1])) / (6 * np.maximum(counts, 1)))
    weights = jnp.asarray(np.clip(weights, .25, 5), jnp.float32)
    tx = optax.chain(optax.clip_by_global_norm(1.), optax.adam(3e-4))
    opt = tx.init(params)
    def loss(params, obs, intent, target):
        output = head.apply({'params': params['head']}, backbone.apply({'params': params['backbone']}, obs))
        output = apply_retouch_guard(output, obs)
        ce = optax.softmax_cross_entropy_with_integer_labels(output.intent_logits, intent)
        # Movement always matters. Ball force / launch / spin only on non-MOVE.
        mask = jnp.concatenate([jnp.ones((intent.shape[0], 2)), jnp.broadcast_to((intent != 0)[:, None], (intent.shape[0], 5))], axis=1)
        mse = jnp.sum(mask * (jnp.tanh(output.continuous_mean) - target)**2, axis=1) / jnp.sum(mask, axis=1)
        # BC learns means; a modest fixed initial exploration is used for PPO.
        return jnp.mean(weights[intent] * (ce + 4 * mse))
    @jax.jit
    def update(params, opt, obs, intent, target):
        value, grad = jax.value_and_grad(loss)(params, obs, intent, target)
        delta, opt = tx.update(grad, opt, params)
        return optax.apply_updates(params, delta), opt, value
    evaluate = jax.jit(loss)
    rng = np.random.default_rng(0)
    best = float('inf'); start = 0
    checkpoint = out / 'fit_state.pkl'
    if checkpoint.exists():
        with checkpoint.open('rb') as f: old = pickle.load(f)
        params, opt = jax.device_put(old['params']), jax.device_put(old['opt'])
        start, best, rng.bit_generator.state = old['update'], old['best'], old['rng']
    # Fix log std in exported initialization; warm-starting never changes action semantics.
    params['head']['continuous_log_std'] = jnp.full_like(params['head']['continuous_log_std'], -2.)
    for u in range(start, a.updates):
        idx = rng.integers(0, len(train[0]), size=a.batch_size)
        params, opt, value = update(params, opt, *(jnp.asarray(x[idx]) for x in train))
        if (u + 1) % 100 == 0 or u + 1 == a.updates:
            values = [float(evaluate(params, *(jnp.asarray(x[i:i+1024]) for x in valid))) * len(valid[0][i:i+1024])
                      for i in range(0, len(valid[0]), 1024)]
            val = sum(values) / len(valid[0])
            if not np.isfinite(val) or not np.isfinite(float(value)): raise ValueError('Nonfinite BC loss')
            if val < best:
                best = val
                atomic_pickle(out / 'actor_best.pkl', dict(kind='attack_bc_v1', actor_params=jax.device_get(params),
                              config=config, validation_loss=best, update=u+1, teacher='bundled_rule_policy', intent_counts=counts))
            atomic_pickle(checkpoint, dict(params=jax.device_get(params), opt=jax.device_get(opt), update=u+1,
                                          best=best, rng=rng.bit_generator.state))
            print(f'BC update={u+1} train={float(value):.5f} heldout={val:.5f} best={best:.5f}', flush=True)
    print('Actor exported:', out / 'actor_best.pkl', flush=True)


if __name__ == '__main__': main()
