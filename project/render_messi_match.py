#!/usr/bin/env python3
"""Render MESSI saved states, or re-evaluate a checkpoint and verify logged fields.

Run from the MESSI repository, in the existing 'messi' conda environment.
No training, weight changes, or video generation happens until YOU run this file.

Goal replay (the Gaussian HOME goal from job 1024700):
  python render_messi_match.py --experiment project/experiments/messi11_entropy_1024700 --arm gaussian
Other recorded goal:
  python render_messi_match.py --experiment project/experiments/messi11_entropy_1024700 --arm bounded
Existing full-state recording, without policy replay:
  python render_messi_match.py --states PATH/after_gaussian/states.pkl --output PATH/video_copy

Default replay video: 15 seconds before through 5 seconds after the first HOME
score increase. Use --full for the entire evaluation. If replay diverges, it is
explicitly labelled as a new evaluation; the old goal is never fabricated.
Requires the original messi11_entropy_patch, footballworld, JAX and ffmpeg.
Only load pickle files that you trust. This file never extracts an archive.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import traceback


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--experiment', type=Path, help='Original experiment directory')
    mode.add_argument('--states', type=Path, help='Full time-stacked states.pkl; NOT evaluation/state.pkl')
    p.add_argument('--arm', choices=['gaussian', 'bounded'], default='gaussian')
    p.add_argument('--package', type=Path, default=Path('messi11_entropy_patch'))
    p.add_argument('--output', type=Path, help='NEW directory; existing directories are never overwritten')
    p.add_argument('--full', action='store_true', help='Render all recorded states instead of a goal clip')
    p.add_argument('--before', type=float, default=15.)
    p.add_argument('--after', type=float, default=5.)
    p.add_argument('--fps', type=float, default=10.)
    p.add_argument('--platform', choices=['cpu', 'cuda'], help='Optional JAX backend override')
    a = p.parse_args()
    if a.before < 0 or a.after <= 0 or a.fps <= 0:
        p.error('--before must be nonnegative; --after and --fps must be positive')
    return a


def recording_collector_source(fn):
    """Add a state output to the ORIGINAL evaluator, without changing any action.

    Executed in memory only. Refuse unknown source rather than silently using a
    different action path. The original terminal pause remains in place.
    """
    source = inspect.getsource(fn)
    changes = [
        ('return carry, (jnp.zeros(len(METRICS), dtype=jnp.int32), jnp.zeros(len(FIELDS),dtype=jnp.float32))',
         'return carry, (jnp.zeros(len(METRICS), dtype=jnp.int32), jnp.zeros(len(FIELDS),dtype=jnp.float32), carry.rollout.state)'),
        ('return next_carry, (counts,row)', 'return next_carry, (counts,row,result.rollout.state)'),
        ('carry, (counts, rows) = jax.lax.scan', 'carry, (counts, rows, states) = jax.lax.scan'),
        ('return carry, jnp.sum(counts, axis=0), rows', 'return carry, jnp.sum(counts, axis=0), rows, states'),
    ]
    for old, new in changes:
        if source.count(old) != 1:
            raise RuntimeError('Unsupported trace.py version: expected evaluator fragment missing. '
                               'Use the ORIGINAL messi11_entropy_patch from this experiment.')
        source = source.replace(old, new, 1)
    compile(source, '<recording_collector>', 'exec')
    return source


def compare_rows(reference, actual, fields, np):
    if reference.ndim != 2 or actual.ndim != 2 or reference.shape[1] != len(fields):
        raise ValueError('Invalid trace matrix')
    n = min(len(reference), len(actual))
    exact = reference.shape == actual.shape and np.array_equal(reference, actual)
    close = np.isclose(reference[:n], actual[:n], rtol=1e-5, atol=1e-5, equal_nan=False)
    mismatch = np.flatnonzero(~close.all(axis=1))
    first = None
    if len(mismatch):
        i = int(mismatch[0])
        bad = np.flatnonzero(~close[i])
        first = dict(row=i, tick=int(reference[i, fields.index('tick')]),
                     fields={fields[j]: dict(reference=float(reference[i, j]), replay=float(actual[i, j]))
                             for j in bad[:12]})
    elif len(reference) != len(actual):
        first = dict(row=n, reason='trace lengths differ')
    return dict(logged_fields_exact_match=bool(exact),
                logged_fields_within_tolerance=bool(reference.shape == actual.shape and close.all()),
                tolerance=dict(rtol=1e-5, atol=1e-5), first_divergence=first,
                reference_rows=len(reference), replay_rows=len(actual),
                identity_limit='Original full player trajectories were not saved. Matching logged '
                               'fields does not establish full-state bitwise identity.')


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    a = arguments()
    # These are the settings from the original job's run_4h.sbatch.
    settings = dict(JAX_THREEFRY_PARTITIONABLE='0', JAX_DEFAULT_MATMUL_PRECISION='highest',
                    MESSI_MOVEMENT_MODE='formation', MESSI_CONTROL_PRIOR='4',
                    MESSI_CONTACT_MODE='settle', MESSI_REWARD_MODE='support',
                    MESSI_ENTROPY_MODE=a.arm, MPLBACKEND='Agg',
                    XLA_PYTHON_CLIENT_PREALLOCATE='false', OPENBLAS_NUM_THREADS='1')
    os.environ.update(settings)
    if a.platform:
        os.environ['JAX_PLATFORMS'] = a.platform
    package = a.package.resolve()
    if not (package / 'vendor/project').is_dir():
        raise FileNotFoundError(f'Original patch not found: {package}. Run from MESSI or use --package.')
    # Ensure the original patch's vendored project is imported before a repository project.
    sys.path.insert(0, str(package / 'vendor'))
    sys.path.insert(1, str(package))
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg is missing. Install it in the messi environment before running.')

    import numpy as np
    import jax
    import jax.numpy as jnp
    from project.envs.messi_11v11 import make_env, reset_env

    if a.output is None:
        a.output = ((a.experiment / f'goal_video_{a.arm}') if a.experiment else
                    (a.states.parent / 'video_python'))
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(mode='saved_states' if a.states else 'checkpoint_reevaluation',
                  arm=a.arm, settings=settings, jax_version=jax.__version__,
                  devices=[str(x) for x in jax.devices()], output=str(out))
    try:
        if a.states:
            states = pickle.loads(a.states.read_bytes())
            if isinstance(states, dict) or not hasattr(states, 'players'):
                raise ValueError('This is not a full state trajectory. evaluation/state.pkl holds '
                                 'only the final carry; use --experiment to re-evaluate.')
            states = jax.tree_util.tree_map(np.asarray, states)
            if np.asarray(states.players.position).ndim != 3:
                raise ValueError('Expected time-stacked player positions [frames,22,2].')
            env, _ = make_env(6400)
            report.update(source_states=str(a.states.resolve()),
                          source_sha256=hashlib.sha256(a.states.read_bytes()).hexdigest())
            video_name = 'saved_match.mp4'
        else:
            experiment = a.experiment.resolve()
            rel = 'ppo_False/6700' if a.arm == 'gaussian' else 'heldout_ppo_False/8701'
            arm = experiment / a.arm
            reference_dir = arm / 'evaluation' / rel
            reference = json.loads((reference_dir / 'result.json').read_text())
            config = json.loads((arm / 'configuration.json').read_text())
            ckpath = arm / 'final_train.pkl'
            seed = int(reference['seed'])
            deterministic = reference['deterministic']
            if not isinstance(deterministic, bool) or reference.get('random_policy', False):
                raise ValueError('Only learned policy evaluations with boolean deterministic flag are supported.')
            chunk = int(config['args']['chunk'])
            steps = int(reference['valid_steps'])
            report.update(seed=seed, deterministic=deterministic, chunk=chunk,
                          reference_directory=str(reference_dir), checkpoint=str(ckpath),
                          reference_score=[reference['goals_for'], reference['goals_against']])
            ck = pickle.loads(ckpath.read_bytes())
            params = jax.device_put(ck['actor_params'])
            ph = hashlib.sha256()
            for leaf in jax.tree_util.tree_leaves(params):
                ph.update(np.asarray(leaf).tobytes())
            report['actor_parameter_sha256'] = ph.hexdigest()
            if ph.hexdigest() != reference['actor_parameter_sha256']:
                raise ValueError('Checkpoint actor hash differs from the goal evaluation. '
                                 'Do not substitute latest.pkl or initial.pkl.')
            # Refuse changed policy/engine code instead of presenting another run as the original.
            import footballworld
            engine = Path(footballworld.__file__).resolve().parent
            changed = []
            for base, mapping in ((package, config.get('source_hashes', {})),
                                  (engine, config.get('environment_hashes', {}))):
                for relative, expected in mapping.items():
                    path = base / relative
                    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                        changed.append(str(path))
            report['changed_source_files'] = changed
            report['jax_version_matches_reference'] = jax.__version__ == config['versions']['jax']
            if changed:
                raise RuntimeError('Original source files changed; restore the original patch/environment '
                                   'first. See replay_report.json for the changed paths.')
            with np.load(reference_dir / 'trace.npz', allow_pickle=False) as z:
                ref_fields, ref_rows = z['fields'].tolist(), z['rows'].copy()
            from project.models.mlp_actor import MLPActorBackbone
            from project.models.action_head import HybridActionHead
            from project.algorithms.rollout_jit import initialize_jit_rollout_carry
            spec = importlib.util.spec_from_file_location('_original_messi_trace', package / 'trace.py')
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            if module.FIELDS != ref_fields:
                raise RuntimeError('Evaluator fields differ from the saved reference trace.')
            namespace = dict(vars(module))
            exec(compile(recording_collector_source(module.make_trace), '<recording_collector>', 'exec'), namespace)
            # Match original training environment construction seed, then evaluation reset seed.
            env, _ = make_env(int(config['args']['seed']))
            reset = reset_env(env, seed)
            roster = env.roster_metadata(reset.rollout)
            si = env.roster_metadata_si(reset.rollout)
            carry = initialize_jit_rollout_carry(env, reset, jax.random.key(seed + 10000))
            collectors = {}
            rows_saved, states_saved = [], [jax.tree_util.tree_map(lambda x: np.asarray(x)[None], carry.rollout.state)]
            recorded = 0
            print(f'REPLAY seed={seed} deterministic={deterministic} steps={steps}; first JIT may take time', flush=True)
            while recorded < steps and not bool(carry.done):
                length = min(chunk, steps - recorded)
                if length not in collectors:
                    collectors[length] = namespace['make_trace'](
                        env, MLPActorBackbone(hidden_dim=256), HybridActionHead(),
                        rollout_length=length, deterministic=deterministic)
                carry, counts, rows, segment = collectors[length](
                    reset.setup, carry, roster, si, params, jax.random.key(seed + 20000), jnp.bool_(True))
                rows = np.asarray(rows)
                valid = rows[:, ref_fields.index('valid')] > 0
                segment = jax.tree_util.tree_map(lambda x: np.asarray(x)[valid], segment)
                rows_saved.append(rows[valid])
                states_saved.append(segment)
                recorded += int(valid.sum())
                print(f'REPLAY {recorded}/{steps} score={np.asarray(carry.rollout.state.score).tolist()}', flush=True)
                if not valid.any():
                    break
            actual = np.concatenate(rows_saved, axis=0)
            states = jax.tree_util.tree_map(lambda *xs: np.concatenate(xs, axis=0), *states_saved)
            report.update(compare_rows(ref_rows, actual, ref_fields, np))
            tick = ref_fields.index('tick')
            gf = ref_fields.index('score_for_after')
            gb = ref_fields.index('score_for_before')
            report['reference_home_goal_transition_ticks'] = ref_rows[ref_rows[:, gf] > ref_rows[:, gb], tick].astype(int).tolist()
            report['replay_home_goal_transition_ticks'] = actual[actual[:, gf] > actual[:, gb], tick].astype(int).tolist()
            report['home_goal_ticks_match'] = (report['reference_home_goal_transition_ticks'] == report['replay_home_goal_transition_ticks'])
            np.savez_compressed(out / 'replay_trace.npz', fields=np.asarray(ref_fields), rows=actual)
            # Save ALL frames even when rendering only the goal clip.
            with (out / 'states.pkl').open('wb') as f:
                pickle.dump(states, f, protocol=pickle.HIGHEST_PROTOCOL)
            matched = report['logged_fields_exact_match']
            video_name = 'logged_trace_matched.mp4' if matched else 'new_evaluation.mp4'
            report['replay_label'] = ('Matches all originally logged fields exactly; full-state identity unverified'
                                      if matched else 'New evaluation; original logged trajectory not exactly reproduced')
            print('COMPARISON', json.dumps({k: report[k] for k in (
                'logged_fields_exact_match', 'logged_fields_within_tolerance', 'first_divergence',
                'reference_home_goal_transition_ticks', 'replay_home_goal_transition_ticks')}), flush=True)

        score = np.asarray(states.score)
        ticks = np.asarray(states.control_tick)
        if len(ticks) < 2 or not np.all(np.diff(ticks) > 0):
            raise ValueError('Need at least two strictly increasing control ticks.')
        if score.shape != (len(ticks), 2):
            raise ValueError(f'Expected scores [frames,2], found {score.shape}')
        goals = np.flatnonzero(np.diff(score[:, 0]) > 0) + 1
        dt = float(env.timebase.control_dt)
        report.update(recorded_score_start=score[0].tolist(), recorded_score_end=score[-1].tolist(),
                      recorded_frames=len(ticks), control_dt=dt,
                      home_goal_post_state_ticks=ticks[goals].astype(int).tolist(),
                      scoring_note='A HOME score increase is a team goal. It does not alone establish '
                                   'the scorer or that a policy SHOT caused the goal.')
        start, stop = 0, len(ticks)
        if not a.full and len(goals):
            goal_tick = ticks[goals[0]]
            start = int(np.searchsorted(ticks, goal_tick - a.before / dt, side='left'))
            stop = int(np.searchsorted(ticks, goal_tick + a.after / dt, side='right'))
        elif not a.full and a.experiment:
            # A failed goal replay still produces a clearly labelled diagnostic opening clip.
            stop = min(len(ticks), int(60 / dt) + 1)
            report['no_goal_reproduced'] = True
        clip = jax.tree_util.tree_map(lambda x: x[start:stop], states)
        report.update(rendered_first_tick=int(ticks[start]), rendered_last_tick=int(ticks[stop - 1]),
                      video=str(out / 'video' / video_name), render_status='pending')
        dump_json(out / 'replay_report.json', report)
        print('RENDER', report['video'], flush=True)
        from footballworld.rendering.renderer import render_mp4
        from footballworld.rendering.defaults import RenderStyle
        render_mp4(clip, out / 'video', video_name=video_name, env=env, fps=a.fps, workers=1,
                   style=RenderStyle(width_px=1280, height_px=720, dpi=100, encoder_threads=2),
                   metadata=report)
        report['render_status'] = 'complete'
        dump_json(out / 'replay_report.json', report)
        print('DONE', report['video'], flush=True)
        print('REPORT', out / 'replay_report.json', flush=True)
    except Exception:
        report['error'] = traceback.format_exc()
        dump_json(out / 'replay_report.json', report)
        raise


if __name__ == '__main__':
    main()
