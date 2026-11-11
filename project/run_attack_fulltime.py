"""GPU preflight -> teacher BC -> MAPPO -> full-match validation -> held-out test.

Re-run the SAME command to resume. No 6,400-step performance gate.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='project/experiments/attack_fulltime_seed0')
    p.add_argument('--phase', choices=['all', 'bc', 'train', 'eval'], default='all')
    p.add_argument('--total-steps', type=int, default=131072)
    p.add_argument('--demo-steps', type=int, default=8192)
    p.add_argument('--bc-updates', type=int, default=3000)
    p.add_argument('--allow-cpu', action='store_true', help='Explicit local testing only')
    a = p.parse_args()
    if min(a.total_steps, a.demo_steps, a.bc_updates) <= 0: p.error('Budgets must be positive')
    if not Path('src/footballworld').is_dir(): p.error('Run from MESSI root')
    # Set backend requirement BEFORE importing JAX. Never quietly train on gate1 CPU.
    if not a.allow_cpu: os.environ['JAX_PLATFORMS'] = 'cuda'
    os.environ.setdefault('JAX_THREEFRY_PARTITIONABLE', '0')
    import jax
    print('Devices:', jax.devices(), flush=True)
    if not a.allow_cpu and not any(d.platform == 'gpu' for d in jax.devices()):
        raise SystemExit('No GPU. Allocate a Slurm GPU node and rerun.')
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out / 'run.lock').open('w')
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit('Another pipeline is using this output directory.')
    # Preserve diagnostic evidence even when a subprocess exits with an error.
    import atexit
    import tarfile
    def bundle_results():
        with tarfile.open(out/'results_bundle.tar.gz', 'w:gz') as tar:
            for pattern in ('*.json', '*.log', '**/results.csv', 'ppo/logs/*.csv'):
                for path in sorted(out.glob(pattern)):
                    tar.add(path, arcname=str(path.relative_to(out)))
    atexit.register(bundle_results)
    config = dict(total_steps=a.total_steps, demo_steps=a.demo_steps, bc_updates=a.bc_updates)
    identity = out / 'run_config.json'
    if identity.exists() and json.loads(identity.read_text()) != config:
        raise ValueError('Run budgets changed; choose a new --output directory')
    identity.write_text(json.dumps(config, indent=2))
    def run(script, arguments, name):
        cmd = [sys.executable, '-u', 'project/' + script] + list(map(str, arguments))
        print('\nRUN:', ' '.join(cmd), flush=True)
        with (out / (name + '.log')).open('a') as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                print(line, end='', flush=True); log.write(line); log.flush()
            if proc.wait(): raise SystemExit(f'{name} failed. See {out / (name + ".log")}')
    bc = out / 'bc' / 'actor_best.pkl'
    if a.phase in ('all', 'bc'):
        run('warmstart_attack.py', ['--output', out/'bc', '--demo-steps', a.demo_steps,
                                   '--updates', a.bc_updates], 'bc')
    if a.phase == 'bc': return
    if not bc.exists(): raise SystemExit('BC actor missing. Run --phase bc or --phase all.')
    name = 'mlp_mappo_shaped_v3_c0p05_a0p05_p0p05_d10_seed_0_retouchguard'
    latest = out / 'ppo' / 'checkpoints' / (name + '_latest.pkl')
    if a.phase in ('all', 'train'):
        args = ['--total-steps', a.total_steps, '--rollout-length', 256,
                '--reward-version', 'v3', '--reward-shaping-coef', .05,
                '--approach-coef', .05, '--possession-coef', .05,
                '--actor-lr', 1e-5, '--critic-lr', 1e-4, '--entropy-coef', .001,
                '--ppo-epochs', 4, '--target-kl', .01, '--save-every', 8192,
                '--log-dir', out/'ppo'/'logs', '--checkpoint-dir', out/'ppo'/'checkpoints']
        if latest.exists(): args += ['--resume', latest]
        else: args += ['--actor-init', bc]
        finished = False
        if latest.exists():
            with latest.open('rb') as f: finished = int(pickle.load(f)['total_env_steps']) >= a.total_steps
        if not finished: run('train_attack.py', args, 'ppo')
        else: print('PPO target already reached; preserving checkpoint bytes.', flush=True)
    if a.phase == 'train': return
    if not latest.exists(): raise SystemExit('PPO checkpoint missing')
    candidates = {'bc': bc, 'bc_mappo': latest}
    baseline = Path('project/experiments/reward_v3_ablation_seed0/train/v3/checkpoints') / (name + '_step_16384.pkl')
    if baseline.exists(): candidates['old_v3'] = baseline
    rows = []
    for tag, checkpoint in candidates.items():
        folder = out / 'validation' / tag
        run('evaluate_fulltime.py', ['--checkpoint', checkpoint, '--output', folder,
                                    '--seeds', 1000, '--deterministic'], 'validation_'+tag)
        with (folder/'results.csv').open() as f: row = next(csv.DictReader(f))
        row['candidate'] = tag; rows.append(row)
    # A validation selection, not an assertion of globally best performance.
    def rank(r):
        return (int(r['goals_for'])-int(r['goals_against']), int(r['goals_for']),
                float(r['lawful_possession_pct']), int(r['shot_releases']))
    selected = max(rows, key=rank)['candidate']
    selection = dict(selected=selected, validation_seed=1000, test_seeds=[1001,1002],
                     criterion='goal difference, goals, lawful possession, shot releases', validation=rows,
                     checkpoint=str(candidates[selected]),
                     checkpoint_sha256=hashlib.sha256(candidates[selected].read_bytes()).hexdigest(),
                     limitation='One validation seed; BC uses bundled rule-policy demonstrations; not pure RL.')
    (out/'selection.json').write_text(json.dumps(selection, indent=2))
    run('evaluate_fulltime.py', ['--checkpoint', candidates[selected], '--output', out/'test',
                                '--seeds', 1001, 1002, '--deterministic'], 'test')
    # Single uploadable evidence bundle, excluding large model/data snapshots.
    bundle_results()
    print('\nFinished. Selected on validation:', selected, flush=True)
    print('Send:', out/'results_bundle.tar.gz', flush=True)


if __name__ == '__main__': main()
