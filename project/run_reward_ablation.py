"""Run the matched fresh v2/v3 experiment from the MESSI repository root.

train: short smoke runs first, then 16k per condition. eval: frozen ON-only
evaluation on three held-out match seeds. Resume requires --resume-existing.
"""
import argparse
from pathlib import Path
import subprocess
import sys

from project.train import reward_run_name
from project.utils.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('smoke', 'train', 'eval'), default='train')
    parser.add_argument('--output-root', default='project/experiments/reward_v3_ablation_seed0')
    parser.add_argument('--resume-existing', action='store_true')
    args = parser.parse_args()
    if not Path('project/train.py').is_file() or not Path('src/footballworld').is_dir():
        parser.error('Run from the MESSI repository root')
    output = Path(args.output_root)
    names = {v: reward_run_name(0, .05, v, .05, .05, 10.) + '_retouchguard' for v in ('v2', 'v3')}

    def execute(command):
        print('\nRUN:', ' '.join(command), flush=True)
        subprocess.run(command, check=True)

    def training(version, smoke=False):
        directory = output / ('smoke' if smoke else 'train') / version
        target = 128 if smoke else 16384
        latest = directory / 'checkpoints' / (names[version] + '_latest.pkl')
        log = directory / 'logs' / (names[version] + '.csv')
        command = [sys.executable, '-u', 'project/train.py', '--seed', '0',
            '--total-steps', str(target), '--rollout-length', '128',
            '--ppo-epochs', '1' if smoke else '4', '--reward-version', version,
            '--reward-shaping-coef', '.05', '--approach-coef', '.05', '--possession-coef', '.05',
            '--approach-distance-scale-m', '10', '--save-every', '4096',
            '--log-dir', str(directory / 'logs'), '--checkpoint-dir', str(directory / 'checkpoints')]
        if latest.exists():
            checkpoint = load_checkpoint(str(latest))
            if int(checkpoint['total_env_steps']) >= target:
                config = checkpoint.get('training_config', {})
                if config.get('reward_version') != version or not checkpoint.get('retouch_guard_enabled'):
                    raise SystemExit(f'Unexpected completed-run configuration: {latest}')
                print(f'Completed {version} {"smoke" if smoke else "training"}; skipping: {latest}', flush=True)
                return
            if not args.resume_existing:
                raise SystemExit('Partial run exists. Rerun with --resume-existing or a new --output-root')
            command += ['--resume', str(latest)]
        elif log.exists():
            raise SystemExit(f'Log exists without a recovery checkpoint: {log}. Choose a new --output-root')
        execute(command)

    if args.phase in ('smoke', 'train'):
        for version in ('v2', 'v3'):
            training(version, smoke=True)
        print('\nBoth smoke runs passed.', flush=True)
        if args.phase == 'train':
            for version in ('v2', 'v3'):
                training(version)
            print('\nMatched training finished. Next run: --phase eval', flush=True)
    else:
        for version in ('v2', 'v3'):
            checkpoint = output / 'train' / version / 'checkpoints' / (names[version] + '_step_16384.pkl')
            if not checkpoint.exists():
                raise SystemExit(f'Training checkpoint missing: {checkpoint}')
            execute([sys.executable, '-u', 'project/evaluate_retouch_guard.py',
                '--checkpoint', str(checkpoint), '--num-steps', '6400', '--rollout-length', '128',
                '--seeds', '1000', '1001', '1002', '--guard-modes', 'ON',
                '--csv', str(output / 'eval' / (version + '_seed1000_1001_1002.csv'))])
        print('\nUpload both evaluation CSVs for the matched comparison.', flush=True)


if __name__ == '__main__':
    main()
