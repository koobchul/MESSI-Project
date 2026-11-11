# Matched reward v2/v3 pilot

Install after retouch_guard_patch. Run from the MESSI repository root.

## Install and run

1. Copy `reward_v3_patch.tar.gz` to the MESSI root.
2. Install: `tar -xzf reward_v3_patch.tar.gz && python reward_v3_patch/install.py`
3. Train, including automatic smoke checks: `JAX_THREEFRY_PARTITIONABLE=0 PYTHONPATH=src:. python -u project/run_reward_ablation.py --phase train`
4. Evaluate after both training runs finish: `JAX_THREEFRY_PARTITIONABLE=0 PYTHONPATH=src:. python -u project/run_reward_ablation.py --phase eval`
5. Send these files back:
   - `project/experiments/reward_v3_ablation_seed0/eval/v2_seed1000_1001_1002.csv`
   - `project/experiments/reward_v3_ablation_seed0/eval/v3_seed1000_1001_1002.csv`
   - Training logs under `project/experiments/reward_v3_ablation_seed0/train/{v2,v3}/logs/` if acquisition remains near zero.

The training command runs v2 then v3 smoke checks (128 steps, one PPO epoch),
followed by two independent fresh 16,384-step runs (128-step rollouts, four
PPO epochs). Both start at seed 0 with identical model initialization, learning
rates, gamma, opponent and guard ON. Smoke weights are not used to initialize
training. The historical 8k checkpoint is not resumed. Progress coefficient is
0.05 in both conditions. This is a short pilot, not a final competitive policy.

If Slurm stops a run, acquire a compute node, activate messi, and run:
`JAX_THREEFRY_PARTITIONABLE=0 PYTHONPATH=src:. python -u project/run_reward_ablation.py --phase train --resume-existing`
Completed conditions are skipped, partial conditions resume from their latest
saved checkpoint (every 4096 steps); unsaved steps must run again. A failure
before the first checkpoint leaves a log without a checkpoint. In that case
select a new `--output-root`, rather than deleting or overwriting that log.
No timing estimate is guaranteed; evaluation can run in a separate allocation.

## Reward definition and limits

v2 retains the existing implementation: goal +1/-1 and coefficient 0.05 times
`gamma * ball_progress_next - ball_progress_current`, suppressing progress
shaping on goal transitions exactly as before.

v3 adds:

- Approach potential `A = exp(-d/10 m)`, where `d` is the SI distance from the
  ball to the nearest active, legally eligible outfield teammate. Adds
  `0.05 * (gamma * A_next - A_current)`.
- Possession potential `P = 1` only in live open play with a physically
  verified carrier on the controlled team; otherwise zero. Adds
  `0.05 * (gamma * P_next - P_current)`.

Possession uses the engine's strict `verified_controlled_carrier`, including
active actor, team, positive control ticks, carry radius and control height.
The fresh-trap grace flag, last-contact TRAP alone, and possession identity
outside the physical envelope do not qualify. A set-piece taker awaiting
another player's touch is excluded from both potentials. Goalkeepers can earn
possession but not the approach incentive. Additional chasers do not multiply
approach reward. Staying in possession has no positive per-tick bonus; this is
an acquisition/loss potential difference, not an incentive to hold forever.

The added signals apply only when both endpoints are live open play, excluding
goal transitions and restart/half-time placement boundaries. A true terminal
uses next potential zero; truncation retains bootstrapping. These phase gates
are heuristics; no claim of optimal-policy invariance is made. The original
v2 progress signal can still be affected by placement transitions. No extra
retouch penalty is added, so the pilot changes only these two signals.

The guard still allows passive contacts during MOVE. This patch does not change
environment rules, observations, action dimensions, or existing weights.
`--reward-version v2` remains the default for direct train.py use. Coefficient
zero disables only the corresponding component; v3 with progress zero may
still have approach/possession shaping.

## Diagnostics and reproducibility

CSV training logs append approach shaping, possession shaping and lawful
possession ticks. Tick counts sample the pre-action state at control frequency,
not every physics substep. Configuration JSON files and checkpoint metadata
record arguments. v3 resumes reject reward/gamma configuration changes and
historical checkpoints lacking a v3 configuration. Run names include v3
coefficients and distance scale, and fresh training refuses to overwrite logs.

Evaluation supports `--guard-modes ON` (default remains OFF and ON), preserving
old stored-possession counters while adding strict lawful possession counters.
The scripted 6400-step evaluations use match seeds 1000/1001/1002 with identical
RNG addressing across checkpoints. `external_cutoff=True` means a partial
match, so scores are diagnostic counts, not full-match wins. The next decision
is based on strict possession, retouches and actual goals/conceded, then longer
training/full-match evaluation if the pilot improves.

## Local validation

13 CPU unit tests passed: original five guard/PPO tests and eight reward tests
covering SI distance, approach direction, no chaser multiplier, keeper handling,
strict possession, prohibited takers, restart/goal teleports, acquisition/loss,
terminal handling and coefficient-zero compatibility. Actual short MESSI
training with rollout/PPO/checkpoint saving was exercised; this does not predict
GPU runtime or learning performance. Tests can be run on the server with:
`JAX_THREEFRY_PARTITIONABLE=0 PYTHONPATH=src:. python -m unittest discover -s project/tests -v`

Installer verifies expected source hashes before writing, backs up changes
under `reward_v3_backups/`, and refuses a changed source tree. No files are
written on a hash mismatch. Installing twice is safe.
