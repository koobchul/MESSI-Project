# Restart retouch guard

## Purpose and scope

Seed 1001 at pre-step ticks 1342 and 2992: team0's set-piece taker made a CONTROL/TRAP before another player touched the released ball. The environment awarded team1 an indirect free kick, cleared possession, and repositioned players. Lean/eventful endpoint comparisons passed. Those two events do not establish a general trap-retention bug.

This patch masks explicit non-MOVE requests only when the player's own observation says it is the untouched set-piece taker in live open play. It does not add geometry masks, change continuous actions, change checkpoint architecture, or change environment rules/rewards. Eligibility is conservative at the policy boundary: another player's touch later in the same frame could otherwise make a subsequent touch legal. MOVE may still cause passive contact, so zero retouch violations is not guaranteed.

## Install

From the MESSI repository root, with the downloaded archive in that directory:

```bash
tar -xzf retouch_guard_patch.tar.gz && python retouch_guard_patch/install.py
```

The installer checks original source hashes before writing and backs up every replaced file under `retouch_guard_backups/<timestamp>/`. It refuses to overwrite changed known source files. Existing checkpoints are untouched.

## First experiment: frozen 8k checkpoint

```bash
JAX_THREEFRY_PARTITIONABLE=0 PYTHONPATH=src:. python -u project/evaluate_retouch_guard.py --checkpoint project/checkpoints/mlp_mappo_shaped_v2_c0p05_seed_0_step_8192.pkl --num-steps 6400 --rollout-length 128 --seeds 1001
```

Runs OFF then ON from the same seed, with identical RNG addressing. Exact substep event collection runs throughout evaluation, so compilation/runtime can differ from the lean diagnostic. Progress prints every ten chunks. The CSV is saved to `project/eval_logs/retouch_guard_8k_seed1001.csv`. Change `--csv` for later experiments to preserve this comparison.

Compare team0 retouch events, non-MOVE requests in prohibited contexts, team possession ticks, physical CONTROL/TRAPs, their same-substep-retouch subset, and goals for/against. Retouch events are credited to the offender team (event.team stores the team awarded the restart). TRAP counts are physical outcomes, not successful lawful possessions. A physical trap without same-substep retouch is still not proof of retained possession. Score/possession summaries are bounded to the requested horizon, and `external_cutoff`, `terminated`, and `truncated` are stored separately. A 6400-step score is not necessarily a completed match score or a win.

Do not claim improved playing strength from masking two events. Extend the comparison to multiple seeds and full matches if the initial result supports it.

## PPO consistency

Sampling and stored rollout log-probability use the guarded distribution. Each Transition records `retouch_guard_enabled`. PPO recomputes both log-probability and entropy using that same flag, including the post-update KL check. Old/default Transition objects are unguarded. Flags are false for padding. Weights and architecture stay compatible with old checkpoints.

The JIT and Python collectors default to guarded operation. `train.py --disable-retouch-guard` selects the historical baseline. Guarded training filenames append `_retouchguard`, and checkpoints record the guard mode. Resuming an unguarded checkpoint in guarded mode is a changed-policy experiment, not an exact continuation; the training script prints this distinction. Start a fresh run or use distinct paths when changing mode.

The two historical `diagnose_trap_*.py` scripts explicitly disable the guard to keep their original traces reproducible. For guard evaluation, use `evaluate_retouch_guard.py`. Other custom inference loops must apply `apply_retouch_guard()` before sampling if they are meant to deploy the guarded policy.

## Validation

Five CPU JAX tests passed: 5v5/11v11 layouts and release conditions; sampling/finite entropy gradients; actual MESSI encoder agreement; PPO ratio/KL consistency with mixed guard flags; JIT evaluator counters and OFF/ON switching using a controlled transition fixture. Syntax checked for all modified sources.

```bash
JAX_THREEFRY_PARTITIONABLE=0 PYTHONPATH=src:. python -m unittest discover -s project/tests -p test_retouch_guard.py -v
```

Validation used temporary local test dependencies; no server dependency changes are required. No real checkpoint or complete new guarded match was available locally. Real rollout performance and event reduction must be verified with the server experiment above.
