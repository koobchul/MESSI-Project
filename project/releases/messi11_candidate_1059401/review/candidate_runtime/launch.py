"""Four-hour unattended comparison; failed performance thresholds do not halt it."""
import argparse,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.output)
root.mkdir(parents=True,exist_ok=False)
start=time.monotonic();failures={}
def save(name,data):(root/name).write_text(json.dumps(data,indent=2))
def call(script,args,mode,limit):
 env={**os.environ,'MESSI_CONTACT_MODE':'settle','MESSI_ENTROPY_MODE':mode}
 cmd=[sys.executable,'-u',str(HERE/script),*args];print('STAGE',mode,cmd,flush=True)
 remaining=13600-(time.monotonic()-start)
 if remaining<90:raise TimeoutError('Overall job ending; outputs remain partial')
 subprocess.run(cmd,env=env,check=True,timeout=min(limit,remaining))
try:
 expected=json.loads((HERE/'environment_expected.json').read_text())
 for name,sha in expected.items():
  path=Path('src/footballworld')/name
  if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise RuntimeError('Environment differs: '+name)
 if hashlib.sha256((HERE/'seed_checkpoint.pkl').read_bytes()).hexdigest()!='2d3436824482de1b6c2251ffa740ecc736ae3032c6cd55fb7a366bf2d3d86848':raise RuntimeError('Seed checksum mismatch')
 save('plan.json',dict(total_allocation_seconds=14400,outer_timeout_seconds=13800,per_arm_updates=2304,nominal_per_arm_steps=294912,per_arm_budget_seconds=6000,training_seed=3,selection_seeds=[6700,6701],heldout_seeds=[8700,8701],
   source='night1022059 INITIAL checkpoint, before regressed pilot; actor/critic same, optimizers reset identically',
   comparison='same settle contact controller; Gaussian-only entropy vs tanh-bounded entropy quadrature',performance_early_stop=False,
   note='Only exceptions, invalid/nonfinite execution or time limit stop a stage. No one-hour hold gate. Bounded evaluation, no fulltime claim.'))
 # Frozen policies provide controller effect separately from the PPO before/after result.
 for mode in ('gaussian',):
  try:call('audit.py',['--output',str(root/('before_'+mode)),'--steps','1024','--render'],mode,240)
  except BaseException:failures['before_'+mode]=traceback.format_exc();save('stage_failures.json',failures)
 # Fixed identical step budgets. Do not redistribute training from a failed arm.
 for mode in ('gaussian','bounded'):
  try:call('worker.py',['--output',str(root/mode),'--steps','294912','--eval-steps','2048','--budget','6000'],mode,6200)
  except BaseException:failures[mode]=traceback.format_exc();save('stage_failures.json',failures)
  ck=root/mode/'final_train.pkl'
  if not ck.exists():ck=root/mode/'latest.pkl'
  if ck.exists():
   try:call('audit.py',['--output',str(root/('after_'+mode)),'--checkpoint',str(ck),'--steps','1024','--render'],mode,240)
   except BaseException:failures['after_'+mode]=traceback.format_exc();save('stage_failures.json',failures)
 comparisons={}
 for mode in ('gaussian','bounded'):
  path=root/mode/'comparison.json'
  if path.exists():comparisons[mode]=json.loads(path.read_text())
 save('comparisons.json',comparisons)
 complete=len(comparisons)==2 and all(c['completed'] and c['nominal_training_steps']==294912 and c['updates']==2304 for c in comparisons.values())
 selected=None;scores={}
 if complete:
  for mode,c in comparisons.items():
   rows=[r for r in c['results'] if r['policy'].startswith('ppo_')]
   scores[mode]=(sum(r['goals_for']-r['goals_against'] for r in rows),sum(r['policy_shot_releases'] for r in rows),sum(r['lawful_team_possession_ticks'] for r in rows),sum(r['policy_pass_releases'] for r in rows))
  selected=max(scores,key=scores.get)
 save('selection.json',dict(selected=selected,scores=scores,basis='development 6700/6701 only; goal difference, policy actual shots, lawful possession, policy actual passes; no automatic deployment',heldout_not_used=True))
 saved_statuses={m:json.loads((root/m/'status.json').read_text()) if (root/m/'status.json').exists() else {} for m in ('gaussian','bounded')}
 all_evaluations_complete=all(s.get('completed') and s.get('heldout_complete') for s in saved_statuses.values())
 save('status.json',dict(comparison_complete=complete,all_evaluations_complete=all_evaluations_complete,all_stages_completed=not failures and all_evaluations_complete,selected=selected,policy_improvement_claimed=False,full_match_verified=False))
except BaseException:
 (root/'failure.txt').write_text(traceback.format_exc());raise
finally:
 subprocess.run([sys.executable,str(HERE/'archive.py'),str(root)],check=False)
