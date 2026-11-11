"""One frozen-opponent evaluation shard: four policies x two seeds; no PPO."""
import argparse,dataclasses,hashlib,json,os,pickle,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--train-root',type=Path,required=True);p.add_argument('--opponent',type=int,choices=[1,2,3],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--pair',type=int,choices=[0,1],required=True);p.add_argument('--cap',type=int,choices=[600,3600],required=True);a=p.parse_args()
 root=a.output;root.mkdir(parents=True,exist_ok=False);records=[]
 try:
  os.environ['MESSI_POOL_MANIFEST']=str((a.train_root/'opponent_manifest.json').resolve())
  os.environ['MESSI_POOL_MODE']='mixed_pool'
  base=(a.train_root/'isolated_source').resolve();sys.path.insert(0,str(base));sys.path.insert(0,str(base/'vendor'))
  import jax,numpy as np
  from flax.core import unfreeze
  sys.path.insert(0,str(HERE))
  from reward_scale import install
  config=json.loads((a.train_root/'runtime/configuration.json').read_text())
  os.environ['MESSI_APPROACH_SCALE']=str(config['approach_output_scale'])
  install(.1,1.)
  import run
  from collector_contract import load_trace,checked
  from recovery_classify import classify
  from project.envs.messi_11v11 import reset_env
  from report import shot_report
  import learned_pool
  learned_pool.set_eval(a.opponent)
  e=run.Experiment(argparse.Namespace(output=str(root/'runtime'),seed=81,chunk=128,budget_seconds=13500,mode='probe',max_train_steps=0))
  (e.root/'initial.pkl').unlink()
  e.env=dataclasses.replace(e.env,match=dataclasses.replace(e.env.match,maximum_added_time_seconds_per_period=float(a.cap)))
  e.reset=reset_env(e.env,81);e.roster=e.env.roster_metadata(e.reset.rollout);e.si=e.env.roster_metadata_si(e.reset.rollout)
  trace=load_trace(HERE/'audit_trace.py');run.FIELDS=trace.FIELDS;run.METRICS=trace.METRICS
  e.stochastic_collect=checked(trace.make_trace(e.env,e.b,e.h,128,deterministic=False))
  e.eval_collect=checked(trace.make_trace(e.env,e.b,e.h,128,deterministic=True))
  learned_pool.set_eval(None)
  paths=dict(incumbent=a.train_root/'runtime/initial.pkl',ppo1024=Path('project/experiments/messi11_winner_pool_probe_1057366/seed_1451/control/runtime/final_train.pkl'))
  params={};hashes={};identities={}
  for name,path in paths.items():
   hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest();ck=pickle.loads(path.read_bytes())
   if name in ('ppo1024','kl512') and ck.get('updates')!={'ppo1024':1024,'kl512':512}[name]:raise RuntimeError('Wrong snapshot update count: '+name)
   params[name]=jax.device_put(unfreeze(ck['actor_params']));run.finite(params[name],name)
   h=hashlib.sha256()
   for leaf in jax.tree_util.tree_leaves(params[name]):h.update(np.asarray(leaf).tobytes())
   identities[name]=dict(path=str(path.resolve()),checkpoint_sha256=hashes[name],actor_sha256=h.hexdigest())
  run.json_write(root/'policy_identities.json',identities)
  # Cache is reused for cold/warm probes AND all eight full matches.
  e.evaluate(params['incumbent'],'speed_cold',191590,4096,deterministic=False)
  warm=e.evaluate(params['incumbent'],'speed_warm',191591,4096,deterministic=False)
  sps=warm['valid_steps']/max(warm['seconds'],1e-9)
  cap_ticks=2*(e.env.match.halftime_seconds+e.env.match.maximum_added_time_seconds_per_period)/e.env.timebase.control_dt
  remaining=e.deadline-time.monotonic();predicted=1.25*4*cap_ticks/sps+600
  gate=dict(passed=warm['evaluation_complete'] and predicted<remaining,warm_ticks_per_second=sps,per_match_cap_ticks=cap_ticks,predicted_eight_match_seconds=predicted,remaining_seconds=remaining,cached_collector=True,ppo_updates=0)
  run.json_write(root/'validation.json',gate);print('EVAL_VALIDATION',json.dumps(gate),flush=True)
  if not gate['passed']:raise RuntimeError('Evaluation shard runtime budget insufficient; no completion criteria relaxed')
  # Same opponent/seed pairing as original experiment, split by opponent.
  seeds=[s for s in range(361600,361612) if 1+s%3==a.opponent]
  phases=[('verification',s) for s in seeds[2*a.pair:2*a.pair+2]]
  for phase,seed in phases:
   for name,ap in params.items():
    record=dict(phase=phase,policy=name,seed=seed,opponent_pool_id=a.opponent,additional_time_cap=a.cap,classification='failed')
    try:
     r=e.evaluate(ap,phase+'_'+name,seed,0,deterministic=False)
     folder=e.root/'evaluation'/(phase+'_'+name)/str(seed)
     with np.load(folder/'trace.npz') as z:r['shot_diagnostics']=shot_report(z['fields'],z['rows'],float(e.env.timebase.control_dt))
     assert r['shot_diagnostics']['adapter_requests']==0
     record.update(classification=classify(r),result=r)
    except Exception:record['error']=traceback.format_exc()
    records.append(record);run.json_write(root/'records.json',records)
    run.json_write(root/'summary.json',dict(opponent_pool_id=a.opponent,records=records,counts={c:sum(x['classification']==c for x in records) for c in ['complete','incomplete','failed']},all4_complete=len(records)==4 and all(x['classification']=='complete' for x in records),ppo_updates=0))
  for name,path in paths.items():
   if hashlib.sha256(path.read_bytes()).hexdigest()!=hashes[name]:raise RuntimeError('Source checkpoint changed')
 except BaseException:
  (root/'failure.txt').write_text(traceback.format_exc());raise
 finally:
  import subprocess
  subprocess.run([sys.executable,str(HERE/'archive.py'),str(root)],check=False)
if __name__=='__main__':main()
