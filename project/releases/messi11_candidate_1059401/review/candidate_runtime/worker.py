"""Fixed-step same-initial-state comparison; no performance-based early exit."""
import argparse,hashlib,json,pickle,time,traceback
from pathlib import Path
import run
import jax,jax.numpy as jnp,numpy as np
from run import Experiment,json_write,pack
from project.envs.messi_11v11 import reset_env
from project.envs.curriculum import training_start
from project.algorithms.rollout_jit import initialize_jit_rollout_carry
from project.models.field_movement import CONTACT_MODE
from project.models.action_sampling import ENTROPY_MODE
class Comparison(Experiment):
 def new_match(self,seed):
  cycle=('acquire','advance','native','acquire','finish','native','gk','acquire')
  self.training_kind='native' if self.training_fraction>=.75 else cycle[self.episode%len(cycle)]
  self.reset=training_start(self.env,reset_env(self.env,seed),seed,self.training_kind);self.segment_steps=0
  self.carry=initialize_jit_rollout_carry(self.env,self.reset,jax.random.key(seed+10000));self.key=jax.random.key(seed+20000)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=294912);p.add_argument('--eval-steps',type=int,default=2048);p.add_argument('--budget',type=int,default=6000);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 args=argparse.Namespace(output=a.output,seed=3,chunk=128,budget_seconds=a.budget,mode='probe',max_train_steps=a.steps)
 if a.steps%128:raise ValueError('Nominal steps must be a multiple of 128')
 target_updates=a.steps//128
 e=Comparison(args);source=Path(__file__).parent/'seed_checkpoint.pkl';ck=pickle.loads(source.read_bytes())
 e.ap=jax.device_put(ck['actor_params']);e.cp=jax.device_put(ck['critic_params'])
 # Equal optimizer restart for the two changed execution conditions.
 e.ast=e.ao.init(e.ap);e.vst=e.vo.init(e.cp);e.initial=jax.tree_util.tree_map(lambda x:x.copy(),e.ap);e.save('initial.pkl')
 e.audit.update(entropy_mode=ENTROPY_MODE,contact_mode=CONTACT_MODE,control_parameter_map='raw' if CONTACT_MODE=='raw' else 'rule CONTROL parameter base +0.05 raw residual only if both request CONTROL',
   checkpoint_reuse='night1022059 initial actor/critic (before failed pilot); same optimizer reset for both arms',source_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
   curriculum='explicit acquire/advance/native/acquire/finish/native/gk/acquire; final quarter native; same step schedule for both arms',
   seeds=dict(training=3,selection=[6700,6701],final=[8700,8701]),matmul_precision='highest',
   attribution='Formation is rule movement + PPO residual. Settling is an additional rule contact parameter map, not a learned PPO improvement. Intent/PASS/SHOT/GK/AWAY/restarts unaffected.')
 json_write(e.root/'configuration.json',e.audit)
 try:
  from movement_checks import check_movement
  json_write(e.root/'movement_checks.json',check_movement(e.env,e.reset))
  rows=[]
  def evaluate(label,params,seeds,steps):
   return [e.evaluate(params,label+'_'+str(det),s,steps,deterministic=det) for s in seeds for det in (True,False)]
  seeds=(6700,) if a.smoke else (6700,6701)
  rows+=evaluate('initial',e.initial,seeds,a.eval_steps)
  while e.updates<target_updates and not run.STOP:
   e.training_fraction=e.updates/target_updates;e.iteration()
   if e.updates%32==0:e.save('latest.pkl')
   # Stop only for computation budget or execution/finite checks, never low hold scores.
   if time.monotonic()>e.deadline-(30 if a.smoke else 1600):raise TimeoutError('Execution budget exhausted; partial checkpoint saved, comparison not complete')
  if run.STOP:raise InterruptedError('Stop requested')
  e.save('final_train.pkl');rows+=evaluate('ppo',e.ap,seeds,a.eval_steps)
  comparison=dict(completed=e.updates==target_updates,additional_training_steps=e.steps,nominal_training_steps=a.steps,updates=e.updates,contact_mode=CONTACT_MODE,entropy_mode=ENTROPY_MODE,results=rows,training_budget='same number of PPO updates; valid steps exclude terminal padding and are reported separately')
  comparison['completed']=comparison['completed'] and all(r['evaluation_complete'] for r in rows)
  json_write(e.root/'comparison.json',comparison)
  if not comparison['completed']:raise RuntimeError('Training/development evaluation incomplete')
  if not a.smoke:
   # Heldout seeds are evaluated only after training and are never used for arm selection.
   held=[]
   for label,params in [('initial',e.initial),('ppo',e.ap)]:held+=evaluate('heldout_'+label,params,(8700,8701),4096)
   json_write(e.root/'heldout.json',held)
   if not all(r['evaluation_complete'] for r in held):
    json_write(e.root/'status.json',dict(completed=False,heldout_complete=False,reason='Bounded evaluation stopped before requested tick count'))
    raise RuntimeError('Heldout evaluation incomplete; must not mark complete')
  json_write(e.root/'status.json',dict(completed=True,heldout_complete=not a.smoke,full_match_verified=False,scope='bounded native evaluation, no score threshold; no reward-based performance claim'))
 except BaseException:(e.root/'failure.txt').write_text(traceback.format_exc());raise
 finally:e.save('latest.pkl');pack(e.root)
if __name__=='__main__':main()
