"""Instrumentation of unchanged PPO: check stored raw actions, not executed controls."""
import json

def install_check(run,root):
 import jax,numpy as np
 from project.models.role_actor import actor_forward
 from project.models.intent_guard import apply_retouch_guard
 from project.models.action_sampling import action_log_prob
 original=run.train_iteration_jit
 count=0
 def wrapped(**kw):
  nonlocal count
  result=original(**kw);count+=1
  if count<=8 or count%128==0:
   batch=result[5];valid=np.asarray(result[6])
   output=actor_forward(kw['actor_params'],kw['backbone'],kw['action_head'],batch.actor_obs)
   output=apply_retouch_guard(output,batch.actor_obs,enabled=batch.retouch_guard_enabled)
   recomputed=np.asarray(action_log_prob(output,batch.intent,batch.pre_tanh_continuous))
   old=np.asarray(batch.log_prob)
   error=float(np.max(np.abs(recomputed[valid]-old[valid])))
   row=dict(update=count,valid_ticks=int(valid.sum()),preupdate_raw_logprob_max_abs_error=error,tolerance=.002,passed=bool(np.isfinite(error) and error<=.002))
   with (root/'likelihood_checks.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   print('LIKELIHOOD_CHECK',json.dumps(row),flush=True)
   if not row['passed']:raise RuntimeError('Stored-action likelihood mismatch; checkpoint retained, no further training')
  return result
 run.train_iteration_jit=wrapped
