"""Bounded native evaluation, exact-state repeats and acquisition telemetry."""
import argparse,pickle,json,hashlib,time
from pathlib import Path
import run
import jax,jax.numpy as jnp,numpy as np
from audit_trace import make_trace
from project.envs.messi_11v11 import make_env
from project.algorithms.rollout_jit import initialize_jit_rollout_carry
from project.models.mlp_actor import MLPActorBackbone
from project.models.action_head import HybridActionHead
from project.models.field_movement import MODE,CONTACT_MODE
from project.models.role_actor import CONTROL_PRIOR
from footballworld.core.contact import INTENT_CONTROL,INTENT_PASS,INTENT_SHOT,OUTCOME_TRAP,OUTCOME_RELEASE,INTENT_SOURCE_POLICY
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--checkpoint',default=str(Path(__file__).parent/'seed_checkpoint.pkl'));p.add_argument('--steps',type=int,default=1280);p.add_argument('--render',action='store_true');a=p.parse_args()
if a.steps%128:raise ValueError('Use multiples of 128')
root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
env,reset=make_env(6400);roster=env.roster_metadata(reset.rollout);si=env.roster_metadata_si(reset.rollout)
ck=pickle.loads(Path(a.checkpoint).read_bytes());params=jax.device_put(ck['actor_params'])
collect=make_trace(env,MLPActorBackbone(hidden_dim=256),HybridActionHead(),128,deterministic=True)
traces=[];saved=[];start=time.monotonic()
for rep in range(2):
 carry=initialize_jit_rollout_carry(env,reset,jax.random.key(16400));chunks=[]
 for i in range(a.steps//128):
  carry,counts,rows,detail,states=collect(reset.setup,carry,roster,si,params,jax.random.key(26400),jnp.bool_(True))
  chunks.append(jax.tree_util.tree_map(np.asarray,detail))
  if rep==0:saved.append(jax.tree_util.tree_map(np.asarray,states))
  if bool(carry.done):raise RuntimeError('Native match terminated within bounded audit; do not count post-terminal frames')
 trace=jax.tree_util.tree_map(lambda *xs:np.concatenate(xs),*chunks);traces.append(trace)
 np.savez_compressed(root/f'trace{rep}.npz',**trace)
 print('AUDIT',MODE,CONTROL_PRIOR,'repeat',rep,'steps',a.steps,flush=True)
t=traces[0];repeat={}
for k in t:
 x,y=t[k],traces[1][k];bad=np.any((x!=y).reshape(len(x),-1),axis=1)
 repeat[k]=dict(first_bit_difference_tick=int(np.flatnonzero(bad)[0]) if bad.any() else None)
near=(t['gap']<2)&t['live']&(t['restart']==0)
held=0;longest=0;segments=0;prev=-1;hold_lengths=[]
for carrier,live,restart in zip(t['carrier'],t['live'],t['restart']):
 valid=0<carrier<11 and live and restart==0
 if not valid or carrier!=prev:
  if held:hold_lengths.append(held*.1)
  held=0
 if valid:
  if held==0:segments+=1
  held+=1;longest=max(longest,held)
 prev=carrier
if held:hold_lengths.append(held*.1)
own=(t['event_actor']>0)&(t['event_actor']<11)&(t['event_source']==INTENT_SOURCE_POLICY)
summary=dict(mode=MODE,contact_mode=CONTACT_MODE,control_prior=CONTROL_PRIOR,steps=a.steps,seed=6400,opponent_key=16400,rollout_key=26400,
 checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),repeat=repeat,
 elapsed_wall_seconds=time.monotonic()-start,field_gk_transform_max=float(t['gk_field_delta'].max()),
 raw_field_mean_move=t['raw'][:,1:11,1:3].mean(axis=(0,1)).tolist(),
 near_ticks=int(near.sum()),available_ticks=int((near&t['control_available']).sum()),chosen_ticks=int((near&t['control_available']&t['control_chosen']).sum()),
 near_relative_speed_median=float(np.median(t['relative_speed'][near])) if near.any() else None,
 policy_control_traps=int(np.sum(own&(t['event_intent']==INTENT_CONTROL)&(t['event_outcome']==OUTCOME_TRAP))),
 policy_pass_releases=int(np.sum(own&(t['event_intent']==INTENT_PASS)&(t['event_outcome']==OUTCOME_RELEASE))),
 policy_shot_releases=int(np.sum(own&(t['event_intent']==INTENT_SHOT)&(t['event_outcome']==OUTCOME_RELEASE))),
 possession_segments=segments,max_hold_seconds=longest*.1,hold_lengths=hold_lengths,
 field_goal_line_fraction=float(np.mean(np.abs(t['position'][:,1:11,0])>49.5)),
 score=np.asarray(carry.rollout.state.score).tolist(),
 snapshots={str(sec):{k:t[k][int(sec*10)].tolist() for k in ('position','velocity','ball','raw','final','direction')} for sec in (0,1,5,10,30) if sec*10<a.steps},
 limitations='2m is a diagnostic window, not the official CONTROL radius; bounded single-seed frozen policy; not PPO improvement or full match; times use control ticks including dead balls')
from funnel import summarize
summary['linked_funnel']=summarize(t)
(root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ('snapshots','repeat')}),flush=True)
states=jax.tree_util.tree_map(lambda *xs:np.concatenate(xs),*saved);(root/'states.pkl').write_bytes(pickle.dumps(states))
if a.render:
 from footballworld.rendering.renderer import render_mp4
 from footballworld.rendering.defaults import RenderStyle
 render_mp4(states,root/'video',video_name='preview.mp4',env=env,fps=10,workers=1,style=RenderStyle(width_px=1280,height_px=720,dpi=100,encoder_threads=2),metadata={'mode':MODE,'contact_mode':CONTACT_MODE,'control_prior':CONTROL_PRIOR,'frozen_checkpoint':True,'seed':6400})
