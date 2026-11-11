#!/usr/bin/env python3
"""One isolated 11v11 experiment: correctness gate -> PPO -> held-out evaluation."""
from evaluation_gate import complete as evaluation_complete
import os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor'))
import argparse,json,time,signal,traceback,tarfile,csv,hashlib,pickle,platform
import numpy as np
import jax
import jax.numpy as jnp
import optax
from project.envs.messi_11v11 import make_env,reset_env,ROLES,POSITIONS
from project.envs.policy_obs import encode_policy_observation
from project.envs.critic_encoder import encode_critic_state
from project.envs.reward11 import CONFIG,NAMES
from project.envs.curriculum import training_start,choose_kind
from project.models.mlp_actor import MLPActorBackbone
from project.models.action_head import HybridActionHead
from project.models.critic import CentralizedCritic
from project.algorithms.rollout_jit import make_jitted_rollout_collector,initialize_jit_rollout_carry
from project.algorithms.train_step import train_iteration_jit
from project.attack_common import atomic_pickle,completion
from trace import make_trace,FIELDS,METRICS
STOP=False

def stopped(signum,frame):
    global STOP
    STOP=True
    print('STOP requested: checkpoint and diagnostics at next chunk boundary',flush=True)
for s in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):signal.signal(s,stopped)

def json_write(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False));os.replace(tmp,path)

def finite(tree,label):
    for x in jax.tree_util.tree_leaves(tree):
        if hasattr(x,'dtype') and jax.dtypes.issubdtype(x.dtype,jax.dtypes.prng_key):continue
        x=np.asarray(x)
        if x.dtype.kind in 'fc' and not np.isfinite(x).all():raise ValueError('Nonfinite '+label)

def pack(root):
    target=root/'diagnostics.tar.gz';tmp=root/'diagnostics.tmp.tar.gz'
    with tarfile.open(tmp,'w:gz') as t:
        for f in sorted(root.rglob('*')):
            if f.is_file() and f not in (target,tmp) and not f.name.endswith('.tmp'):
                t.add(f,arcname=str(f.relative_to(root)))
    os.replace(tmp,target)

class Experiment:
    def __init__(self,args):
        self.a=args;self.root=Path(args.output).resolve()
        self.root.mkdir(parents=True,exist_ok=False) # never overwrite previous results
        self.start=time.monotonic();self.deadline=self.start+args.budget_seconds
        self.env,self.reset=make_env(args.seed);self.roster=self.env.roster_metadata(self.reset.rollout)
        self.si=self.env.roster_metadata_si(self.reset.rollout)
        teams=np.asarray(self.roster.team_id);gk=np.asarray(self.roster.is_goalkeeper)
        assert [int(sum(teams==t)) for t in (0,1)]==[11,11]
        assert [int(sum((teams==t)&gk)) for t in (0,1)]==[1,1]
        assert np.flatnonzero((teams==0)&gk).tolist()==[0]
        assert np.flatnonzero((teams==1)&gk).tolist()==[11]
        assert np.array_equal(np.flatnonzero(teams==0),np.arange(11))
        self.b=MLPActorBackbone(hidden_dim=256);self.h=HybridActionHead();self.c=CentralizedCritic(hidden_dim=256)
        obs=encode_policy_observation(self.env,self.reset.rollout,self.roster)
        co=encode_critic_state(self.env.global_state_view(self.reset.rollout),controlled_team=0)
        assert obs.shape == (22,784), obs.shape
        assert co.shape == (747,), co.shape
        keys=jax.random.split(jax.random.key(args.seed),5)
        bp=self.b.init(keys[0],obs)['params'];hp=self.h.init(keys[1],self.b.apply({'params':bp},obs))['params']
        gbp=self.b.init(keys[3],obs)['params'];ghp=self.h.init(keys[4],self.b.apply({'params':gbp},obs))['params']
        self.ap={'field':{'backbone':bp,'head':hp},'gk':{'backbone':gbp,'head':ghp}};self.cp=self.c.init(keys[2],co[None,:])['params']
        self.initial=jax.tree_util.tree_map(lambda x:x.copy(),self.ap)
        role_opt=lambda:optax.chain(optax.clip_by_global_norm(.5),optax.adam(3e-5))
        labels={role:jax.tree_util.tree_map(lambda _:role,self.ap[role]) for role in ('field','gk')}
        self.ao=optax.multi_transform({'field':role_opt(),'gk':role_opt()},labels);self.vo=optax.chain(optax.clip_by_global_norm(.5),optax.adam(1e-4))
        self.ast=self.ao.init(self.ap);self.vst=self.vo.init(self.cp)
        self.episode=0;self.steps=0;self.updates=0;self.training_fraction=0.;self.segment_steps=0;self.new_match(args.seed)
        self.collect=make_jitted_rollout_collector(self.env,self.b,self.h,self.c,args.chunk,jnp.arange(11),reward_gamma=.995)
        self.eval_collect=make_trace(self.env,self.b,self.h,args.chunk,deterministic=True)
        self.stochastic_collect=make_trace(self.env,self.b,self.h,args.chunk,deterministic=False)
        self.random_collect=make_trace(self.env,self.b,self.h,args.chunk,deterministic=True,random_policy=True)
        import footballworld
        files={str(f.relative_to(HERE)):hashlib.sha256(f.read_bytes()).hexdigest() for f in HERE.rglob('*.py')}
        env_path=Path(footballworld.__file__).resolve().parent
        env_hashes={str(f.relative_to(env_path)):hashlib.sha256(f.read_bytes()).hexdigest() for f in env_path.rglob('*.py')}
        self.audit=dict(teams=[11,11],controlled_team=0,controlled_slots=list(range(11)),learned_GK_slot=0,opponent='fixed environment rule policy',
            actor_observation_dim=int(obs.shape[-1]),critic_dim=int(co.shape[-1]),critic_outputs=2,intents=6,continuous=7,roles=list(ROLES),formation=list(POSITIONS),
            goalkeeper='independent learned actor with goal-area target map in training AND evaluation; restart exception',credit='independent GK and field actor trees and optimizer clipping; two role value heads and GAE; equal role losses',reward=CONFIG,
            seeds=dict(training=args.seed,selection=[6600,6601],final=[8600,8601]),
            devices=[str(d) for d in jax.devices()],versions=dict(python=platform.python_version(),jax=jax.__version__,optax=optax.__version__),
            env_path=str(env_path),source_hashes=files,environment_hashes=env_hashes,
            checkpoint_reuse='constructor initialization; entrypoint records actual warm-start actor/critic reuse in configuration.json',
            curriculum='training-only loose-ball starts in early budget, 1024-step segments before final phase then uninterrupted native matches; last 25 percent budget native only; all evaluation native',
            final_constraint_permission='assignment did not explicitly settle constrained-action final submission eligibility; confirm with instructor before tournament',
            fulltime_seconds=self.env.match.fulltime_seconds,minimum_players=list(self.env.match.minimum_team_players),args=vars(args))
        json_write(self.root/'configuration.json',self.audit);finite((obs,co,self.ap,self.cp),'initialization')
        self.save('initial.pkl')
        print('CONFIG',json.dumps({k:v for k,v in self.audit.items() if k not in ('source_hashes','environment_hashes')}),flush=True)

    def new_match(self,seed):
        self.training_kind=choose_kind(self.episode,self.training_fraction)
        self.reset=training_start(self.env,reset_env(self.env,seed),seed,self.training_kind)
        self.segment_steps=0
        self.carry=initialize_jit_rollout_carry(self.env,self.reset,jax.random.key(seed+10000));self.key=jax.random.key(seed+20000)

    def save(self,name):
        atomic_pickle(self.root/name,dict(kind='messi11_independent_ppo_v2',actor_params=jax.device_get(self.ap),critic_params=jax.device_get(self.cp),
          actor_opt_state=jax.device_get(self.ast),critic_opt_state=jax.device_get(self.vst),carry=jax.device_get(self.carry),
          steps=self.steps,updates=self.updates,episode=self.episode,training_kind=self.training_kind,training_fraction=self.training_fraction,rollout_key=self.key,seed=self.a.seed))

    def iteration(self):
        if bool(self.carry.done) or (self.segment_steps>=1024 and (self.training_fraction<.75 or self.training_kind!='native')):
            with (self.root/'training_matches.jsonl').open('a') as f:
                f.write(json.dumps(dict(episode=self.episode,training_kind=self.training_kind,curriculum_cutoff=not bool(self.carry.done),score=np.asarray(self.carry.rollout.state.score).tolist(),**completion(self.env,self.carry.rollout.state,bool(self.carry.done))))+'\n')
            self.episode+=1;self.new_match(1000000+self.a.seed+self.episode)
        start=time.monotonic()
        result=train_iteration_jit(env=self.env,setup=self.reset.setup,carry=self.carry,collector=self.collect,roster=self.roster,si_roster=self.si,
            backbone=self.b,action_head=self.h,critic=self.c,actor_params=self.ap,critic_params=self.cp,actor_opt_state=self.ast,critic_opt_state=self.vst,
            actor_optimizer=self.ao,critic_optimizer=self.vo,rollout_key=self.key,gamma=.995,num_epochs=4,entropy_coef=.002,target_kl=.02)
        self.ap,self.cp,self.ast,self.vst,self.carry,batch,valid,adv,returns,metrics=result
        jax.block_until_ready(self.ap)
        finite((self.ap,self.cp,self.ast,self.vst,adv,returns,batch,metrics),'PPO update')
        valid=np.asarray(valid);intent=np.asarray(batch.intent)[valid];action=np.tanh(np.asarray(batch.pre_tanh_continuous)[valid])
        assert intent.shape[1]==11 and np.all((intent>=0)&(intent<6))
        assert action.shape[-1]==7 and np.max(np.abs(action))<=1
        n=int(valid.sum());assert n>0
        assert np.asarray(batch.reward).shape==(self.a.chunk,2)
        np.testing.assert_allclose(np.asarray(batch.reward)[valid,1],np.asarray(batch.base_reward)[valid]+np.asarray(batch.reward_details)[valid,7:11].sum(-1),atol=1e-6)
        np.testing.assert_allclose(np.asarray(batch.reward)[valid,0],np.asarray(batch.base_reward)[valid]+(np.asarray(batch.reward_details)[valid,:7].sum(-1)+np.asarray(batch.reward_details)[valid,11]),atol=1e-6)
        self.steps+=n;self.segment_steps+=n;self.updates+=1
        row=dict(field_raw_action_saturation=float(np.mean(np.abs(action[:,1:11,:])>.98)),field_raw_move_saturation=float(np.mean(np.abs(action[:,1:11,:2])>.98)),field_raw_dimension_saturation=np.mean(np.abs(action[:,1:11,:])>.98,axis=(0,1)).tolist(),steps=self.steps,updates=self.updates,training_kind=self.training_kind,training_fraction=self.training_fraction,seconds=time.monotonic()-start,score=np.asarray(self.carry.rollout.state.score).tolist(),
                 base_reward=float(np.asarray(batch.base_reward)[valid].sum()),lawful_ticks=float(np.asarray(batch.lawful_possession)[valid].sum()),
                 reward_components=dict(zip(NAMES,np.asarray(batch.reward_details)[valid].sum(0).astype(float).tolist())),
                 **{k:float(v) for k,v in metrics[-1]._asdict().items()})
        with (self.root/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print('PPO',json.dumps(row),flush=True)
        return row

    def clock_gate(self):
        # Boundary fixtures use the real default 90-minute engine, no altered physics.
        # These test clock logic, not a naturally completed full trajectory.
        full,half=self.env.match.clock_ticks(self.env.timebase)
        carry=initialize_jit_rollout_carry(self.env,self.reset,jax.random.key(99))
        native=reset_env(self.env,999)
        carry=initialize_jit_rollout_carry(self.env,native,jax.random.key(99))
        st=carry.rollout.state
        st=st._replace(control_tick=jnp.int32(half),dead_ball_control_ticks=jnp.int32(0),first_half_live_extension_ticks=jnp.int32(0))
        carry=carry._replace(rollout=carry.rollout._replace(state=st))
        out,_,_=self.eval_collect(self.reset.setup,carry,self.roster,self.si,self.initial,jax.random.key(77),jnp.bool_(True))
        assert np.allclose(np.asarray(out.rollout.state.attack_direction),-np.asarray(st.attack_direction)), 'halftime direction did not flip'
        assert int(out.rollout.state.first_half_wall_end_tick)>=0
        st=out.rollout.state._replace(control_tick=jnp.int32(full),dead_ball_control_ticks=jnp.int32(0),first_half_live_extension_ticks=jnp.int32(0),penalty_completion_active=jnp.bool_(False))
        carry=out._replace(rollout=out.rollout._replace(state=st),done=jnp.bool_(False),terminated=jnp.bool_(False),truncated=jnp.bool_(False))
        done,_,_=self.eval_collect(self.reset.setup,carry,self.roster,self.si,self.initial,jax.random.key(77),jnp.bool_(True))
        status=completion(self.env,done.rollout.state,bool(done.done))
        assert bool(done.truncated) and status['full_duration_complete'], status
        return dict(halftime_direction_flip=True,fulltime_boundary=status,natural_full_match_verified=False,fixture='clock counters advanced in test state only; training reset untouched')

    def evaluate(self,params,label,seed,max_steps=0,random=False,deterministic=True):
        folder=self.root/'evaluation'/label/str(seed);folder.mkdir(parents=True,exist_ok=True)
        reset=reset_env(self.env,seed);carry=initialize_jit_rollout_carry(self.env,reset,jax.random.key(seed+10000))
        totals=np.zeros(len(METRICS),np.int64);chunks=[];collector=self.random_collect if random else (self.eval_collect if deterministic else self.stochastic_collect)
        start=time.monotonic();chunk_id=0
        while not bool(carry.done) and (not max_steps or totals[0]<max_steps):
            if STOP or time.monotonic()>self.deadline-90:break
            carry,counts,rows=collector(reset.setup,carry,self.roster,self.si,params,jax.random.key(seed+20000),jnp.bool_(True))
            counts=np.asarray(counts);rows=np.asarray(rows);rows=rows[rows[:,0]>0]
            finite((rows,carry.rollout.state),'evaluation');totals+=counts;chunks.append(rows);chunk_id+=1
            # Periodic trace and state survive failures, including an unfinished match.
            if chunk_id%16==0 or bool(carry.done) or (max_steps and totals[0]>=max_steps):
                np.savez_compressed(folder/'trace_partial.npz',fields=np.asarray(FIELDS),rows=np.concatenate(chunks))
                atomic_pickle(folder/'state.pkl',dict(carry=jax.device_get(carry),totals=totals))
                print('EVAL',label,seed,'steps',totals[0],'score',np.asarray(carry.rollout.state.score).tolist(),flush=True)
        rows=np.concatenate(chunks) if chunks else np.empty((0,len(FIELDS)))
        np.savez_compressed(folder/'trace.npz',fields=np.asarray(FIELDS),rows=rows)
        g=lambda field: rows[:,FIELDS.index(field)]
        live=g('ball_live')>0
        ph=hashlib.sha256()
        for leaf in jax.tree_util.tree_leaves(params): ph.update(np.asarray(leaf).tobytes())
        row=dict(policy=label,actor_parameter_sha256=ph.hexdigest(),seed=seed,random_policy=random,deterministic=(deterministic and not random),max_steps=max_steps,seconds=time.monotonic()-start,
                 **dict(zip(METRICS,totals.astype(int).tolist())),goals_for=int(carry.rollout.state.score[0]),goals_against=int(carry.rollout.state.score[1]),
                 external_cutoff=not bool(carry.done),terminated=bool(carry.terminated),truncated=bool(carry.truncated),**completion(self.env,carry.rollout.state,bool(carry.done)))
        row.update(gk_live_outside_pct=float(100*np.mean(g('gk_outside_own_penalty_area')[live])) if live.any() else None,
                   gk_mean_goal_distance_m=float(np.mean(g('gk_distance_to_goal_center_m'))) if len(rows) else None,
                   lawful_possession_pct=100*row['lawful_team_possession_ticks']/max(1,row['valid_steps']),
                   outcome='win' if row['goals_for']>row['goals_against'] else ('loss' if row['goals_for']<row['goals_against'] else 'draw'))
        row.update(live_seconds=float(live.sum()*self.env.timebase.control_dt),
            radius_interventions=int(g('radius_intervention').sum()),radius_restart_exceptions=int(g('radius_restart_exception').sum()),
            gk_goal_area_outside_live_pct=float(100*np.mean(((g('gk_depth_from_goal_line_m')<0)|(g('gk_depth_from_goal_line_m')>self.env.stadium.goal_area_length)|(np.abs(g('gk_y_m'))>self.env.stadium.goal_area_width/2))[live])) if live.any() else None)
        row['control_parameter_interventions']=int(g('control_parameter_interventions').sum())
        row['field_goal_line_fraction']=float(np.mean(g('field_goal_line_fraction'))) if len(rows) else None
        row['field_mean_width_m']=float(np.mean(g('field_width_m'))) if len(rows) else None
        row['field_mean_depth_m']=float(np.mean(g('field_depth_m'))) if len(rows) else None
        held=0;segments=0;longest=0;previous=-1
        for carrier,live_,restart in zip(g('verified_carrier'),g('ball_live'),g('restart_kind')):
            valid=0<carrier<11 and live_>0 and restart==0
            if valid:
                if held==0 or carrier!=previous:segments+=1;held=0
                held+=1;longest=max(longest,held)
            else:held=0
            previous=carrier
        row['field_acquisition_segments']=segments
        row['field_max_hold_seconds']=longest*float(self.env.timebase.control_dt)
        row['conceded_per_90_live_minutes']=row['goals_against']*5400/max(row['live_seconds'],1e-6)
        row['evaluation_complete']=evaluation_complete(row['valid_steps'],max_steps,row['full_duration_complete'],row['external_cutoff'])
        if row['external_cutoff']: row['outcome']='unfinished'
        elif row['terminated']: row['outcome']='abandoned'
        # Include causal pre-transition context, not only positions after goal resets.
        incidents=[]
        for i in np.flatnonzero(g('conceded_this_transition')>0):
            incidents.append(dict(tick=int(g('tick')[i]),fields=FIELDS,pre_goal_10s=rows[max(0,i-100):i+1].tolist()))
        json_write(folder/'pregoal.json',incidents);json_write(folder/'result.json',row)
        with (self.root/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print('RESULT',json.dumps(row),flush=True)
        return row

    def run(self):
        from verify import verify_invariants
        invariants=verify_invariants(self)
        initial_leaves=jax.tree_util.tree_leaves(self.initial)
        probe=[self.iteration() for _ in range(8)]
        changed=any(not np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(initial_leaves,jax.tree_util.tree_leaves(self.ap)))
        assert changed,'PPO did not change actor parameters'
        role_changed={k:any(not np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(jax.tree_util.tree_leaves(self.initial[k]),jax.tree_util.tree_leaves(self.ap[k]))) for k in ('field','gk')}
        assert all(role_changed.values()),role_changed
        clock=self.clock_gate()
        # Compile both evaluation modes before speed/budget estimate.
        self.evaluate(self.initial,'probe_initial',6102,self.a.chunk)
        self.evaluate(self.initial,'probe_random',6102,self.a.chunk,random=True)
        ev=self.evaluate(self.initial,'probe_speed',6103,4096)
        train_sps=self.a.chunk/max(r['seconds'] for r in probe[1:])
        eval_sps=ev['valid_steps']/max(ev['seconds'],1e-6)
        if ev['valid_steps'] != 4096: raise RuntimeError('Speed probe incomplete; inspect probe result')
        # 2*(45+10)min gives nominal 66k control ticks. Pending penalties can extend.
        # 1.5x time reserve, measured I/O included; not a guarantee of match duration.
        reserve=1.5*(6*66000+4*4096)/eval_sps+300
        available=self.deadline-time.monotonic()-reserve
        if STOP: raise RuntimeError('Interrupted during preflight; long training not started')
        report=dict(passed=True,invariants=invariants,role_actor_changed=role_changed,role_credit_isolation=True,actor_changed_by_PPO=changed,finite=True,action_ranges=True,team_indices=True,
                    clock=clock,train_steps_per_second=train_sps,eval_steps_per_second=eval_sps,
                    final_evaluation_reserved_seconds=reserve,remaining_training_seconds=max(0,available),
                    estimated_additional_training_steps=int(max(0,available)*train_sps),performance_gate=False)
        json_write(self.root/'validation.json',report);self.save('probe_after.pkl')
        print('VALIDATION',json.dumps(report),flush=True)
        if self.a.mode=='probe':return
        if available<600:raise RuntimeError('Measured throughput leaves <10 minutes PPO after reserving evaluation; long run not started. See validation.json.')
        # Probe updates belong to the PPO trajectory; no BC and no reset to demo weights.
        training_start_time=time.monotonic();stop_train=training_start_time+available;mid_at=training_start_time+available*.5;mid_saved=False
        while not STOP and time.monotonic()<stop_train and (not self.a.max_train_steps or self.steps<self.a.max_train_steps):
            self.training_fraction=(time.monotonic()-training_start_time)/max(available,1.)
            self.iteration()
            if self.updates%16==0:self.save('latest.pkl')
            if not mid_saved and time.monotonic()>=mid_at:self.save('mid.pkl');mid_saved=True
        self.save('latest.pkl');self.save('final_train.pkl')
        if STOP:return
        if not mid_saved:self.save('mid.pkl')
        candidates={name:jax.device_put(pickle.loads((self.root/(name+'.pkl')).read_bytes())['actor_params']) for name in ('mid','final_train')}
        scored={}
        for name,params in candidates.items():
            rows=[self.evaluate(params,'selection_'+name,s,4096) for s in (6500,6501)]
            if any(r['valid_steps']<4096 and r['external_cutoff'] for r in rows):raise RuntimeError('Selection interrupted; no selected model')
            scored[name]=(sum((r['goals_for']-r['goals_against'])*5400/max(r['live_seconds'],1.) for r in rows),sum(r['policy_shot_releases'] for r in rows),
                          sum(r['lawful_team_possession_ticks'] for r in rows),-sum(r['gk_live_outside_pct'] or 0 for r in rows))
        best=max(scored,key=lambda x:scored[x]);json_write(self.root/'selection.json',dict(scores=scored,selected=best,scope='bounded 4096-step selection, goal difference per live time first; noisy proxy, NOT full match performance',heldout_seeds=[8500,8501]))
        atomic_pickle(self.root/'selected_actor.pkl',dict(kind='messi11_independent_ppo_v2',actor_params=jax.device_get(candidates[best]),selected=best))
        final=[]
        # Pair conditions per seed; no final seed is used to select or tune a policy.
        for seed in (8500,8501):
            for label,params,random in [('random',self.initial,True),('initial',self.initial,False),('ppo',candidates[best],False)]:
                final.append(self.evaluate(params,label,seed,random=random))
        with (self.root/'comparison.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(final[0]));w.writeheader();w.writerows(final)
        complete=all(r['full_duration_complete'] and not r['external_cutoff'] for r in final)
        json_write(self.root/'final_status.json',dict(full_regulation_comparison_complete=complete,
            policy_improvement_claimed=False,reason='Judge goals/concessions/WDL on paired held-out seeds; reward increases are not evidence.',
            limitations='Two held-out seeds are a baseline, not statistical proof; added-time cap results are labeled separately.'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--mode',choices=('auto','probe'),default='auto')
    p.add_argument('--budget-seconds',type=int,default=42300);p.add_argument('--seed',type=int,default=0);p.add_argument('--chunk',type=int,default=128);p.add_argument('--max-train-steps',type=int,default=0,help='0: use measured wall-time budget; otherwise cap PPO steps, including preflight updates')
    a=p.parse_args()
    if not 0<a.budget_seconds<=42300:raise ValueError('Budget must be in (0,42300] seconds')
    if a.max_train_steps<0:raise ValueError('max-train-steps must be nonnegative')
    if a.chunk<1 or 4096%a.chunk:raise ValueError('chunk must divide 4096')
    if Path(a.output).exists():raise SystemExit('Output already exists; choose a new directory. Nothing was modified.')
    e=None
    try:
        e=Experiment(a);e.run()
    except BaseException:
        root=Path(a.output)
        if root.exists():(root/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if e is not None:
            try:e.save('last_state.pkl')
            except Exception:(e.root/'checkpoint_error.txt').write_text(traceback.format_exc())
        root=Path(a.output)
        if root.exists():pack(root)
if __name__=='__main__':main()
