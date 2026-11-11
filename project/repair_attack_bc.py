"""Fix MOVE-body-aim supervision, balance replay, and collect both teacher teams.

Runs BC only. Independent offline and closed-loop gates precede full-match eval.
No teacher actions are used in student evaluation.
"""
import argparse,json,pickle,hashlib
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
import optax
from footballworld.policies.rule_based import make_rule_based_policy
from project.algorithms.rollout_jit import initialize_jit_rollout_carry
from project.envs.messi_5v5 import make_5v5_env,reset_5v5_env
from project.envs.obs_encoder import encode_mlp_observation
from project.models.intent_guard import retouch_prohibited_from_mlp
from project.models.mlp_actor import MLPActorBackbone
from project.models.action_head import HybridActionHead
from project.attack_common import atomic_pickle
from project.attack_repair_core import own_team_first,replay_pools,balanced_indices,repair_loss,diagnostics,coverage_ok,offline_gate


def make_collector(env,length=128):
    teacher=make_rule_based_policy(env)
    @jax.jit
    def collect(setup,carry,roster,si):
        n=roster.team_id.shape[0]
        def step(c,_):
            def advance(_):
                raw=encode_mlp_observation(env.observe_all(c.rollout),roster)
                t=teacher.step(env.observe_all_si(c.rollout),si,c.rule_state,c.opponent_key)
                action=t.action._replace(intent=jnp.where(retouch_prohibited_from_mlp(raw),0,t.action.intent))
                obs=own_team_first(raw,roster.team_id)
                result=env.step(c.rollout,setup,action,jax.random.fold_in(c.opponent_key,c.rollout.state.control_tick))
                nxt=c._replace(rollout=result.rollout,rule_state=t.state,terminated=result.terminated,truncated=result.truncated,done=result.done)
                return nxt,(obs,action.intent,action.as_continuous_array(),c.rollout.state.players.active)
            def pause(_):
                return c,(jnp.zeros((n,n*30+114)),jnp.zeros(n,jnp.int32),jnp.zeros((n,7)),jnp.zeros(n,jnp.bool_))
            return jax.lax.cond(c.done,pause,advance,None)
        return jax.lax.scan(step,carry,None,length=length)
    return collect


def collect_seed(env,collect,seed,steps,path):
    if path.exists():
        with path.open('rb') as f: return pickle.load(f)
    reset=reset_5v5_env(env,seed)
    roster=env.roster_metadata(reset.rollout);si=env.roster_metadata_si(reset.rollout)
    carry=initialize_jit_rollout_carry(env,reset,jax.random.key(seed+70000))
    partial=path.with_suffix('.partial.pkl');chunks=[];done=0
    if partial.exists():
        with partial.open('rb') as f: old=pickle.load(f)
        carry,chunks,done=jax.device_put(old['carry']),old['chunks'],old['steps']
    while done<steps and not bool(carry.done):
        carry,batch=collect(reset.setup,carry,roster,si)
        x,y,a,v=jax.device_get(batch);v=np.array(v,copy=True);v[min(128,steps-done):]=False
        chunks.append((x[v],y[v],a[v]));done+=min(128,steps-done)
        if done%1024==0 or done>=steps or bool(carry.done):
            atomic_pickle(partial,dict(carry=jax.device_get(carry),chunks=chunks,steps=done))
            print(f'demo seed={seed} steps={done}/{steps} score={np.asarray(carry.rollout.state.score).tolist()}',flush=True)
    data=tuple(np.concatenate([c[i] for c in chunks]) for i in range(3))
    atomic_pickle(path,data);partial.unlink(missing_ok=True)
    print('demo counts',seed,np.bincount(data[1],minlength=6).tolist(),flush=True)
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='project/experiments/attack_fulltime_seed0/bc')
    p.add_argument('--output',default='project/experiments/attack_repair_seed0')
    p.add_argument('--demo-steps',type=int,default=16384)
    p.add_argument('--validation-steps',type=int,default=8192)
    p.add_argument('--updates',type=int,default=3000)
    p.add_argument('--batch-size',type=int,default=512)
    a=p.parse_args()
    if min(a.demo_steps,a.validation_steps,a.updates,a.batch_size)<=0:p.error('Positive budgets required')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);source=Path(a.source)
    basebytes=(source/'actor_best.pkl').read_bytes();base=pickle.loads(basebytes)
    config=dict(vars(a),base_sha256=hashlib.sha256(basebytes).hexdigest(),train_seeds=[20,21],validation_seed=22)
    identity=out/'fit_config.json'
    if identity.exists() and json.loads(identity.read_text())!=config:raise ValueError('Config changed; use new output')
    identity.write_text(json.dumps(config,indent=2))
    if (out/'fit_complete.json').exists():
        print('Fit already complete:',out/'fit_complete.json');return
    env,_=make_5v5_env(20);collect=make_collector(env)
    training=[]
    for seed in (10,11,12,13):
        path=source/f'demo_seed{seed}.pkl'
        if path.exists():
            with path.open('rb') as f:training.append(pickle.load(f))
    for seed in (20,21):training.append(collect_seed(env,collect,seed,a.demo_steps,out/f'demo_seed{seed}.pkl'))
    val=collect_seed(env,collect,22,a.validation_steps,out/'demo_seed22.pkl')
    data=tuple(np.concatenate([d[i] for d in training]) for i in range(3))
    coverage=dict(train_counts=np.bincount(data[1],minlength=6).tolist(),validation_counts=np.bincount(val[1],minlength=6).tolist(),
                  passed=coverage_ok(data,val),note='Raw frame labels, not unique successful skills; heuristic minimum counts.')
    (out/'coverage.json').write_text(json.dumps(coverage,indent=2))
    print('COVERAGE',coverage,flush=True)
    # Even if rare shots are insufficient, fix the confirmed MOVE bug. Promotion stays blocked.
    backbone,head=MLPActorBackbone(hidden_dim=256),HybridActionHead()
    params=jax.device_put(base['actor_params']);before=diagnostics(params,val,backbone,head)
    tx=optax.chain(optax.clip_by_global_norm(1),optax.adam(1e-4));opt=tx.init(params)
    rng=np.random.default_rng(0);start=0;best=float('inf');statepath=out/'fit_state.pkl'
    if statepath.exists():
        with statepath.open('rb') as f:s=pickle.load(f)
        params,opt=jax.device_put(s['params']),jax.device_put(s['opt']);start,best=s['update'],s['best'];rng.bit_generator.state=s['rng']
    pools=replay_pools(data[0],data[1])
    @jax.jit
    def update(params,opt,x,y,target):
        value,grad=jax.value_and_grad(repair_loss)(params,x,y,target,backbone,head)
        updates,opt=tx.update(grad,opt,params)
        return optax.apply_updates(params,updates),opt,value
    for u in range(start,a.updates):
        idx=balanced_indices(rng,pools,a.batch_size)
        params,opt,loss=update(params,opt,*(jnp.asarray(d[idx]) for d in data))
        if (u+1)%250==0 or u+1==a.updates:
            report=diagnostics(params,val,backbone,head)
            score=report['near_move_rmse']+report['move_body_aim_rmse']+.25*report['near_opposite_fraction']+.25*report['near_body_aim_opposite_fraction']+.5*(1.-np.mean(report['recall'][1:4]))
            if not np.isfinite(score) or not np.isfinite(float(loss)):raise ValueError('Nonfinite fit')
            passed=coverage['passed'] and offline_gate(before,report)
            # Prefer any passing candidate over a failing one; then diagnostic score.
            objective=score if passed else score+10.
            if objective<best:
                best=objective
                atomic_pickle(out/'actor_best.pkl',dict(kind='attack_bc_v1',actor_params=jax.device_get(params),repair='move_body_aim_v2',update=u+1,config=config,validation=report,offline_gate_passed=passed))
                (out/'diagnostics.json').write_text(json.dumps(dict(before=before,after=report,coverage=coverage,offline_gate_passed=passed),indent=2))
            atomic_pickle(statepath,dict(params=jax.device_get(params),opt=jax.device_get(opt),update=u+1,best=best,rng=rng.bit_generator.state))
            print(f'update={u+1} loss={float(loss):.4f} near_rmse={report["near_move_rmse"]:.4f} body_rmse={report["move_body_aim_rmse"]:.4f} pass={passed}',flush=True)
    (out/'fit_complete.json').write_text(json.dumps(dict(updates=a.updates,best_score=best)))
    print('Saved',out/'diagnostics.json',flush=True)


if __name__=='__main__':main()
