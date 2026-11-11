#!/usr/bin/env python3
"""PPO from the six-win curriculum incumbent: unchanged vs weaker field possession potential."""
import os
import argparse,hashlib,importlib.util,json,pickle,shutil,sys,time,traceback
from pathlib import Path
HERE = Path(__file__).resolve().parent


from selection import rank


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=['rule_only','mixed_pool'],required=True)
    p.add_argument('--train-seed',type=int,required=True);p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--updates',type=int,default=4096);p.add_argument('--budget',type=int,default=19500)
    a=p.parse_args()
    if a.updates<8 or a.updates%2: p.error('updates must be an even integer >=8')
    sys.path.insert(0,str(a.base.resolve()));sys.path.insert(0,str(a.base.resolve()/'vendor'))
    import jax,numpy as np,optax
    from reward_scale import install,scale_values
    factor=.1
    install(factor,1.)
    import run
    from run import json_write,finite
    spec=importlib.util.spec_from_file_location('_old_equal_budget_worker',a.base/'worker.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    from project.envs.messi_11v11 import reset_env
    from project.algorithms.rollout_jit import initialize_jit_rollout_carry
    import dataclasses
    native_make_env=run.make_env
    def make_env(seed):
        env,_=native_make_env(seed)
        env=dataclasses.replace(env,match=dataclasses.replace(env.match,maximum_added_time_seconds_per_period=3600.))
        return env,reset_env(env,seed)
    run.make_env=make_env
    def native_match(self,seed):
        self.training_kind='native';self.segment_steps=0
        self.reset=reset_env(self.env,seed)
        self.carry=initialize_jit_rollout_carry(self.env,self.reset,jax.random.key(seed+10000));self.key=jax.random.key(seed+20000)
    old.Comparison.new_match=native_match
    e=None
    try:
        e=old.Comparison(argparse.Namespace(output=str(a.output),seed=a.train_seed,chunk=128,
            budget_seconds=a.budget,mode='probe',max_train_steps=128*a.updates))
        ck=pickle.loads(a.source.read_bytes())
        if ck.get('updates')!=4096:raise RuntimeError('Incumbent checkpoint must be original curriculum weak candidate at4096 additional updates')
        e.ap=jax.device_put(ck['actor_params']);e.cp=jax.device_put(ck['critic_params'])
        e.ast=jax.device_put(ck['actor_opt_state']);e.vst=jax.device_put(ck['critic_opt_state'])
        h=hashlib.sha256()
        for x in jax.tree_util.tree_leaves(e.ap):h.update(np.asarray(x).tobytes())
        if h.hexdigest()!='526324f5b56d48134adb1b6f6389dde0d94ccceeab3febfe3bacb9fad6e4cf00':
            raise RuntimeError('Wrong starting policy: use messi11_possession_balance_1040542/weak_acquire/final_train.pkl')
        e.training_fraction=1.
        source_actor=jax.tree_util.tree_map(lambda x:x.copy(),e.ap)
        from initialization import shot_initialization
        e.ap=shot_initialization(e.ap,2.)
        for x,y in zip(jax.tree_util.tree_leaves(source_actor['gk']),jax.tree_util.tree_leaves(e.ap['gk'])):np.testing.assert_array_equal(np.asarray(x),np.asarray(y))
        lr=1e-5
        # Identical Adam state schema, labels, clipping and moment history.
        # Actor and critic optimizers stay identical across reward conditions.
        role_opt=lambda:optax.chain(optax.clip_by_global_norm(.5),optax.adam(lr))
        labels={r:jax.tree_util.tree_map(lambda _:r,e.ap[r]) for r in ('field','gk')}
        e.ao=optax.multi_transform({'field':role_opt(),'gk':role_opt()},labels)
        e.ast=e.ao.init(e.ap);e.vst=e.vo.init(e.cp)
        e.initial=jax.tree_util.tree_map(lambda x:x.copy(),e.ap)
        finite((e.ap,e.cp,e.ast,e.vst),'source')
        e.save('initial.pkl')
        # Read-only extended evaluation diagnostics; no action adapter enabled.
        sys.path.insert(0,str(HERE))
        from collector_contract import load_trace,checked
        trace=load_trace(HERE/'audit_trace.py')
        json_write(e.root/'collector_contract.json',dict(module_file=str(Path(trace.__file__).resolve()),expected_returns=3,base_audit_trace_not_used=True))
        import report as shot_report_module
        run.FIELDS=trace.FIELDS;run.METRICS=trace.METRICS
        e.eval_collect=checked(trace.make_trace(e.env,e.b,e.h,128,deterministic=True))
        e.stochastic_collect=checked(trace.make_trace(e.env,e.b,e.h,128,deterministic=False))
        from cache import CollectorCache
        def build_collector(fixed):
            import learned_pool
            learned_pool.set_eval(fixed)
            try:return checked(trace.make_trace(e.env,e.b,e.h,128,deterministic=False))
            finally:learned_pool.set_eval(None)
        collector_cache=CollectorCache(build_collector)
        original_evaluate=e.evaluate
        def evaluate_with_shot_audit(*args,**kwargs):
            import learned_pool
            seed=args[2]
            fixed=1+(seed%3)
            e.stochastic_collect=collector_cache.get(fixed)
            r=original_evaluate(*args,**kwargs)
            r['opponent_pool_id']=fixed
            learned_pool.set_eval(None)
            folder=e.root/'evaluation'/r['policy']/str(r['seed'])
            with np.load(folder/'trace.npz') as data:
                r['shot_diagnostics']=shot_report_module.shot_report(data['fields'],data['rows'],float(e.env.timebase.control_dt))
            assert r['shot_diagnostics']['adapter_requests']==0
            json_write(folder/'result.json',r)
            return r
        e.evaluate=evaluate_with_shot_audit
        # Recompute stored raw-action log probability with the PRE-update actor.
        # Diagnostic only: it does not change gradients, rewards or actions.
        from likelihood_check import install_check
        install_check(run,e.root)
        from anchor import install_anchor
        install_anchor(e,float(os.environ['MESSI_ANCHOR_COEF']))
        source_h=hashlib.sha256();init_h=hashlib.sha256()
        for leaf in jax.tree_util.tree_leaves(source_actor):source_h.update(np.asarray(leaf).tobytes())
        for leaf in jax.tree_util.tree_leaves(e.initial):init_h.update(np.asarray(leaf).tobytes())
        json_write(e.root/'initialization.json',dict(original_actor_sha256=source_h.hexdigest(),initialized_actor_sha256=init_h.hexdigest(),field_SHOT_bias_boost=2.,optimizer_reset=True,forced_SHOT=False))

        dev=(281600,281601,281602);final=()
        e.audit.update(arm=('control' if os.environ['MESSI_ANCHOR_COEF']=='0' else 'anchored'),anchor_coefficient=float(os.environ['MESSI_ANCHOR_COEF']),approach_output_scale=float(os.environ['MESSI_APPROACH_SCALE']),actor_lr=lr,critic_lr=1e-4,shot_credit_requires_verified_carrier=True,field_possession_potential_scale=factor,effective_field_acquire=.2*factor,effective_useful_shot=.02,source_checkpoint=str(a.source),
            source_actor_sha256=h.hexdigest(),source_file_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
            single_change='Only reference-policy KL penalty coefficient differs: 0 vs0.02; both approach output0.1, stable likelihood, same initialized weights and1024 PPO updates',
            checkpoint_reuse='Original weak5263 actor/critic; fieldSHOTbias+2 in both arms; Adam reset; original/initialized/trained evaluated separately',
            seeds=dict(training=a.train_seed,selection=list(dev),final=list(final)),
            training_budget=dict(updates=a.updates,nominal_ticks=128*a.updates),
            curriculum='Native full matches only; reset only at done; match additional-time ceiling3600seconds per half',
            selection='Development only: original/initialized/mid512/final1024 on three new paired seeds. No final-test or automatic promotion',
            evaluation='Nine development matches, three policies x three seeds211600..211602; internal frozen opponents, strict5400seconds; no final test',
            reproducibility='same-process repeat checked; cross-process path identity is not a stop condition',
            attribution='formation/settling rules active for all policies; goals are team scores, not automatically policy-shot goals')
        json_write(e.root/'configuration.json',e.audit)
        json_write(e.root/'status.json',dict(stage='preflight',completed=False))
        from pool_checks import check_pool
        json_write(e.root/'pool_checks.json',check_pool(e))
        from movement_checks import check_movement
        json_write(e.root/'movement_checks.json',check_movement(e.env,e.reset))
        json_write(e.root/'clock_checks.json',e.clock_gate())
        import jax.numpy as jnp
        for sign in (1.,-1.):
            fixture=(jnp.arange(12,dtype=jnp.float32)+1)*sign
            shot_factor=1.
            actual=np.asarray(jax.jit(scale_values)(fixture,factor,shot_factor))
            np.testing.assert_array_equal(np.delete(actual,[1,4]),np.delete(np.asarray(fixture),[1,4]))
            np.testing.assert_allclose(actual[1],float(fixture[1])*factor,rtol=1e-6)
            np.testing.assert_allclose(actual[4],float(fixture[4])*shot_factor,rtol=1e-6)
        json_write(e.root/'reward_scale_checks.json',dict(passed=True,scale=factor,
            positive_negative_scaled_equally=True,other_10_components_unchanged=True,scaled_components=[1,4],shot_component_factor=shot_factor,effective_field_acquire=.2*factor,
            base_reward_untouched=True))
        # A warm-up then two SAME-PROCESS frozen-policy repeats; no optimizer updates.
        e.evaluate(e.initial,'warmup',181590,1024,deterministic=False)
        speed=[]
        for tag in ('repeat_A','repeat_B'):
            speed.append(e.evaluate(e.initial,tag,181591,4096,deterministic=False))
        with np.load(e.root/'evaluation/repeat_A/181591/trace.npz') as x,np.load(e.root/'evaluation/repeat_B/181591/trace.npz') as y:
            identical=bool(np.array_equal(x['fields'],y['fields']) and np.array_equal(x['rows'],y['rows']))
        if not identical:raise RuntimeError('Same-process frozen-policy repeat failed; long training not started')
        if not all(r['evaluation_complete'] for r in speed):raise RuntimeError('Speed probe incomplete')
        eval_sps=speed[-1]['valid_steps']/max(speed[-1]['seconds'],1e-9)
        probe=[e.iteration() for _ in range(8)]
        changed={r:any(not np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(
            jax.tree_util.tree_leaves(e.initial[r]),jax.tree_util.tree_leaves(e.ap[r]))) for r in ('field','gk')}
        if not all(changed.values()):raise RuntimeError('Actor update missing in a role')
        # Reserve ALL full development and final matches before continuing.
        # Full matches run in separate evaluation jobs; keep a save/exit reserve.
        reserve=900.+1.25*12*126000/eval_sps
        predicted_train=(a.updates-8)*max(r['seconds'] for r in probe[1:])
        remaining=e.deadline-time.monotonic()
        report=dict(passed=predicted_train+reserve<remaining,
                    same_process_trace_equal=identical,role_actor_changed=changed,
                    actor_lr=lr,eval_ticks_per_second=eval_sps,
                    predicted_training_seconds=predicted_train,reserved_evaluation_save_seconds=reserve,full_evaluation_in_separate_jobs=False,probe_seconds=[r['seconds'] for r in speed],
                    remaining_seconds=remaining,performance_gate=False)
        json_write(e.root/'validation.json',report);print('VALIDATION',json.dumps(report),flush=True)
        if not report['passed']:raise RuntimeError('Measured runtime exceeds reserved budget; inspect validation.json')
        if a.preflight_only:
            json_write(e.root/'status.json',dict(stage='preflight_passed',updates=e.updates,passed=True,long_training_started=False))
            return
        json_write(e.root/'status.json',dict(stage='PPO',completed=False,updates_target=a.updates))
        while e.updates<a.updates:
            if run.STOP:raise InterruptedError('Stop requested; latest retained')
            if time.monotonic()>e.deadline-reserve:raise TimeoutError('Training consumed reserved full-evaluation budget; latest preserved')
            e.training_fraction=1.
            e.iteration()
            if e.updates%128==0:e.save('latest.pkl')
            if e.updates==a.updates//2:e.save('mid.pkl')
        e.save('final_train.pkl')
        e.audit['evaluation_budget']='9 full development games; incomplete and failed retained; no cap changes'
        json_write(e.root/'configuration.json',e.audit)
        mid=pickle.loads((e.root/'mid.pkl').read_bytes())['actor_params']
        records=[]
        from recovery_classify import classify
        for seed in dev:
            for name,params in [('original',source_actor),('initialized',e.initial),('mid',jax.device_put(mid)),('trained',e.ap)]:
                record=dict(seed=seed,policy=name,opponent_pool_id=1+seed%3,classification='failed')
                try:
                    if time.monotonic()>e.deadline-120:raise TimeoutError('Evaluation budget exhausted')
                    r=e.evaluate(params,'development_'+name,seed,0,deterministic=False)
                    record.update(classification=classify(r),result=r)
                except Exception:record['error']=traceback.format_exc()
                records.append(record)
                json_write(e.root/'evaluation_records.json',records)
        json_write(e.root/'probe_summary.json',dict(records=records,counts={c:sum(r['classification']==c for r in records) for c in ['complete','incomplete','failed']},ppo_updates=e.updates,automatic_promotion=False))

        json_write(e.root/'status.json',dict(stage='training_finished',training_complete=True,updates=e.updates,
            full_evaluation_in_separate_jobs=False,policy_improvement_claimed=False))
    except BaseException:
        if e is not None:(e.root/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if e is not None:
            try:e.save('latest.pkl')
            except BaseException:(e.root/'checkpoint_error.txt').write_text(traceback.format_exc())

if __name__=='__main__':main()
