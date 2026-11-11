"""Create isolated source-only runtime; never modify installed environment/base."""
import hashlib,json,shutil
from pathlib import Path

def prepare(root):
 previous=Path('project/experiments/messi11_possession_balance_1040542')
 cfg=json.loads((previous/'weak_acquire/configuration.json').read_text())
 base=Path('messi11_entropy_patch');envbase=Path(cfg['env_path'])
 if not envbase.exists():envbase=Path('src/footballworld')
 for folder,mapping in [(base,cfg['source_hashes']),(envbase,cfg['environment_hashes'])]:
  for rel,h in mapping.items():
   p=folder/rel
   if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise RuntimeError('Source changed: '+str(p))
 isolated=root/'isolated_source';isolated.mkdir()
 for rel in cfg['source_hashes']:
  p=isolated/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/rel,p)
 p=isolated/'vendor/project/algorithms/rollout_jit.py';s=p.read_text()
 needle='''    rule_policy = make_rule_based_policy(
        env
    )'''
 if s.count(needle)!=1:raise RuntimeError('Collector factory source not recognized')
 s=s.replace(needle,needle+'\n    from learned_pool import make_controller\n    pool_action=make_controller(env,backbone,action_head)')
 needle='''                opponent_action = (
                    rule_step.action
                )'''
 if s.count(needle)!=1:raise RuntimeError('Opponent action source not recognized')
 s=s.replace(needle,needle+'\n                opponent_action=pool_action(rollout,roster,opponent_action,carry.opponent_key,rollout.state.control_tick.astype(jnp.uint32))')
 p.write_text(s)
 # Stable exact tanh Jacobian in the isolated copy, shared by collector/update.
 q=isolated/'vendor/project/models/action_sampling.py';v=q.read_text()
 old="""    squashed = jnp.tanh(
        pre_tanh_continuous
    )

    correction = jnp.log(
        1.0 - squashed**2 + 1e-6
    )"""
 new="""    # log(sech(z)^2), stable for saturated actions; no epsilon density floor.
    correction = 2.0 * (jnp.log(2.0) - pre_tanh_continuous
                        - jax.nn.softplus(-2.0 * pre_tanh_continuous))"""
 if v.count(old)!=1:raise RuntimeError('Unrecognized tanh logprob source; stop without touching original')
 q.write_text(v.replace(old,new))
 (root/'likelihood_source_change.json').write_text(json.dumps(dict(file=str(q),before_sha256=hashlib.sha256(v.encode()).hexdigest(),after_sha256=hashlib.sha256(q.read_bytes()).hexdigest(),formula='2*(log2-z-softplus(-2z))',original_environment_untouched=True),indent=2))
 entries=[]
 for name,path in [('original',previous/'weak_acquire/final_train.pkl'),('verified_ppo1024',Path('project/experiments/messi11_anchor_probe_1057075/seed_1351/control/runtime/final_train.pkl')),('verified_kl512',Path('project/experiments/messi11_anchor_probe_1057075/seed_1251/anchored/runtime/mid.pkl'))]:
  if not path.is_file():raise FileNotFoundError(path)
  entries.append(dict(name=name,checkpoint=str(path.resolve()),file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),origin='our own frozen historical policy, NOT another student'))
 manifest=root/'opponent_manifest.json';manifest.write_text(json.dumps(dict(opponents=entries,training_mix='rule25percent + each learned25percent',homogeneous_schema='our role MLP784, 6intents+7continuous',student_opponents_available=False),indent=2))
 (root/'source_changes.json').write_text(json.dumps(dict(original_source_hashes=cfg['source_hashes'],changed_file=str(p),isolated_source=str(isolated),environment_source_modified=False,modified_sha256=hashlib.sha256(p.read_bytes()).hexdigest()),indent=2))
 return isolated.resolve(),manifest.resolve()
