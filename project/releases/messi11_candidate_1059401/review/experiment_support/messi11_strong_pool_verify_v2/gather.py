import argparse,json,tarfile,collections
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
root=Path('project/experiments')/('messi11_strong_pool_verify_v2_'+a.job);root.mkdir(parents=True,exist_ok=True)
records=[];missing=[];shard_failures=[]
for i in range(6):
 f=root/f'shard_{i}/summary.json'
 if f.exists():records.extend(json.loads(f.read_text())['records'])
 else:
  missing.append(i)
  failure=root/f'shard_{i}/failure.txt'
  if failure.exists():shard_failures.append(dict(shard=i,error=failure.read_text()))
stats={}
for name in ['original','incumbent','ppo1024','kl512']:
 rs=[r['result'] for r in records if r['policy']==name and r['classification']=='complete']
 stats[name]=dict(complete=len(rs),wins=sum(r['outcome']=='win' for r in rs),draws=sum(r['outcome']=='draw' for r in rs),losses=sum(r['outcome']=='loss' for r in rs),goals_for=sum(r['goals_for'] for r in rs),goals_against=sum(r['goals_against'] for r in rs))
d=dict(shard_failures=shard_failures,evaluation_started_shards=6-len(missing),records=records,missing_shards=missing,counts={c:sum(r['classification']==c for r in records) for c in ['complete','incomplete','failed']},complete_only_statistics=stats,automatic_promotion=False,ppo_updates=0)
(root/'verification.json').write_text(json.dumps(d,indent=2))
with tarfile.open(root/'verification_results.tar.gz','w:gz') as t:
 t.add(root/'verification.json',arcname='verification.json')
 for f in root.rglob('*'):
  if f.is_file() and f.name in ['failure.txt','validation.json','policy_identities.json']:t.add(f,arcname=str(f.relative_to(root)))
print('UPLOAD',root/'verification_results.tar.gz')
