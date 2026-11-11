import argparse,json,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
root=Path('project/experiments')/('messi11_clock_protocol_check_'+a.job);root.mkdir(parents=True,exist_ok=True)
records=[];missing=[];failures=[]
for i in range(12):
 f=root/f'shard_{i}/summary.json'
 if f.exists():records.extend(json.loads(f.read_text())['records'])
 else:
  missing.append(i);err=root/f'shard_{i}/failure.txt'
  if err.exists():failures.append(dict(shard=i,error=err.read_text()))
stats={}
for cap in (600,3600):
 stats[str(cap)]={}
 for policy in ('incumbent','ppo1024'):
  allrows=[r for r in records if r['additional_time_cap']==cap and r['policy']==policy]
  rs=[r['result'] for r in allrows if r['classification']=='complete']
  stats[str(cap)][policy]=dict(complete=len(rs),incomplete=sum(r['classification']=='incomplete' for r in allrows),failed=sum(r['classification']=='failed' for r in allrows),wins=sum(r['outcome']=='win' for r in rs),draws=sum(r['outcome']=='draw' for r in rs),losses=sum(r['outcome']=='loss' for r in rs),goals_for=sum(r['goals_for'] for r in rs),goals_against=sum(r['goals_against'] for r in rs))
d=dict(records=records,missing_shards=missing,shard_failures=failures,complete_only_by_protocol=stats,ppo_updates=0,automatic_promotion=False,note='600 is engine default cap, not confirmed tournament configuration. Incomplete matches are NOT relabeled complete or draws.')
(root/'verification.json').write_text(json.dumps(d,indent=2))
with tarfile.open(root/'verification_results.tar.gz','w:gz') as t:
 t.add(root/'verification.json',arcname='verification.json')
 for f in root.rglob('*'):
  if f.is_file() and f.name in ['failure.txt','validation.json','policy_identities.json']:t.add(f,arcname=str(f.relative_to(root)))
print('UPLOAD',root/'verification_results.tar.gz')
