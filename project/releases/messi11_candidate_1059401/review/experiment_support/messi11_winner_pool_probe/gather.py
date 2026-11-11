import argparse,json,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
root=Path('project/experiments')/('messi11_winner_pool_probe_'+a.job)
root.mkdir(parents=True,exist_ok=True);rows=[]
for seed in (1451,1551):
 for arm in ('control','anchored'):
  f=root/f'seed_{seed}/{arm}/runtime/probe_summary.json'
  rows.append(dict(seed=seed,arm=arm,summary=json.loads(f.read_text()) if f.exists() else None))
(root/'campaign_summary.json').write_text(json.dumps(rows,indent=2))
with tarfile.open(root/'probe_results.tar.gz','w:gz') as t:
 t.add(root/'campaign_summary.json',arcname='campaign_summary.json')
 for f in sorted(root.rglob('*')):
  if f.name in ('probe_summary.json','configuration.json','status.json','failure.txt','likelihood_checks.jsonl','training.jsonl','training_matches.jsonl','validation.json'):
   if f.is_file() and f.stat().st_size<4*1024*1024:t.add(f,arcname=str(f.relative_to(root)))
print('UPLOAD',root/'probe_results.tar.gz')
