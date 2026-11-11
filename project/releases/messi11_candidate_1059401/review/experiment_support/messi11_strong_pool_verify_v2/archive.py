import tarfile,sys,json,os
from pathlib import Path
r=Path(sys.argv[1]);r.mkdir(parents=True,exist_ok=True)
names=['likelihood_source_change.json','runtime/likelihood_checks.jsonl','runtime/training_matches.jsonl','records.json','policy_identities.json','validation.json','training_status.json','summary.json','failure.txt','opponent_manifest.json','source_changes.json','runtime/status.json','runtime/summary.json','runtime/configuration.json','runtime/validation.json','runtime/pool_checks.json','runtime/initialization.json','runtime/collector_contract.json','runtime/reward_scale_checks.json','runtime/failure.txt','runtime/evaluation_records.json']
with tarfile.open(r/'summary_light.tmp.tar.gz','w:gz') as t:
 size=0
 for n in names:
  p=r/n
  if p.is_file() and size+p.stat().st_size<8*1024**2:t.add(p,arcname=n);size+=p.stat().st_size
os.replace(r/'summary_light.tmp.tar.gz',r/'summary_light.tar.gz');print('UPLOAD',r/'summary_light.tar.gz')
