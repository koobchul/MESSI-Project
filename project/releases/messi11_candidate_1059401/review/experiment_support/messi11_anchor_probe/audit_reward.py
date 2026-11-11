"""Read existing traces only; test whether shot credit lacked prior normal possession."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('project/experiments/messi11_eval_recovery_1051819'));p.add_argument('--output',type=Path,default=Path('shot_credit_audit.json'));a=p.parse_args()
records=[]
for task in (1,7,9):
 for f in sorted((a.root/f'task_{task}'/'matches/comparison/trained').glob('*/runtime/evaluation/*/*/trace.npz')):
  with np.load(f) as z:
   fields=list(z['fields']);rows=z['rows'];idx={s:i for i,s in enumerate(fields)}
   pay=rows[:,idx['reward_actual_useful_shot']]>0
   normal=rows[:,idx['field_possession_before']]>.5
   records.append(dict(trace=str(f),shot_credit_positive_ticks=int(pay.sum()),shot_credit_without_normal_prepossession_ticks=int((pay&~normal).sum()),normal_prepossession_ticks=int(normal.sum()),ticks=len(rows)))
a.output.write_text(json.dumps(dict(records=records,note='Read-only recorded evaluation rewards, not PPO advantages. Recovery logged common0.02, training arm may have used0.10. Tests eligibility issue, does not prove gradient causality.'),indent=2));print(a.output,'traces',len(records))
