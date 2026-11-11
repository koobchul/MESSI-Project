def classify(r):
 return 'complete' if (r.get('evaluation_complete') is True and r.get('full_duration_complete') is True and r.get('external_cutoff') is False and r.get('terminal_reason')=='regulation_complete' and r.get('regulation_elapsed_seconds',0)>=5400 and not r.get('terminated',False)) else 'incomplete'

def summarize(records):
 groups={}
 for phase in ['development','comparison']:
  for pol in ['original','initialized','mid','trained']:
   rr=[x['result'] for x in records if x['phase']==phase and x['policy']==pol and x['classification']=='complete']
   groups[phase+'_'+pol]=dict(matches=len(rr),wins=sum(r['goals_for']>r['goals_against'] for r in rr),draws=sum(r['goals_for']==r['goals_against'] for r in rr),goals_for=sum(r['goals_for'] for r in rr),goals_against=sum(r['goals_against'] for r in rr),policy_shots=sum(r['policy_shot_releases'] for r in rr),policy_passes=sum(r['policy_pass_releases'] for r in rr),lawful_possession_pct=100*sum(r['lawful_team_possession_ticks'] for r in rr)/max(1,sum(r['valid_steps'] for r in rr)))
 return dict(groups=groups,records=records,counts={c:sum(r['classification']==c for r in records) for c in ['complete','incomplete','failed']},all24_complete=len(records)==24 and all(r['classification']=='complete' for r in records),note='Paired opponent and seed common to original/initialized/trained; compare only common complete cases. Internal policies, not other students. No automatic promotion.')
