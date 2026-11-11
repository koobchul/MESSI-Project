def rank(rows):
 return (sum(3*(r['goals_for']>r['goals_against'])+(r['goals_for']==r['goals_against']) for r in rows),sum(r['goals_for']-r['goals_against'] for r in rows),sum(r['goals_for'] for r in rows))
def complete(rows,seeds):
 return ([r['seed'] for r in rows]==list(seeds) and all(r['evaluation_complete'] and r['full_duration_complete'] and not r['external_cutoff'] for r in rows))
def choose(dev):
 return max(('incumbent','mid','final_train'),key=lambda n:rank(dev[n]))
