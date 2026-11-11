"""Per-tick linked nearest-player funnel, not unmatched event totals."""
import numpy as np
from footballworld.core.contact import INTENT_CONTROL,OUTCOME_TRAP,INTENT_SOURCE_POLICY

def summarize(t):
 near=(t['gap']<2)&t['live']&(t['restart']==0)&(t['nearest']>0)
 available=near&t['control_available'];selected=available&t['control_chosen']
 event=(t['event_actor']==t['nearest'][:,None])&(t['event_intent']==INTENT_CONTROL)&(t['event_source']==INTENT_SOURCE_POLICY)
 contact=selected&np.any(event,axis=1)
 trap=contact&np.any(event&(t['event_outcome']==OUTCOME_TRAP),axis=1)
 # Next pre-state is the prior transition's post-state; final tick is censored.
 after=np.r_[t['carrier'][1:],-1];afterlive=np.r_[t['live'][1:],False];afterrestart=np.r_[t['restart'][1:],-1]
 acquired=trap&(after==t['nearest'])&afterlive&(afterrestart==0)
 return dict(near_ticks=int(near.sum()),available_ticks=int(available.sum()),selected_ticks=int(selected.sum()),
   contact_ticks=int(contact.sum()),trap_ticks=int(trap.sum()),verified_next_tick=int(acquired.sum()),
   relative_speed_median_at_selection=float(np.median(t['relative_speed'][selected])) if selected.any() else None,
   note='Same nearest eligible player and same transition; last tick censored. Repeated occupied ticks are not independent trials. Deceleration failure cannot be diagnosed from speed alone.')
