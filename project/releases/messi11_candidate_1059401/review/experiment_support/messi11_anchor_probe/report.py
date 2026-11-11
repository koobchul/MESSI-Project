"""Separate adapter releases from neural-policy and environment releases."""
import numpy as np

def shot_report(fields,rows,dt=.1):
    idx={str(f):i for i,f in enumerate(fields)}
    def g(name):return rows[:,idx[name]]
    chance=g('box_opportunity')>.5
    carrier=g('box_carrier')
    previous=np.r_[False,chance[:-1]]
    previous_carrier=np.r_[-1,carrier[:-1]]
    starts=chance&(~previous|(carrier!=previous_carrier))
    def count(name):return int(g(name).sum())
    requests=count('box_adapter_request');releases=count('box_adapter_release')
    return dict(opportunity_ticks=int(chance.sum()),opportunity_seconds=float(chance.sum()*dt),
        opportunity_episodes=int(starts.sum()),policy_shot_selected_opportunity_ticks=count('box_policy_shot_selected'),
        mean_policy_shot_probability_on_opportunity=float(g('box_policy_shot_probability')[chance].mean()) if chance.any() else None,
        adapter_requests=requests,adapter_release_ticks=releases,adapter_releases=count('box_adapter_release_events'),
        adapter_release_tick_per_request=releases/requests if requests else None,
        policy_shot_releases_all=count('box_policy_release_nonadapter'),policy_shot_releases_on_opportunity=count('box_policy_shot_release'),
        adapter_post_tick_mouth_trajectory_proxy=count('box_adapter_release_mouth_trajectory'),
        note='Opportunities/requests are correlated ticks, not independent attacks. Proxy uses post-tick velocity/height and is NOT actual shots-on-target. Full action traces retain raw and executed controls. Adapter events have engine source POLICY but are excluded from neural-policy shot counts. No scorer or causal-goal attribution.')

def augment_goals(goals,fields,rows,dt=.1):
    idx={str(f):i for i,f in enumerate(fields)}
    ticks=rows[:,idx['tick']]
    for goal in goals:
        mask=(ticks<=goal['tick'])&(ticks>=goal['tick']-round(10/dt))
        goal['adapter_shot_release_ticks_previous_10s']=ticks[mask&(rows[:,idx['box_adapter_release']]>.5)].astype(int).tolist()
        goal['note']+=' Adapter releases listed separately; temporal context is not causal credit.'
