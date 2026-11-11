"""v2 role reward: SI state and actual policy contact events only.
Heuristic gated shaping, NOT a policy-invariance claim. No request reward.
"""
import os
import jax.numpy as jnp
from footballworld.core.contact import INTENT_SHOT, INTENT_PASS, OUTCOME_RELEASE, OUTCOME_CATCH, INTENT_SOURCE_POLICY
from footballworld.dynamics.contact_predicates import verified_controlled_carrier
from project.envs.reward_v3 import control_potentials, extra_control_shaping, open_play, prohibited_taker_mask
NAMES=('field_approach','field_acquire','field_hold','field_progress','actual_useful_shot',
       'actual_forward_pass','field_crowding','gk_return','gk_local_approach','gk_acquire','gk_hold','field_support_cover')
CONFIG=dict(gamma=.995,field_approach=.1,field_acquire=.2,field_hold_per_second=.003,
 field_forward_per_pitch=.2,policy_shot_release=.02,policy_forward_pass_release=.002,
 crowd_pair_per_second=.0002,gk_distance_per_m_second=.0005,gk_outside_per_second=.002,
 gk_approach=.05,gk_acquire=.2,gk_policy_catch=.2,gk_hold_per_second=.002,
 gk_ball_max_speed=12.,gk_ball_max_height=1.2,shot_min_x=20.,shot_max_abs_y=25.)

MODE=os.environ.get('MESSI_REWARD_MODE','support')
assert MODE in ('baseline','context','support')
CONFIG.update(mode=MODE,context_approach=.1,support_cover=.02,position_scale_m=20.,target_frozen_per_transition=True)

def carrier(env,s):
    return verified_controlled_carrier(s,env.ball,env.reach,env.action_scale)

def possession(env,s,gk=False):
    c=carrier(env,s);i=jnp.clip(c,0,21)
    return (c>=0)&(s.players.team_id[i]==0)&(s.players.is_goalkeeper[i]==gk)&~prohibited_taker_mask(s)[i]&open_play(s)

def gk_local(env,s):
    i=jnp.argmax((s.players.team_id==0)&s.players.is_goalkeeper)
    depth=s.ball.position[0]*s.attack_direction[0]+env.stadium.half_length
    eligible=(open_play(s)&s.players.active[i]&(depth>=0)&(depth<=env.stadium.goal_area_length)
      &(jnp.abs(s.ball.position[1])<=env.stadium.goal_width/2-.5)
      &(s.ball.position[2]<=CONFIG['gk_ball_max_height'])
      &(jnp.linalg.norm(s.ball.velocity)<=CONFIG['gk_ball_max_speed']))
    potential=jnp.exp(-jnp.linalg.norm(s.players.position[i]-s.ball.position[:2])/3.)
    return eligible,potential

def reward11(env,state,result,gamma=.995):
    nxt=result.rollout.state;dt=env.timebase.control_dt
    valid=open_play(state)&open_play(nxt)&(state.attack_direction[0]==nxt.attack_direction[0])&(result.reward[0]==0)
    a,_=extra_control_shaping(state,nxt,result.terminated,result.reward[0],gamma,.1,0.,0,10.,env.ball,env.reach,env.action_scale)
    a=jnp.where(valid,a,0.)
    b=possession(env,state);n=possession(env,nxt)
    # Loss of verified possession pays back its acquisition potential.
    acquire=jnp.where(valid,.2*(gamma*n.astype(jnp.float32)-b.astype(jnp.float32)),0.)
    hold=jnp.where(valid&b&n,.003*dt,0.)
    dx=(nxt.ball.position[0]-state.ball.position[0])*state.attack_direction[0]
    progress=jnp.where(valid&b&n,.2*jnp.clip(dx,-2,2)/(2*env.stadium.half_length),0.)
    c=result.events.deliberate_contact;idx=jnp.clip(c.actor,0,21)
    release=(c.actor>=0)&(state.players.team_id[idx]==0)&~state.players.is_goalkeeper[idx]&(c.outcome==OUTCOME_RELEASE)&(c.intent_source==INTENT_SOURCE_POLICY)
    # Event-source filter excludes automatic releases. Velocity is a post-tick proxy,
    # not a shots-on-target label; collision within the tick may change it.
    bx=state.ball.position[0]*state.attack_direction[0];by=state.ball.position[1]
    vx=nxt.ball.velocity[0]*state.attack_direction[0];vy=nxt.ball.velocity[1]*state.attack_direction[0]
    y_at_goal=by*state.attack_direction[0]+vy*(env.stadium.half_length-bx)/jnp.maximum(vx,.01)
    aimed=(vx>1.)&(jnp.abs(y_at_goal)<env.stadium.goal_width/2+1.)
    shot=jnp.where(open_play(state)&(bx>20)&(jnp.abs(by)<25)&aimed&jnp.any(release&(c.intent==INTENT_SHOT)),.02,0.)
    passed=jnp.where(open_play(state)&b&(vx>1)&jnp.any(release&(c.intent==INTENT_PASS)),.002,0.)
    p=state.players;eligible=(p.team_id==0)&~p.is_goalkeeper&p.active
    delta=p.position[:,None,:]-p.position[None,:,:]
    close=jnp.sum(delta*delta,axis=-1)<4.
    near=jnp.linalg.norm(p.position-state.ball.position[:2],axis=-1)<8.
    pairs=close&eligible[:,None]&eligible[None,:]&near[:,None]&near[None,:]&jnp.triu(jnp.ones((22,22),dtype=bool),1)
    crowd=jnp.where(open_play(state),-.0002*dt*jnp.minimum(jnp.sum(pairs),10),0.)
    gk=jnp.argmax((p.team_id==0)&p.is_goalkeeper);pos=p.position[gk]
    depth=pos[0]*state.attack_direction[0]+env.stadium.half_length
    distance=jnp.sqrt(depth**2+pos[1]**2)
    outside=(depth<0)|(depth>env.stadium.penalty_area_length)|(jnp.abs(pos[1])>env.stadium.penalty_area_width/2)
    ret=jnp.where(open_play(state)&p.active[gk],-dt*(.002*outside+.0005*jnp.maximum(distance-5.,0.)),0.)
    eb,pb=gk_local(env,state);en,pn=gk_local(env,nxt)
    ga=jnp.where(valid&eb&en,.05*(gamma*pn-pb),0.)
    gb=possession(env,state,True);gn=possession(env,nxt,True)
    caught=jnp.any((c.actor==gk)&(c.outcome==OUTCOME_CATCH)&(c.intent_source==INTENT_SOURCE_POLICY))
    # Catch opens RK_GK_HOLD; do not incorrectly gate the catch out merely
    # because the next state is a lawful goalkeeper-hold restart.
    gc=jnp.where(open_play(state)&caught,.2,jnp.where(valid,.2*(gamma*gn.astype(jnp.float32)-gb.astype(jnp.float32)),0.))
    gh=jnp.where(valid&gb&gn,.002*dt,0.)
    from project.envs.tactical_reward import contextual_terms
    approach, support = contextual_terms(env,state,nxt,valid)
    if MODE != 'baseline': a=approach
    if MODE != 'support': support=jnp.float32(0.)
    return jnp.asarray([a,acquire,hold,progress,shot,passed,crowd,ret,ga,gc,gh,support],jnp.float32)
