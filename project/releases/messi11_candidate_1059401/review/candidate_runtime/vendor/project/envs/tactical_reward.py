"""Small team-level shaping; frozen assignment/targets avoid target-motion credit.
Not potential-based policy-invariance: gamma intentionally absent from progress.
No movement override, no teacher labels. GK excluded from both terms.
"""
import jax.numpy as jnp
from project.envs.reward_v3 import open_play, prohibited_taker_mask
from footballworld.dynamics.contact_predicates import verified_controlled_carrier
from project.envs.messi_11v11 import POSITIONS

def assignment(env,s):
    p=s.players;direction=s.attack_direction[0]
    c=verified_controlled_carrier(s,env.ball,env.reach,env.action_scale)
    safe=jnp.clip(c,0,21)
    own=(c>=0)&(p.team_id[safe]==0)&~prohibited_taker_mask(s)[safe]
    eligible=(p.team_id==0)&~p.is_goalkeeper&p.active&~prohibited_taker_mask(s)
    gap=jnp.linalg.norm(p.position-s.ball.position[:2],axis=-1)
    chaser=jnp.argmin(jnp.where(eligible,gap,jnp.inf))
    # Anchor in attacking frame, translated by ball and bounded inside the pitch.
    anchor=jnp.asarray(POSITIONS)[jnp.arange(22)%11]
    ball=s.ball.position[:2]*direction
    shift=jnp.array([.45*ball[0]+jnp.where(own,10.,-5.),.35*ball[1]])
    target=anchor+shift
    target=target.at[:,0].set(jnp.clip(target[:,0],-45.,42.))
    target=target.at[:,1].set(jnp.clip(target[:,1],-28.,28.))
    # Defensive cover remains goal-side of the ball; no fixed target for carrier.
    target=target.at[:,0].set(jnp.clip(jnp.where(own,target[:,0],jnp.minimum(target[:,0],ball[0]-5.)),-45.,42.))
    support=eligible & jnp.where(own,jnp.arange(22)!=c,jnp.arange(22)!=chaser)
    return own,eligible,chaser,target*direction,support

def frozen_progress(before,after,target,mask,scale):
    d0=jnp.linalg.norm(before-target,axis=-1)
    d1=jnp.linalg.norm(after-target,axis=-1)
    # Bounded per-transition progress, not standing-position rent.
    return jnp.sum(jnp.where(mask,jnp.clip(d0-d1,-1.,1.)/scale,0.))/jnp.maximum(mask.sum(),1)

def contextual_terms(env,s,nxt,valid):
    own,eligible,chaser,target,support=assignment(env,s)
    same_roster=jnp.all(s.players.active==nxt.players.active)
    gate=valid&same_roster&open_play(s)&open_play(nxt)
    chase_mask=eligible&(jnp.arange(22)==chaser)
    # The old ball location is held fixed: ball flying toward a stationary
    # player is not credited as the player's approach.
    a=.1*frozen_progress(s.players.position,nxt.players.position,
                       s.ball.position[:2],chase_mask,10.)
    b=.02*frozen_progress(s.players.position,nxt.players.position,target,support,20.)
    return jnp.where(gate&~own,a,0.),jnp.where(gate,b,0.)
