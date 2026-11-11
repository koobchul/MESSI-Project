"""Policy-side goal-area target mapping; never changes environment positions.

This map is used identically in PPO rollouts and all evaluation policies.
The raw GK move selects a target inside the six-yard box. PD velocity control
tracks it. Inertia/collisions/referee relocation can still cause excursions.
"""
import jax.numpy as jnp

def limit_gk_move(env,state,action):
    gk=jnp.argmax((state.players.team_id==0)&state.players.is_goalkeeper)
    direction=state.attack_direction[0]
    pos=state.players.position[gk]*direction
    vel=state.players.velocity[gk]*direction
    depth=pos[0]+env.stadium.half_length
    width=env.stadium.goal_area_width/2
    length=env.stadium.goal_area_length
    target_half_width=min(width-1.,env.stadium.goal_width/2-.5)
    raw=jnp.clip(action.move[gk],-1,1)
    # One metre margin to the sides/front and 0.8m to the goal line.
    lo=.8;hi=length-1.
    target=jnp.asarray([-env.stadium.half_length+(lo+hi)/2+(hi-lo)*raw[0]/2,
                        target_half_width*raw[1]])
    desired=2.*(target-pos)-vel
    speed=jnp.maximum(state.players.max_speed[gk],.1)
    velocity=desired/jnp.maximum(jnp.linalg.norm(desired)/speed,1.)
    # Invert L-infinity radial decode: direction*max(abs(control)).
    unit=velocity/speed
    move=unit*jnp.linalg.norm(unit)/jnp.maximum(jnp.max(jnp.abs(unit)),1e-6)
    move=jnp.clip(move,-1.,1.) # engine rotates team-frame actions into world frame
    outside=(depth<0)|(depth>length)|(jnp.abs(pos[1])>width)
    exception=(state.restart.kind!=0)&(state.restart.taker==gk)&outside
    enabled=state.players.active[gk]&~exception
    updated=action._replace(move=action.move.at[gk].set(jnp.where(enabled,move,action.move[gk])))
    intervention=enabled&(jnp.max(jnp.abs(move-action.move[gk]))>1e-6)
    return updated,intervention,exception
