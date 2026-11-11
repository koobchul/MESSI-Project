"""Training-only loose-ball starts; native match physics and 22 active players.
No fabricated possession/contact or teacher actions. All evaluation uses reset_env.
"""
import numpy as np
import jax.numpy as jnp
from footballworld.core.state import RestartState,RestartReleaseProvenance

def training_start(env,reset,seed,kind):
    if kind=='native':return reset
    rng=np.random.default_rng(seed);s=reset.rollout.state;direction=float(s.attack_direction[0])
    pos=np.asarray(s.players.position).copy()
    if kind=='gk':
        slot=0;xy=np.array([-49.5,rng.uniform(-2,2)],np.float32)*direction
    else:
        slot=int(rng.integers(5,11))
        x={'acquire':-5.,'advance':12.,'finish':31.}[kind]
        xy=np.array([x+rng.uniform(-3,3),rng.uniform(-8,8)],np.float32)*direction
    # Keep all players and abilities. Avoid placing the selected player on another.
    for i in range(22):
        if i!=slot and np.linalg.norm(pos[i]-xy)<5.:
            pos[i,1]=np.clip(pos[i,1]+(6. if pos[i,1]>=xy[1] else -6.),-30,30)
    pos[slot]=xy
    gap=rng.uniform(.35,.65) if kind in ('finish','advance','gk') else rng.uniform(.8,1.6)
    ball=np.array([xy[0]+direction*gap,xy[1],env.ball.radius],np.float32)
    s=s._replace(players=s.players._replace(position=jnp.asarray(pos),velocity=jnp.zeros_like(s.players.velocity)),
        ball=s.ball._replace(position=jnp.asarray(ball),velocity=jnp.zeros_like(s.ball.velocity),spin=jnp.zeros_like(s.ball.spin),live=jnp.bool_(True)),
        restart=RestartState(jnp.int32(0),jnp.int32(-1),jnp.int32(0),jnp.int32(-1),jnp.bool_(False)),
        restart_release=RestartReleaseProvenance(jnp.bool_(False),jnp.bool_(False),jnp.int32(0),jnp.int32(-1),jnp.int32(-1),jnp.bool_(False),jnp.bool_(False)),
        possession=s.possession._replace(team=jnp.int32(-1),player=jnp.int32(-1),previous_team=jnp.int32(-1),control_ticks=jnp.int32(0)),
        restart_layout_ready=jnp.bool_(False))
    return reset._replace(rollout=reset.rollout._replace(state=s))

def choose_kind(episode,fraction):
    # Fractions are elapsed training budget, not performance. Before 75%,
    # each kind (including native) is a 1024-step bootstrapped segment.
    if fraction>=.75:return 'native'
    native_count=2 if fraction<.25 else (4 if fraction<.5 else 8)
    if episode%10<native_count:return 'native'
    return ('acquire','advance','finish','gk')[episode%4]
