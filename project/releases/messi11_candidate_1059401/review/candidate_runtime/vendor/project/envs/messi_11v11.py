"""Fixed 4-4-2; positions are in each team's attacking frame."""
import jax
from footballworld import FootballWorld, MatchConfig
from project.envs.messi_5v5 import make_player
POSITIONS = ((-50.,0.),(-34.,-24.),(-36.,-8.),(-36.,8.),(-34.,24.),
             (-20.,-24.),(-23.,-8.),(-23.,8.),(-20.,24.),(-7.,-9.),(-7.,9.))
ROLES = (0,1,1,1,1,2,2,2,2,3,3) # GK, DEF, MID, FWD; no substitutions

def rosters():
    return tuple(tuple(make_player(base+i, x,y, goalkeeper=i==0)
                       for i,(x,y) in enumerate(POSITIONS)) for base in (1,101))

def reset_env(env, seed):
    return env.reset(*rosters(), key=jax.random.key(seed))

def make_env(seed=0):
    env=FootballWorld(match=MatchConfig()) # official default min=7, 90 min
    return env, reset_env(env, seed)
