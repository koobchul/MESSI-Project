import jax
import jax.numpy as jnp
from footballworld.core.action import IntentAction

from footballworld import (
    FootballWorld,
    MatchConfig,
    Player,
    PlayerProfile,
    neutral_action,
)

def player(player_id, x, y, goalkeeper=False):
    return Player(
        profile=PlayerProfile(
            player_id=player_id,
            is_goalkeeper=goalkeeper,
        ),
        initial_position=(x, y),
    )

home = (
    player(1, -30.0, -8.0, goalkeeper=True),
    player(2, -10.0, 8.0),
)

away = (
    player(3, -30.0, -8.0, goalkeeper=True),
    player(4, -10.0, 8.0),
)

env = FootballWorld(
    match=MatchConfig(minimum_team_players=(2, 2))
)

reset = env.reset(
    home,
    away,
    key=jax.random.key(0),
)

obs = env.observe_all(reset.rollout)

print("Observation type:")
print(type(obs))

print("\nObservation tree:")
print(jax.tree_util.tree_map(
    lambda x: (x.shape, x.dtype),
    obs
))

action = IntentAction.move_only(
    jnp.array(
        [
            [0.5, 0.0],
            [0.5, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
)

print("\nAction tree:")
print(jax.tree_util.tree_map(
    lambda x: (x.shape, x.dtype),
    action
))

result = env.step(
    reset.rollout,
    reset.setup,
    action,
    jax.random.key(1),
)

print("\nReward:")
print(result.reward)

print("\nDone:")
print(result.done)
