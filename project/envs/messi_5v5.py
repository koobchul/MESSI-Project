import jax

from footballworld import (
    FootballWorld,
    MatchConfig,
    Player,
    PlayerProfile,
)


def make_player(player_id, x, y, goalkeeper=False):
    return Player(
        profile=PlayerProfile(
            player_id=player_id,
            is_goalkeeper=goalkeeper,
        ),
        initial_position=(x, y),
    )


def make_5v5_env(seed=0):
    # Home team: 5 players
    home = (
        make_player(1, -50.0, 0.0, goalkeeper=True),
        make_player(2, -30.0, -15.0),
        make_player(3, -30.0, 15.0),
        make_player(4, -10.0, -8.0),
        make_player(5, -10.0, 8.0),
    )

    # Away team: 5 players
    away = (
        make_player(101, -50.0, 0.0, goalkeeper=True),
        make_player(102, -30.0, -15.0),
        make_player(103, -30.0, 15.0),
        make_player(104, -10.0, -8.0),
        make_player(105, -10.0, 8.0),
    )

    env = FootballWorld(
        match=MatchConfig(
            minimum_team_players=(5, 5)
        )
    )

    reset = env.reset(
        home,
        away,
        key=jax.random.key(seed),
    )

    return env, reset