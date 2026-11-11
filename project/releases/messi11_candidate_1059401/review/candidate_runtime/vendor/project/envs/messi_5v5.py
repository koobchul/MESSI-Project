import jax

from footballworld import (
    FootballWorld,
    MatchConfig,
    Player,
    PlayerProfile,
)


def make_player(
    player_id,
    x,
    y,
    goalkeeper=False,
):
    return Player(
        profile=PlayerProfile(
            player_id=player_id,
            is_goalkeeper=goalkeeper,
        ),
        initial_position=(
            x,
            y,
        ),
    )


def make_5v5_rosters():
    """
    Create the fixed 5v5 player rosters used by the project.

    Returns
    -------
    home:
        tuple of 5 Player objects

    away:
        tuple of 5 Player objects
    """

    home = (
        make_player(
            1,
            -50.0,
            0.0,
            goalkeeper=True,
        ),
        make_player(
            2,
            -30.0,
            -15.0,
        ),
        make_player(
            3,
            -30.0,
            15.0,
        ),
        make_player(
            4,
            -10.0,
            -8.0,
        ),
        make_player(
            5,
            -10.0,
            8.0,
        ),
    )

    away = (
        make_player(
            101,
            -50.0,
            0.0,
            goalkeeper=True,
        ),
        make_player(
            102,
            -30.0,
            -15.0,
        ),
        make_player(
            103,
            -30.0,
            15.0,
        ),
        make_player(
            104,
            -10.0,
            -8.0,
        ),
        make_player(
            105,
            -10.0,
            8.0,
        ),
    )

    return (
        home,
        away,
    )


def make_5v5_env(
    seed=0,
):
    """
    Create the default project 5v5 environment and initial reset.

    This keeps the original public interface unchanged:

        env, reset = make_5v5_env(seed=0)
    """

    home, away = make_5v5_rosters()

    env = FootballWorld(
        match=MatchConfig(
            minimum_team_players=(
                5,
                5,
            )
        )
    )

    reset = env.reset(
        home,
        away,
        key=jax.random.key(seed),
    )

    return (
        env,
        reset,
    )


def reset_5v5_env(
    env,
    seed,
):
    """
    Reset an EXISTING 5v5 FootballWorld instance.

    This is used by the training loop when one match ends and
    another match begins.

    Important
    ---------
    The environment object itself is reused.

    Therefore the JIT rollout collector can continue using the
    same static environment configuration instead of constructing
    a new FootballWorld for every episode.
    """

    home, away = make_5v5_rosters()

    return env.reset(
        home,
        away,
        key=jax.random.key(seed),
    )