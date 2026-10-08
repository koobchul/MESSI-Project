import jax.numpy as jnp

from footballworld.core.action import IntentAction


def combine_team_actions(
    learned_action: IntentAction,
    opponent_action: IntentAction,
    team_id,
    controlled_team=0,
):
    """
    Combine learned-policy actions for one team with
    opponent-policy actions for the other team.

    Parameters
    ----------
    learned_action:
        Full-slot IntentAction produced by our actor.

    opponent_action:
        Full-slot IntentAction produced by the fixed opponent.

    team_id:
        Shape [num_players].
        Team ID for every roster slot.

    controlled_team:
        Team controlled by the learned policy.

    Returns
    -------
    combined_action:
        Full IntentAction with one action per player.
    """

    controlled = (
        team_id == controlled_team
    )

    return IntentAction(
        intent=jnp.where(
            controlled,
            learned_action.intent,
            opponent_action.intent,
        ),

        move=jnp.where(
            controlled[:, None],
            learned_action.move,
            opponent_action.move,
        ),

        force_to_ball=jnp.where(
            controlled[:, None],
            learned_action.force_to_ball,
            opponent_action.force_to_ball,
        ),

        launch=jnp.where(
            controlled,
            learned_action.launch,
            opponent_action.launch,
        ),

        spin=jnp.where(
            controlled[:, None],
            learned_action.spin,
            opponent_action.spin,
        ),
    )