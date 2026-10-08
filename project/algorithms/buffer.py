from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp


class Transition(NamedTuple):
    """
    One MAPPO rollout transition.

    Shapes for 5v5 controlled Team 0:

    actor_obs:
        [5, 414]

    critic_obs:
        [411]

    intent:
        [5]

    pre_tanh_continuous:
        [5, 7]

    log_prob:
        [5]

    value:
        scalar

    reward:
        scalar

    terminated:
        scalar bool

    truncated:
        scalar bool

    done:
        scalar bool
    """

    actor_obs: jax.Array
    critic_obs: jax.Array

    intent: jax.Array
    pre_tanh_continuous: jax.Array

    log_prob: jax.Array

    value: jax.Array
    reward: jax.Array

    terminated: jax.Array
    truncated: jax.Array
    done: jax.Array


def stack_transitions(
    transitions: Sequence[Transition],
) -> Transition:
    """
    Stack a sequence of transitions along the time axis.

    Example
    -------
    T transitions:

        actor_obs [5, 414]

    become:

        actor_obs [T, 5, 414]
    """

    if len(transitions) == 0:
        raise ValueError(
            "transitions must not be empty"
        )

    return jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=0),
        *transitions,
    )