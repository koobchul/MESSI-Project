#buffer.py

from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp

class Transition(NamedTuple):
    """
    One MAPPO rollout transition.

    reward:
        Reward actually used by GAE / PPO.

        reward
        =
        base_reward
        +
        shaping_reward

    base_reward:
        Original sparse MESSI goal reward.

    shaping_reward:
        Potential-based dense shaping component.
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

    # Diagnostic fields.
    #
    # Defaults preserve compatibility with older code that constructs
    # Transition without explicitly providing these fields.
    base_reward: jax.Array = 0.0
    shaping_reward: jax.Array = 0.0

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