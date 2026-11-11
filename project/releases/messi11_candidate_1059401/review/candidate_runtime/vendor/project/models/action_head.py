#action_head.py

from typing import NamedTuple

import jax.numpy as jnp

from flax import linen as nn

from footballworld.core.constants import INTENT_ACTION_CONTINUOUS_DIM
from footballworld.core.contact import ACTION_INTENT_COUNT


class ActorOutput(NamedTuple):
    """
    Distribution parameters produced by the actor.

    intent_logits:
        [batch_size, 6]

    continuous_mean:
        [batch_size, 7]

    continuous_log_std:
        [batch_size, 7]
    """

    intent_logits: jnp.ndarray
    continuous_mean: jnp.ndarray
    continuous_log_std: jnp.ndarray


class HybridActionHead(nn.Module):
    """
    Hybrid MESSI action head.

    Input
    -----
    latent:
        [batch_size, latent_dim]

    Output
    ------
    ActorOutput containing:

        categorical intent logits
        Gaussian continuous-action mean
        Gaussian continuous-action log standard deviation
    """

    initial_log_std: float = -0.5

    @nn.compact
    def __call__(self, latent):
        # --------------------------------------------------------
        # 1. Discrete intent
        #
        # MOVE / CONTROL / PASS / SHOT / CLEAR / CHALLENGE
        # --------------------------------------------------------
        intent_logits = nn.Dense(
            ACTION_INTENT_COUNT,
            name="intent_head",
        )(latent)

        # --------------------------------------------------------
        # 2. Continuous action mean
        #
        # move          : 2
        # force_to_ball : 2
        # launch        : 1
        # spin          : 2
        #
        # total         : 7
        #
        # Do NOT apply tanh here.
        #
        # Later we will sample a Gaussian in unconstrained space
        # and squash it with tanh.
        # --------------------------------------------------------
        continuous_mean = nn.Dense(
            INTENT_ACTION_CONTINUOUS_DIM,
            name="continuous_mean_head",
        )(latent)

        # --------------------------------------------------------
        # 3. State-independent learned log standard deviation
        # --------------------------------------------------------
        log_std = self.param(
            "continuous_log_std",
            nn.initializers.constant(
                self.initial_log_std
            ),
            (
                INTENT_ACTION_CONTINUOUS_DIM,
            ),
        )

        continuous_log_std = jnp.broadcast_to(
            log_std,
            continuous_mean.shape,
        )

        return ActorOutput(
            intent_logits=intent_logits,
            continuous_mean=continuous_mean,
            continuous_log_std=continuous_log_std,
        )