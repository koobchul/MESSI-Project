from flax import linen as nn


class MLPActorBackbone(nn.Module):
    """
    Shared MLP backbone for the actor.

    Input
    -----
    obs:
        [batch_size, input_dim]

    For 5v5:
        [10, 414]

    Output
    ------
    latent:
        [batch_size, hidden_dim]

    Default architecture:
        input -> 256 -> 256
    """

    hidden_dim: int = 256

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.hidden_dim)(obs)
        x = nn.tanh(x)

        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)

        return x