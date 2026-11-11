from flax import linen as nn


class CentralizedCritic(nn.Module):
    """
    Centralized value network for MAPPO.

    Input
    -----
    critic_input:
        [batch_size, input_dim]

    Output
    ------
    value:
        [batch_size, 2] (outfield return, GK return)
    """

    hidden_dim: int = 256

    @nn.compact
    def __call__(self, critic_input):
        x = nn.Dense(self.hidden_dim)(critic_input)
        x = nn.tanh(x)

        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)

        value = nn.Dense(2)(x)

        return value # [batch, 2]: outfield team return, goalkeeper return