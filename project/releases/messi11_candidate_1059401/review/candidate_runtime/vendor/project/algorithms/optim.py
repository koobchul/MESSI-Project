import optax


def make_optimizer(
    learning_rate=3e-4,
    max_grad_norm=0.5,
):
    """
    PPO/MAPPO optimizer.

    Gradient clipping is applied before Adam.
    """

    return optax.chain(
        optax.clip_by_global_norm(
            max_grad_norm
        ),
        optax.adam(
            learning_rate
        ),
    )