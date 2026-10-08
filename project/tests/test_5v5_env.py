import jax

from footballworld import neutral_action
from project.envs.messi_5v5 import make_5v5_env


def test_5v5_environment():
    env, reset = make_5v5_env(seed=0)

    # Observation
    obs = env.observe_all(reset.rollout)

    print("\n=== Observation ===")
    print("valid shape:", obs.valid.shape)

    assert obs.valid.shape == (10,)

    # Neutral action for all 10 players
    action = neutral_action(player_count=10)

    print("\n=== Action ===")
    print("intent:", action.intent.shape)
    print("move:", action.move.shape)
    print("force_to_ball:", action.force_to_ball.shape)
    print("launch:", action.launch.shape)
    print("spin:", action.spin.shape)

    assert action.intent.shape == (10,)
    assert action.move.shape == (10, 2)
    assert action.force_to_ball.shape == (10, 2)
    assert action.launch.shape == (10,)
    assert action.spin.shape == (10, 2)

    # One environment step
    result = env.step(
        reset.rollout,
        reset.setup,
        action,
        jax.random.key(1),
    )

    print("\n=== Step Result ===")
    print("reward:", result.reward)
    print("done:", result.done)

    assert result.reward.shape == (2,)