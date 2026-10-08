import jax.numpy as jnp

from project.algorithms.buffer import (
    Transition,
    stack_transitions,
)


def test_rollout_buffer_shapes():
    # ============================================================
    # 1. Create dummy transitions
    # ============================================================
    transitions = []

    for _ in range(4):
        transition = Transition(
            actor_obs=jnp.zeros(
                (5, 414),
                dtype=jnp.float32,
            ),

            critic_obs=jnp.zeros(
                (411,),
                dtype=jnp.float32,
            ),

            intent=jnp.zeros(
                (5,),
                dtype=jnp.int32,
            ),

            pre_tanh_continuous=jnp.zeros(
                (5, 7),
                dtype=jnp.float32,
            ),

            log_prob=jnp.zeros(
                (5,),
                dtype=jnp.float32,
            ),

            value=jnp.array(
                0.0,
                dtype=jnp.float32,
            ),

            reward=jnp.array(
                0.0,
                dtype=jnp.float32,
            ),

            terminated=jnp.array(
                False,
                dtype=jnp.bool_,
            ),

            truncated=jnp.array(
                False,
                dtype=jnp.bool_,
            ),

            done=jnp.array(
                False,
                dtype=jnp.bool_,
            ),
        )

        transitions.append(
            transition
        )

    # ============================================================
    # 2. Stack over time
    # ============================================================
    batch = stack_transitions(
        transitions
    )

    # ============================================================
    # 3. Print shapes
    # ============================================================
    print("\n=== Rollout Buffer ===")

    print(
        "actor_obs:",
        batch.actor_obs.shape,
    )

    print(
        "critic_obs:",
        batch.critic_obs.shape,
    )

    print(
        "intent:",
        batch.intent.shape,
    )

    print(
        "pre_tanh_continuous:",
        batch.pre_tanh_continuous.shape,
    )

    print(
        "log_prob:",
        batch.log_prob.shape,
    )

    print(
        "value:",
        batch.value.shape,
    )

    print(
        "reward:",
        batch.reward.shape,
    )

    print(
        "terminated:",
        batch.terminated.shape,
    )

    print(
        "truncated:",
        batch.truncated.shape,
    )

    print(
        "done:",
        batch.done.shape,
    )

    # ============================================================
    # 4. Assertions
    # ============================================================
    assert batch.actor_obs.shape == (
        4,
        5,
        414,
    )

    assert batch.critic_obs.shape == (
        4,
        411,
    )

    assert batch.intent.shape == (
        4,
        5,
    )

    assert batch.pre_tanh_continuous.shape == (
        4,
        5,
        7,
    )

    assert batch.log_prob.shape == (
        4,
        5,
    )

    assert batch.value.shape == (4,)
    assert batch.reward.shape == (4,)

    assert batch.terminated.shape == (4,)
    assert batch.truncated.shape == (4,)
    assert batch.done.shape == (4,)