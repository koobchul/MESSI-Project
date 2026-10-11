import jax
import jax.numpy as jnp

from project.algorithms.buffer import Transition

from project.algorithms.optim import (
    make_optimizer,
)

from project.algorithms.update import (
    ppo_update_step,
)

from project.models.action_head import (
    HybridActionHead,
)

from project.models.action_sampling import (
    action_log_prob,
    sample_action,
)

from project.models.critic import (
    CentralizedCritic,
)

from project.models.mlp_actor import (
    MLPActorBackbone,
)


def tree_l2_distance(
    tree_a,
    tree_b,
):
    """
    L2 distance between two parameter pytrees.
    """

    leaves_a = jax.tree_util.tree_leaves(
        tree_a
    )

    leaves_b = jax.tree_util.tree_leaves(
        tree_b
    )

    return jnp.sqrt(
        sum(
            jnp.sum(
                jnp.square(
                    a - b
                )
            )
            for a, b in zip(
                leaves_a,
                leaves_b,
            )
        )
    )


def make_stored_action_data(
    backbone,
    backbone_variables,
    action_head,
    head_variables,
    actor_obs,
    key,
):
    """
    Build stored rollout actions and old log probabilities
    for one fixed-shape actor observation batch.
    """

    latent = backbone.apply(
        backbone_variables,
        actor_obs,
    )

    actor_output = action_head.apply(
        head_variables,
        latent,
    )

    sampled = sample_action(
        actor_output,
        key,
    )

    intent = sampled.action.intent

    pre_tanh_continuous = (
        sampled.pre_tanh_continuous
    )

    old_log_prob = action_log_prob(
        actor_output,
        intent,
        pre_tanh_continuous,
    )

    return (
        intent,
        pre_tanh_continuous,
        old_log_prob,
    )


def test_masked_update_ignores_padding_contents():
    # ============================================================
    # 1. Random keys
    # ============================================================

    keys = jax.random.split(
        jax.random.key(0),
        8,
    )

    (
        real_actor_key,
        padding_actor_a_key,
        padding_actor_b_key,
        real_critic_key,
        padding_critic_a_key,
        padding_critic_b_key,
        action_key,
        _,
    ) = keys

    # ============================================================
    # 2. REAL actor observations
    #
    # These first two rows are identical in both batches.
    # ============================================================

    real_actor_obs = jax.random.normal(
        real_actor_key,
        (
            2,
            5,
            414,
        ),
    )

    # ============================================================
    # 3. Two completely different padding regions
    # ============================================================

    padding_actor_obs_a = (
        20.0
        * jax.random.normal(
            padding_actor_a_key,
            (
                2,
                5,
                414,
            ),
        )
    )

    padding_actor_obs_b = (
        50.0
        * jax.random.normal(
            padding_actor_b_key,
            (
                2,
                5,
                414,
            ),
        )
    )

    actor_obs_a = jnp.concatenate(
        [
            real_actor_obs,
            padding_actor_obs_a,
        ],
        axis=0,
    )

    actor_obs_b = jnp.concatenate(
        [
            real_actor_obs,
            padding_actor_obs_b,
        ],
        axis=0,
    )

    # ============================================================
    # 4. Actor initialization
    #
    # Parameter shapes do not depend on batch size/content.
    # ============================================================

    backbone = MLPActorBackbone(
        hidden_dim=256,
    )

    backbone_variables = backbone.init(
        jax.random.key(1),
        actor_obs_a,
    )

    latent_a = backbone.apply(
        backbone_variables,
        actor_obs_a,
    )

    action_head = HybridActionHead()

    head_variables = action_head.init(
        jax.random.key(2),
        latent_a,
    )

    actor_params = {
        "backbone":
            backbone_variables["params"],

        "head":
            head_variables["params"],
    }

    # ============================================================
    # 5. Stored actions for Batch A
    # ============================================================

    (
        intent_a,
        pre_tanh_a,
        old_log_prob_a,
    ) = make_stored_action_data(
        backbone=backbone,
        backbone_variables=backbone_variables,

        action_head=action_head,
        head_variables=head_variables,

        actor_obs=actor_obs_a,

        key=action_key,
    )

    # ============================================================
    # 6. Stored actions for Batch B
    #
    # Same key + same full shape.
    #
    # Therefore the REAL first two rows receive the same random
    # samples. Only padding observations differ.
    # ============================================================

    (
        intent_b,
        pre_tanh_b,
        old_log_prob_b,
    ) = make_stored_action_data(
        backbone=backbone,
        backbone_variables=backbone_variables,

        action_head=action_head,
        head_variables=head_variables,

        actor_obs=actor_obs_b,

        key=action_key,
    )

    # ============================================================
    # 7. Verify real actor rollout data are identical
    # ============================================================

    assert bool(
        jnp.all(
            intent_a[:2]
            == intent_b[:2]
        )
    )

    assert jnp.allclose(
        pre_tanh_a[:2],
        pre_tanh_b[:2],
        atol=1e-7,
    )

    assert jnp.allclose(
        old_log_prob_a[:2],
        old_log_prob_b[:2],
        atol=1e-7,
    )

    # ============================================================
    # 8. REAL centralized critic observations
    # ============================================================

    real_critic_obs = jax.random.normal(
        real_critic_key,
        (
            2,
            411,
        ),
    )

    # ============================================================
    # 9. Different critic padding
    # ============================================================

    padding_critic_obs_a = (
        20.0
        * jax.random.normal(
            padding_critic_a_key,
            (
                2,
                411,
            ),
        )
    )

    padding_critic_obs_b = (
        50.0
        * jax.random.normal(
            padding_critic_b_key,
            (
                2,
                411,
            ),
        )
    )

    critic_obs_a = jnp.concatenate(
        [
            real_critic_obs,
            padding_critic_obs_a,
        ],
        axis=0,
    )

    critic_obs_b = jnp.concatenate(
        [
            real_critic_obs,
            padding_critic_obs_b,
        ],
        axis=0,
    )

    # ============================================================
    # 10. Critic initialization
    # ============================================================

    critic = CentralizedCritic(
        hidden_dim=256,
    )

    critic_variables = critic.init(
        jax.random.key(3),
        critic_obs_a,
    )

    critic_params = (
        critic_variables["params"]
    )

    # ============================================================
    # 11. Fixed valid mask
    #
    # Only first two timesteps are real.
    # ============================================================

    valid_mask = jnp.array(
        [
            True,
            True,
            False,
            False,
        ]
    )

    # ============================================================
    # 12. Batch A
    # ============================================================

    batch_a = Transition(
        actor_obs=actor_obs_a,

        critic_obs=critic_obs_a,

        intent=intent_a,

        pre_tanh_continuous=(
            pre_tanh_a
        ),

        log_prob=old_log_prob_a,

        value=jnp.zeros(
            (4,),
            dtype=jnp.float32,
        ),

        reward=jnp.zeros(
            (4,),
            dtype=jnp.float32,
        ),

        terminated=jnp.array(
            [
                False,
                False,
                False,
                False,
            ]
        ),

        truncated=jnp.array(
            [
                False,
                True,
                False,
                False,
            ]
        ),

        done=jnp.array(
            [
                False,
                True,
                True,
                True,
            ]
        ),
    )

    # ============================================================
    # 13. Batch B
    #
    # Same REAL rows.
    # Different padding contents.
    # ============================================================

    batch_b = Transition(
        actor_obs=actor_obs_b,

        critic_obs=critic_obs_b,

        intent=intent_b,

        pre_tanh_continuous=(
            pre_tanh_b
        ),

        log_prob=old_log_prob_b,

        value=jnp.zeros(
            (4,),
            dtype=jnp.float32,
        ),

        reward=jnp.zeros(
            (4,),
            dtype=jnp.float32,
        ),

        terminated=jnp.array(
            [
                False,
                False,
                True,
                True,
            ]
        ),

        truncated=jnp.array(
            [
                False,
                True,
                True,
                False,
            ]
        ),

        done=jnp.array(
            [
                False,
                True,
                False,
                False,
            ]
        ),
    )

    # ============================================================
    # 14. Advantages
    #
    # Real rows are identical.
    #
    # Padding values are deliberately wildly different.
    # ============================================================

    advantages_a = jnp.array(
        [
            1.0,
            -1.0,
            1000.0,
            -1000.0,
        ],
        dtype=jnp.float32,
    )

    advantages_b = jnp.array(
        [
            1.0,
            -1.0,
            -9000.0,
            7000.0,
        ],
        dtype=jnp.float32,
    )

    # ============================================================
    # 15. Returns
    # ============================================================

    returns_a = jnp.array(
        [
            0.5,
            -0.25,
            1000.0,
            -1000.0,
        ],
        dtype=jnp.float32,
    )

    returns_b = jnp.array(
        [
            0.5,
            -0.25,
            -8000.0,
            6000.0,
        ],
        dtype=jnp.float32,
    )

    # ============================================================
    # 16. Optimizers
    # ============================================================

    actor_optimizer = make_optimizer(
        learning_rate=3e-5,
        max_grad_norm=0.5,
    )

    critic_optimizer = make_optimizer(
        learning_rate=3e-5,
        max_grad_norm=0.5,
    )

    actor_opt_state = actor_optimizer.init(
        actor_params
    )

    critic_opt_state = critic_optimizer.init(
        critic_params
    )

    # ============================================================
    # 17. PPO Update A
    # ============================================================

    (
        actor_params_a,
        critic_params_a,
        _,
        _,
        metrics_a,
    ) = ppo_update_step(
        actor_params=actor_params,
        critic_params=critic_params,

        actor_opt_state=actor_opt_state,
        critic_opt_state=critic_opt_state,

        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,

        backbone=backbone,
        action_head=action_head,
        critic=critic,

        batch=batch_a,

        advantages=advantages_a,
        returns=returns_a,

        clip_eps=0.2,
        entropy_coef=0.01,

        valid_mask=valid_mask,
    )

    # ============================================================
    # 18. PPO Update B
    # ============================================================

    (
        actor_params_b,
        critic_params_b,
        _,
        _,
        metrics_b,
    ) = ppo_update_step(
        actor_params=actor_params,
        critic_params=critic_params,

        actor_opt_state=actor_opt_state,
        critic_opt_state=critic_opt_state,

        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,

        backbone=backbone,
        action_head=action_head,
        critic=critic,

        batch=batch_b,

        advantages=advantages_b,
        returns=returns_b,

        clip_eps=0.2,
        entropy_coef=0.01,

        valid_mask=valid_mask,
    )

    # ============================================================
    # 19. Parameter comparison
    # ============================================================

    actor_difference = tree_l2_distance(
        actor_params_a,
        actor_params_b,
    )

    critic_difference = tree_l2_distance(
        critic_params_a,
        critic_params_b,
    )

    print(
        "\n=== Padding Invariance Test ==="
    )

    print(
        "actor parameter difference:",
        actor_difference,
    )

    print(
        "critic parameter difference:",
        critic_difference,
    )

    print(
        "\n=== Metrics A ==="
    )

    print(
        "actor loss:",
        metrics_a.actor_loss,
    )

    print(
        "policy loss:",
        metrics_a.policy_loss,
    )

    print(
        "entropy:",
        metrics_a.entropy,
    )

    print(
        "critic loss:",
        metrics_a.critic_loss,
    )

    print(
        "KL:",
        metrics_a.approx_kl,
    )

    print(
        "clip fraction:",
        metrics_a.clip_fraction,
    )

    print(
        "\n=== Metrics B ==="
    )

    print(
        "actor loss:",
        metrics_b.actor_loss,
    )

    print(
        "policy loss:",
        metrics_b.policy_loss,
    )

    print(
        "entropy:",
        metrics_b.entropy,
    )

    print(
        "critic loss:",
        metrics_b.critic_loss,
    )

    print(
        "KL:",
        metrics_b.approx_kl,
    )

    print(
        "clip fraction:",
        metrics_b.clip_fraction,
    )

    # ============================================================
    # 20. Assertions
    #
    # Same shape + same real rows + same mask.
    #
    # Changing padding content must not change optimization.
    # ============================================================

    assert actor_difference < 1e-6

    assert critic_difference < 1e-6

    assert jnp.allclose(
        metrics_a.actor_loss,
        metrics_b.actor_loss,
        atol=1e-6,
    )

    assert jnp.allclose(
        metrics_a.policy_loss,
        metrics_b.policy_loss,
        atol=1e-6,
    )

    assert jnp.allclose(
        metrics_a.entropy,
        metrics_b.entropy,
        atol=1e-6,
    )

    assert jnp.allclose(
        metrics_a.critic_loss,
        metrics_b.critic_loss,
        atol=1e-6,
    )

    assert jnp.allclose(
        metrics_a.approx_kl,
        metrics_b.approx_kl,
        atol=1e-6,
    )

    assert jnp.allclose(
        metrics_a.clip_fraction,
        metrics_b.clip_fraction,
        atol=1e-6,
    )