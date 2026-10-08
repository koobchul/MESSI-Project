import jax
import jax.numpy as jnp

from footballworld.policies.rule_based import make_rule_based_policy

from project.envs.messi_5v5 import make_5v5_env
from project.envs.obs_encoder import encode_mlp_observation

from project.models.action_head import HybridActionHead
from project.models.action_sampling import sample_action
from project.models.mlp_actor import MLPActorBackbone

from project.utils.team_action import combine_team_actions


def test_learned_team_vs_rule_based_opponent():
    # ============================================================
    # 1. Environment
    # ============================================================
    env, reset = make_5v5_env(seed=0)

    # ============================================================
    # 2. Learned policy observation
    # ============================================================
    obs = env.observe_all(
        reset.rollout
    )

    roster = env.roster_metadata(
        reset.rollout
    )

    mlp_obs = encode_mlp_observation(
        obs,
        roster,
    )

    # ============================================================
    # 3. Learned MLP actor
    # ============================================================
    backbone = MLPActorBackbone(
        hidden_dim=256,
    )

    backbone_variables = backbone.init(
        jax.random.key(0),
        mlp_obs,
    )

    latent = backbone.apply(
        backbone_variables,
        mlp_obs,
    )

    action_head = HybridActionHead()

    head_variables = action_head.init(
        jax.random.key(1),
        latent,
    )

    actor_output = action_head.apply(
        head_variables,
        latent,
    )

    sampled = sample_action(
        actor_output,
        jax.random.key(2),
    )

    learned_action = sampled.action

    # ============================================================
    # 4. MESSI built-in rule-based opponent
    #
    # RuleBasedPolicy explicitly consumes SI observations.
    # ============================================================
    rule_policy = make_rule_based_policy(
        env
    )

    si_obs = env.observe_all_si(
        reset.rollout
    )

    si_roster = env.roster_metadata_si(
        reset.rollout
    )

    rule_state = rule_policy.initialize(
        si_obs,
        si_roster,
    )

    rule_step = rule_policy.step(
        si_obs,
        si_roster,
        rule_state,
        jax.random.key(3),
    )

    opponent_action = rule_step.action

    # ============================================================
    # 5. Combine
    #
    # Team 0 -> learned policy
    # Team 1 -> rule-based policy
    # ============================================================
    action = combine_team_actions(
        learned_action,
        opponent_action,
        roster.team_id,
        controlled_team=0,
    )

    print("\n=== Team Assignment ===")
    print("team_id:", roster.team_id)

    print("\n=== Learned Action ===")
    print("intent:", learned_action.intent)

    print("\n=== Rule-based Action ===")
    print("intent:", opponent_action.intent)

    print("\n=== Combined Action ===")
    print("intent:", action.intent)

    # ============================================================
    # 6. Verify action ownership
    # ============================================================
    controlled_mask = (
        roster.team_id == 0
    )

    opponent_mask = (
        roster.team_id == 1
    )

    assert jnp.all(
        action.intent[controlled_mask]
        == learned_action.intent[controlled_mask]
    )

    assert jnp.all(
        action.intent[opponent_mask]
        == opponent_action.intent[opponent_mask]
    )

    # ============================================================
    # 7. Environment step
    # ============================================================
    result = env.step(
        reset.rollout,
        reset.setup,
        action,
        jax.random.key(4),
    )

    print("\n=== Environment Step ===")
    print("reward:", result.reward)
    print("done:", result.done)

    assert result.reward.shape == (2,)