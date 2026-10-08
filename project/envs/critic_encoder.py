import jax
import jax.numpy as jnp

from footballworld.core.constants import (
    NO_TEAM,
    RESTART_COUNT,
)

from footballworld.core.contact import (
    INTENT_COUNT,
    INTENT_SOURCE_COUNT,
    LAW11_EFFECT_COUNT,
    MECHANISM_COUNT,
    OUTCOME_COUNT,
)


CRITIC_PLAYER_FEATURE_DIM = 28
CRITIC_CONTEXT_DIM = 39
CRITIC_CATEGORICAL_DIM = 92


def _relative_team_one_hot(team, controlled_team):
    """
    Encode an absolute team ID relative to the learning team.

    [1, 0, 0] = no team
    [0, 1, 0] = own team
    [0, 0, 1] = opponent team
    """

    no_team = team == NO_TEAM
    own_team = (~no_team) & (team == controlled_team)
    opponent_team = (~no_team) & (team != controlled_team)

    return jnp.stack(
        [
            no_team,
            own_team,
            opponent_team,
        ],
        axis=-1,
    ).astype(jnp.float32)


def build_global_player_features(
    global_state,
    controlled_team=0,
):
    """
    Build centralized player features for the critic.

    Returns
    -------
    player_features:
        [num_players, 28]

    For 5v5:
        [10, 28]
    """

    state = global_state.state
    players = state.players

    num_players = players.position.shape[0]

    # ============================================================
    # 1. Continuous physical state
    # ============================================================

    continuous_features = [
        players.position,                         # 2
        players.velocity,                         # 2
        players.body_forward,                     # 2

        players.max_speed[..., None],             # 1
        players.reach_height[..., None],          # 1
        players.height[..., None],                # 1
        players.ball_control[..., None],          # 1
        players.endurance_factor[..., None],      # 1

        players.stamina_long[..., None],          # 1
        players.stamina_short[..., None],         # 1

        players.challenge_recovery_substeps[
            ..., None
        ],                                        # 1

        players.contact_lock_substeps[
            ..., None
        ],                                        # 1

        players.aerial_recovery_substeps[
            ..., None
        ],                                        # 1

        players.possession_loss_lock_substeps[
            ..., None
        ],                                        # 1

        players.yellow_cards.astype(
            jnp.float32
        )[..., None],                             # 1
    ]

    # ============================================================
    # 2. Player status
    # ============================================================

    status_features = [
        players.on_pitch.astype(
            jnp.float32
        )[..., None],

        players.sent_off.astype(
            jnp.float32
        )[..., None],

        players.is_goalkeeper.astype(
            jnp.float32
        )[..., None],
    ]

    # ============================================================
    # 3. Team relationship
    #
    # Raw team_id is not used as a numerical feature.
    # ============================================================

    is_own_team = (
        players.team_id == controlled_team
    )

    is_opponent = (
        players.team_id != controlled_team
    )

    team_features = [
        is_own_team.astype(
            jnp.float32
        )[..., None],

        is_opponent.astype(
            jnp.float32
        )[..., None],
    ]

    # ============================================================
    # 4. Semantic player identity flags
    #
    # Do not feed raw player_id / slot index.
    #
    # possession.player etc. are converted to flags attached
    # directly to the corresponding player entity.
    # ============================================================

    player_slots = jnp.arange(
        num_players,
        dtype=jnp.int32,
    )

    is_possessor = (
        player_slots
        == state.possession.player
    )

    is_restart_taker = (
        player_slots
        == state.restart.taker
    )

    is_release_taker = (
        player_slots
        == state.restart_release.taker
    )

    is_last_actor = (
        player_slots
        == state.possession.last_contact.actor
    )

    is_offside = global_state.offside.flagged

    semantic_flags = [
        is_possessor.astype(
            jnp.float32
        )[..., None],

        is_restart_taker.astype(
            jnp.float32
        )[..., None],

        is_release_taker.astype(
            jnp.float32
        )[..., None],

        is_last_actor.astype(
            jnp.float32
        )[..., None],

        is_offside.astype(
            jnp.float32
        )[..., None],
    ]

    # ============================================================
    # 5. Concatenate
    # ============================================================

    player_features = jnp.concatenate(
        continuous_features
        + status_features
        + team_features
        + semantic_flags,
        axis=-1,
    )

    return player_features


def build_global_context_features(
    global_state,
    controlled_team=0,
):
    """
    Build continuous and boolean centralized match context.

    Returns
    -------
    context_features:
        [39]
    """

    state = global_state.state
    clock = state.clock

    opponent_team = 1 - controlled_team

    # ============================================================
    # Score relative to the learning team
    # ============================================================

    relative_score = jnp.stack(
        [
            state.score[controlled_team],
            state.score[opponent_team],
        ]
    ).astype(jnp.float32)

    # ============================================================
    # Attack direction relative to learning team
    # ============================================================

    relative_attack_direction = jnp.stack(
        [
            state.attack_direction[controlled_team],
            state.attack_direction[opponent_team],
        ]
    ).astype(jnp.float32)

    features = [
        # --------------------------------------------------------
        # Match time
        # --------------------------------------------------------
        state.control_tick[None],

        clock.period_regulation_progress[None],
        clock.match_regulation_progress[None],
        clock.dead_ball_accrued_fraction[None],

        clock.added_time_active.astype(
            jnp.float32
        )[None],

        clock.added_time_elapsed_fraction[None],
        clock.added_time_remaining_fraction[None],
        clock.period_dead_ball_counter[None],
        clock.added_time_elapsed_counter[None],
        clock.added_time_remaining_counter[None],

        # --------------------------------------------------------
        # Ball
        # --------------------------------------------------------
        state.ball.position,                     # 3
        state.ball.velocity,                     # 3
        state.ball.spin,                         # 3

        state.ball.live.astype(
            jnp.float32
        )[None],

        # --------------------------------------------------------
        # Team orientation
        # --------------------------------------------------------
        relative_attack_direction,               # 2

        # --------------------------------------------------------
        # Possession
        # --------------------------------------------------------
        state.possession.control_ticks[None],

        state.possession.last_contact.kick_applied.astype(
            jnp.float32
        )[None],

        # --------------------------------------------------------
        # Restart
        # --------------------------------------------------------
        state.restart.substeps_remaining[None],

        state.restart.indirect.astype(
            jnp.float32
        )[None],

        state.restart.opened_control_tick[None],

        # --------------------------------------------------------
        # Score
        # --------------------------------------------------------
        relative_score,                          # 2

        # --------------------------------------------------------
        # Restart release
        # --------------------------------------------------------
        state.restart_release.active.astype(
            jnp.float32
        )[None],

        state.restart_release.untouched.astype(
            jnp.float32
        )[None],

        state.restart_release.indirect.astype(
            jnp.float32
        )[None],

        state.restart_release.law11_direct_exempt.astype(
            jnp.float32
        )[None],

        # --------------------------------------------------------
        # Additional match state
        # --------------------------------------------------------
        state.dead_ball_control_ticks[None],

        state.first_half_wall_end_tick[None],

        state.first_half_wall_end_known.astype(
            jnp.float32
        )[None],

        state.first_half_live_extension_ticks[None],

        state.penalty_completion_active.astype(
            jnp.float32
        )[None],

        state.restart_layout_ready.astype(
            jnp.float32
        )[None],
    ]

    return jnp.concatenate(
        features,
        axis=-1,
    )


def build_global_categorical_features(
    global_state,
    controlled_team=0,
):
    """
    Build one-hot categorical features for the centralized critic.

    Returns
    -------
    categorical_features:
        [92]
    """

    state = global_state.state

    def one_hot(value, num_classes):
        return jax.nn.one_hot(
            value,
            num_classes=num_classes,
            dtype=jnp.float32,
        )

    # ============================================================
    # 1. Match period
    #
    # 1 -> first period
    # 2 -> second period
    # ============================================================

    period = one_hot(
        state.clock.period - 1,
        2,
    )

    # ============================================================
    # 2. Team-valued categorical fields
    # ============================================================

    kickoff_team = _relative_team_one_hot(
        state.kickoff_team,
        controlled_team,
    )

    possession_team = _relative_team_one_hot(
        state.possession.team,
        controlled_team,
    )

    possession_previous_team = _relative_team_one_hot(
        state.possession.previous_team,
        controlled_team,
    )

    restart_team = _relative_team_one_hot(
        state.restart.team,
        controlled_team,
    )

    restart_release_team = _relative_team_one_hot(
        state.restart_release.team,
        controlled_team,
    )

    gk_backpass_team = _relative_team_one_hot(
        state.gk_backpass_team,
        controlled_team,
    )

    penalty_completion_team = _relative_team_one_hot(
        state.penalty_completion_team,
        controlled_team,
    )

    offside_direct_exempt_team = _relative_team_one_hot(
        global_state.offside.direct_exempt_team,
        controlled_team,
    )

    # ============================================================
    # 3. Last-contact categorical state
    # ============================================================

    last_contact = state.possession.last_contact

    last_contact_mechanism = one_hot(
        last_contact.mechanism,
        MECHANISM_COUNT,
    )

    last_contact_intent = one_hot(
        last_contact.intent,
        INTENT_COUNT,
    )

    last_contact_outcome = one_hot(
        last_contact.outcome,
        OUTCOME_COUNT,
    )

    last_contact_restart_kind = one_hot(
        last_contact.restart_kind,
        RESTART_COUNT,
    )

    last_contact_law11_effect = one_hot(
        last_contact.law11_effect,
        LAW11_EFFECT_COUNT,
    )

    last_contact_intent_source = one_hot(
        last_contact.intent_source,
        INTENT_SOURCE_COUNT,
    )

    # ============================================================
    # 4. Restart categories
    # ============================================================

    restart_kind = one_hot(
        state.restart.kind,
        RESTART_COUNT,
    )

    restart_release_kind = one_hot(
        state.restart_release.kind,
        RESTART_COUNT,
    )

    restart_release_mechanism = one_hot(
        state.restart_release.release_mechanism,
        MECHANISM_COUNT,
    )

    # ============================================================
    # 5. Concatenate
    # ============================================================

    categorical_features = jnp.concatenate(
        [
            period,                         # 2

            kickoff_team,                  # 3
            possession_team,               # 3
            possession_previous_team,      # 3

            last_contact_mechanism,        # 7
            last_contact_intent,           # 7
            last_contact_outcome,          # 10
            last_contact_restart_kind,     # 9
            last_contact_law11_effect,     # 5
            last_contact_intent_source,    # 3

            restart_kind,                  # 9
            restart_team,                  # 3

            restart_release_kind,          # 9
            restart_release_team,          # 3
            restart_release_mechanism,     # 7

            gk_backpass_team,              # 3
            penalty_completion_team,       # 3
            offside_direct_exempt_team,    # 3
        ],
        axis=-1,
    )

    return categorical_features


def encode_critic_state(
    global_state,
    controlled_team=0,
):
    """
    Build final centralized critic input.

    5v5:
        10 * 28 + 39 + 92 = 411

    Returns
    -------
    critic_input:
        [411] for 5v5
    """

    player_features = build_global_player_features(
        global_state,
        controlled_team,
    )

    context_features = build_global_context_features(
        global_state,
        controlled_team,
    )

    categorical_features = build_global_categorical_features(
        global_state,
        controlled_team,
    )

    player_flat = player_features.reshape(-1)

    critic_input = jnp.concatenate(
        [
            player_flat,
            context_features,
            categorical_features,
        ],
        axis=-1,
    )

    return critic_input