import math

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


# ============================================================
# Feature dimensions
# ============================================================

PLAYER_FEATURE_DIM = 30
CONTEXT_FEATURE_DIM = 34
CATEGORICAL_FEATURE_DIM = 80


# ============================================================
# Diagnostic utilities
# ============================================================

def _feature_dim_per_observer(shape, num_observers):
    """
    Observation leaf가 observer 축을 첫 번째 축으로 가지는 경우,
    observer 한 명 기준 scalar 개수를 계산한다.
    """
    if len(shape) == 0:
        return None

    if shape[0] != num_observers:
        return None

    if len(shape) == 1:
        return 1

    return math.prod(shape[1:])


def describe_observation(obs):
    """
    MESSI observation PyTree의 모든 leaf에 대해

    - field path
    - shape
    - dtype
    - observer 한 명 기준 scalar feature 개수

    를 출력한다.

    실제 neural-network encoder가 아니라
    observation schema 분석용 diagnostic utility이다.
    """

    num_observers = int(obs.valid.shape[0])

    leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(obs)

    print("\n" + "=" * 90)
    print("MESSI OBSERVATION STRUCTURE")
    print("=" * 90)
    print(f"Number of observers: {num_observers}\n")

    total_float_dim = 0
    total_int_dim = 0
    total_bool_dim = 0

    for path, value in leaves_with_path:
        array = jnp.asarray(value)

        path_str = jax.tree_util.keystr(path)

        feature_dim = _feature_dim_per_observer(
            array.shape,
            num_observers,
        )

        if jnp.issubdtype(array.dtype, jnp.floating):
            dtype_group = "float"

            if feature_dim is not None:
                total_float_dim += feature_dim

        elif jnp.issubdtype(array.dtype, jnp.integer):
            dtype_group = "int"

            if feature_dim is not None:
                total_int_dim += feature_dim

        elif jnp.issubdtype(array.dtype, jnp.bool_):
            dtype_group = "bool"

            if feature_dim is not None:
                total_bool_dim += feature_dim

        else:
            dtype_group = "other"

        feature_text = (
            str(feature_dim)
            if feature_dim is not None
            else "-"
        )

        print(
            f"{path_str:55s} "
            f"shape={str(array.shape):18s} "
            f"dtype={str(array.dtype):10s} "
            f"group={dtype_group:6s} "
            f"per_observer={feature_text}"
        )

    print("\n" + "-" * 90)
    print("PER-OBSERVER RAW FEATURE COUNT")
    print("-" * 90)

    print(f"float features : {total_float_dim}")
    print(f"int features   : {total_int_dim}")
    print(f"bool features  : {total_bool_dim}")

    print(
        "raw total      :",
        total_float_dim
        + total_int_dim
        + total_bool_dim,
    )

    print("=" * 90)

    return {
        "num_observers": num_observers,
        "float_dim": total_float_dim,
        "int_dim": total_int_dim,
        "bool_dim": total_bool_dim,
        "raw_total_dim": (
            total_float_dim
            + total_int_dim
            + total_bool_dim
        ),
    }


# ============================================================
# Shared helper functions
# ============================================================

def _observer_team(obs, roster):
    """
    Return each observer's team ID safely.

    obs.self_state.player_index is normally valid for observe_all(),
    but clipping makes the encoder robust to invalid observation rows.
    """

    player_index = obs.self_state.player_index

    safe_player_index = jnp.clip(
        player_index,
        0,
        roster.team_id.shape[0] - 1,
    )

    return roster.team_id[safe_player_index]


def _relative_team_one_hot(team, observer_team):
    """
    Convert an absolute MESSI team ID into observer-relative semantics.

    Output:
        [1, 0, 0] = no team
        [0, 1, 0] = observer's own team
        [0, 0, 1] = opponent team
    """

    no_team = team == NO_TEAM

    own_team = (
        (~no_team)
        & (team == observer_team)
    )

    opponent_team = (
        (~no_team)
        & (team != observer_team)
    )

    return jnp.stack(
        [
            no_team,
            own_team,
            opponent_team,
        ],
        axis=-1,
    ).astype(jnp.float32)


# ============================================================
# Player entity encoder
# ============================================================

def build_player_features(obs, roster):
    """
    Build player-entity features shared by the MLP and Transformer.

    Returns
    -------
    player_features:
        [num_observers, num_players, PLAYER_FEATURE_DIM]

    5v5:
        [10, 10, 30]

    11v11:
        [22, 22, 30]
    """

    num_observers = obs.valid.shape[0]
    num_players = roster.team_id.shape[0]

    # --------------------------------------------------------
    # 1. Continuous player state
    # --------------------------------------------------------

    continuous_features = [
        # Position / velocity relative to observer
        obs.players.relative_position,       # [..., 2]
        obs.players.relative_velocity,       # [..., 2]

        # Body direction
        obs.players.facing_sin[..., None],
        obs.players.facing_cos[..., None],

        # Stamina
        obs.players.stamina_long[..., None],
        obs.players.stamina_short[..., None],

        # Recovery / lock states
        obs.players.challenge_recovery_substeps[..., None],
        obs.players.contact_lock_substeps[..., None],
        obs.players.aerial_recovery_substeps[..., None],
        obs.players.possession_loss_lock_substeps[..., None],
    ]

    # --------------------------------------------------------
    # 2. Player status
    # --------------------------------------------------------

    status_features = [
        # Numeric count
        obs.players.yellow_cards.astype(jnp.float32)[..., None],

        # Boolean player states
        obs.players.on_pitch.astype(jnp.float32)[..., None],
        obs.players.sent_off.astype(jnp.float32)[..., None],
        obs.players.offside.astype(jnp.float32)[..., None],
        obs.players.possessor.astype(jnp.float32)[..., None],
        obs.players.restart_taker.astype(jnp.float32)[..., None],
        obs.players.release_taker.astype(jnp.float32)[..., None],
        obs.players.last_actor.astype(jnp.float32)[..., None],
        obs.players.contact_may_occur_this_frame.astype(
            jnp.float32
        )[..., None],
    ]

    # --------------------------------------------------------
    # 3. Roster metadata
    # --------------------------------------------------------

    def broadcast_roster_feature(x):
        x = jnp.asarray(
            x,
            dtype=jnp.float32,
        )

        return jnp.broadcast_to(
            x[None, :, None],
            (
                num_observers,
                num_players,
                1,
            ),
        )

    roster_features = [
        broadcast_roster_feature(
            roster.is_goalkeeper
        ),
        broadcast_roster_feature(
            roster.max_speed
        ),
        broadcast_roster_feature(
            roster.reach_height
        ),
        broadcast_roster_feature(
            roster.height
        ),
        broadcast_roster_feature(
            roster.ball_control
        ),
        broadcast_roster_feature(
            roster.endurance_factor
        ),
    ]

    # --------------------------------------------------------
    # 4. Observer-player relationship
    #
    # Do not expose raw player_id or team_id.
    # --------------------------------------------------------

    observer_team = _observer_team(
        obs,
        roster,
    )

    player_slots = jnp.arange(
        num_players,
        dtype=jnp.int32,
    )[None, :]

    observer_slots = jnp.clip(
        obs.self_state.player_index,
        0,
        num_players - 1,
    )[:, None]

    is_self = (
        player_slots
        == observer_slots
    )

    is_same_team = (
        roster.team_id[None, :]
        == observer_team[:, None]
    )

    is_teammate = (
        is_same_team
        & (~is_self)
    )

    is_opponent = (
        ~is_same_team
    )

    relationship_features = [
        is_self.astype(
            jnp.float32
        )[..., None],

        is_teammate.astype(
            jnp.float32
        )[..., None],

        is_opponent.astype(
            jnp.float32
        )[..., None],
    ]

    # --------------------------------------------------------
    # 5. Concatenate
    # --------------------------------------------------------

    player_features = jnp.concatenate(
        continuous_features
        + status_features
        + roster_features
        + relationship_features,
        axis=-1,
    )

    # Invalid observer rows must contain no usable information.
    valid_mask = obs.valid.astype(
        jnp.float32
    )[:, None, None]

    player_features = (
        player_features
        * valid_mask
    )

    return player_features


# ============================================================
# Context encoder
# ============================================================

def build_context_features(obs, roster):
    """
    Build non-player continuous / boolean context features.

    Returns
    -------
    context_features:
        [num_observers, CONTEXT_FEATURE_DIM]

    Current dimension:
        34

    Contains:
        - observer's own state
        - ball state
        - possession timing
        - restart state
        - restart-release state
        - observer-relative score
        - match time
    """

    observer_team = _observer_team(
        obs,
        roster,
    )

    # --------------------------------------------------------
    # Observer-relative score
    #
    # MESSI score is [team_0_score, team_1_score].
    # Convert it to:
    #
    # [own_score, opponent_score]
    # --------------------------------------------------------

    score = obs.match.score.astype(
        jnp.float32
    )

    safe_team = jnp.clip(
        observer_team,
        0,
        1,
    )

    own_score = jnp.take_along_axis(
        score,
        safe_team[:, None],
        axis=1,
    )

    opponent_team = (
        1 - safe_team
    )

    opponent_score = jnp.take_along_axis(
        score,
        opponent_team[:, None],
        axis=1,
    )

    relative_score = jnp.concatenate(
        [
            own_score,
            opponent_score,
        ],
        axis=-1,
    )

    features = [
        # ----------------------------------------------------
        # 1. Observer's own state
        # ----------------------------------------------------

        obs.self_state.position,       # [N, 2]
        obs.self_state.velocity,       # [N, 2]

        # ----------------------------------------------------
        # 2. Ball
        # ----------------------------------------------------

        obs.ball.relative_state,       # [N, 9]

        obs.ball.live.astype(
            jnp.float32
        )[..., None],                  # [N, 1]

        # ----------------------------------------------------
        # 3. Possession
        # ----------------------------------------------------

        obs.possession.control_ticks[
            ..., None
        ],

        obs.possession.last_contact.kick_applied.astype(
            jnp.float32
        )[..., None],

        # ----------------------------------------------------
        # 4. Restart
        # ----------------------------------------------------

        obs.restart.substeps_remaining[
            ..., None
        ],

        obs.restart.indirect.astype(
            jnp.float32
        )[..., None],

        # ----------------------------------------------------
        # 5. Restart release
        # ----------------------------------------------------

        obs.restart_release.active.astype(
            jnp.float32
        )[..., None],

        obs.restart_release.untouched.astype(
            jnp.float32
        )[..., None],

        obs.restart_release.indirect.astype(
            jnp.float32
        )[..., None],

        obs.restart_release.law11_direct_exempt.astype(
            jnp.float32
        )[..., None],

        # ----------------------------------------------------
        # 6. Match state
        # ----------------------------------------------------

        relative_score,               # [N, 2]

        obs.match.control_tick[
            ..., None
        ],

        # ----------------------------------------------------
        # 7. Match clock
        # ----------------------------------------------------

        obs.match.clock.period_regulation_progress[
            ..., None
        ],

        obs.match.clock.match_regulation_progress[
            ..., None
        ],

        obs.match.clock.dead_ball_accrued_fraction[
            ..., None
        ],

        obs.match.clock.added_time_active.astype(
            jnp.float32
        )[..., None],

        obs.match.clock.added_time_elapsed_fraction[
            ..., None
        ],

        obs.match.clock.added_time_remaining_fraction[
            ..., None
        ],

        obs.match.clock.period_dead_ball_counter[
            ..., None
        ],

        obs.match.clock.added_time_elapsed_counter[
            ..., None
        ],

        obs.match.clock.added_time_remaining_counter[
            ..., None
        ],
    ]

    context_features = jnp.concatenate(
        features,
        axis=-1,
    )

    valid_mask = obs.valid.astype(
        jnp.float32
    )[:, None]

    context_features = (
        context_features
        * valid_mask
    )

    return context_features


# ============================================================
# Categorical encoder
# ============================================================

def build_categorical_features(obs, roster):
    """
    Encode categorical observation fields using one-hot encoding.

    Returns
    -------
    categorical_features:
        [num_observers, CATEGORICAL_FEATURE_DIM]

    Current dimension:
        80

    Team-related categories use observer-relative semantics:

        [no team, own team, opponent team]

    Raw integer category IDs are never interpreted as continuous
    magnitudes.
    """

    observer_team = _observer_team(
        obs,
        roster,
    )

    # --------------------------------------------------------
    # Helper
    # --------------------------------------------------------

    def one_hot(
        value,
        num_classes,
    ):
        return jax.nn.one_hot(
            value,
            num_classes=num_classes,
            dtype=jnp.float32,
        )

    # --------------------------------------------------------
    # 1. Possession
    # --------------------------------------------------------

    possession_previous_team = (
        _relative_team_one_hot(
            obs.possession.previous_team,
            observer_team,
        )
    )

    last_contact_mechanism = one_hot(
        obs.possession.last_contact.mechanism,
        MECHANISM_COUNT,
    )

    last_contact_intent = one_hot(
        obs.possession.last_contact.intent,
        INTENT_COUNT,
    )

    last_contact_intent_source = one_hot(
        obs.possession.last_contact.intent_source,
        INTENT_SOURCE_COUNT,
    )

    last_contact_outcome = one_hot(
        obs.possession.last_contact.outcome,
        OUTCOME_COUNT,
    )

    last_contact_restart_kind = one_hot(
        obs.possession.last_contact.restart_kind,
        RESTART_COUNT,
    )

    last_contact_law11_effect = one_hot(
        obs.possession.last_contact.law11_effect,
        LAW11_EFFECT_COUNT,
    )

    # --------------------------------------------------------
    # 2. Current restart
    # --------------------------------------------------------

    restart_kind = one_hot(
        obs.restart.kind,
        RESTART_COUNT,
    )

    restart_team = (
        _relative_team_one_hot(
            obs.restart.team,
            observer_team,
        )
    )

    # --------------------------------------------------------
    # 3. Restart-release provenance
    # --------------------------------------------------------

    restart_release_kind = one_hot(
        obs.restart_release.kind,
        RESTART_COUNT,
    )

    restart_release_mechanism = one_hot(
        obs.restart_release.release_mechanism,
        MECHANISM_COUNT,
    )

    # --------------------------------------------------------
    # 4. Match-level team categories
    # --------------------------------------------------------

    offside_direct_exempt_team = (
        _relative_team_one_hot(
            obs.match.offside_direct_exempt_team,
            observer_team,
        )
    )

    gk_handling_restricted_team = (
        _relative_team_one_hot(
            obs.match.gk_handling_restricted_team,
            observer_team,
        )
    )

    # --------------------------------------------------------
    # 5. Match period
    #
    # MESSI:
    #
    #   0 = invalid observation sentinel
    #   1 = period 1
    #   2 = period 2
    #
    # Only 1 and 2 are real period categories.
    #
    # period 1 -> index 0 -> [1, 0]
    # period 2 -> index 1 -> [0, 1]
    #
    # invalid 0 -> index -1 -> [0, 0]
    # --------------------------------------------------------

    match_period = one_hot(
        obs.match.clock.period - 1,
        2,
    )

    # --------------------------------------------------------
    # 6. Concatenate
    # --------------------------------------------------------

    categorical_features = jnp.concatenate(
        [
            possession_previous_team,          # 3
            last_contact_mechanism,            # 7
            last_contact_intent,               # 7
            last_contact_intent_source,        # 3
            last_contact_outcome,              # 10
            last_contact_restart_kind,         # 9
            last_contact_law11_effect,         # 5
            restart_kind,                      # 9
            restart_team,                      # 3
            restart_release_kind,              # 9
            restart_release_mechanism,         # 7
            offside_direct_exempt_team,        # 3
            gk_handling_restricted_team,       # 3
            match_period,                      # 2
        ],
        axis=-1,
    )

    valid_mask = obs.valid.astype(
        jnp.float32
    )[:, None]

    categorical_features = (
        categorical_features
        * valid_mask
    )

    return categorical_features


# ============================================================
# MLP observation encoder
# ============================================================

def encode_mlp_observation(obs, roster):
    """
    Build the final flat observation for the MLP actor.

    Player entity structure is flattened only here.

    Returns
    -------
    mlp_obs:
        [num_observers, input_dim]

    5v5:
        num_players = 10

        10 * 30
        + 34
        + 80
        = 414

        shape = [10, 414]

    11v11:
        num_players = 22

        22 * 30
        + 34
        + 80
        = 774

        shape = [22, 774]
    """

    player_features = build_player_features(
        obs,
        roster,
    )

    context_features = build_context_features(
        obs,
        roster,
    )

    categorical_features = build_categorical_features(
        obs,
        roster,
    )

    # --------------------------------------------------------
    # [N, num_players, 30]
    # ->
    # [N, num_players * 30]
    # --------------------------------------------------------

    player_flat = player_features.reshape(
        player_features.shape[0],
        -1,
    )

    # --------------------------------------------------------
    # Final MLP observation
    # --------------------------------------------------------

    mlp_obs = jnp.concatenate(
        [
            player_flat,
            context_features,
            categorical_features,
        ],
        axis=-1,
    )

    return mlp_obs


def mlp_input_dim(num_players):
    """
    Return the MLP actor input dimension for a given roster size.

    Examples
    --------
    5v5:
        num_players = 10
        -> 414

    11v11:
        num_players = 22
        -> 774
    """

    return (
        num_players * PLAYER_FEATURE_DIM
        + CONTEXT_FEATURE_DIM
        + CATEGORICAL_FEATURE_DIM
    )