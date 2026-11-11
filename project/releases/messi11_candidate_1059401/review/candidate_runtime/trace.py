from project.models.field_movement import field_movement
from footballworld.dynamics.contact_predicates import verified_controlled_carrier
from project.models.gk_movement import limit_gk_move
from project.models.role_actor import actor_forward
from project.envs.policy_obs import encode_policy_observation
"""SI pre-transition goalkeeper/ball traces, paired goalkeeper intervention."""
from project.evaluate_fulltime import *
from footballworld.core.contact import INTENT_SOURCE_POLICY, INTENT_SOURCE_ENVIRONMENT_FORCED
METRICS = METRICS + ("policy_shot_releases", "policy_pass_releases", "forced_shot_releases", "forced_pass_releases", "gk_verified_control_ticks", "field_verified_control_ticks")
FIELDS=['valid', 'tick', 'conceded_this_transition', 'score_for_before', 'score_against_before', 'gk_x_m', 'gk_y_m', 'ball_x_m', 'ball_y_m', 'ball_z_m', 'own_goal_x_m', 'attack_direction', 'gk_depth_from_goal_line_m', 'gk_distance_to_goal_center_m', 'gk_ball_distance_m', 'gk_outside_own_penalty_area', 'ball_live', 'restart_kind', 'gk_active', 'possession_team', 'gk_intent', 'gk_move_x', 'gk_move_y', 'gk_force_x', 'gk_force_y', 'score_for_after', 'score_against_after', 'radius_intervention', 'radius_restart_exception', 'verified_carrier', 'field_players_near_ball']

FIELDS += ["field_goal_line_fraction","field_width_m","field_depth_m","control_parameter_interventions"]

def make_trace(env, backbone, head, rollout_length=128, deterministic=True, rule_goalkeeper=False, random_policy=False):
    opponent = make_rule_based_policy(env)

    @jax.jit
    def collect(setup, carry, roster, si_roster, params, rollout_key, guard_enabled):
        actor_key, env_key = jax.random.split(rollout_key, 2)

        def scan_step(carry, _):
            def pause(_):
                return carry, (jnp.zeros(len(METRICS), dtype=jnp.int32), jnp.zeros(len(FIELDS),dtype=jnp.float32))

            def advance(_):
                state = carry.rollout.state
                tick = state.control_tick.astype(jnp.uint32)
                obs = encode_policy_observation(env, carry.rollout, roster)
                output = actor_forward(params, backbone, head, obs)
                output = apply_retouch_guard(output, obs, enabled=guard_enabled)
                sampled = sample_action(output, jax.random.fold_in(actor_key, tick))
                if random_policy:
                    k1,k2=jax.random.split(jax.random.fold_in(actor_key,tick))
                    logits=jnp.where(obs[:,-6:]>.5,0.,-1e9)
                    # Same legality/re-touch mask as learned policies, uniform legal intent.
                    uniform=output._replace(intent_logits=logits)
                    uniform=apply_retouch_guard(uniform,obs,enabled=guard_enabled)
                    intent=jax.random.categorical(k1,uniform.intent_logits)
                    continuous=jax.random.uniform(k2,(22,7),minval=-1.,maxval=1.)
                    sampled=sampled._replace(action=IntentAction.from_array(intent,continuous))
                if deterministic and not random_policy:
                    action_mean = IntentAction.from_array(jnp.argmax(output.intent_logits, axis=-1), jnp.tanh(output.continuous_mean))
                    sampled = sampled._replace(action=action_mean)
                si_obs = env.observe_all_si(carry.rollout)
                rule_step = opponent.step(si_obs, si_roster, carry.rule_state, carry.opponent_key)
                action = combine_team_actions(
                    sampled.action, rule_step.action, roster.team_id, controlled_team=0,
                )
                gk_mask = (roster.team_id == 0) & roster.is_goalkeeper
                gk = jnp.argmax(gk_mask)
                if rule_goalkeeper:
                    def choose(a,b):
                        mask = gk_mask.reshape(gk_mask.shape+(1,)*(a.ndim-1))
                        return jnp.where(mask,b,a)
                    action = jax.tree_util.tree_map(choose,action,rule_step.action)
                action = field_movement(state, action, rule_step.action)
                action, intervention, exception = limit_gk_move(env, state, action)
                result = env.step_with_events(
                    carry.rollout, setup, action, jax.random.fold_in(env_key, tick),
                )
                retouch = result.events.retouch
                event_actor = result.events.deliberate_contact.actor
                safe_actor = jnp.clip(event_actor, 0, roster.team_id.shape[0] - 1)
                trap = (
                    (event_actor >= 0) & (roster.team_id[safe_actor] == 0)
                    & (result.events.deliberate_contact.intent == INTENT_CONTROL)
                    & (result.events.deliberate_contact.outcome == OUTCOME_TRAP)
                )
                context = retouch_prohibited_from_mlp(obs) & (roster.team_id == 0)
                c = result.events.deliberate_contact
                own_release = (event_actor >= 0) & (roster.team_id[safe_actor] == 0) & (c.outcome == OUTCOME_RELEASE)
                verified = verified_controlled_carrier(state,env.ball,env.reach,env.action_scale)
                verified_own = (verified>=0)&(roster.team_id[jnp.clip(verified,0,21)]==0)&state.ball.live&(state.restart.kind==0)
                counts = jnp.asarray([
                    1,
                    state.possession.team == 0,
                    state.possession.team == 1,
                    jnp.sum(retouch.occurred & (retouch.offender>=0) & (roster.team_id[jnp.clip(retouch.offender,0,21)] == 0)),
                    jnp.sum(retouch.occurred & (retouch.offender>=0) & (roster.team_id[jnp.clip(retouch.offender,0,21)] == 1)),
                    jnp.sum(trap),
                    jnp.sum(trap & retouch.occurred & (retouch.offender == event_actor)),
                    jnp.sum(context),
                    jnp.sum(context & (sampled.action.intent != 0)),
                    ~state.ball.live,
                    result.restart_opened,
                    control_potentials(state, controlled_team=0).possession > 0,
                    control_potentials(state, controlled_team=1).possession > 0,
                    jnp.sum((sampled.action.intent == INTENT_SHOT) & (roster.team_id == 0) & state.players.active),
                    jnp.sum((event_actor >= 0) & (roster.team_id[safe_actor] == 0) & (result.events.deliberate_contact.intent == INTENT_SHOT) & (result.events.deliberate_contact.outcome == OUTCOME_RELEASE)),
                    jnp.sum((sampled.action.intent == INTENT_PASS) & (roster.team_id == 0) & state.players.active),
                    jnp.sum((event_actor >= 0) & (roster.team_id[safe_actor] == 0) & (result.events.deliberate_contact.intent == INTENT_PASS) & (result.events.deliberate_contact.outcome == OUTCOME_RELEASE)),
                    jnp.sum(own_release & (c.intent == INTENT_SHOT) & (c.intent_source == INTENT_SOURCE_POLICY)),
                    jnp.sum(own_release & (c.intent == INTENT_PASS) & (c.intent_source == INTENT_SOURCE_POLICY)),
                    jnp.sum(own_release & (c.intent == INTENT_SHOT) & (c.intent_source == INTENT_SOURCE_ENVIRONMENT_FORCED)),
                    jnp.sum(own_release & (c.intent == INTENT_PASS) & (c.intent_source == INTENT_SOURCE_ENVIRONMENT_FORCED)),
                    verified_own & (verified==gk),
                    verified_own & (verified!=gk),
                ], dtype=jnp.int32)
                next_carry = carry._replace(
                    rollout=result.rollout, rule_state=rule_step.state,
                    terminated=result.terminated, truncated=result.truncated, done=result.done,
                )
                pos=state.players.position[gk]
                ball=state.ball.position
                direction=state.attack_direction[0]
                goal_x=-direction*env.stadium.half_length
                depth=(pos[0]-goal_x)*direction
                outside=(depth<0) | (depth>env.stadium.penalty_area_length) | (jnp.abs(pos[1])>env.stadium.penalty_area_width/2)
                row=jnp.asarray([
                    1,state.control_tick,result.rollout.state.score[1]-state.score[1],
                    state.score[0],state.score[1],pos[0],pos[1],ball[0],ball[1],ball[2],
                    goal_x,direction,depth,jnp.sqrt((pos[0]-goal_x)**2+pos[1]**2),
                    jnp.linalg.norm(pos-ball[:2]),outside,state.ball.live,state.restart.kind,
                    state.players.active[gk],state.possession.team,
                    action.intent[gk],action.move[gk,0],action.move[gk,1],
                    action.force_to_ball[gk,0],action.force_to_ball[gk,1],
                    result.rollout.state.score[0],result.rollout.state.score[1],intervention,exception,verified,
                    jnp.sum((roster.team_id==0)&~roster.is_goalkeeper&state.players.active&(jnp.linalg.norm(state.players.position-state.ball.position[:2],axis=-1)<8.)),
                ],dtype=jnp.float32)
                fp=state.players.position[1:11]
                row=jnp.concatenate([row,jnp.asarray([jnp.mean(jnp.abs(fp[:,0])>49.5),jnp.ptp(fp[:,1]),jnp.ptp(fp[:,0]),jnp.sum(jnp.any(jnp.abs(jnp.concatenate([action.force_to_ball[1:11]-sampled.action.force_to_ball[1:11],(action.launch[1:11]-sampled.action.launch[1:11])[:,None],action.spin[1:11]-sampled.action.spin[1:11]],axis=-1))>1e-6,axis=-1))])])
                return next_carry, (counts,row)

            return jax.lax.cond(carry.done, pause, advance, operand=None)

        carry, (counts, rows) = jax.lax.scan(scan_step, carry, None, length=rollout_length)
        return carry, jnp.sum(counts, axis=0), rows

    return collect

