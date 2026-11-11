"""Explicit diagnostic action adapter, not a trained policy. Array backend injectable."""
def box_shot(env, state, action, obs, carrier, normal_possession, enabled, xp=None, intent_only=False):
    if xp is None:
        import jax.numpy as xp
    slots=xp.arange(action.intent.shape[0])
    direction=state.attack_direction[0]
    depth=env.stadium.half_length-direction*state.ball.position[0]
    # SI world coordinates; no hardcoded pitch size or halftime sign.
    inside=(depth>0)&(depth<=env.stadium.penalty_area_length)&(xp.abs(state.ball.position[1])<=env.stadium.penalty_area_width/2)
    legal=obs[:,-6:][:,3]>.5  # documented six-intent order, SHOT=3
    eligible=((slots==carrier)&(state.players.team_id==0)&~state.players.is_goalkeeper&state.players.active
              &legal&normal_possession&state.ball.live&(state.restart.kind==0)&inside&(state.ball.position[2]<=1.2))
    inject=eligible & enabled & (action.intent!=3)
    if not enabled:
        return action,eligible,inject
    if intent_only:
        return action._replace(intent=xp.where(inject,3,action.intent)),eligible,inject
    target=xp.asarray([direction*env.stadium.half_length,0.],dtype=xp.float32)
    delta=target-state.ball.position[:2]
    # Full-power L-infinity radial encoding, aimed at goal center (no keeper avoidance).
    force=delta/xp.maximum(xp.max(xp.abs(delta)),1e-6)
    height=xp.clip((state.ball.position[2]-env.ball.radius)/max(env.action_scale.ground_launch_down_reference_height_m-env.ball.radius,1e-6),0.,1.)
    floor=-(env.action_scale.ground_launch_down_max_radians+(env.action_scale.launch_max_radians-env.action_scale.ground_launch_down_max_radians)*height)
    # Physical launch angle +0.05 rad, converted to the environment's signed control.
    launch=xp.clip(2*(.05-floor)/xp.maximum(env.action_scale.launch_max_radians-floor,1e-6)-1,-1.,1.)
    final=action._replace(intent=xp.where(inject,3,action.intent),
        force_to_ball=xp.where(inject[:,None],force,action.force_to_ball),
        launch=xp.where(inject,launch,action.launch),spin=xp.where(inject[:,None],0.,action.spin))
    return final,eligible,inject
