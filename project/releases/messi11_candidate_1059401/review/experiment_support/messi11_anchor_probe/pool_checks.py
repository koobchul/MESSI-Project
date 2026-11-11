def check_pool(e):
 import jax,numpy as np
 import learned_pool
 from footballworld.policies.rule_based import make_rule_based_policy
 p=make_rule_based_policy(e.env);si=e.env.observe_all_si(e.reset.rollout)
 state=p.initialize(si,e.si);step=p.step(si,e.si,state,jax.random.key(909))
 checked=[]
 for idx in (1,2,3):
  learned_pool.set_eval(idx)
  ctrl=learned_pool.make_controller(e.env,e.b,e.h)
  action=jax.jit(ctrl)(e.reset.rollout,e.roster,step.action,jax.random.key(908),e.reset.rollout.state.control_tick)
  for original,actual in zip(jax.tree_util.tree_leaves(step.action),jax.tree_util.tree_leaves(action)):
   x=np.asarray(actual);np.testing.assert_array_equal(np.asarray(original)[:11],x[:11])
   assert np.isfinite(x).all()
  for leaf in (action.move,action.force_to_ball,action.launch,action.spin):assert np.max(np.abs(np.asarray(leaf)))<=1.000001
  assert np.all((np.asarray(action.intent)>=0)&(np.asarray(action.intent)<6))
  checked.append(idx)
 learned_pool.set_eval(None)
 return dict(passed=True,opponents_checked=checked,HOME_rule_rows_unchanged=True,AWAY_actions_finite_and_bounded=True,
  observation='AWAY player feature blocks reordered own11-first; context/categories retain observer-relative semantics',
  limitation='Adapter contract smoke check, not full side-swap equivalence or student-policy compatibility proof')
