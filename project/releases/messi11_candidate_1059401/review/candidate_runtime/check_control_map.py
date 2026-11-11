"""Leakage and action-contract checks for the additional CONTROL mapping."""
from types import SimpleNamespace
import jax.numpy as jnp,numpy as np
from footballworld.core.action import IntentAction
from project.models import field_movement as m
m.CONTACT_MODE='settle'
s=SimpleNamespace(players=SimpleNamespace(team_id=jnp.repeat(jnp.arange(2),11),is_goalkeeper=jnp.array([True]+[False]*10+[True]+[False]*10),active=jnp.ones(22,dtype=bool)),ball=SimpleNamespace(live=jnp.bool_(True)),restart=SimpleNamespace(kind=jnp.int32(0)))
raw=IntentAction.from_array(jnp.ones(22,dtype=jnp.int32),jnp.full((22,7),.6))
rule=IntentAction.from_array(jnp.ones(22,dtype=jnp.int32),jnp.full((22,7),-.1))
a=m.field_movement(s,raw,rule)
for key in ('force_to_ball','launch','spin'):
 x=np.asarray(getattr(a,key));r=np.asarray(getattr(raw,key));np.testing.assert_array_equal(x[[0,*range(11,22)]],r[[0,*range(11,22)]])
 assert np.max(np.abs(x))<=1
np.testing.assert_array_equal(a.intent,raw.intent)
# PASS/SHOT and a policy/rule intent disagreement retain learned contact parameters.
for intent in (2,3,0):
 b=m.field_movement(s,raw._replace(intent=raw.intent.at[5].set(intent)),rule)
 np.testing.assert_array_equal(b.force_to_ball[5],raw.force_to_ball[5])
b=m.field_movement(s,raw,rule._replace(intent=rule.intent.at[5].set(0)))
np.testing.assert_array_equal(b.force_to_ball[5],raw.force_to_ball[5])
s.restart.kind=jnp.int32(1);b=m.field_movement(s,raw,rule)
np.testing.assert_array_equal(b.force_to_ball,raw.force_to_ball)
print('CONTROL map contracts passed: intent, GK, AWAY, other intents, mismatch, restarts, ranges')
