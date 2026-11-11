def eligible(normal_field,control_ticks,carrier,actors,intents,sources,outcomes,shot,policy,release,xp=None):
 if xp is None:
  import jax.numpy as xp
 same_release=xp.any((actors==carrier)&(carrier>=0)&(intents==shot)&(sources==policy)&(outcomes==release))
 return normal_field&(control_ticks>=3)&same_release
