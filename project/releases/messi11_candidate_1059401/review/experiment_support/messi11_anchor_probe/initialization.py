def shot_initialization(params,boost=5.):
    from flax.core import unfreeze
    import jax
    import jax.numpy as jnp
    p=unfreeze(params)
    p=jax.tree_util.tree_map(lambda x:jnp.array(x),p)
    bias=p['field']['head']['intent_head']['bias']
    if bias.shape!=(6,):raise ValueError('Expected six intent biases')
    p['field']['head']['intent_head']['bias']=bias.at[3].add(boost)
    return p
