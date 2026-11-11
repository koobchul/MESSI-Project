"""Coverage, balanced replay, and near-ball movement diagnostics for 5v5 BC."""
import numpy as np
import jax
import jax.numpy as jnp
import optax
from project.models.intent_guard import apply_retouch_guard


def own_team_first(obs, team_id):
    """Canonicalize entity order for teacher team1; attack-frame values already rotated.

    Context and categorical features are observer-relative. Only the entity slot
    order changes. Team0's inference observation remains exactly unchanged.
    """
    n = team_id.shape[0]
    if obs.shape[-1] != n * 30 + 114 or n != 10:
        raise ValueError('Repair supports the supplied 5v5 encoder only')
    order = jnp.argsort(team_id[None, :] != team_id[:, None], axis=1, stable=True)
    entities = obs[:, :n * 30].reshape(n, n, 30)
    reordered = jnp.take_along_axis(entities, order[..., None], axis=1)
    return jnp.concatenate([reordered.reshape(n, n*30), obs[:, n*30:]], axis=1)


def near_mask(obs):
    # Fixed normalized-feature neighbourhood, not an invented metre distance.
    return np.linalg.norm(obs[:, 304:306], axis=1) < .05


def replay_pools(obs, intent):
    near = near_mask(obs)
    masks = [(intent == 0) & ~near, (intent == 0) & near] + [intent == c for c in range(1,6)]
    return [np.flatnonzero(m) for m in masks]


def balanced_indices(rng, pools, size):
    # Retain plenty of MOVE context; avoid extreme uniform six-class balancing.
    weights = np.array([.25,.25,.25,.15,.05,.02,.03])
    weights *= np.array([len(p)>0 for p in pools]); weights /= weights.sum()
    buckets = rng.choice(len(pools), size=size, p=weights)
    result = np.empty(size, dtype=np.int64)
    for k, pool in enumerate(pools):
        mask = buckets == k
        if mask.any(): result[mask] = rng.choice(pool, size=int(mask.sum()))
    return result


def apply_actor(params, obs, backbone, head):
    return apply_retouch_guard(head.apply({'params':params['head']},
                               backbone.apply({'params':params['backbone']},obs)), obs)


def repair_loss(params, obs, intent, target, backbone, head):
    output = apply_actor(params,obs,backbone,head)
    mean = jnp.tanh(output.continuous_mean)
    ce = optax.softmax_cross_entropy_with_integer_labels(output.intent_logits,intent)
    move_mse = jnp.mean((mean[:,:2]-target[:,:2])**2,axis=1)
    # Unlike v1, movement is not diluted by the five ball-control coordinates.
    # MOVE uses force_to_ball as body aim; these two axes are ALWAYS supervised.
    force_mse = jnp.mean((mean[:,2:4]-target[:,2:4])**2,axis=1)
    contact_mse = jnp.mean((mean[:,4:]-target[:,4:])**2,axis=1) * (intent != 0)
    target_norm = jnp.sqrt(jnp.sum(target[:,:2]**2,axis=1)+1e-8)
    mean_norm = jnp.sqrt(jnp.sum(mean[:,:2]**2,axis=1)+1e-4)
    cosine = jnp.sum(mean[:,:2]*target[:,:2],axis=1)/(target_norm*mean_norm)
    direction = jnp.where(target_norm > .25,1.-cosine,0.)
    return jnp.mean(ce + 4.*move_mse + 2.*force_mse + contact_mse + .5*direction)


def diagnostics(params, data, backbone, head):
    obs,intent,target = data
    @jax.jit
    def predict(x):
        o=apply_actor(params,x,backbone,head)
        return jax.nn.softmax(o.intent_logits),jnp.tanh(o.continuous_mean)
    probs=[];means=[]
    for i in range(0,len(obs),1024):
        pr,m=predict(obs[i:i+1024]);probs.append(np.asarray(pr));means.append(np.asarray(m))
    probs=np.concatenate(probs);mean=np.concatenate(means);pred=probs.argmax(1)
    confusion=np.zeros((6,6),int);np.add.at(confusion,(intent,pred),1)
    near=near_mask(obs);moving=np.linalg.norm(target[:,:2],axis=1)>.25
    dot=np.sum(target[:,:2]*mean[:,:2],axis=1)
    mask=near & moving
    aim_mask=(intent==0) & near & (np.linalg.norm(target[:,2:4],axis=1)>.1)
    aim_dot=np.sum(target[:,2:4]*mean[:,2:4],axis=1)
    recall=[float(confusion[i,i]/max(1,confusion[i].sum())) for i in range(6)]
    return dict(samples=len(obs),true_counts=np.bincount(intent,minlength=6).tolist(),
                predicted_counts=np.bincount(pred,minlength=6).tolist(),confusion=confusion.tolist(),recall=recall,
                near_move_rmse=float(np.sqrt(np.mean((target[near,:2]-mean[near,:2])**2))) if near.any() else None,
                near_opposite_fraction=float(np.mean(dot[mask]<0)) if mask.any() else None,
                near_moving_samples=int(mask.sum()),
                move_body_aim_rmse=float(np.sqrt(np.mean((target[intent==0,2:4]-mean[intent==0,2:4])**2))),
                near_body_aim_opposite_fraction=float(np.mean(aim_dot[aim_mask]<0)) if aim_mask.any() else None,
                control_move_rmse=float(np.sqrt(np.mean((target[intent==1,:2]-mean[intent==1,:2])**2))) if (intent==1).any() else None)


def coverage_ok(train, validation):
    t=np.bincount(train[1],minlength=6);v=np.bincount(validation[1],minlength=6)
    # Explicit engineering gates, not a statistical sufficiency theorem.
    return bool(t[1]>=200 and t[2]>=100 and t[3]>=16 and v[1]>=50 and v[2]>=20 and v[3]>=4)


def offline_gate(before, after):
    # Gate the confirmed body-aim repair while preventing large movement regressions.
    return bool(after['near_move_rmse'] is not None and before['near_move_rmse'] is not None
                and after['near_body_aim_opposite_fraction'] is not None
                and before['near_body_aim_opposite_fraction'] is not None
                and after['move_body_aim_rmse'] <= .75 * before['move_body_aim_rmse']
                and after['near_body_aim_opposite_fraction'] <= .5 * before['near_body_aim_opposite_fraction']
                and after['near_move_rmse'] <= 1.05 * before['near_move_rmse']
                and after['recall'][1] >= .5 and after['recall'][2] >= .3 and after['recall'][3] >= .25)
