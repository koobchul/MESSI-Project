import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
import jax,jax.numpy as jnp,numpy as np
from project.models.action_head import ActorOutput
from project.models import action_sampling as s
from evaluation_gate import complete

def out(mu):return ActorOutput(jnp.zeros((1,6)),jnp.full((1,1),mu),jnp.full((1,1),-.5))
s.ENTROPY_MODE='gaussian'
g0=float(s.action_entropy(out(0.))[0]);g5=float(s.action_entropy(out(5.))[0]);assert g0==g5
s.ENTROPY_MODE='bounded'
b0=float(s.action_entropy(out(0.))[0]);b5=float(s.action_entropy(out(5.))[0]);assert b5<b0-7
assert float(jax.grad(lambda mu:s.action_entropy(out(mu)).sum())(5.))<0
assert float(jax.grad(lambda mu:s.action_entropy(out(mu)).sum())(-5.))>0
rng=np.random.default_rng(9);z=rng.normal(0,np.exp(-.5),200000);jac=2*(np.log(2)-z-np.logaddexp(0,-2*z));mc=np.log(6)+.5*np.log(2*np.pi*np.e)-.5+jac.mean();assert abs(b0-mc)<.01
assert not complete(0,4096);assert not complete(768,4096);assert not complete(1792,4096);assert complete(4096,4096)
assert not complete(4096,0,False,False);assert complete(64000,0,True,False)
r=dict(gaussian_entropy_mu0=g0,gaussian_entropy_mu5=g5,bounded_entropy_mu0=b0,bounded_entropy_mu5=b5,MC_mu0=mc,mean_gradient_points_inward=True,partial_eval_cases_rejected=True)
print(json.dumps(r));Path('entropy_checks.json').write_text(json.dumps(r,indent=2))
