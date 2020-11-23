from collections import namedtuple
import copy

from jax import device_put, lax, random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.handlers import substitute, trace, seed
from numpyro.infer import MCMC, NUTS, log_likelihood
from numpyro.infer.mcmc import MCMCKernel
from numpyro.util import identity

HMC_ECS_State = namedtuple("HMC_ECS_State", "uz, hmc_state, accept_prob, rng_key")
"""
 - **uz** - a dict of current subsample indices and the current latent values
 - **hmc_state** - current hmc_state
 - **accept_prob** - acceptance probability of the proposal subsample indices
 - **rng_key** - random key to generate new subsample indices
"""

def _wrap_model(model):
    def fn(*args, **kwargs):
        subsample_values = kwargs.pop("_subsample_sites", {})
        with substitute(data=subsample_values):
            model(*args, **kwargs)
    return fn


class HMC_ECS(MCMCKernel):
    sample_field = "uz"

    def __init__(self, inner_kernel):
        self.inner_kernel = copy.copy(inner_kernel)
        self.inner_kernel._model = _wrap_model(inner_kernel.model)
        self._plate_sizes = None

    @property
    def model(self):
        return self.inner_kernel._model

    def postprocess_fn(self, args, kwargs):
        def fn(uz):
            z = {k: v for k, v in uz.items() if k not in self._plate_sizes}
            return self.inner_kernel.postprocess_fn(args, kwargs)(z)

        return fn

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        model_kwargs = {} if model_kwargs is None else model_kwargs.copy()
        rng_key, key_u, key_z = random.split(rng_key, 3)
        prototype_trace = trace(seed(self.model, key_u)).get_trace(*model_args, **model_kwargs)
        u = {name: site["value"] for name, site in prototype_trace.items()
             if site["type"] == "plate" and site["args"][0] > site["args"][1]}
        self._plate_sizes = {name: prototype_trace[name]["args"] for name in u}
        model_kwargs["_subsample_sites"] = u
        hmc_state = self.inner_kernel.init(key_z, num_warmup, init_params,
                                           model_args, model_kwargs)
        uz = {**u, **hmc_state.z}
        return device_put(HMC_ECS_State(uz, hmc_state, 1., rng_key))

    def sample(self, state, model_args, model_kwargs):
        model_kwargs = {} if model_kwargs is None else model_kwargs.copy()
        rng_key, key_u = random.split(state.rng_key)
        u = {k: v for k, v in state.uz.items() if k in self._plate_sizes}
        u_new = {}
        for name, (size, subsample_size) in self._plate_sizes.items():
            key_u, subkey = random.split(key_u)
            u_new[name] = random.choice(subkey, size, (subsample_size,), replace=False)
        sample = self.postprocess_fn(model_args, model_kwargs)(state.hmc_state.z)
        u_loglik = log_likelihood(self.model, sample, *model_args, batch_ndims=0,
                                  **model_kwargs, _subsample_sites=u)
        u_loglik = sum(v.sum() for v in u_loglik.values())
        u_new_loglik = log_likelihood(self.model, sample, *model_args, batch_ndims=0,
                                      **model_kwargs, _subsample_sites=u_new)
        u_new_loglik = sum(v.sum() for v in u_new_loglik.values())
        accept_prob = jnp.clip(jnp.exp(u_new_loglik - u_loglik), a_max=1.0)
        u = lax.cond(random.bernoulli(key_u, accept_prob), u_new, identity, u, identity)
        model_kwargs["_subsample_sites"] = u
        hmc_state = self.inner_kernel.sample(state.hmc_state, model_args, model_kwargs)
        uz = {**u, **hmc_state.z}
        return HMC_ECS_State(uz, hmc_state, accept_prob, rng_key)


def model(data):
    x = numpyro.sample("x", dist.Normal(0, 1))
    with numpyro.plate("N", data.shape[0], subsample_size=100):
        batch = numpyro.subsample(data, event_dim=0)
        numpyro.sample("obs", dist.Normal(x, 1), obs=batch)

kernel = HMC_ECS(NUTS(model))
mcmc = MCMC(kernel, 500, 500)
data = random.normal(random.PRNGKey(1), (10000,)) + 1
mcmc.run(random.PRNGKey(0), data, extra_fields=("accept_prob",))
# there is a bug when exclude_deterministic=True, which will be fixed upstream
mcmc.print_summary(exclude_deterministic=False)