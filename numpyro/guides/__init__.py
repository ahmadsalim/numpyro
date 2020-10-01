from abc import ABC, abstractmethod
from numpyro import handlers
from numpyro.infer.util import find_valid_initial_params, init_to_uniform
from numpyro.distributions.constraints import real
from numpyro.distributions.transforms import biject_to
import jax.random

__all__ = ['ReinitGuide', 'WrappedGuide']


class ReinitGuide(ABC):
    @abstractmethod
    def init_params(self):
        raise NotImplementedError

    @abstractmethod
    def find_params(self, rng_keys, *args, **kwargs):
        raise NotImplementedError


class WrappedGuide(ReinitGuide):
    def __init__(self, fn, reinit_hide_fn=lambda site: site['name'].endswith("$params"),
                 init_strategy=init_to_uniform()):
        self.fn = fn
        self.reinit_hide_fn = reinit_hide_fn
        self._init_params = None
        self.init_strategy = init_strategy

    def init_params(self):
        return self._init_params

    def find_params(self, rng_keys, *args, **kwargs):
        guide_trace = handlers.trace(handlers.seed(self.fn, rng_keys[0])).get_trace(*args, **kwargs)

        def _find_valid_params(rng_key):
            guide_trace = handlers.trace(handlers.seed(self.fn, rng_key)).get_trace(*args, **kwargs)
            params = {name: site['value'] for name, site in guide_trace.items()
                      if site['type'] == 'param' and self.reinit_hide_fn(site)}
            for site in guide_trace.values():
                if site['type'] == 'param' and not self.reinit_hide_fn(site):
                    params[site['name']] = self.init_strategy(site, reinit_param=lambda _: True)
            return params

        if rng_keys.ndim > 1:
            init_params = jax.vmap(_find_valid_params)(rng_keys)
        else:
            init_params = _find_valid_params(rng_keys)
        params = {}
        for name, site in guide_trace.items():
            if site['type'] == 'param':
                constraint = site['kwargs'].pop('constraint', real)
                param_val = biject_to(constraint)(init_params[name])
                params[name] = (name, param_val, constraint)
        self._init_params = {param: (val, constr) for param, val, constr in params.values()}

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
