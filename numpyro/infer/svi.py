# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import namedtuple, partial

import jax
from jax import random, value_and_grad

from numpyro.contrib.funsor import enum, config_enumerat
from numpyro.distributions import Distribution, Delta
from numpyro.handlers import seed, trace, replay
from numpyro.infer import VI
from numpyro.infer.util import transform_fn, get_parameter_transform, _guess_max_plate_nesting

SVIState = namedtuple('SVIState', ['optim_state', 'rng_key'])
"""
A :func:`~collections.namedtuple` consisting of the following fields:
 - **optim_state** - current optimizer's state.
 - **rng_key** - random number generator seed used for the iteration.
"""


class SVI(VI):
    """
    Stochastic Variational Inference given an ELBO loss objective.

    **References**

    1. *SVI Part I: An Introduction to Stochastic Variational Inference in Pyro*,
       (http://pyro.ai/examples/svi_part_i.html)

    **Example:**

    .. doctest::

        >>> from jax import lax, random
        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.distributions import constraints
        >>> from numpyro.infer import SVI, ELBO

        >>> def model(data):
        ...     f = numpyro.sample("latent_fairness", dist.Beta(10, 10))
        ...     with numpyro.plate("N", data.shape[0]):
        ...         numpyro.sample("obs", dist.Bernoulli(f), obs=data)

        >>> def guide(data):
        ...     alpha_q = numpyro.param("alpha_q", 15., constraint=constraints.positive)
        ...     beta_q = numpyro.param("beta_q", 15., constraint=constraints.positive)
        ...     numpyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q))

        >>> data = jnp.concatenate([jnp.ones(6), jnp.zeros(4)])
        >>> optimizer = numpyro.optim.Adam(step_size=0.0005)
        >>> svi = SVI(model, guide, optimizer, loss=ELBO())
        >>> init_state = svi.init(random.PRNGKey(0), data)
        >>> state = lax.fori_loop(0, 2000, lambda i, state: svi.update(state, data)[0], init_state)
        >>> # or to collect losses during the loop
        >>> # state, losses = lax.scan(lambda state, i: svi.update(state, data), init_state, jnp.arange(2000))
        >>> params = svi.get_params(state)
        >>> inferred_mean = params["alpha_q"] / (params["alpha_q"] + params["beta_q"])

    :param model: Python callable with Pyro primitives for the model.
    :param guide: Python callable with Pyro primitives for the guide
        (recognition network).
    :param optim: an instance of :class:`~numpyro.optim._NumpyroOptim`.
    :param loss: ELBO loss, i.e. negative Evidence Lower Bound, to minimize.
    :param static_kwargs: static arguments for the model / guide, i.e. arguments
        that remain constant during fitting.
    :return: tuple of `(init_fn, update_fn, evaluate)`.
    """

    def __init__(self, model, guide, optim, loss, enum=True, **static_kwargs):
        super().__init__(model, guide, optim, loss, **static_kwargs, name='SVI')
        self._inference_model = model
        self.model = model
        self.guide = guide
        self.loss = loss
        self.optim = optim
        self.enum = enum
        self.static_kwargs = static_kwargs
        self.constrain_fn = None

    def init(self, rng_key, *args, **kwargs):
        """

        :param jax.random.PRNGKey rng_key: random number generator seed.
        :param args: arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: keyword arguments to the model / guide (these can possibly vary
            during the course of fitting).
        :return: tuple containing initial :data:`SVIState`, and `get_params`, a callable
            that transforms unconstrained parameter values from the optimizer to the
            specified constrained domain
        """
        rng_key, model_seed, guide_seed = random.split(rng_key, 3)
        model_init = seed(self.model, model_seed)
        guide_init = seed(self.guide, guide_seed)
        guide_trace = trace(guide_init).get_trace(*args, **kwargs, **self.static_kwargs)
        model_trace = trace(replay(model_init, guide_trace)).get_trace(*args, **kwargs, **self.static_kwargs)
        params = {}
        inv_transforms = {}
        should_enum = False
        # NB: params in model_trace will be overwritten by params in guide_trace
        for site in list(model_trace.values()) + list(guide_trace.values()):
            if site['type'] == 'param':
                transform = get_parameter_transform(site)
                inv_transforms[site['name']] = transform
                params[site['name']] = transform.inv(site['value'])
            if isinstance(site['fn'], Distribution) and site['fn'].is_discrete:
                if (isinstance(site['fn'], Delta) or site['fn'].has_enumerate_support) and self.enum:
                    should_enum = True
                else:
                    raise Exception("Cannot enumerate model with discrete variables without enumerate support")

        if should_enum:
            mpn = _guess_max_plate_nesting(model_trace)
            self._inference_model = enum(config_enumerate(self.model), - mpn - 1)

        self.constrain_fn = partial(transform_fn, inv_transforms)
        return SVIState(self.optim.init(params), rng_key)

    def get_params(self, svi_state):
        """
        Gets values at `param` sites of the `model` and `guide`.

        :param svi_state: current state of the optimizer.
        """
        params = self.constrain_fn(self.optim.get_params(svi_state.optim_state))
        return params

    def update(self, svi_state, *args, **kwargs):
        """
        Take a single step of SVI (possibly on a batch / minibatch of data),
        using the optimizer.

        :param svi_state: current state of SVI.
        :param args: arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: keyword arguments to the model / guide (these can possibly vary
            during the course of fitting).
        :return: tuple of `(svi_state, loss)`.
        """
        rng_key, rng_key_step = random.split(svi_state.rng_key)
        params = self.optim.get_params(svi_state.optim_state)
        loss_val, grads = value_and_grad(
            lambda x: self.loss.loss(rng_key_step, self.constrain_fn(x), self._inference_model, self.guide,
                                     *args, **kwargs, **self.static_kwargs))(params)
        optim_state = self.optim.update(grads, svi_state.optim_state)
        return SVIState(optim_state, rng_key), loss_val

    def evaluate(self, svi_state, *args, **kwargs):
        """
        Take a single step of SVI (possibly on a batch / minibatch of data).

        :param svi_state: current state of SVI.
        :param args: arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: keyword arguments to the model / guide.
        :return: evaluate ELBO loss given the current parameter values
            (held within `svi_state.optim_state`).
        """
        # we split to have the same seed as `update_fn` given an svi_state
        _, rng_key_eval = random.split(svi_state.rng_key)
        params = self.get_params(svi_state)
        return self.loss.loss(rng_key_eval, params, self._inference_model, self.guide,
                              *args, **kwargs, **self.static_kwargs)

    def predict(self, state, *args, num_samples=1, **kwargs):
        _, rng_key_predict = jax.random.split(state.rng_key)
        params = self.get_params(state)
        if num_samples == 1:
            return self._predict_model(rng_key_predict, params, *args, **kwargs)
        else:
            return jax.vmap(lambda rk: self._predict_model(rk, params)
                            )(jax.random.split(rng_key_predict, num_samples))
