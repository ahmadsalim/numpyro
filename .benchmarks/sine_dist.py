from time import time

import torch
import jax.numpy as jnp
from jax import random

from numpyro.distributions import Sine, SineSkewed

# https://pytorch.org/tutorials/recipes/recipes/benchmark.html

if __name__ == '__main__':
    loc = jnp.array([0.])
    conc = jnp.array([1.])
    corr = jnp.array([.6])
    sine = Sine(loc, loc, conc, conc, corr)
    skewness = jnp.array([.3, -.2])
    ss = SineSkewed(sine, skewness)

    key = random.PRNGKey(0)
    timings = []
    for samples in [1, 5, 25, 125, 625, 3125, 15625, 78125, 390625, 1953125]:
        times = []
        for _ in range(10):
            key, sample_key = random.split(key)
            start = time()
            data = sine.sample(sample_key, (samples,))
            times.append(time() - start)
        timings.append(times)
    print(timings)
