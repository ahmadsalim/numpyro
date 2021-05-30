from time import time

import jax.numpy as jnp
# https://pytorch.org/tutorials/recipes/recipes/benchmark.html
from jax import random

from numpyro import set_platform
from numpyro.distributions import Sine, SineSkewed

if __name__ == '__main__':
    set_platform('gpu')

    loc = jnp.array([0.])
    conc = jnp.array([1.])
    corr = jnp.array([.6])
    sine = Sine(loc, loc, conc, conc, corr)
    skewness = jnp.array([.3, -.2])
    ss = SineSkewed(sine, skewness)
    key = random.PRNGKey(0)
    first_timings = []
    after_timings = []
    for samples in [1, 5, 25, 125, 625, 3125, 15625, 78125, 390625, 1953125, 5 * 1953125]:
        times = []
        print(samples)
        for _ in range(11):
            key, sample_key = random.split(key)
            start = time()
            data = sine.sample(sample_key, (samples,))
            times.append(time() - start)
        first_timings.append(times[1:])
        after_timings.append(times[:1])
    print(first_timings)
    print(after_timings)
