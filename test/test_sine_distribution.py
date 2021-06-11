import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose
from scipy.special import iv

from numpyro.distributions.directional import log_I1
import numpy as np


@pytest.mark.parametrize("order", [0, 1, 5, 10, 20])
@pytest.mark.parametrize("value", [0.01, 0.1, 1.0, 10.0, 100.0])
def test_log_I1(order, value):
    expected = jnp.array([np.log(iv(i, value)) for i in range(order + 1)])
    value = jnp.array([value])
    actual = log_I1(order, value).reshape(-1)
    assert_allclose(actual, expected, atol=0.05)
