import pytest
import jax.numpy as jnp
from numpy.testing import assert_allclose
from scipy.special import iv, gammaln
from numpyro.distributions.directional import log_im, log_I1

@pytest.mark.parametrize('order', [0, 1, 5, 10, 20])
@pytest.mark.parametrize('value', [0.01, .1, 1., 10., 100.])
def test_log_I1(order, value):
    expected = jnp.log(jnp.array([iv(i, value) for i in range(order + 1)]))
    value = jnp.array([value])
    actual = log_im(order, value).reshape(-1)
    assert_allclose(actual, expected)