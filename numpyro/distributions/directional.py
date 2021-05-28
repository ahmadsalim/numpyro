# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools
import math
import operator
import warnings
from math import pi

import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
from jax.scipy.special import erf, i0e, i1e, logsumexp

from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import (
    is_prng_key,
    promote_shapes,
    safe_normalize,
    validate_sample,
    von_mises_centered, lazy_property,
)


def _numel(shape):
    return functools.reduce(operator.mul, shape, 1)


def log_im(order, x):  ## x is a parameter, like k1 or k2
    # Based on '_log_modified_bessel_fn'
    # Tanabe, A., Fukumizu, K., Oba, S., Takenouchi, T., & Ishii, S. (2007).
    # Parameter estimation for von Mises–Fisher distributions. Computational Statistics, 22(1), 145-157.
    """ terms to sum over, 10 by 'shape of x' and sums over the first dimension """
    """ vectorized logarithmic Im """
    s = jnp.arange(0, 251).reshape(251, 1)
    fs = 2 * s * (jnp.log(x) - math.log(2)) - jax.scipy.special.gammaln(s + 1.) - jax.scipy.special.gammaln(
        order + s + 1.)

    return order * (jnp.log(x) - math.log(2)) + logsumexp(fs, -2)


def log_I1(orders: int, value, terms=250):
    r""" Compute first n log modified bessel function of first kind
    .. math ::

        \log(I_v(z)) = v*\log(z/2) + \log(\sum_{k=0}^\inf \exp\left[2*k*\log(z/2) - \sum_kk^k log(kk)
        - \lgamma(v + k + 1)\right])

    :param orders: orders of the log modified bessel function.
    :param value: values to compute modified bessel function for
    :param terms: truncation of summation
    :return: 0 to orders modified bessel function
    """
    orders = orders + 1
    if value.ndim == 0:
        vshape = jnp.shape([1])
    else:
        vshape = value.shape
    value = value.reshape(-1, 1)
    flat_vshape = _numel(vshape)

    k = jnp.arange(terms)
    lgammas_all = jax.scipy.special.gammaln(jnp.arange(1., terms + orders + 1.))
    assert lgammas_all.shape == (orders + terms,)  # lgamma(0) = inf => start from 1

    lvalues = jnp.log(value / 2) * k.reshape(1, -1)
    assert lvalues.shape == (flat_vshape, terms)

    lfactorials = lgammas_all[:terms]
    assert lfactorials.shape == (terms,)

    lgammas = lgammas_all.repeat(orders).reshape(orders, -1)
    assert lgammas.shape == (orders, terms + orders)  # lgamma(0) = inf => start from 1

    indices = k[:orders].reshape(-1, 1) + k.reshape(1, -1)
    assert indices.shape == (orders, terms)

    seqs = logsumexp(
        2 * lvalues[None, :, :] - lfactorials[None, None, :] - jnp.take_along_axis(lgammas, indices, axis=1)[:, None,
                                                               :], -1)
    assert seqs.shape == (orders, flat_vshape)

    i1s = lvalues[..., :orders].T + seqs
    assert i1s.shape == (orders, flat_vshape)
    return i1s.reshape(-1, *vshape)


class VonMises(Distribution):
    arg_constraints = {"loc": constraints.real, "concentration": constraints.positive}
    reparametrized_params = ["loc"]
    support = constraints.interval(-math.pi, math.pi)

    def __init__(self, loc, concentration, validate_args=None):
        """von Mises distribution for sampling directions.

        :param loc: center of distribution
        :param concentration: concentration of distribution
        """
        self.loc, self.concentration = promote_shapes(loc, concentration)

        batch_shape = lax.broadcast_shapes(jnp.shape(concentration), jnp.shape(loc))

        super(VonMises, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        """Generate sample from von Mises distribution

        :param key: random number generator key
        :param sample_shape: shape of samples
        :return: samples from von Mises
        """
        assert is_prng_key(key)
        samples = von_mises_centered(
            key, self.concentration, sample_shape + self.shape()
        )
        samples = samples + self.loc  # VM(0, concentration) -> VM(loc,concentration)
        samples = (samples + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

        return samples

    @validate_sample
    def log_prob(self, value):
        return -(
                jnp.log(2 * jnp.pi) + jnp.log(i0e(self.concentration))
        ) + self.concentration * (jnp.cos((value - self.loc) % (2 * jnp.pi)) - 1)

    @property
    def mean(self):
        """Computes circular mean of distribution. NOTE: same as location when mapped to support [-pi, pi]"""
        return jnp.broadcast_to(
            (self.loc + jnp.pi) % (2.0 * jnp.pi) - jnp.pi, self.batch_shape
        )

    @property
    def variance(self):
        """Computes circular variance of distribution"""
        return jnp.broadcast_to(
            1.0 - i1e(self.concentration) / i0e(self.concentration), self.batch_shape
        )


class ProjectedNormal(Distribution):
    """
    Projected isotropic normal distribution of arbitrary dimension.

    This distribution over directional data is qualitatively similar to the von
    Mises and von Mises-Fisher distributions, but permits tractable variational
    inference via reparametrized gradients.

    To use this distribution with autoguides and HMC, use ``handlers.reparam``
    with a :class:`~numpyro.infer.reparam.ProjectedNormalReparam`
    reparametrizer in the model, e.g.::

        @handlers.reparam(config={"direction": ProjectedNormalReparam()})
        def model():
            direction = numpyro.sample("direction",
                                       ProjectedNormal(zeros(3)))
            ...

    .. note:: This implements :meth:`log_prob` only for dimensions {2,3}.

    [1] D. Hernandez-Stumpfhauser, F.J. Breidt, M.J. van der Woerd (2017)
        "The General Projected Normal Distribution of Arbitrary Dimension:
        Modeling and Bayesian Inference"
        https://projecteuclid.org/euclid.ba/1453211962
    """

    arg_constraints = {"concentration": constraints.real_vector}
    reparametrized_params = ["concentration"]
    support = constraints.sphere

    def __init__(self, concentration, *, validate_args=None):
        assert jnp.ndim(concentration) >= 1
        self.concentration = concentration
        batch_shape = concentration.shape[:-1]
        event_shape = concentration.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def mean(self):
        """
        Note this is the mean in the sense of a centroid in the submanifold
        that minimizes expected squared geodesic distance.
        """
        return safe_normalize(self.concentration)

    @property
    def mode(self):
        return safe_normalize(self.concentration)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        eps = random.normal(key, shape=shape)
        return safe_normalize(self.concentration + eps)

    def log_prob(self, value):
        if self._validate_args:
            event_shape = value.shape[-1:]
            if event_shape != self.event_shape:
                raise ValueError(
                    f"Expected event shape {self.event_shape}, "
                    f"but got {event_shape}"
                )
            self._validate_sample(value)
        dim = int(self.concentration.shape[-1])
        if dim == 2:
            return _projected_normal_log_prob_2(self.concentration, value)
        if dim == 3:
            return _projected_normal_log_prob_3(self.concentration, value)
        raise NotImplementedError(
            f"ProjectedNormal.log_prob() is not implemented for dim = {dim}. "
            "Consider using handlers.reparam with ProjectedNormalReparam."
        )

    @staticmethod
    def infer_shapes(concentration):
        batch_shape = concentration[:-1]
        event_shape = concentration[-1:]
        return batch_shape, event_shape


def _projected_normal_log_prob_2(concentration, value):
    def _dot(x, y):
        return (x[..., None, :] @ y[..., None])[..., 0, 0]

    # We integrate along a ray, factorizing the integrand as a product of:
    # a truncated normal distribution over coordinate t parallel to the ray, and
    # a univariate normal distribution over coordinate r perpendicular to the ray.
    t = _dot(concentration, value)
    t2 = t * t
    r2 = _dot(concentration, concentration) - t2
    perp_part = (-0.5) * r2 - 0.5 * math.log(2 * math.pi)

    # This is the log of a definite integral, computed by mathematica:
    # Integrate[x/(E^((x-t)^2/2) Sqrt[2 Pi]), {x, 0, Infinity}]
    # = (t + Sqrt[2/Pi]/E^(t^2/2) + t Erf[t/Sqrt[2]])/2
    para_part = jnp.log(
        (jnp.exp((-0.5) * t2) * ((2 / math.pi) ** 0.5) + t * (1 + erf(t * 0.5 ** 0.5)))
        / 2
    )

    return para_part + perp_part


class Sine(Distribution):
    r""" Unimodal distribution of two dependent angles on the 2-torus (S^1 ⨂ S^1) given by

    .. math::

        C^{-1}\exp(\kappa_1\cos(x-\mu_1) + \kappa_2\cos(x_2 -\mu_2) + \rho\sin(x_1 - \mu_1)\sin(x_2 - \mu_2))

    and

    .. math::

        C = (2\pi)^2 \sum_{i=0} {2i \choose i}
        \left(\frac{\rho^2}{4\kappa_1\kappa_2}\right)^i I_i(\kappa_1)I_i(\kappa_2),

    where I_i(\cdot) is the modified bessel function of first kind, mu's are the locations of the distribution,
    kappa's are the concentration and rho gives the correlation between angles x_1 and x_2.

    This distribution is helpful for modeling coupled angles such as torsion angles in peptide chains.
    To infer parameters, use :class:`~pyro.infer.NUTS` or :class:`~pyro.infer.HMC` with priors that
    avoid parameterizations where the distribution becomes bimodal; see note below.

    .. note:: Sample efficiency drops as

        .. math::

            \frac{\rho}{\kappa_1\kappa_2} \rightarrow 1

        because the distribution becomes increasingly bimodal.

    .. note:: The correlation and weighted_correlation params are mutually exclusive.

    .. note:: In the context of :class:`~pyro.infer.SVI`, this distribution can be used as a likelihood but not for
        latent variables.

    ** References: **
      1. Probabilistic model for two dependent circular variables Singh, H., Hnizdo, V., and Demchuck, E. (2002)

    :param jnp.Tensor phi_loc: location of first angle
    :param jnp.Tensor psi_loc: location of second angle
    :param jnp.Tensor phi_concentration: concentration of first angle
    :param jnp.Tensor psi_concentration: concentration of second angle
    :param jnp.Tensor correlation: correlation between the two angles
    :param jnp.Tensor weighted_correlation: set correlation to weigthed_corr * sqrt(phi_conc*psi_conc)
        to avoid bimodality (see note).
    """

    arg_constraints = {'phi_loc': constraints.real, 'psi_loc': constraints.real,
                       'phi_concentration': constraints.positive, 'psi_concentration': constraints.positive,
                       'correlation': constraints.real}
    support = constraints.independent(constraints.real, 1)
    max_sample_iter = 1000

    def __init__(self, phi_loc, psi_loc, phi_concentration, psi_concentration, correlation=None,
                 weighted_correlation=None, validate_args=None):

        assert (correlation is None) != (weighted_correlation is None)

        if weighted_correlation is not None:
            correlation = weighted_correlation * jnp.sqrt(phi_concentration * psi_concentration) + 1e-8

        self.phi_loc, self.psi_loc, self.phi_concentration, self.psi_concentration, self.correlation = promote_shapes(
            phi_loc, psi_loc,
            phi_concentration,
            psi_concentration,
            correlation)
        event_shape = jnp.shape([2])
        batch_shape = lax.broadcast_shapes(phi_loc.shape, psi_loc.shape, phi_concentration.shape,
                                           psi_concentration.shape, correlation.shape)

        super().__init__(batch_shape, event_shape, validate_args)

        if self._validate_args and jnp.any(phi_concentration * psi_concentration <= correlation ** 2):
            warnings.warn(
                f'{self.__class__.__name__} bimodal due to concentration-correlation relation, '
                f'sampling will likely fail.', UserWarning)

    @lazy_property
    def norm_const(self):
        corr = self.correlation.reshape(1, -1) + 1e-8
        conc = jnp.stack((self.phi_concentration, self.psi_concentration), axis=-1).reshape(-1, 2)
        m = jnp.arange(50).reshape(-1, 1)
        num = lax.lgamma(2 * m + 1.)
        den = lax.lgamma(m + 1.)
        lbinoms = num - 2 * den

        fs = lbinoms.reshape(-1, 1) + 2 * m * jnp.log(corr) - m * jnp.log(4 * jnp.prod(conc, axis=-1))
        fs += log_I1(50, conc, terms=51).sum(-1)
        mfs = fs.max()
        norm_const = 2 * jnp.log(jnp.array(2 * pi)) + mfs + logsumexp(fs - mfs, 0)
        return norm_const.reshape(self.phi_loc.shape)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        indv = self.phi_concentration * jnp.cos(value[..., 0] - self.phi_loc) + self.psi_concentration * jnp.cos(
            value[..., 1] - self.psi_loc)
        corr = self.correlation * jnp.sin(value[..., 0] - self.phi_loc) * jnp.sin(value[..., 1] - self.psi_loc)
        return indv + corr - self.norm_const

    def sample(self, key, sample_shape=()):
        """
        ** References: **
            1. A New Unified Approach for the Simulation of aWide Class of Directional Distributions
               John T. Kent, Asaad M. Ganeiber & Kanti V. Mardia (2018)
        """
        phi_key, psi_key = random.split(key)
        sample_shape = jnp.shape(sample_shape)

        corr = self.correlation
        conc = jnp.stack((self.phi_concentration, self.psi_concentration))

        eig = 0.5 * (conc[0] - corr ** 2 / conc[1])
        eig = jnp.stack((jnp.zeros_like(eig), eig))
        eigmin = jnp.where(eig[1] < 0, eig[1], jnp.zeros_like(eig[1], dtype=eig.dtype))
        eig = eig - eigmin
        b0 = self._bfind(eig)

        total = sample_shape.numel()
        missing = total * jnp.ones((self.batch_shape.numel(),), dtype=int)
        start = jnp.zeros_like(missing)
        phi = jnp.zeros((2, *missing.shape, total), dtype=corr.dtype)

        max_iter = Sine.max_sample_iter

        # flatten batch_shape
        conc = conc.reshape(2, -1, 1)
        eigmin = eigmin.reshape(-1, 1)
        corr = corr.reshape(-1, 1)
        eig = eig.reshape(2, -1)
        b0 = b0.reshape(-1)
        phi_den = log_I1(0, conc[1]).reshape(-1, 1)
        lengths = jnp.arange(total).reshape(1, -1)

        while jnp.any(missing > 0) and max_iter:
            accept_key, acg_key, phi_key = random.split(phi_key, 3)
            curr_conc = conc[:, missing > 0, :]
            curr_corr = corr[missing > 0]
            curr_eig = eig[:, missing > 0]
            curr_b0 = b0[missing > 0]
            min_left = missing[missing > 0].min()

            x = random.normal(acg_key, (min_left,), 0., jnp.sqrt(1 + 2 * curr_eig / curr_b0)).reshape((2, -1, min_left))
            x /= jnp.linalg.norm(x, axis=0)[None, ...]  # Angular Central Gaussian distribution

            lf = curr_conc[0] * (x[0] - 1) + eigmin[missing > 0] + log_I1(0, jnp.sqrt(
                curr_conc[1] ** 2 + (curr_corr * x[1]) ** 2)).squeeze(0) - phi_den[missing > 0]
            assert lf.shape == ((missing > 0).sum(), missing[missing > 0].min())

            lg_inv = 1. - curr_b0.reshape(-1, 1) / 2 + jnp.log(
                curr_b0.reshape(-1, 1) / 2 + (curr_eig.reshape(2, -1, 1) * x ** 2).sum(0))
            assert lg_inv.shape == lf.shape

            accepted = random.uniform(accept_key, lf.shape, 0., jnp.ones(())) < (lf + lg_inv).exp()

            phi_mask = jnp.zeros((*missing.shape, total), dtype=int, )
            phi_mask[missing > 0] += jnp.logical_and(lengths < (start[missing > 0] + accepted.sum(-1)).reshape(-1, 1),
                                                     lengths >= start[missing > 0].reshape(-1, 1))

            phi[:, phi_mask] += x[:, accepted]

            start[missing > 0] += jnp.sum(accepted, -1)
            missing[missing > 0] -= jnp.sum(accepted, -1)
            max_iter -= 1

        if max_iter == 0 or jnp.any(missing > 0):
            raise ValueError("maximum number of iterations exceeded; "
                             "try increasing `SineBivariateVonMises.max_sample_iter`")

        phi = lax.atan2(phi[0], phi[1])

        alpha = jnp.sqrt(conc[1] ** 2 + (corr * jnp.sin(phi)) ** 2)
        beta = lax.atan(corr / conc[1] * jnp.sin(phi))

        psi = VonMises(beta, alpha).sample()

        phi_psi = jnp.stack(((phi + self.phi_loc.reshape((-1, 1)) + pi) % (2 * pi) - pi,
                             (psi + self.psi_loc.reshape((-1, 1)) + pi) % (2 * pi) - pi), axis=-1).permute(1, 0, 2)
        return phi_psi.reshape(*sample_shape, *self.batch_shape, *self.event_shape)

    @property
    def mean(self):
        return jnp.stack((self.phi_loc, self.psi_loc), axis=-1)

    def _bfind(self, eig):
        b = eig.shape[0] / 2 * jnp.ones(self.batch_shape, dtype=eig.dtype)
        g1 = jnp.sum(1 / (b + 2 * eig) ** 2, axis=0)
        g2 = jnp.sum(-2 / (b + 2 * eig) ** 3, axis=0)
        return jnp.where(jnp.linalg.norm(eig, 0) != 0, b - g1 / g2, b)


def _projected_normal_log_prob_3(concentration, value):
    def _dot(x, y):
        return (x[..., None, :] @ y[..., None])[..., 0, 0]

    # We integrate along a ray, factorizing the integrand as a product of:
    # a truncated normal distribution over coordinate t parallel to the ray, and
    # a bivariate normal distribution over coordinate r perpendicular to the ray.
    t = _dot(concentration, value)
    t2 = t * t
    r2 = _dot(concentration, concentration) - t2
    perp_part = (-0.5) * r2 - math.log(2 * math.pi)

    # This is the log of a definite integral, computed by mathematica:
    # Integrate[x^2/(E^((x-t)^2/2) Sqrt[2 Pi]), {x, 0, Infinity}]
    # = t/(E^(t^2/2) Sqrt[2 Pi]) + ((1 + t^2) (1 + Erf[t/Sqrt[2]]))/2
    para_part = jnp.log(
        t * jnp.exp((-0.5) * t2) / (2 * math.pi) ** 0.5
        + (1 + t2) * (1 + erf(t * 0.5 ** 0.5)) / 2
    )

    return para_part + perp_part
