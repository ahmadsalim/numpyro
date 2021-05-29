import pickle
import sys
from math import pi
from pathlib import Path

import numpy as np
from jax import numpy as jnp
from jax import random

import numpyro
from numpyro.contrib.funsor import config_enumerate
from numpyro.distributions import Dirichlet, Gamma, Uniform, VonMises, Beta, Categorical, Sine, SineSkewed
from numpyro.infer import NUTS, init_to_median, MCMC

AMINO_ACIDS = ['M', 'N', 'I', 'F', 'E', 'L', 'R', 'D', 'G', 'K', 'Y', 'T', 'H', 'S', 'P', 'A', 'V', 'Q', 'W', 'C']


@config_enumerate
def sine_model(data, num_mix_comp=2):
    # Mixture prior
    mix_weights = numpyro.sample('mix_weights', Dirichlet(jnp.ones((num_mix_comp,))))

    # Hprior BvM
    # Bayesian Inference and Decision Theory by Kathryn Blackmond Laskey
    beta_mean_phi = numpyro.sample('beta_mean_phi', Uniform(0., 1.))
    beta_prec_phi = numpyro.sample('beta_prec_phi', Gamma(1., 1 / 20.))  # shape, rate
    halpha_phi = beta_mean_phi * beta_prec_phi
    beta_mean_psi = numpyro.sample('beta_mean_psi', Uniform(0, 1.))
    beta_prec_psi = numpyro.sample('beta_prec_psi', Gamma(1., 1 / 20.))  # shape, rate
    halpha_psi = beta_mean_psi * beta_prec_psi

    with numpyro.plate('mixture', num_mix_comp):
        # BvM priors
        phi_loc = numpyro.sample('phi_loc', VonMises(pi, 2.))
        psi_loc = numpyro.sample('psi_loc', VonMises(-pi / 2, .2))
        phi_conc = numpyro.sample('phi_conc', Beta(halpha_phi, beta_prec_phi - halpha_phi))
        psi_conc = numpyro.sample('psi_conc', Beta(halpha_psi, beta_prec_psi - halpha_psi))
        corr_scale = numpyro.sample('corr_scale', Beta(2., 5.))

    with numpyro.plate('obs_plate', len(data), dim=-1):
        assign = numpyro.sample('mix_comp', Categorical(mix_weights), infer={"enumerate": "parallel"})
        sine = Sine(phi_loc=phi_loc[assign], psi_loc=psi_loc[assign],
                    phi_concentration=750 * phi_conc[assign],
                    psi_concentration=750 * psi_conc[assign],
                    weighted_correlation=corr_scale[assign])
        return numpyro.sample('phi_psi', sine, obs=data)


@config_enumerate
def ss_model(data, num_mix_comp=2):
    # Mixture prior
    mix_weights = numpyro.sample('mix_weights', Dirichlet(jnp.ones((num_mix_comp,))))

    # Hprior BvM
    # Bayesian Inference and Decision Theory by Kathryn Blackmond Laskey
    beta_mean_phi = numpyro.sample('beta_mean_phi', Uniform(0., 1.))
    beta_prec_phi = numpyro.sample('beta_prec_phi', Gamma(1., 1 / 20.))  # shape, rate
    halpha_phi = beta_mean_phi * beta_prec_phi
    beta_mean_psi = numpyro.sample('beta_mean_psi', Uniform(0, 1.))
    beta_prec_psi = numpyro.sample('beta_prec_psi', Gamma(1., 1 / 20.))  # shape, rate
    halpha_psi = beta_mean_psi * beta_prec_psi

    with numpyro.plate('mixture', num_mix_comp):
        # BvM priors
        phi_loc = numpyro.sample('phi_loc', VonMises(pi, 2.))
        psi_loc = numpyro.sample('psi_loc', VonMises(-pi / 2, .2))
        phi_conc = numpyro.sample('phi_conc', Beta(halpha_phi, beta_prec_phi - halpha_phi))
        psi_conc = numpyro.sample('psi_conc', Beta(halpha_psi, beta_prec_psi - halpha_psi))
        corr_scale = numpyro.sample('corr_scale', Beta(2., 5.))

        skew_phi = numpyro.sample('skew_phi', Uniform(-1., 1.))
        psi_bound = 1 - jnp.abs(skew_phi)
        skew_psi = numpyro.sample('skew_psi', Uniform(-1., 1.))
        skewness = jnp.stack((skew_phi, psi_bound * skew_psi), axis=-1)
        assert skewness.shape == (num_mix_comp, 2)

    with numpyro.plate('obs_plate', len(data), dim=-1):
        assign = numpyro.sample('mix_comp', Categorical(mix_weights), infer={"enumerate": "parallel"})
        sine = Sine(phi_loc=phi_loc[assign], psi_loc=psi_loc[assign],
                    phi_concentration=750 * phi_conc[assign],
                    psi_concentration=750 * psi_conc[assign],
                    weighted_correlation=corr_scale[assign])
        return numpyro.sample('phi_psi', SineSkewed(sine, skewness[assign]))


def run_hmc(model, data, num_mix_comp, num_samples):
    rng_key = random.PRNGKey(0)
    kernel = NUTS(model, init_strategy=init_to_median())
    mcmc = MCMC(kernel, num_samples=num_samples, num_warmup=num_samples // 2)
    mcmc.run(rng_key, data, num_mix_comp)
    mcmc.print_summary()
    post_samples = mcmc.get_samples()
    return post_samples


def fetch_aa_dihedrals(split='train', subsample_to=1000_000):
    file = Path(__file__).parent / 'data/9mer_fragments_processed.pkl'
    data = pickle.load(file.open('rb'))[split]['sequences']
    data_aa = np.argmax(data[..., :20], -1)
    data = {aa: data[..., -2:][data_aa == i] for i, aa in enumerate(AMINO_ACIDS)}
    [np.random.shuffle(v) for v in data.values()]
    data = {aa: aa_data[:min(subsample_to, aa_data.shape[0])] for aa, aa_data in data.items()}
    data = {aa: jnp.array(aa_data, dtype=float) for aa, aa_data in data.items()}
    return data


def main(num_mix_start=20, num_mix_end=45, num_samples=4, aas=('S', 'G', 'P'), capture_std=True,
         rerun_inference=True):
    for aa in aas:
        for num_mix_comp in range(num_mix_start, num_mix_end):
            if capture_std:
                sys.stdout = (Path(__file__).parent / "runs" /
                              f'ssbvm_bmixture_aa{aa}_comp{num_mix_comp}_steps{num_samples}.out').open('w')

            chain_file = Path(__file__).parent / "runs" / \
                         f'ssbvm_bmixture_aa{aa}_comp{num_mix_comp}_steps{num_samples}.pkl'

            if rerun_inference or not chain_file.exists():
                data = fetch_aa_dihedrals(subsample_to=10)
                posterior_samples = {aa: {'sine': run_hmc(sine_model, data[aa], num_mix_comp, num_samples),
                                          'ss': run_hmc(ss_model, data[aa], num_mix_comp, num_samples), }}
                pickle.dump(posterior_samples, chain_file.open('wb'))
            else:
                posterior_samples = pickle.load(chain_file.open('rb'))

            if capture_std:
                sys.stdout.close()


if __name__ == '__main__':
    main()
