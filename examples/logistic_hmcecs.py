""" Logistic regression model as implemetned in https://arxiv.org/pdf/1708.00955.pdf with Higgs Dataset """
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, Predictive
from numpyro.contrib.hmcecs import HMC
from sklearn.datasets import load_breast_cancer


# TODO: import Higgs data! ---> http://archive.ics.uci.edu/ml/machine-learning-databases/00280/
# https://towardsdatascience.com/identifying-higgs-bosons-from-background-noise-pyspark-d7983234207e

def model(feats, obs):
    """  Logistic regression model

    """
    n, m = feats.shape
    precision = numpyro.sample('precision', dist.continuous.Uniform(0, 4))
    theta = numpyro.sample('theta', dist.continuous.Normal(jnp.zeros(m), precision * jnp.ones(m)))

    numpyro.sample('obs', dist.Bernoulli(logits=jnp.matmul(feats, theta)), obs=obs)


def infer_nuts(rng_key, feats, obs, samples=10, warmup=5, ):
    kernel = NUTS(model=model)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples)
    mcmc.run(rng_key, feats, obs)
    # mcmc.print_summary()
    return mcmc.get_samples()


def infer_hmcecs(rng_key, feats, obs, g=2, samples=10, warmup=5, ):
    hmcecs_key, map_key = jax.random.split(rng_key)
    n, _ = feats.shape
    model_args = (feats, obs)
    print("Running Nuts for map estimation")
    z_map = {key: value.mean(0) for key, value in infer_nuts(map_key, feats, obs).items()}
    print("Running MCMC subsampling")
    kernel = HMC(model=model)
    mcmc = MCMC(kernel,num_warmup=warmup,num_samples=samples)
    mcmc.run(rng_key,observations=obs,features=feats,subsample_method="perturb",m=4,g=2,z_map = z_map)
    return mcmc.get_samples()



def breast_cancer_data():
    dataset = load_breast_cancer()
    feats = dataset.data
    feats = (feats - feats.mean(0)) / feats.std(0)
    feats = jnp.hstack((feats, jnp.ones((feats.shape[0], 1))))
    return feats, dataset.target.reshape((-1, 1))


def higgs_data():
    return


if __name__ == '__main__':
    rng_key = jax.random.PRNGKey(37)
    rng_key, feat_key, obs_key = jax.random.split(rng_key, 3)
    n = 100
    m = 10

    feats, obs = breast_cancer_data()

    from jax.config import config

    config.update('jax_disable_jit', True)
    est_posterior = infer_hmcecs(rng_key, feats=feats, obs=obs)

    exit()
    predictions = Predictive(model, posterior_samples=est_posterior)(rng_key, feats, None)['obs']

    # for i, y in enumerate(obs):
    #     print(i, y[0], jnp.sum(predictions[i]) > 50)