import sys
from math import sqrt
from random import randint

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import stax

import numpyro
from numpyro import distributions as dist
from numpyro.callbacks import Progbar, History
from numpyro.examples.datasets import load_dataset, FASHION_MNIST
from numpyro.infer import SVI, ELBO
from numpyro.optim import Adam

import matplotlib.pyplot as plt


def Reshape(new_shape):
    def init_fun(_rng_key, input_shape):
        batch_size, *rest_shape = input_shape
        assert np.prod(new_shape) == np.prod(rest_shape)
        return (batch_size, *new_shape), ()

    def apply_fun(_params, inputs, **_kwargs):
        return jnp.reshape(inputs, (-1, *new_shape))

    return init_fun, apply_fun


def guide(mnist_like, latent_dim=30):
    batch_size, *data_sizes = mnist_like.shape
    encoder = numpyro.module('encoder',
                             stax.serial(stax.Conv(32, (3, 3), (2, 2), padding='SAME'),
                                         stax.Relu,
                                         stax.BatchNorm(),
                                         stax.Conv(32, (3, 3), (2, 2), padding='SAME'),
                                         stax.Relu,
                                         stax.BatchNorm(),
                                         stax.Conv(64, (3, 3), (2, 2), padding='SAME'),
                                         stax.BatchNorm(),
                                         stax.Relu,
                                         stax.Flatten,
                                         stax.Dense(latent_dim * 2)),
                             input_shape=(batch_size, *data_sizes, 1))
    with numpyro.plate('data', batch_size):
        mnist_like = jnp.reshape(mnist_like, (batch_size, *data_sizes, 1))
        enc_params = jnp.reshape(encoder(mnist_like), (batch_size, latent_dim, 2))
        mus = enc_params[..., 0]
        sigmas = jnp.exp(enc_params[..., 1])
        _zs = numpyro.sample('zs', dist.Normal(mus, sigmas).to_event(1))


def model(mnist_like, latent_dim=30):
    batch_size, *data_sizes = mnist_like.shape
    decoder = numpyro.module('decoder',
                             stax.serial(stax.Dense(7 * 7 * 32),
                                         stax.Relu,
                                         Reshape((7, 7, 32)),
                                         stax.ConvTranspose(64, (3, 3), (2, 2), padding='SAME'),
                                         stax.Relu,
                                         stax.BatchNorm(),
                                         stax.ConvTranspose(1, (3, 3), (2, 2), padding='SAME')),
                             input_shape=(batch_size, latent_dim))
    with numpyro.plate('data', mnist_like.shape[0]):
        zs = numpyro.sample('zs', dist.Normal(0, 1).expand_by((latent_dim,)).to_event(1))
        logits = decoder(zs)[..., 0]
        numpyro.sample('xs', dist.Bernoulli(logits=logits).to_event(2), obs=(mnist_like > 0.5) * 1)


def _show_mnist(images):
    rc = int(sqrt(len(images)))
    fig, ax = plt.subplots(nrows=rc, ncols=rc)
    for i in range(rc):
        for j in range(rc):
            ax[i, j].axis('off')
            ax[i, j].imshow(images[rc * i + j])
    plt.show()


def main(_argv):
    numpyro.set_platform('gpu')
    numpyro.enable_validation()
    num_steps = 50000
    rng_key = jax.random.PRNGKey(randint(0, 10000))
    svi = SVI(model, guide, Adam(1e-3), ELBO())
    init, get_batch = load_dataset(FASHION_MNIST, 32)
    num_batches, idxs = init()
    test_batch, _ = get_batch(idxs=idxs)

    ## %
    def batch_fun(step):
        i = step % num_batches
        epoch = step // num_batches
        is_last = i == (num_batches - 1)
        batch, _ = get_batch(i, idxs)
        return (batch,), {}, epoch, is_last

    _show_mnist(test_batch)
    history = History()
    svi_state, loss = svi.train(rng_key, num_steps, batch_fun=batch_fun,
                                callbacks=[Progbar(), history])
    plt.plot(history.training_history)
    plt.show()


if __name__ == '__main__':
    main(sys.argv)
