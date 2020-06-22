import os
import pickle
import random

import jax
import jax.numpy as np
from jax.config import config
from jax.experimental import stax
from tqdm import tqdm

import numpyro
import numpyro.distributions as dist
from numpyro.examples.datasets import load_dataset, JSBCHORALES
from numpyro.infer.guide import WrappedGuide
from numpyro.infer import ELBO
from numpyro.infer.kernels import RBFKernel
from numpyro.infer.stein import SVGD
from numpyro.optim import ClippedAdam



def _one_hot_chorales(seqs, num_nodes=88):
    return np.sum(np.array((seqs[..., None] == np.arange(num_nodes + 1)), 'int'), axis=-2)[..., 1:]


def _reverse_padded(padded, lengths):
    def _reverse_single(p, l):
        new = np.zeros_like(p)
        reverse = np.roll(p[::-1], l, axis=0)
        return jax.ops.index_update(new, jax.ops.index[:], reverse)

    return jax.vmap(_reverse_single)(padded, lengths)


def Emitter(hidden_dim1, hidden_dim2, out_dim):
    return stax.serial(
        stax.Dense(hidden_dim1), stax.Relu,
        stax.Dense(hidden_dim2), stax.Relu,
        stax.Dense(out_dim), stax.Sigmoid
    )


def Transition(gate_hidden_dim, prop_mean_hidden_dim, out_dim):
    gate_init_fun, gate_apply_fun = stax.serial(
        stax.Dense(gate_hidden_dim), stax.Relu,
        stax.Dense(out_dim), stax.Sigmoid
    )

    prop_mean_init_fun, prop_mean_apply_fun = stax.serial(
        stax.Dense(prop_mean_hidden_dim), stax.Relu,
        stax.Dense(out_dim)
    )

    mean_init_fun, mean_apply_fun = stax.Dense(out_dim)

    stddev_init_fun, stddev_apply_fun = stax.serial(
        stax.Relu, stax.Dense(out_dim),
        stax.Softplus
    )

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2, k3, k4 = jax.random.split(rng, num=4)
        _, gate_params = gate_init_fun(k1, input_shape)
        prop_mean_output_shape, prop_mean_params = prop_mean_init_fun(k2, input_shape)
        _, mean_params = mean_init_fun(k3, input_shape)
        _, stddev_params = stddev_init_fun(k4, prop_mean_output_shape)
        return (output_shape, output_shape), (gate_params, prop_mean_params,
                                              mean_params, stddev_params)

    def apply_fun(params, inputs, **kwargs):
        gate_params, prop_mean_params, mean_params, stddev_params = params
        gt = gate_apply_fun(gate_params, inputs)
        ht = prop_mean_apply_fun(prop_mean_params, inputs)
        mut = (1 - gt) * mean_apply_fun(mean_params, inputs) + gt * ht
        sigmat = stddev_apply_fun(stddev_params, ht)
        return mut, sigmat

    return init_fun, apply_fun


def Combiner(hidden_dim, out_dim):
    mean_init_fun, mean_apply_fun = stax.Dense(out_dim)

    stddev_init_fun, stddev_apply_fun = stax.serial(
        stax.Dense(out_dim),
        stax.Softplus
    )

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2 = jax.random.split(rng, num=2)
        _, mean_params = mean_init_fun(k1, input_shape)
        _, stddev_params = stddev_init_fun(k2, input_shape)
        return (output_shape, output_shape), (mean_params, stddev_params)

    def apply_fun(params, inputs, **kwargs):
        mean_params, stddev_params = params
        mut = mean_apply_fun(mean_params, inputs)
        sigmat = stddev_apply_fun(stddev_params, inputs)
        return mut, sigmat

    return init_fun, apply_fun


def GRU(hidden_dim, W_init=stax.glorot_normal()):
    # Inspired by https://github.com/google/jax/pull/2298
    input_update_init_fun, input_update_apply_fun = stax.Dense(hidden_dim)
    input_reset_init_fun, input_reset_apply_fun = stax.Dense(hidden_dim)
    input_output_init_fun, input_output_apply_fun = stax.Dense(hidden_dim)

    def init_fun(rng, input_shape):
        indv_input_shape = input_shape[1:]
        output_shape = input_shape[:-1] + (hidden_dim,)
        rng, k1, k2 = jax.random.split(rng, num=3)
        hidden_update_w = W_init(k1, (hidden_dim, hidden_dim))
        _, input_update_params = input_update_init_fun(k2, indv_input_shape)

        rng, k1, k2 = jax.random.split(rng, num=3)
        hidden_reset_w = W_init(k1, (hidden_dim, hidden_dim))
        _, input_reset_params = input_reset_init_fun(k2, indv_input_shape)

        rng, k1, k2 = jax.random.split(rng, num=3)
        hidden_output_w = W_init(k1, (hidden_dim, hidden_dim))
        _, input_output_params = input_output_init_fun(k2, indv_input_shape)

        return output_shape, (hidden_update_w, input_update_params,
                              hidden_reset_w, input_reset_params,
                              hidden_output_w, input_output_params)

    def apply_fun(params, inputs, **kwargs):
        (hidden_update_w, input_update_params,
         hidden_reset_w, input_reset_params,
         hidden_output_w, input_output_params) = params
        inps, lengths, init_hidden = inputs

        def apply_fun_single(prev_hidden, inp):
            i, inpv = inp
            inp_update = input_update_apply_fun(input_update_params, inpv)
            hidden_update = np.dot(prev_hidden, hidden_update_w)
            update_gate = stax.sigmoid(inp_update + hidden_update)
            reset_gate = stax.sigmoid(input_reset_apply_fun(input_reset_params, inpv) +
                                      np.dot(prev_hidden, hidden_reset_w))
            output_gate = update_gate * prev_hidden + (1 - update_gate) * np.tanh(
                input_output_apply_fun(input_output_params, inpv) +
                np.dot(reset_gate * prev_hidden, hidden_output_w))
            hidden = np.where((i < lengths)[:, None], output_gate, np.zeros_like(prev_hidden))
            return hidden, hidden

        return jax.lax.scan(apply_fun_single, init_hidden, (np.arange(inps.shape[0]), inps))

    return init_fun, apply_fun


def model(seqs, seqs_rev, lengths, *,
          latent_dim=100, emission_dim=100, transition_dim=200,
          data_dim=88, gru_dim=400, annealing_factor=1.0):
    batch_size, max_seq_length, *_ = seqs.shape

    transition = numpyro.module('transition', Transition(transition_dim, transition_dim, latent_dim),
                                input_shape=(batch_size, latent_dim))
    emitter = numpyro.module('emitter', Emitter(emission_dim, emission_dim, data_dim),
                             input_shape=(batch_size, latent_dim))

    z0 = numpyro.param('z0', np.zeros((batch_size, 1, latent_dim)))
    ones = np.ones((batch_size, max_seq_length, latent_dim))

    masks = np.repeat(np.expand_dims(np.arange(max_seq_length), axis=0), batch_size, axis=0) < \
            np.expand_dims(lengths, axis=-1)
    with numpyro.plate('data', batch_size):
        # NB: Mask is to avoid scoring 'z' using distribution at this point
        z = numpyro.sample('z', dist.Normal(0.0, ones).mask(False).to_event(2))
        z_shift = np.concatenate([z0, z[:, :-1, :]], axis=-2)
        z_loc, z_scale = transition(z_shift)

        with numpyro.handlers.scale(scale_factor=annealing_factor):
            # Actually score 'z'
            numpyro.sample('z_aux', dist.Normal(z_loc, z_scale).mask(np.expand_dims(masks, axis=-1))
                           .to_event(2), obs=z)

        emission_probs = emitter(z)
        oh_x = _one_hot_chorales(seqs)
        numpyro.sample('obs_x', dist.Bernoulli(emission_probs).mask(np.expand_dims(masks, axis=-1))
                       .to_event(2), obs=oh_x)


def guide(seqs, seqs_rev, lengths, *,
          latent_dim=100, emission_dim=100, transition_dim=200,
          data_dim=88, gru_dim=400, annealing_factor=1.0):
    batch_size, max_seq_length, *_ = seqs.shape
    seqs_rev = np.transpose(seqs_rev, axes=(1, 0, 2))
    gru = numpyro.module('gru', GRU(gru_dim), input_shape=(max_seq_length, batch_size, data_dim))
    combiner = numpyro.module('combiner', Combiner(gru_dim, latent_dim),
                              input_shape=(batch_size, gru_dim))

    masks = np.repeat(np.expand_dims(np.arange(max_seq_length), axis=0), batch_size, axis=0) < \
            np.expand_dims(lengths, axis=-1)

    h0 = numpyro.param('h0', np.zeros((batch_size, gru_dim)))
    _, hs = gru((_one_hot_chorales(seqs_rev), lengths, h0))
    hs = _reverse_padded(np.transpose(hs, axes=(1, 0, 2)), lengths)
    z_loc, z_scale = combiner(hs)
    with numpyro.plate('data', batch_size):
        with numpyro.handlers.scale(scale_factor=annealing_factor):
            numpyro.sample('z', dist.Normal(z_loc, z_scale).mask(np.expand_dims(masks, axis=-1)).to_event(2))

if __name__ == '__main__':

    ## CONFIGURE JAX TO USE TPU
    #config.FLAGS.jax_xla_backend = "tpu_driver"
    #config.FLAGS.jax_backend_target = "grpc://" + os.environ['TPU_ADDR']

    ## FETCH DATASET
    batch_size = 32
    init, get_batch = load_dataset(JSBCHORALES, batch_size=batch_size, split='train')
    ds_count, ds_indxs = init()
    seqs, seqs_rev, lengths = get_batch(0, ds_indxs)

    learning_rate = 0.0003
    beta1 = 0.96
    beta2 = 0.999
    clip_norm = 10
    lr_decay = 0.99996
    adam_params = {"step_size": lambda i: learning_rate * lr_decay ** i,
                   "b1": beta1, "b2": beta2,
                   "clip_norm": clip_norm}
    svgd = SVGD(model, WrappedGuide(guide, reinit_hide_fn=lambda site: site['name'].endswith('$params')),
                ClippedAdam(**adam_params), ELBO(), RBFKernel(), num_particles=10,
                repulsion_temperature=batch_size)

    init, get_batch = load_dataset(JSBCHORALES, batch_size=batch_size, split='train')
    ds_count, ds_indxs = init()

    num_epochs = 3  # 5000
    check_point_freq = 2
    test_eval_freq = 2

    rng_key = jax.random.PRNGKey(seed=142)
    seqs, seqs_rev, lengths = get_batch(0, ds_indxs)

    try:
        state = svgd.load_latest_checkout(rng_key)
    except:
        state = svgd.init(rng_key, seqs, seqs_rev, lengths)

    init_test, get_batch_test = load_dataset(JSBCHORALES, batch_size=batch_size, split='test')
    ds_count_test, ds_indxs_test = init_test()


    test_losses = []

    pbar = tqdm(range(num_epochs))

    for j in pbar:
        random.shuffle(ds_indxs)
        if j % check_point_freq == 0:
            svgd.store_checkout(state)
        loss = 0.
        for i in range(ds_count):
            i = j % ds_count
            seqs, seqs_rev, lengths = get_batch(i, ds_indxs)
            state, batch_loss = jax.jit(svgd.update)(state, seqs, seqs_rev, lengths)
            loss += batch_loss
        if j % test_eval_freq == 0:
            loss = 0
            for i in range(ds_count_test):
                seqs, seqs_rev, lengths = get_batch_test(i, ds_indxs_test)
                loss += svgd.evaluate(state, seqs, seqs_rev, lengths)
            test_losses.append(loss)
            with open(f'test_losses_nep{j:04}.pkl', 'wb') as f:
                pickle.dump(test_losses, f)
        pbar.set_description(f'SVGD loss (test={j % test_eval_freq == 0}): {loss}', True)
