import jax
import jax.numpy as jnp
from jax.experimental import stax

import numpyro


def _one_hot_chorales(seqs, num_nodes=88):
    return jnp.sum(jnp.array((seqs[..., None] == jnp.arange(num_nodes + 1)), 'int'), axis=-2)[..., 1:]


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
            hidden_update = jnp.dot(prev_hidden, hidden_update_w)
            update_gate = stax.sigmoid(inp_update + hidden_update)
            reset_gate = stax.sigmoid(input_reset_apply_fun(input_reset_params, inpv) +
                                      jnp.dot(prev_hidden, hidden_reset_w))
            output_gate = update_gate * prev_hidden + (1 - update_gate) * jnp.tanh(
                input_output_apply_fun(input_output_params, inpv) +
                jnp.dot(reset_gate * prev_hidden, hidden_output_w))
            hidden = jnp.where((i < lengths)[:, None], output_gate, jnp.zeros_like(prev_hidden))
            return hidden, hidden

        return jax.lax.scan(apply_fun_single, init_hidden, (jnp.arange(inps.shape[0]), inps))

    return init_fun, apply_fun


def model(seqs, lengths, gru_dim=300, decoder_dim=500, stochastic_dim=100,
          data_dim=88):
    batch_size, max_seq_length, *_ = seqs.shape
    gru = numpyro.module('gru', GRU(gru_dim), input_shape=(max_seq_length, batch_size, data_dim))
    decoder_mean = numpyro.module('decoder_mean', stax.serial(
        stax.Dense(decoder_dim),
        stax.Relu(),
        stax.Dense(stochastic_dim)
    ), input_shape=(batch_size, gru_dim + stochastic_dim))
    decoder_scale = numpyro.module('decoder_scale', stax.serial(
        stax.Dense(decoder_dim),
        stax.Relu(),
        stax.Dense(stochastic_dim),
        stax.Softplus()
    ))
    h0 = jnp.zeros((batch_size, data_dim))
    z0 = jnp.zeros((batch_size, data_dim))
    ds = gru((_one_hot_chorales(seqs), lengths, h0))
    # numpyro.scan()