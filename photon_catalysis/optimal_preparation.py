"""
This module contains functions related to the state preparation using direct optimization on tensors. Theoretically,
it corresponds to the solution of the decomposition in Theorem 2 in https://arxiv.org/abs/2507.19397, or, alternatively,
it could be viewed as a generalization of Kopulov's method proposed in https://doi.org/10.1103/sv6z-v1gk.
"""
from functools import partial

import jax
import numpy as np
import optax
import scipy.optimize as opt

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import logging
from photon_catalysis.utils import *


logger = logging.getLogger(__name__)

def optimal_preparation(
        target_state: StateDict,
        extra_photons: int = 1,
        num_decompositions: int=10,
        start_seed: int=0,
        num_iterations: int = 10000,
        lr:float =0.01,
        optimize_prob: bool = True):
    """
    Yields tuples of the form ``(W, p, f)``, where ``W`` is a matrix which rows define set of linear forms for the
    multiport interferometer. ``p`` is the probability of successfully conditioning on having ``extra_photons`` in the
    ancillary mode, ``f`` is the fidelity with the target state.

    :param target_state:
    :param extra_photons:
    :param num_decompositions: Number of trials (each uses different random seed derived from ``start_seed``).
    :param start_seed: Random seed.
    :param num_iterations: Number of steps of gradient descent.
    :param lr: Learning rate.
    """

    target_state = normalized_state(target_state)

    keys = list(target_state.keys())
    degree = sum(keys[0])
    num_modes = len(keys[0])

    helper = StateOptimizationHelper(target_state, extra_photons)
    loss_fn = helper.get_loss_fn()
    prob_fn = helper.get_prob_fn()

    @jax.jit
    def prob_loss_fn(w, s):
        return -prob_fn(w, s)

    @jax.jit
    def update_step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        grads = jnp.conj(grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def update_prob_step(scale, w, opt_state):
        loss, grads = jax.value_and_grad(prob_loss_fn, argnums=1)(w, scale)
        grads = jnp.conj(grads)
        updates, opt_state = optimizer.update(grads, opt_state, scale)
        scale = optax.apply_updates(scale, updates)
        return scale, opt_state, loss

    for seed_value in range(num_decompositions):
        key = jax.random.PRNGKey(start_seed + seed_value)
        key1, key2 = jax.random.split(key)
        param_scale = 0.5

        shape = (degree + extra_photons, num_modes + 1)
        x = jax.random.normal(key1, shape) * param_scale
        y = jax.random.normal(key2, shape) * param_scale
        w = x + 1j * y

        params = w
        optimizer = optax.adam(learning_rate=lr)
        opt_state = optimizer.init(params)

        with logging_redirect_tqdm():
            progress = tqdm(range(num_iterations), desc='Optimizing fidelity...', leave=True)
            for it in progress:
                params, opt_state, loss_value = update_step(params, opt_state)
                if it % 100 == 0:
                    progress.set_postfix(loss=loss_value)

        w = params
        if optimize_prob:
            w = normalize_W(w)
            scale = jax.random.normal(key1, ()) + 1.j * jax.random.normal(key2, ())
            optimizer = optax.adam(learning_rate=lr)
            opt_state = optimizer.init(scale)
            with logging_redirect_tqdm():
                progress = tqdm(range(num_iterations), desc='Optimizing probability...', leave=True)
                for it in progress:
                    scale, opt_state, prob_value = update_prob_step(scale, w, opt_state)
                    if it % 100 == 0:
                        progress.set_postfix(prob=-prob_value, scale=scale)
            w = normalize_W(w, complex(scale))

        p = prob_fn(w, 1)
        f = 1 - loss_fn(w)

        # to get the state after projection
        # R = helper.conditioned_projector(W_to_stellar_tensor(w) * helper.renorm_tensor)
        # R = state_array_to_dict(R)
        # print(state_to_string(R, 1e-3))

        yield w, p, f
