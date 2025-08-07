"""
This module contains functions related to the state preparation using direct optimization on tensors. Theoretically,
it corresponds to the solution of the decomposition in Theorem 2 in https://arxiv.org/abs/2507.19397, or, alternatively,
it could be viewed as a generalization of Kopulov's method proposed in https://doi.org/10.1103/sv6z-v1gk.
"""
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
        lr:float =0.01):
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
    def update_step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        grads = jnp.conj(grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

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
            progress = tqdm(range(num_iterations), desc='Optimizing...', leave=True)
            for it in progress:
                params, opt_state, loss_value = update_step(params, opt_state)
                if it % 100 == 0:
                    progress.set_postfix(loss=loss_value)

        w = params
        p = prob_fn(w)
        f = 1 - loss_fn(w)

        # to get the state after projection
        # R = helper.conditioned_projector(W_to_stellar_tensor(w) * helper.renorm_tensor)
        # R = state_array_to_dict(R)
        # print(state_to_string(R, 1e-3))

        yield w, p, f
