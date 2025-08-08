"""
This module contains functions related to the state preparation using Waring decomposition, i.e. implementing the
solution of the Theorem 1 from https://arxiv.org/abs/2507.19397
"""
from typing import Iterable

import jax
import numpy as np
import optax
import scipy.optimize as opt

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import logging
from photon_catalysis.utils import *


logger = logging.getLogger(__name__)

def symmetric_tensor_decomposition(
        T: jax.Array,
        rank: int,
        seed: int = 0,
        num_iterations: int = 10000,
        tolerance: float = 1e-10,
        lr: float = 0.01) -> tuple[np.ndarray, float]:
    """
    Implements symmetric tensor decomposition (Waring decomposition for polynomials) using gradient descent.
    :param T: The tensor, must be symmetric, i.e. ``T[idx] == T[s(idx)]`` for any permutation of indices ``s``
    :param rank: The candidate rank of the decomposition
    :param seed: Seed for random initialization
    :param num_iterations: Number of steps of gradient descent
    :param tolerance: When the error is smaller than ``tolerance``, stop the optimization
    :param lr: Learning rate
    :return: Tuple ``(W, l)``, where ``W`` is the matrix corresponding to the decomposition (rows of the matrix define
        linear forms, i.e. ``W.shape[0] == rank``) and ``l`` is the loss.
    """
    degree = len(T.shape)
    num_modes = T.shape[0]
    assert (list(set(T.shape))==[num_modes])

    optimizer = optax.adam(learning_rate=lr)
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 2)
    param_scale = 1.0/rank
    x = jax.random.normal(keys[0], (rank, num_modes), dtype=jnp.float32)*param_scale
    y = jax.random.normal(keys[1], (rank, num_modes), dtype=jnp.float32)*param_scale
    factors = x+1j*y
    params = factors
    opt_state = optimizer.init(params)
    all_indices = jnp.array(list(itertools.product(range(num_modes), repeat=degree)), dtype=jnp.int32)

    T_jax = jnp.asarray(T)
    T_flattened = T_jax[tuple(all_indices.T)]

    @jax.jit
    def loss_fn(params):
        coll = jnp.take(params, all_indices, axis=1)
        products = jnp.prod(coll, axis=2)
        products = products.T
        reconstructed_T = jnp.sum(products, axis=-1)
        error = T_flattened-reconstructed_T
        loss = jnp.sum(jnp.real(error*jnp.conj(error)))
        return loss

    @jax.jit
    def update_step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        # Optax requires the conjugate gradient here
        # https://github.com/jax-ml/jax/issues/9110
        grads = jnp.conj(grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    with logging_redirect_tqdm():
        progress = tqdm(range(num_iterations), desc='Optimizing...', leave=True)
        done = False
        for it in progress:
            if not done:
                params, opt_state, loss_value = update_step(params, opt_state)
                loss_value = float(loss_value)
                if loss_value<=tolerance:
                    done = True
            if it % 100==0:
                progress.set_postfix(loss=loss_value)

    # Convert back to numpy arrays and return
    return np.array(params), float(loss_value)


def decomposition_to_linear_forms(dec: np.ndarray, degree: int) -> np.ndarray:
    rank, num_modes = dec.shape
    m = np.zeros((rank * degree, num_modes + 1), dtype=np.complex128)
    w = np.exp(2 * np.pi * 1j / degree)
    phi = np.exp(np.pi * 1j / degree) if degree % 2 == 0 else 1
    for i in range(rank):
        for j in range(degree):
            m[i * degree + j, :] = np.concatenate((np.array([1]), phi * (w ** j) * dec[i, :]))
    return np.array(m)


def waring_preparation(
        target_state: StateDict,
        candidate_ranks: Iterable[int]=(5),
        num_decompositions: int=10,
        start_seed: int=0,
        num_iterations: int=10000,
        num_rank_attempts: int=4,
        tolerance: float=1e-10,
        lr:float =0.01):
    """
    Yields tuples of the form ``(W, p, f)``, where ``W`` is a matrix which rows define set of linear forms for the
    multiport interferometer. ``p`` is the probability of successfully conditioning on having ``n`` photons in the
    ancillary mode, where ``n = rank*(modes - 1)``. ``f`` is the fidelity with the target state.

    :param target_state:
    :param candidate_ranks: Different ranks to try
    :param num_decompositions: Number of trials (each uses different random seed derived from ``start_seed``).
    :param start_seed: Random seed.
    :param num_iterations: Number of steps of gradient descent.
    :param num_rank_attempts: Number of attempts to find a decomposition of the specified rank.
    :param tolerance: When the error is smaller than ``tolerance``, consider it as a success.
    :param lr: Learning rate.
    """
    target_state = normalized_state(target_state)
    target_state_array = state_dict_to_array(target_state)
    T = state_to_tensor(target_state)
    degree = len(T.shape)
    num_modes = T.shape[0]
    T = T / get_renorm_tensor(num_modes, degree)[0]
    assert (list(set(T.shape))==[num_modes])

    rank_found = False
    for rank in candidate_ranks:

        # This could, theoretically, be used, but it consumes too much memory and works too slow
        # Instead we use SymPy to expand the product of linear forms on the level of polynomials
        # helper = StateOptimizationHelper(target_state, degree*(rank-1))
        # infidelity_fn = helper.get_loss_fn()
        # prob_fn = helper.get_prob_fn()

        for attempt in range(num_decompositions):
            logger.info(f'Searching for a decomposition of rank {rank}, attempt {attempt+1}/{num_decompositions}')
            W, loss = symmetric_tensor_decomposition(T, rank, start_seed + attempt, num_iterations, tolerance, lr)
            if loss < tolerance:
                rank_found = True
                W = decomposition_to_linear_forms(W, degree)

                logger.debug('Optimizing success probability')
                alpha = sp.Symbol('\\alpha')
                p_success, final_state = projection_prob(np.asarray(W), degree*(rank - 1), alpha)
                s, p = optimize_probability_by_scaling(p_success, alpha)
                final_state = normalized_state({k: v.subs({alpha: s}) for k, v in final_state.items()})
                final_state_array = state_dict_to_array(final_state)

                yield W, p, 1 - infidelity(final_state_array, target_state_array)

            if attempt == num_rank_attempts and not rank_found:
                logger.warning(f'Unable to find a decomposition of rank {rank}')
                break
        if rank_found:
            logger.info(f'Stopping at rank: {rank}')
            break
