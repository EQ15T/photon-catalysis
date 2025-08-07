"""
This modules contains functions shared by different methods discussed in the paper.

In general, there are 5 ways utilized to represent core states:

    * list of kets, i.e. ``[(0, 4, 0), (1, 2, 1), (2, 0, 2)]``. This type is somewhat defective as it only supports
      kets to have coefficient 1 or 0, but it may be sometimes convenient.
    * ``StateDict``, i.e. ``{(0, 4, 0): 1, (1, 2, 1): 1, (2, 0, 2): 1}``. Should be pretty clear.
    * State Array. For a core state having ``d`` photons in ``M`` modes, this type is an array of shape
      ``(d+1, d+1, ..., d+1)`` (M times). It could be thought as of dense representation of ``StateDict``.
    * State Tensor. For a core state having ``d`` photons in ``M`` modes, this type is an array of shape
      ``(M, ..., M)`` (d times). This is the "true" symmetric tensor corresponding to the state, i.e. we have
      the following property: ``T[idx] == T[s(idx)]`` for any permutation of indexes ``s``. Intuitively, it encodes
      "which indices to multiply together", i.e. the element ``(0, ..., 0)`` stores the coefficient of |d, 0, ..., 0>.
    * `sympy.Poly`. When polynomial is used, it corresponds directly to the stellar polynomial. Note that the conversion
       between states and polynomials requires "renormalization" with square roots of factorials!

    State tensor representation is used only when expanding the product of linear forms, i.e. after ``L_to_tensor``.
"""
import copy

import math
import sympy as  sp
import numpy as np
import jax
import jax.numpy as jnp

from functools import reduce
from operator import mul
import itertools


StateDict = dict[tuple[int, ...], sp.Basic]



def kets_to_state_dict(kets: list[tuple[int, ...]]) -> StateDict:
    """
    Converts list of kets to the state dict

    >>> kets_to_state_dict([(0, 4, 0), (1, 2, 1), (2, 0, 2)])
    {(0, 4, 0): 1, (1, 2, 1): 1, (2, 0, 2): 1}

    :param kets: List of tuples giving the kets that has weight 1
    :return: The dictionary describing the state
    """
    return dict(zip(kets, [sp.sympify(1)] * len(kets)))

def state_norm(state_dict: StateDict) -> sp.Basic:
    """
    :return: Norm of the state dict
    """
    return sp.sqrt(sum([abs(v) ** 2 for v in state_dict.values()], start=sp.sympify(0)))

def normalized_state(state_dict: StateDict) -> StateDict:
    """
    :return: Normalized state
    """
    ret = copy.deepcopy(state_dict)
    p_sum = state_norm(state_dict)
    for k in state_dict.keys():
        ret[k] = state_dict[k] / p_sum
    return ret

def state_dict_to_array(state_dict: StateDict) -> jax.Array:
    """

    >>> T = state_dict_to_array({(0, 4, 0): 1, (1, 2, 1): 1, (2, 0, 2): 1})
    >>> assert(T[0, 4, 0] == 1)
    >>> assert(T[0, 0, 4] == 0)

    :return: JAX array representing the state
    """
    state_dict = normalized_state(state_dict)

    keys = list(state_dict.keys())
    degree = sum(keys[0])
    num_modes = len(keys[0])

    T = np.zeros(tuple([degree + 1] * num_modes), dtype=jnp.complex64)
    for ket, amplitude in state_dict.items():
        T[ket] = complex(amplitude)
    return jnp.array(T)

def state_array_to_dict(state_array: jax.Array, cutoff: float=1e-5) -> StateDict:
    degree = state_array.shape[0] - 1
    num_modes = len(state_array.shape)
    ret = {}
    for idx in itertools.product(range(degree + 1), repeat=num_modes):
        val = state_array[idx]
        if abs(val) > cutoff:
            ret[idx] = val
    return ret


def state_to_string(state_dict: StateDict, cutoff: float=1e-4) -> str:
    """
    :param state_dict: Dictionary describing the state
    :param cutoff: If the coefficient is less than cutoff, it is not printed
    :return: String representation of the state
    """
    def ket_str(ket: tuple[int], separator: str='') -> str:
        s = separator.join(map(str, list(ket)))
        return f'|{s}>'
    def cut(v):
        re = v.real if abs(v.real) > cutoff else 0
        im = v.imag if abs(v.imag) > cutoff else 0
        return re + 1j * im if im != 0 else re
    return ' + '.join(f'({cut(complex(v)):.3f}){ ket_str(k) }' for k, v in state_dict.items() if abs(v) >= cutoff)

def make_variables(num_modes: int) -> list[sp.Symbol]:
    return sp.symbols(f"a^\\dagger_1:{num_modes + 1}")

def polynomial_to_state(polynomial: sp.Poly) -> StateDict:
    state_dict = {}
    for term, coefficient in polynomial.terms():
        scale = sp.sqrt(reduce(mul, [sp.factorial(c) for c in term]))
        state_dict[term] = scale * coefficient
    return state_dict

def state_to_polynomial(state_dict: dict[tuple[int], sp.Basic]) -> tuple[sp.Poly, list[sp.Symbol]]:
    num_modes = max(len(k) for k in state_dict.keys())
    variables = make_variables(num_modes)
    p = 0
    for ket, coefficient in state_dict.items():
        monomial = reduce(mul, (variables[i]**k for i, k in enumerate(ket)))
        scale = sp.sqrt(reduce(mul, [sp.factorial(k) for k in ket]))
        p += coefficient / scale * monomial
    return sp.Poly(p, extension=True), variables

def state_to_tensor(state_dict: StateDict) -> jax.Array:
    num_modes = max(len(k) for k in state_dict.keys())
    degree = sum(list(state_dict.keys())[0])
    T = np.zeros(tuple([num_modes] * degree), dtype=complex)
    for monomial, coefficient in state_dict.items():
        indices = []
        for i, count in enumerate(monomial):
            indices.extend([i] * count)
        permutations = list(itertools.permutations(indices))
        scale = 1 / len(permutations)
        for permutation_index in permutations:
            T[permutation_index] += coefficient * scale
    return jnp.array(T)


@jax.jit
def normalized_state_array(s: jax.Array) -> tuple[jax.Array, jnp.complex64]:
    """
    Normalizes a state represented as an array
    :return: The normalized state and the norm
    """
    p = jnp.sum(s * s.conj())
    return s / jnp.sqrt(p), p

@jax.jit
def infidelity(x: jax.Array, y: jax.Array):
    """Returns the infidelity between two states represented as arrays"""
    return 1 - jnp.abs(jnp.sum(x * jnp.conj(y))) ** 2

@jax.jit
def W_to_stellar_tensor(w: jax.Array) -> jax.Array:
    """
    Compute the tensor corresponding to the stellar polynomial of L, i.e. of the product of linear forms.
    Note that this tensor does not contain factorial factors that emerge when going from stellar polynomial
    to the actual core state. To get these factors multiply by the renormalization tensor, i.e. by ``get_renorm_tensor()``

    :param w: The matrix W, where each row defines a linear form
    :return: The tensor ("state tensor") corresponding to the product of linear forms
    """
    p = w[0]
    for v in w[1:]:
        p = jnp.tensordot(p, v, axes=0)
    return p

@jax.jit
def W_to_monic(w: jax.Array) -> jax.Array:
    """
    Rescales given linear forms such that their product represent a monic polynomial, without changing the norm

    :param w: The matrix W, where each row defines a linear form
    :return: Same W, except that the resulting polynomial is monic
    """
    r = w
    for i in range(w.shape[0]):
        r[i] = w[i] / w[i, 0]
    return normalized_state_array(r)

def get_renorm_tensor(num_modes: int, degree: int) -> tuple[jax.Array, jax.Array]:
    """
    Computes renormalization tensor from stellar tensor (stellar polynomial) to state tensor.

    :param num_modes: Number of modes
    :param degree: Degree of the polynomial (total number of photons)
    :return: The tuple ``(T, I)``. ``T`` is The tensor that could be multiplied by a state tensor of
        a stellar polynomial to get the state tensor of the actual state. ``I`` is the list of indices for ``T``.
    """
    renorm_tensor = np.zeros(tuple([num_modes]*degree), dtype=np.float32)
    idx = list(itertools.product(range(num_modes), repeat=degree))
    for index in idx:
        ket = [0]*(num_modes)
        for i in index:
            ket[i] += 1
        scale = np.sqrt(reduce(mul, [math.factorial(k) for k in ket]))
        renorm_tensor[index] = scale
    return jnp.array(renorm_tensor), jnp.array(idx)


class StateOptimizationHelper:
    def __init__(self, target_state: StateDict, extra_photons: int):
        self.target_state_array = state_dict_to_array(target_state)
        self.extra_photons = extra_photons
        self.degree = self.target_state_array.shape[0] - 1
        self.num_modes = len(self.target_state_array.shape)

        self.renorm_tensor, self.source_indices =\
            get_renorm_tensor(self.num_modes + 1, self.degree + self.extra_photons) # +1 -- ancillary mode

        self.conditioned_projector = self._state_tensor_to_state_array(0)
        self.projector = self._state_tensor_to_state_array()


    def get_loss_fn(self):
        @jax.jit
        def loss_fn(w):
            state_array, _ = normalized_state_array(self.conditioned_projector(W_to_stellar_tensor(w) * self.renorm_tensor))
            return jnp.real(infidelity(state_array, self.target_state_array))
        return loss_fn

    def get_prob_fn(self):
        @jax.jit
        def compute_prob_success(w):
            w_tensor = W_to_stellar_tensor(w) * self.renorm_tensor
            _, p_success = normalized_state_array(self.conditioned_projector(w_tensor))
            _, p_success_scale = normalized_state_array(self.projector(w_tensor))
            return jnp.real(p_success / p_success_scale)
        return compute_prob_success

    def _state_tensor_to_state_array(self, projection_mode_index=None):
        if projection_mode_index is None:
            size = (self.degree + self.extra_photons + 1,) * (self.num_modes + 1)
        else:
            size = (self.degree + 1,) * self.num_modes

        @jax.jit
        def project(T: jnp.array) -> jnp.array:
            def tensor_index_to_ket(index):
                # Convert tensor indices to ket
                # Eg:
                # (1, 0, 0) represents a_0^\dagger * a_1^\dagger * a_1^\dagger
                # and thus |120>
                ket = jnp.zeros(self.num_modes + 1, dtype=jnp.int32)
                ket = ket.at[index].add(1)
                if projection_mode_index is not None:
                    # If the target mode has the target number of photon,
                    # keep in the projection, otherwise discard
                    matches_projection = ket[projection_mode_index] == self.extra_photons
                    ket = jnp.where(matches_projection, ket, jnp.zeros_like(ket))
                    ket = jnp.concatenate([ket[:projection_mode_index], ket[projection_mode_index + 1:]])
                    amplitude = jnp.where(matches_projection, T[tuple(index)], 0.0)
                else:
                    amplitude = T[tuple(index)]
                return ket, amplitude

            kets, amplitudes = jax.vmap(tensor_index_to_ket)(self.source_indices)
            T_state = jnp.zeros(size, dtype=jnp.complex64)
            T_state = T_state.at[tuple(kets.T)].add(amplitudes)
            return T_state

        return project

