"""
This module contains functions related to the state preparation using Corollary 3 from https://arxiv.org/abs/2507.19397,
i.e. exact solution to the optimal preparation of 2 photon state in M modes.

No heuristics are used in this case and we actually are able to provide a precise symbolical solution using linear algebra
functions from SymPy. However, practically, for large number of modes it works much slower than using NumPy to get a
solution numerically.
"""
from typing import Union

import numpy as np
import sympy as sp

from photon_catalysis.utils import *

import logging
logger = logging.getLogger(__name__)


def symmetric_part(A: sp.Matrix) -> sp.Matrix:
    """
    :param A: Matrix
    :return: Symmetric part of the matrix
    """
    return (A + A.T) / sp.sympify(2)

def qpoly2mat(poly: sp.Poly) -> sp.Matrix:
    """
    :return: The matrix of the given quadratic homogeneous polynomial
    """
    def qmon2idx(mon):
        mon = list(mon)
        i = 0
        for k, v in enumerate(mon):
            if v != 0:
                i = k
                mon[k] -= 1
                break
        j = 0
        for k, v in enumerate(mon):
            if v != 0:
                j = k
                mon[k] -= 1
                break
        assert (sum(mon)==0)
        return i, j

    A = sp.zeros(len(poly.gens), len(poly.gens))
    for mon in poly.as_dict():
        i, j = qmon2idx(mon)
        A[i,j] = poly.coeff_monomial(mon)
    A = symmetric_part(A)
    poly_rec = sp.Poly((sp.Matrix(list(poly.gens)).T * A * sp.Matrix(list(poly.gens)))[0], *poly.gens, extension=True)
    assert(poly_rec == poly)
    return A

def get_e2_mat(n) -> sp.Matrix:
    """
    :param n: Number of variables
    :return: Matrix of second degree elementary symmetric polynomial of n variables
    """
    return ( sp.ones(n, n) - sp.eye(n, n) ) / sp.simplify(2)



def esp(k, xs, return_poly=True) -> Union[sp.Expr, sp.Poly]:
    """
    Returns SymPy polynomial representing elementary symmetric polynomial of degree k in variables xs
    :param return_poly: if True, casts expression to polynomial in suitable field extension
    :return:
    """
    def esp_rec(k, xs) -> sp.Expr:
        if k == 0:
            return sp.sympify(1)
        if k == 1:
            return sum(xs, sp.sympify(0))
        s = sp.sympify(0)
        for j in range(len(xs) - k + 1):
            s += xs[j] * esp_rec(k - 1, xs[j + 1:])
        return s

    r = esp_rec(k, xs)
    if not return_poly:
        return r
    return sp.Poly(r.expand(), extension=True)



def mat_sqrt_sp(A: sp.Matrix) -> sp.Matrix:
    """
    Returns symbolical square root of a matrix

    :return: A matrix S, such that ``S.T @ S == A``
    """
    return A**(1 / sp.sympify(2))

def mat_sqrt_np(A: np.ndarray) -> np.ndarray:
    """
    Returns numerical square root of a matrix

    :return: A matrix S, such that ``S.T @ S == A``
    """
    evalues, evectors = np.linalg.eig(np.asarray(A).astype(complex))
    return evectors * np.sqrt(evalues) @ np.linalg.inv(evectors)


SymbolicOrNumeric = Union[sp.Matrix, np.ndarray]

def solve_e2(B: SymbolicOrNumeric) -> SymbolicOrNumeric:
    """
    Finds linear transformation W such that e_2( W*x ) == x.T * B * x
    """
    if isinstance(B, sp.Matrix):
        mat_sqrt = mat_sqrt_sp
        mat_inv = lambda x: x.inv()
    else:
        mat_sqrt = mat_sqrt_np
        mat_inv = np.linalg.inv

    A = get_e2_mat(B.shape[0])
    Sa = mat_sqrt(A)
    Sb = mat_sqrt(B)
    W = mat_inv(Sa) @ Sb
    return W


def e2_preparation(state: StateDict, symbolical: bool = False):
    """
    Returns a tuple ``(W, p, f)`` where ``W`` is a matrix which rows define set of linear forms for the
    multiport interferometer. ``p`` is the probability of successfully conditioning on having ``M - 2`` photons, where
    ``M`` is the number of modes.

    :param state: Any 2 photon state
    :param symbolical: Whether to use symbolic calculations. Might be slow, but could be used to verify result exactly
    :return: Tuple ``(W, p, f)``
    """
    state = normalized_state(state)
    poly = state_to_polynomial(state)[0]
    B = qpoly2mat(poly)
    if not symbolical:
        B = np.asarray(B).astype(complex)
    W = solve_e2(B)
    if symbolical:
        W = W.col_insert(0, sp.ones(W.shape[0], 1))
    else:
        W = np.concatenate((np.ones((W.shape[0], 1)), W), axis=1)

    # for probability calculation always use numpy, as otherwise it's too slow
    W_np = np.asarray(W).astype(complex)
    logger.debug('Optimizing success probability')
    alpha = sp.Symbol('\\alpha')
    p_success, final_state = projection_prob(W_np, W_np.shape[1] - 3, alpha)
    s, p = optimize_probability_by_scaling(p_success, alpha)
    final_state = normalized_state({k: v.subs({alpha: s}) for k, v in final_state.items()})
    final_state_array = state_dict_to_array(final_state)

    return W, p, 1 - infidelity(final_state_array, state_dict_to_array(state))
