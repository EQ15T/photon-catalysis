import copy

import pytest

from states import all_states
from photon_catalysis.quadratic_preparation import e2_preparation, esp, qpoly2mat
from photon_catalysis.utils import StateDict, state_to_polynomial, normalized_state

import sympy as sp


quadratic_states = {name: state for name, state in all_states.items() if sum(list(state.keys())[0]) == 2}


def verify_exact_e2(target_state: StateDict, W: sp.Matrix) -> bool:
    """
    Verifies result of solve_e2
    :return: True of False
    """
    B = qpoly2mat(state_to_polynomial(target_state)[0])
    W = copy.copy(W)
    W.col_del(0)

    xs = sp.symbols(f'x_1:{W.shape[1] + 1}')
    xs_vec = sp.Matrix(xs)
    p = esp(2, W * xs_vec, True)
    q = sp.Poly(( xs_vec.T * B * xs_vec )[0], *xs, extension=True)

    # in the symbolic case we expect exact match
    return p.expr == q.expr


@pytest.mark.parametrize('state_name,state', quadratic_states.items())
def test_quadratic_preparation(state_name, state):
    state = normalized_state(state)
    W, _, fid = e2_preparation(state, False)
    assert fid > 0.999

    W, _, fid = e2_preparation(state, True)
    assert verify_exact_e2(state, W)
