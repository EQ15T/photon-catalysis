import copy

import pytest
import sympy as sp

from photon_catalysis.benchmark_states import benchmark_states
from photon_catalysis.quadratic_preparation import e2_preparation, esp, qpoly2mat
from photon_catalysis.utils import StateDict, normalized_state, state_to_polynomial

quadratic_states = [s for s in benchmark_states if s.num_photons == 2]


def verify_exact_e2(target_state: StateDict, W: sp.Matrix) -> bool:
    """
    Verifies result of solve_e2
    :return: True of False
    """
    B = qpoly2mat(state_to_polynomial(target_state)[0])
    W = copy.copy(W)
    W.col_del(0)

    xs = sp.symbols(f"x_1:{W.shape[1] + 1}")
    xs_vec = sp.Matrix(xs)
    p = esp(2, W * xs_vec, True)
    q = sp.Poly((xs_vec.T * B * xs_vec)[0], *xs, extension=True)

    # in the symbolic case we expect exact match
    return p.expr == q.expr


@pytest.mark.parametrize("quadratic_state", quadratic_states, ids=lambda s: s.name)
def test_quadratic_preparation(quadratic_state):
    state = normalized_state(quadratic_state.state)
    W, _, fid = e2_preparation(state, False)
    assert fid > 0.999

    W, _, fid = e2_preparation(state, True)
    assert verify_exact_e2(state, W)
