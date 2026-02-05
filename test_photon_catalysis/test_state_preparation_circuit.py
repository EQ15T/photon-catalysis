import numpy as np
import pytest
import sympy as sp

from photon_catalysis.quadratic_preparation import e2_preparation
from photon_catalysis.state_preparation_circuit import StatePreparationCircuit
from photon_catalysis.utils import (
    StateDict,
    infidelity,
    polynomial_to_state,
    state_dict_to_array,
)


def test_state_preparation_circuit_end_to_end():
    state = {(2, 0, 0): 0.5, (0, 2, 0): 0.5, (0, 0, 2): 0.5, (1, 0, 1): 0.5}
    W, _, _ = e2_preparation(state, False)

    # Extract the unitaries
    circuit = StatePreparationCircuit(W, state)

    # Heisenberg-style simulation tracking the evolution of creation operators
    variables = sp.symbols(f"a^\\dagger_0:{circuit.num_modes}")
    ancilia = variables[0]
    p = 1
    for u in circuit.unitaries:
        # Photon addition
        p *= ancilia
        # Unitary
        product = sp.Matrix(u).T * sp.Matrix(variables)
        substitutions = zip(variables, list(product.col(0)))
        p = p.subs(substitutions, simultaneous=True)

    # Projection
    p = sp.diff(p, ancilia).subs(ancilia, 0)

    target_state = state_dict_to_array(state)
    final_state = state_dict_to_array(polynomial_to_state(sp.Poly(p)))
    assert infidelity(target_state, final_state) <= 1e-8


def test_unitary_completion():
    w = np.array([0.0, 0.6, 0.8j, 0.0])

    def same_up_to_phase(a, b):
        support = a != 0
        if not np.any(support):
            return False
        ratio = b[support] / a[support]
        return np.allclose(ratio / ratio[0], 1)

    # Completion that does not preserve sparsity
    u = StatePreparationCircuit._complete_unitary(w, keep_sparse=False).T
    assert np.all(np.isclose(u.conj().T @ u, np.eye(4)))
    assert same_up_to_phase(u[0], w)
    assert np.count_nonzero(u[1]) == 3
    assert np.count_nonzero(u[2]) == 3
    assert np.count_nonzero(u[3]) == 1

    # Completion that affects the least number of modes
    u = StatePreparationCircuit._complete_unitary(w, keep_sparse=True).T
    assert np.all(np.isclose(u.conj().T @ u, np.eye(4)))
    assert same_up_to_phase(u[0], w)
    assert np.count_nonzero(u[1]) == 1
    assert np.count_nonzero(u[2]) == 2
    assert np.count_nonzero(u[3]) == 1
