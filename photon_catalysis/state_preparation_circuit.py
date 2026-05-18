"""
This module contains an abstract representation of the state preparation circuit, as a list
of unitaries to be sandwiched with photon addition. This representation also contains all
the side-information (number of modes, PNR number) to allow a conversion to an actual
implementation (eg as boson sampling).
"""

from typing import List

import numpy as np

from photon_catalysis.utils import StateDict, state_to_string

class StatePreparationCircuit:
    """
    Abstract representation of a state preparation circuit
    """

    def __init__(self, w: np.array, state: StateDict, corr_addition: float = 1):
        """
        :param corr_addition: The correction for the photon addition
        """
        num_target_photons = sum(list(state.items())[0][0])
        self._unitaries = self._linear_forms_to_unitaries(np.asarray(w), corr_addition)
        self._num_additions, self._num_modes = w.shape
        self._num_target_photons = num_target_photons
        self._pnr = self._num_additions - num_target_photons
        self._name = state_to_string(state)
        self._state = state

    @property
    def state(self) -> StateDict:
        return self._state

    @property
    def unitaries(self) -> List[np.array]:
        return self._unitaries

    @property
    def num_additions(self) -> int:
        return self._num_additions

    @property
    def num_modes(self) -> int:
        return self._num_modes

    @property
    def pnr(self) -> int:
        return self._pnr

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def _linear_forms_to_unitaries(w: np.ndarray, corr_addition: float = 1) -> List[np.ndarray]:
        """
        Converts a list of linear form to a list of unitaries.
        This implements algorithm 1 from Appendix C. https://arxiv.org/pdf/2507.19397
        """
        S_inv = np.eye(w.shape[1])
        S_inv[0, 0] = 1 / corr_addition
        w = w.copy()
        n, _ = w.shape
        unitaries = []
        for i in range(n):
            # Find a unitary matrix whose first row corresponds to the linear form
            u = StatePreparationCircuit._complete_unitary(w[i, :], False)
            # Back-propagate the basis change to the previous linear forms
            for j in range(i + 1, n):
                w[j, :] = w[j, :] @ u.conj() @ S_inv
            unitaries.append(u)
        return unitaries[::-1]

    @staticmethod
    def _complete_unitary(w: np.ndarray, keep_sparse: bool = False) -> np.ndarray:
        """
        Build a full unitary that transforms the first mode into the superposition
        of modes described in w (normalized), acts unitarily on the other affected
        modes, and trivially on the remaining modes.

        :param w: linear combination of modes the first mode is transformed into
        :param keep_sparse: if True, the orthogonalization will preserve the block
           structure of the matrix and will only act on the modes in the support of w

        This implements algorithm 2 from Appendix C. https://arxiv.org/pdf/2507.19397
        """
        n = len(w)

        # Initialize the unitary
        unitary = np.zeros((n, n), dtype=complex)
        if not keep_sparse:
            # If we don't care about producing a non-sparse unitary, or if
            # w is not sparse anyway, we can just orthogonalize the full
            # matrix instead of just orthogonalizeing the non-trivial
            # subspace
            unitary[:, 0] = w
            q, _ = np.linalg.qr(unitary)
            return q

        # Check in which rows we will have to add orthogonal elements
        support = [i for i in range(n) if w[i] != 0]
        n_s = len(support)

        # Orthogonalize the non-trivial subspace
        sub_u = np.eye(n_s, dtype=complex)
        sub_u[:, 0] = w[support]
        q, _ = np.linalg.qr(sub_u)
        sub_u = q.T

        # Update the non-trivial subspace
        affected_cols = support
        affected_rows = [0] + support[1:]
        unitary[np.ix_(affected_rows, affected_cols)] = sub_u

        # Update the trivial subspace
        trivial_space = np.setdiff1d(np.arange(1, n), support)
        unitary[trivial_space, trivial_space] = 1

        if 0 not in affected_cols:
            unitary[support[0], 0] = 1

        return unitary.T

    def to_perceval(
        self, photon_addition_r: float = 0.9, decompose_unitaries: bool = True
    ):
        """
        Convert the circuit to its DV boson sampling representation, as a Perceval circuit

        :param photon_addition_r: The reflectivity of the beam-splitter performing photon addition
        :param decompose_unitaries: Whether the unitaries should be broken down into individual BS/PS
        :return: Tuple (circuit: Circuit, input_state: BasicState, post_select: PostSelect) for simulating
            the circuit with Perceval
        """
        try:
            import perceval as pcvl
            from perceval import Matrix
            from perceval.components import BS, PERM, PS, Unitary
            from perceval.utils.postselect import PostSelect
        except ModuleNotFoundError:
            raise SystemExit(
                "This method requires Perceval to be installed"
                "Please install the optional dependencies, eg with pip install -e .[boson_sampling]"
            )

        num_additions = self.num_additions
        unitaries = self.unitaries
        num_modes = self.num_modes
        pnr = self.pnr

        num_total_modes = num_modes + num_additions

        circuit = pcvl.Circuit(m=num_total_modes, name=self.name)

        for i in range(num_additions):
            # Photon addition with a beam-splitter
            bs = BS(BS.r_to_theta(photon_addition_r))
            circuit //= (num_additions - 1, bs)

            # Shuffle the photon addition modes
            if i != num_additions - 1:
                permutation = list(range(num_additions))
                swap = (num_additions - 1, num_additions - 2 - i)
                permutation[swap[0]] = swap[1]
                permutation[swap[1]] = swap[0]
                circuit //= PERM(permutation)

            m = Matrix(unitaries[i])
            if decompose_unitaries:
                unitary_subcircuit = pcvl.Circuit.decomposition(
                    m,
                    BS(theta=pcvl.P("theta"), phi_tr=pcvl.P("phi")),
                    phase_shifter_fn=PS,
                )
            else:
                unitary_subcircuit = Unitary(m)
            circuit //= (num_additions, unitary_subcircuit)

        # The single photons are initially in the addition anciliary
        # modes (Figure 2 of the paper)
        input_state = pcvl.BasicState([1] * num_additions + [0] * num_modes)

        # Create a post-selection rule that checks that there are no
        # photons on the photon addition ancilia modes, and that
        # all catalysis photons are retrieved
        post_select = PostSelect("&".join(f"[{i}] == 0" for i in range(num_additions)))
        post_select.merge(PostSelect(f"[{num_additions}] == {pnr}"))

        return circuit, input_state, post_select

    def to_sf(self, squeezing_r: float = 0.1):
        """
        Convert the circuit to its Gaussian boson sampling representation, as a StrawberryField program

        :param squeezing_r: The squeezing parameter of the TMS implementing photon addition
        :return: Tuple (program: Program, indexer: Tuple) with the SF program and the indexer to
            post-select the result
        """
        try:
            import strawberryfields as sf
            from strawberryfields import ops
        except ModuleNotFoundError:
            raise SystemExit(
                "This method requires StrawberryFields to be installed"
                "Please install the optional dependencies, eg with pip install -e .[gaussian_boson_sampling]"
            )
        num_additions = self.num_additions
        unitaries = self.unitaries
        num_modes = self.num_modes
        num_total_modes = num_modes + num_additions
        program = sf.Program(num_total_modes)
        with program.context as q:
            for i in range(num_additions):
                ops.S2gate(squeezing_r, 0) | (q[i], q[num_additions])
                ops.Interferometer(unitaries[i]) | [
                    q[i] for i in range(num_additions, num_total_modes)
                ]
        post_select = [1] * num_additions + [self.pnr]
        post_select_indexer = tuple(post_select) + (slice(None),) * (
            num_total_modes - len(post_select)
        )
        return program, post_select_indexer



class ExactStatePreparationCircuitBS(StatePreparationCircuit):
    def __init__(self, w: np.array, state: StateDict, r_addition: float = 0.99):
        super().__init__(w, state, np.sqrt(r_addition))
        self._r_addition = r_addition
    
    def to_perceval(self, decompose_unitaries: bool = True):
        return super().to_perceval(self._r_addition, decompose_unitaries)


class ExactStatePreparationCircuitSQ(StatePreparationCircuit):
    def __init__(self, w: np.array, state: StateDict, r_addition: float = 0.99):
        super().__init__(w, state, 1 / np.cosh(r_addition))
        self._r_addition = r_addition
    
    def to_sf(self):
        return super().to_sf(self._r_addition)

