from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from photon_catalysis.optimal_preparation import optimal_preparation
from photon_catalysis.state_preparation_circuit import StatePreparationCircuit
from photon_catalysis.utils import (
    StateDict,
    kets_to_state_dict,
    normalized_state,
    state_to_string,
)

try:
    import perceval as pcvl
except ImportError:
    raise SystemExit(
        "This example requires Perceval to be installed"
        "Please install the optional dependencies, eg with pip install -e .[boson_sampling]"
    )

from exqalibur import StateVector
from perceval import Matrix
from perceval.components import BS, PERM, PS, Unitary
from perceval.utils.postselect import PostSelect


def circuit_to_perceval_simulation(
    state_preparation: StatePreparationCircuit,
    photon_addition_r: float = 0.9,
    decompose_unitary: bool = False,
) -> Tuple[pcvl.Circuit, pcvl.Simulator, pcvl.BasicState]:
    """
    Converts an abstract description of a state preparation circuit into a Perceval circuit

    :param state_preparation: A state preparation circuit object
    :photon_addition_r: The reflectivity of the beam-splitter performing photon addition
    :decompose_unitary: Whether the unitaries should be broken down into individual BS/PS
    """
    num_additions = state_preparation.num_additions
    unitaries = state_preparation.unitaries
    num_modes = state_preparation.num_modes
    pnr = state_preparation.pnr

    num_total_modes = num_modes + num_additions

    circuit = pcvl.Circuit(m=num_total_modes, name=state_preparation.name)

    for i in range(num_additions):
        # Photon addition with a beam-splitter
        bs = BS(BS.r_to_theta(photon_addition_r))
        circuit //= (num_additions - 1, bs)

        # Shuffle the photon addition modes
        if i != num_additions - 1:
            permutation = list(range(num_total_modes))
            swap = (num_additions - 1, num_additions - 2 - i)
            permutation[swap[0]] = swap[1]
            permutation[swap[1]] = swap[0]
            circuit //= PERM(permutation)

        m = Matrix(unitaries[i]).T
        if decompose_unitary:
            unitary_subcircuit = pcvl.Circuit.decomposition(
                m, BS(theta=pcvl.P("theta"), phi_tr=pcvl.P("phi")), phase_shifter_fn=PS
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
    if pnr:
        post_select.merge(PostSelect(f"[{num_additions}] == {pnr}"))

    simulation = pcvl.Simulator(pcvl.SLOSBackend())
    simulation.set_circuit(circuit)
    simulation.set_postselection(post_select)

    # from perceval.rendering import Format
    # p = pcvl.Processor("SLOS", circuit)
    # p.with_input(input_state)
    # p.set_postselection(post_select)
    # pcvl.pdisplay_to_file(p, "circuit.pdf", output_format=Format.MPLOT, recursive=True)

    return circuit, simulation, input_state


def fidelity(x: StateDict, y: StateVector) -> float:
    """
    Compute the fidelity between states stored as a StateDict or a Perceval state
    vector simulation result.
    """
    num_modes = len(list(x.keys())[0])
    overlap = 0
    for k, v in y:
        overlap += x.get(tuple(k)[-num_modes:], 0) * v.conjugate()
    return np.abs(overlap) ** 2


def simulate_with_perceval(
    circuit: StatePreparationCircuit, addition_r: float
) -> Tuple[float, float]:
    """
    Runs a full state vector simulation with Perceval and outputs probability
    of success and fidelity
    """
    pcvl_circuit, simulation, input_state = circuit_to_perceval_simulation(
        circuit, photon_addition_r=addition_r
    )
    final_state = simulation.evolve(input_state)
    f = fidelity(circuit.state, final_state)
    p_success = simulation.logical_perf
    return f, p_success


def plot_results(r_values, results, state):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    ax1 = axs[0]
    (l1,) = ax1.plot(r_values, results[:, 0], color="C0", label="Fidelity")
    ax1.set_xlabel("BS reflectivity")
    ax1.set_ylabel("Fidelity")
    ax1.legend()
    ax2 = ax1.twinx()
    (l2,) = ax2.semilogy(r_values, results[:, 1], color="C1", label="$P_{success}$")
    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower center")

    axs[1].loglog(1 - results[:, 0], results[:, 1])
    axs[1].set_xlabel("Infidelity")
    axs[1].set_ylabel("$P_{success}$")

    fig.suptitle(f"Preparation of {state_to_string(state)}")
    fig.tight_layout()
    plt.show()


def main():
    # state = normalized_state(kets_to_state_dict([(2, 0, 0), (0, 2, 0), (0, 0, 2)]))
    # state = normalized_state(kets_to_state_dict([(3, 0, 0), (0, 3, 0), (0, 0, 3)]))
    state = normalized_state(kets_to_state_dict([(2, 1, 0), (0, 2, 1)]))
    w, _, _ = next(optimal_preparation(state, extra_photons=1, num_decompositions=1))
    circuit = StatePreparationCircuit(w, state)

    num_r_values = 20
    r_values = 1 - 0.5 * 10 ** np.linspace(0, -1.5, num_r_values)
    results = np.zeros((num_r_values, 2))
    for i in range(num_r_values):
        results[i, :] = simulate_with_perceval(circuit, addition_r=r_values[i])

    plot_results(r_values, results, state)


if __name__ == "__main__":
    main()
