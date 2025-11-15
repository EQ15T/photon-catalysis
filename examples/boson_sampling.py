"""
This scripts converts an optimal state preparation circuit into a boson sampling
scheme employing beam-splitter and single-photon sources to implement photon
addition. The scheme is simulated with Perceval, to retrieve probability of
success and fidelity. Running this script will create a results directory
with a fidelity/probability success plot for each state, and a global
all_states.csv file storing the generated data.
"""

import os
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

RESULTS_DIR = "results"



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
    pcvl_circuit, input_state, post_select = circuit.to_perceval(
        photon_addition_r=addition_r, decompose_unitary=False
    )
    simulation = pcvl.Simulator(pcvl.SLOSBackend())
    simulation.set_circuit(pcvl_circuit)
    simulation.set_postselection(post_select)

    final_state = simulation.evolve(input_state)
    f = fidelity(circuit.state, final_state)
    p_success = simulation.logical_perf
    return f, p_success


def plot_results(
    r_values: np.ndarray, results: np.ndarray, state: StateDict, output_file: str
):
    """
    Plot the reflectivity/fidelity/probability of success curves
    """
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
    plt.savefig(output_file)


def render_circuit(
    circuit: StatePreparationCircuit,
    addition_r: float = 0.9,
    output_file: str = "circuit.pdf",
):
    """
    Saves a graphical representation of the circuit
    """
    pcvl_circuit, input_state, _ = circuit.to_perceval(
        photon_addition_r=addition_r, decompose_unitary=True
    )

    from perceval.rendering import Format
    from perceval.rendering.circuit import SymbSkin

    p = pcvl.Processor("SLOS", pcvl_circuit)
    p.with_input(input_state)
    symbolic_skin = SymbSkin(compact_display=True)
    pcvl.pdisplay_to_file(
        p, output_file, output_format=Format.MPLOT, recursive=True, skin=symbolic_skin
    )


all_states = {
    "psi_1": kets_to_state_dict([(2, 0, 0), (0, 2, 0), (0, 0, 2)]),
    "psi_2": kets_to_state_dict([(3, 0, 0), (0, 3, 0), (0, 0, 3)]),
    "psi_3": kets_to_state_dict([(4, 0, 0), (0, 4, 0), (0, 0, 4)]),
    "psi_4": kets_to_state_dict(
        [(2, 0, 0, 0), (0, 2, 0, 0), (0, 0, 2, 0), (0, 0, 0, 2)]
    ),
    "psi_5": kets_to_state_dict(
        [(0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (1, 0, 2), (2, 1, 0)]
    ),
    "psi_6": kets_to_state_dict([(1, 1, 0), (1, 0, 1), (0, 1, 1)]),
    "psi_7": kets_to_state_dict([(2, 2, 0), (2, 0, 2), (0, 2, 2)]),
    "psi_8": kets_to_state_dict([(2, 0, 0, 0), (0, 1, 1, 0), (0, 0, 0, 2)]),
    "psi_9": kets_to_state_dict(
        [(3, 0, 0, 0), (0, 2, 1, 0), (0, 1, 2, 0), (0, 0, 0, 3)]
    ),
    "psi_10": kets_to_state_dict([(0, 4, 0), (1, 2, 1), (2, 0, 2)]),
    "R4": kets_to_state_dict([(3, 0, 0), (0, 3, 0), (0, 0, 3), (1, 1, 1)]),
    "R5": kets_to_state_dict([(2, 1, 0), (0, 2, 1)]),
    "R2": {
        (3, 0, 0): np.sqrt(13) / 13,
        (1, 2, 0): np.sqrt(39) / 13,
        (1, 1, 1): np.sqrt(78) / 13,
        (1, 0, 2): np.sqrt(39) / 13,
    },
    "K3": {
        (3, 0, 0, 0): 1,
        (2, 1, 0, 0): 1,
        (2, 0, 1, 0): 1,
        (2, 0, 0, 1): 1,
        (1, 1, 1, 0): -1,
        (1, 1, 0, 1): -1,
        (1, 0, 1, 1): -1,
        (0, 1, 1, 1): -1,
    },
}

expected_extra_photons = {
    "psi_1": 1,
    "psi_2": 1,
    "psi_3": 2,
    "psi_4": 2,
    "psi_5": 1,
    "psi_6": 1,
    "psi_7": 1,
    "psi_8": 2,
    "psi_9": 2,
    "psi_10": 2,
    "R4": 1,
    "R5": 1,
    "R2": 1,
    "K3": 2,
}


def boson_sampling_state_preparation(
    state: StateDict,
    name: str,
    num_catalysis_photons: int,
    num_decompositions: int = 5,
    save_plots: bool = True,
):
    # Find the optimal preparation
    state = normalized_state(state)
    w, _, _ = max(
        optimal_preparation(
            state,
            extra_photons=num_catalysis_photons,
            num_decompositions=num_decompositions,
        ),
        key=lambda t: abs(t[1]),
    )
    circuit = StatePreparationCircuit(w, state)

    # Prepare the probability/fidelity plot
    num_r_values = 10
    t_values = (np.linspace(0.5, 0.05, num_r_values)) ** 0.5
    r_values = (1 - t_values**2) ** 0.5
    results = np.zeros((num_r_values, 2))
    for i in range(num_r_values):
        results[i, :] = simulate_with_perceval(circuit, addition_r=r_values[i])

    if save_plots:
        render_circuit(
            circuit, addition_r=0.9, output_file=f"{RESULTS_DIR}/{name}_circuit.pdf"
        )
        plot_results(r_values, results, state, f"{RESULTS_DIR}/{name}_prob_plot.pdf")

    return [
        {
            "name": name,
            "r": r_values[i],
            "fidelity": results[i, 0],
            "probability": results[i, 1],
        }
        for i in range(num_r_values)
    ]


def plot_all_states():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []
    for name, state_dict in all_states.items():
        num_catalysis_photons = expected_extra_photons[name]
        results += boson_sampling_state_preparation(
            state_dict, name, num_catalysis_photons
        )
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "all_states.csv"), index=False)


def plot_comparison_for_paper():
    results_file = os.path.join(RESULTS_DIR, "selected_states.csv")

    force_run = False

    if force_run or not os.path.exists(results_file):
        candidates = ["psi_1", "psi_2", "psi_8", "psi_10", "psi_10 N=5"]
        expected_extra_photons["psi_10 N=5"] = 1
        all_states["psi_10 N=5"] = all_states["psi_10"]
        results = []
        for name in candidates:
            num_catalysis_photons = expected_extra_photons[name]
            results += boson_sampling_state_preparation(
                all_states[name], name, num_catalysis_photons, save_plots=False
            )
        df = pd.DataFrame(results)
        df.to_csv(results_file, index=False)

    df = pd.read_csv(results_file)
    tex_labels = {
        "psi_1": "$|\\Psi\\rangle_1$",
        "psi_2": "$|\\Psi\\rangle_2$",
        "psi_8": "$|\\Psi\\rangle_8$",
        "psi_10": "$|\\Psi\\rangle_{10}$",
        "psi_10 N=5": "$|\\Psi\\rangle_{10}$ N=5",
    }

    matplotlib.use("pgf")
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "font.serif": [],
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )

    fig, axs = plt.subplots(figsize=(4, 3.5))
    for name, d in df.groupby("name", sort=False):
        plt.loglog(
            1 - d["fidelity"].values,
            d["probability"],
            marker="*",
            label=tex_labels[name],
        )
    plt.grid(visible=True)
    plt.xlabel("Distance to target state $1 - F$")
    plt.axvline(
        x=0.01, color="black", linestyle=":", linewidth=1.5, label="99\\% Fidelity"
    )
    plt.ylabel("Probability of success")
    plt.legend(loc="lower center", ncols=3, bbox_to_anchor=(0.5, 1.05), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "selected_states.pgf"))
    # plt.show()


if __name__ == "__main__":
    # plot_all_states()
    plot_comparison_for_paper()
