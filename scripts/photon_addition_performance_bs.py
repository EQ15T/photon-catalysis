"""
This script evaluates the fidelity/probability success trade-off of the boson
sampling implementation, using Perceval for simulations.
"""

import os
from dataclasses import replace
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import perceval as pcvl
except ImportError:
    raise SystemExit(
        "This example requires Perceval to be installed"
        "Please install the optional dependencies, eg with pip install -e .[boson_sampling]"
    )
from exqalibur import StateVector
from perceval.rendering import Format
from perceval.rendering.circuit import SymbSkin

from photon_catalysis.benchmark_states import benchmark_states_dict
from photon_catalysis.optimal_preparation import optimal_preparation
from photon_catalysis.state_preparation_circuit import StatePreparationCircuit
from photon_catalysis.utils import StateDict, normalized_state

RESULTS_DIR = "results"
FILENAME = os.path.basename(__file__.replace(".py", ""))
RESULTS_FILE = os.path.join(RESULTS_DIR, FILENAME + ".csv")
FIGURE_FILES = [
    os.path.join(RESULTS_DIR, FILENAME + "." + ext) for ext in ["pdf", "pgf"]
]


def fidelity(x: StateDict, y: StateVector) -> float:
    """
    Compute the fidelity between states stored as a StateDict or a Perceval state
    vector simulation result.
    """
    num_modes = len(list(x.keys())[0])
    overlap = sum(x.get(tuple(k)[-num_modes:], 0) * v.conjugate() for k, v in y)
    return np.abs(overlap) ** 2


def simulate_state_preparation_circuit(
    circuit: StatePreparationCircuit
) -> Tuple[float, float]:
    """
    Runs a full state vector simulation with Perceval and outputs probability
    of success and fidelity
    """
    pcvl_circuit, input_state, post_select = circuit.to_perceval(
        decompose_unitaries=False
    )
    simulation = pcvl.Simulator(pcvl.SLOSBackend())
    simulation.set_circuit(pcvl_circuit)
    simulation.set_postselection(post_select)

    final_state = simulation.evolve(input_state)
    f = fidelity(circuit.state, final_state)
    p_success = simulation.logical_perf
    return f, p_success


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

    p = pcvl.Processor("SLOS", pcvl_circuit)
    p.with_input(input_state)
    symbolic_skin = SymbSkin(compact_display=True)
    pcvl.pdisplay_to_file(
        p, output_file, output_format=Format.MPLOT, recursive=True, skin=symbolic_skin
    )


def state_preparation_with_boson_sampling(
    state: StateDict,
    name: str,
    num_catalysis_photons: int,
    num_decompositions: int = 5,
    render_circuit_to_pdf: bool = False,
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

    # if render_circuit_to_pdf:
    #     render_circuit(
    #         circuit, addition_r=0.9, output_file=f"{RESULTS_DIR}/{name}_circuit.pdf"
    #     )

    num_r_values = 20
    t_values = (np.linspace(0.95, 0.05, num_r_values)) ** 0.5
    r_values = (1 - t_values**2) ** 0.5
    for i in range(num_r_values):
        circuit = StatePreparationCircuit(w, state, r_values[i])
        f, p_success = simulate_state_preparation_circuit(
            circuit
        )
        yield {
            "kind": "DV",
            "name": name,
            "r": r_values[i],
            "fidelity": f,
            "probability": p_success,
        }


def main():
    force_run = False
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if force_run or not os.path.exists(RESULTS_FILE):
        benchmark_states_dict["psi_10 N=5"] = replace(
            benchmark_states_dict["psi_10"], name="psi_10 N=5", extra_photons=1
        )

        candidates = ["psi_1", "psi_2", "psi_8", "psi_10", "psi_10 N=5"]

        results = []
        for name in candidates:
            state = benchmark_states_dict[name]
            results += list(
                state_preparation_with_boson_sampling(
                    state.state, name, state.extra_photons
                )
            )
        df = pd.DataFrame(results)
        df.to_csv(RESULTS_FILE, index=False)

    df = pd.read_csv(RESULTS_FILE)
    df = df[df["kind"] == "DV"]
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
    plt.xlim([1e-10, 1e-1])
    plt.ylim([1e-10, 1e-1])
    plt.xlabel("Distance to target state $1 - F$")
    plt.axvline(
        x=0.01, color="black", linestyle=":", linewidth=1.5, label="99\\% Fidelity"
    )
    plt.ylabel("Probability of success")
    plt.legend(loc="lower center", ncols=3, bbox_to_anchor=(0.5, 1.05), borderaxespad=0)
    plt.tight_layout()
    for f in FIGURE_FILES:
        plt.savefig(f)


if __name__ == "__main__":
    main()
