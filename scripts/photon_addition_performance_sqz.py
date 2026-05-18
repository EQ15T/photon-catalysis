"""
This script evaluates the fidelity/probability success trade-off of the gaussian boson
sampling implementation, using StrawberryFields for simulations.
"""

import os
from dataclasses import replace
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import strawberryfields as sf

from photon_catalysis.benchmark_states import benchmark_states_dict
from photon_catalysis.optimal_preparation import optimal_preparation
from photon_catalysis.state_preparation_circuit import StatePreparationCircuitSq
from photon_catalysis.utils import StateDict, normalized_state

RESULTS_DIR = "results_sqz"
FILENAME = os.path.basename(__file__.replace(".py", ""))
RESULTS_FILE = os.path.join(RESULTS_DIR, FILENAME + ".csv")
FIGURE_FILES = [
    os.path.join(RESULTS_DIR, FILENAME + "." + ext) for ext in ["pdf", "pgf"]
]


def fidelity(x: StateDict, y: np.array) -> float:
    overlap = sum(y[ket].conj() * amplitude for ket, amplitude in x.items())
    return np.abs(overlap) ** 2


def simulate_state_preparation_circuit(
    circuit: StatePreparationCircuitSq, fock_cutoff: int = 6
) -> Tuple[float, float]:
    prg, post_select = circuit.to_sf()
    eng = sf.Engine("fock", backend_options={"cutoff_dim": fock_cutoff})
    output_state = eng.run(prg).state
    cond_state = output_state.ket()[post_select]
    p_success = np.sum(np.abs(cond_state) ** 2)
    cond_state_normalized = cond_state / np.sqrt(p_success)
    f = fidelity(circuit.state, cond_state_normalized)
    return f, p_success


def state_preparation_with_gaussian_boson_sampling(
    state: StateDict,
    name: str,
    num_catalysis_photons: int,
    num_decompositions: int = 5,
    exact_addition: bool = False,
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
    num_r_values = 5
    squeezing_r_values = 10.0 ** (np.linspace(-0.2, -0.8, num_r_values))
    for i in range(num_r_values):
        r = squeezing_r_values[i]
        circuit = StatePreparationCircuitSq(w, state, r, exact_addition)
        f, p_success = simulate_state_preparation_circuit(circuit)
        yield {
            "kind": "CV",
            "name": name,
            "r": r,
            "fidelity": f,
            "probability": p_success,
        }


def main():
    force_run = True
    exact_addition = False

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
                state_preparation_with_gaussian_boson_sampling(
                    state.state,
                    name,
                    state.extra_photons,
                    exact_addition=exact_addition,
                )
            )
        df = pd.DataFrame(results)
        df.to_csv(RESULTS_FILE, index=False)

    df = pd.read_csv(RESULTS_FILE)
    df = df[df["kind"] == "CV"]
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
    plt.xlim([1e-10 if exact_addition else 1e-5, 1e-1])
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
