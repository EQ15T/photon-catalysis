"""
This script evaluates the fidelity/probability success trade-off of the gaussian boson
sampling implementation, using StrawberryFields for simulations.
"""

import os
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import strawberryfields as sf

from photon_catalysis.optimal_preparation import optimal_preparation
from photon_catalysis.state_preparation_circuit import StatePreparationCircuit
from photon_catalysis.utils import (
    StateDict,
    kets_to_state_dict,
    normalized_state,
    state_to_string,
)

RESULTS_DIR = "results"
FILENAME = os.path.basename(__file__.replace(".py", ""))
RESULTS_FILE = os.path.join(RESULTS_DIR, FILENAME + ".csv")
FIGURE_FILES = [
    os.path.join(RESULTS_DIR, FILENAME + "." + ext) for ext in ["pdf", "pgf"]
]


def fidelity(x: StateDict, y: np.array) -> float:
    overlap = sum(y[ket].conj() * amplitude for ket, amplitude in x.items())
    return np.abs(overlap) ** 2


def simulate_state_preparation_circuit(
    circuit: StatePreparationCircuit, squeezing_r: float
) -> Tuple[float, float]:
    total_modes = circuit.num_modes + circuit.num_additions
    prg, post_select = circuit.to_sf(squeezing_r=squeezing_r)
    eng = sf.Engine("fock", backend_options={"cutoff_dim": 6})
    output_state = eng.run(prg).state
    cond_state = output_state.ket()[post_select]
    p_success = np.sum(np.abs(cond_state) ** 2)
    cond_state_normalized = cond_state / np.sqrt(p_success)
    f = fidelity(circuit.state, cond_state_normalized)
    return f, p_success


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
    circuit = StatePreparationCircuit(w, state)

    if render_circuit_to_pdf:
        render_circuit_pcvl(
            circuit, addition_r=0.9, output_file=f"{RESULTS_DIR}/{name}_circuit.pdf"
        )

    num_r_values = 5
    squeezing_r_values = 10.0 ** (np.linspace(-0.2, -0.8, num_r_values))
    for i in range(num_r_values):
        f, p_success = simulate_state_preparation_circuit(
            circuit, squeezing_r=squeezing_r_values[i]
        )
        yield {
            "kind": "CV",
            "name": name,
            "r": squeezing_r_values[i],
            "fidelity": f,
            "probability": p_success,
        }


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


def main():
    force_run = False
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if force_run or not os.path.exists(RESULTS_FILE):
        candidates = ["psi_1", "psi_2", "psi_8", "psi_10", "psi_10 N=5"]
        expected_extra_photons["psi_10 N=5"] = 1
        all_states["psi_10 N=5"] = all_states["psi_10"]
        results = []
        for name in candidates:
            num_catalysis_photons = expected_extra_photons[name]
            results += list(
                state_preparation_with_boson_sampling(
                    all_states[name], name, num_catalysis_photons
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
    plt.xlim([1e-5, 1e-1])
    plt.ylim([1e-10, 1e-2])
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
