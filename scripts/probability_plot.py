from argparse import ArgumentParser
from dataclasses import asdict as dataclass_asdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp

from photon_catalysis.benchmark_states import benchmark_states_dict
from photon_catalysis.optimal_preparation import optimal_preparation
from photon_catalysis.utils import normalize_W, projection_prob
from photon_catalysis.waring_preparation import waring_preparation


@dataclass
class ProbPlotRecord:
    x: float
    y: float


def prob_plot(num_decompositions=25) -> dict[str, list[ProbPlotRecord]]:
    def make_expr_th1(state):
        d = sum(list(state.keys())[0])
        W, _, _ = max(
            waring_preparation(
                state, [2, 3, 4, 5, 6], num_decompositions=num_decompositions
            ),
            key=lambda t: abs(t[1]),
        )
        N = W.shape[0]
        W = normalize_W(W)
        alpha = sp.Symbol("\\alpha")
        p_success, _ = projection_prob(np.asarray(W), N - d, alpha)
        return sp.lambdify(alpha, p_success)

    def make_expr_th2(state, extra_photons):
        d = sum(list(state.keys())[0])
        W, _, _ = max(
            optimal_preparation(
                state, extra_photons, num_decompositions=num_decompositions
            ),
            key=lambda t: abs(t[1]),
        )
        W = normalize_W(W)
        N = W.shape[0]
        alpha = sp.Symbol("\\alpha")
        p_success, _ = projection_prob(np.asarray(W), N - d, alpha)
        return sp.lambdify(alpha, p_success)

    def optimize_expr(prob_fn):
        res = []
        for s in np.arange(0, 5, 0.05):
            res.append(ProbPlotRecord(x=s, y=prob_fn(s)))
        return res

    res = {}
    for candidate in ["psi_4", "psi_8", "psi_9"]:
        s = benchmark_states_dict[candidate]
        res[f"e1_{s.name}"] = optimize_expr(make_expr_th1(s.state))
        res[f"e2_{s.name}"] = optimize_expr(make_expr_th2(s.state, s.extra_photons))
    return res


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("results/probs"),
        help="Output directory",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    for n, d in prob_plot().items():
        df = pd.json_normalize([dataclass_asdict(r) for r in d])
        df.to_csv(args.output / f"{n}.csv", index=False)


if __name__ == "__main__":
    main()
