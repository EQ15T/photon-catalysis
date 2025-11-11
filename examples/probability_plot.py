import numpy as np
import sympy as sp
import pandas as pd

from photon_catalysis.utils import kets_to_state_dict, normalize_W, projection_prob
from photon_catalysis.optimal_preparation import optimal_preparation
from photon_catalysis.waring_preparation import waring_preparation

from dataclasses import dataclass, asdict as dataclass_asdict

from pathlib import Path
from argparse import ArgumentParser



all_states = {
    'psi_1': kets_to_state_dict([(2, 0, 0), (0, 2, 0), (0, 0, 2)]),
    'psi_2' : kets_to_state_dict([(3, 0, 0), (0, 3, 0), (0, 0, 3)]),
    'psi_3' : kets_to_state_dict([(4, 0, 0), (0, 4, 0), (0, 0, 4)]),
    'psi_4' : kets_to_state_dict([(2, 0, 0, 0), (0, 2, 0, 0), (0, 0, 2, 0), (0, 0, 0, 2)]),
    'psi_5' : kets_to_state_dict([(0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (1, 0, 2), (2, 1, 0)]),
    'psi_6' : kets_to_state_dict([(1, 1, 0), (1, 0, 1), (0, 1, 1)]),
    'psi_7' : kets_to_state_dict([(2, 2, 0), (2, 0, 2), (0, 2, 2)]),
    'psi_8' : kets_to_state_dict([(2, 0, 0, 0), (0, 1, 1, 0), (0, 0, 0, 2)]),
    'psi_9' : kets_to_state_dict([(3, 0, 0, 0), (0, 2, 1, 0), (0, 1, 2, 0), (0, 0, 0, 3)]),
    'psi_10' : kets_to_state_dict([(0, 4, 0), (1, 2, 1), (2, 0, 2)]),
    'R4' : kets_to_state_dict([(3, 0, 0), (0, 3, 0), (0, 0, 3), (1, 1, 1)]),
    'R5' : kets_to_state_dict([(2, 1, 0), (0, 2, 1)]),
    'R2':
        {
             (3, 0, 0): sp.sqrt(13)/13,
             (1, 2, 0): sp.sqrt(39)/13,
             (1, 1, 1): sp.sqrt(78)/13,
             (1, 0, 2): sp.sqrt(39)/13
         },
    'K3' : {
         (3, 0, 0, 0) : 1,
         (2, 1, 0, 0) : 1,
         (2, 0, 1, 0) : 1,
         (2, 0, 0, 1) : 1,
         (1, 1, 1, 0) : -1,
         (1, 1, 0, 1) : -1,
         (1, 0, 1, 1) : -1,
         (0, 1, 1, 1) : -1 }
}
expected_extra_photons = {
    'psi_1': 1,
    'psi_2': 1,
    'psi_3': 2,
    'psi_4': 2,
    'psi_5': 1,
    'psi_6': 1,
    'psi_7': 1,
    'psi_8': 2,
    'psi_9': 2,
    'psi_10': 2,
    'R4': 1,
    'R5': 1,
    'R2': 1,
    'K3': 2
}
assert(expected_extra_photons.keys() == all_states.keys())


@dataclass
class ProbPlotRecord:
    x: float
    y: float

def prob_plot(num_decompositions=25) -> dict[str, list[ProbPlotRecord]]:
    def make_expr_th1(state):
        d = sum(list(state.keys())[0])
        W, _, _ = max(
            waring_preparation(state, [2, 3, 4, 5, 6], num_decompositions=num_decompositions),
            key=lambda t: abs(t[1])
        )
        N = W.shape[0]
        W = normalize_W(W)
        alpha = sp.Symbol('\\alpha')
        p_success, _ = projection_prob(np.asarray(W), N - d, alpha)
        return sp.lambdify(alpha, p_success)

    def make_expr_th2(state, extra_photons):
        d = sum(list(state.keys())[0])
        W, _, _ = max(
            optimal_preparation(state, extra_photons, num_decompositions=num_decompositions),
            key=lambda t: abs(t[1])
        )
        W = normalize_W(W)
        N = W.shape[0]
        alpha = sp.Symbol('\\alpha')
        p_success, _ = projection_prob(np.asarray(W), N - d, alpha)
        return sp.lambdify(alpha, p_success)

    def optimize_expr(prob_fn):
        res = []
        for s in np.arange(0, 5, 0.05):
            res.append(ProbPlotRecord(x=s, y=prob_fn(s)))
        return res

    res = {}
    res['e1_psi_4'] = optimize_expr(make_expr_th1(all_states['psi_4']))
    res['e2_psi_4'] = optimize_expr(make_expr_th2(all_states['psi_4'], expected_extra_photons['psi_4']))
    res['e1_psi_8'] = optimize_expr(make_expr_th1(all_states['psi_8']))
    res['e2_psi_8'] = optimize_expr(make_expr_th2(all_states['psi_8'], expected_extra_photons['psi_8']))
    res['e1_psi_9'] = optimize_expr(make_expr_th1(all_states['psi_9']))
    res['e2_psi_9'] = optimize_expr(make_expr_th2(all_states['psi_9'], expected_extra_photons['psi_9']))

    return res


def main():
    parser = ArgumentParser()
    parser.add_argument('-o', '--output', type=Path, default=Path('probs'), help='Output directory')
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    for n, d in prob_plot().items():
        df = pd.json_normalize([dataclass_asdict(r) for r in d])
        df.to_csv(args.output / f'{n}.csv', index=False)


if __name__ == '__main__':
    main()
