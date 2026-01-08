from argparse import ArgumentParser
from dataclasses import asdict as dataclass_asdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp

from photon_catalysis.benchmark_states import benchmark_states
from photon_catalysis.optimal_preparation import optimal_preparation
from photon_catalysis.utils import kets_to_state_dict
from photon_catalysis.waring_preparation import waring_preparation


@dataclass
class MainTableMethodRecord:
    add: int
    pnr: int
    p_min: float
    p_med: float
    p_max: float
    F: float


@dataclass
class MainTableRecord:
    etat: str
    d: int
    M: int
    baseline: MainTableMethodRecord
    waring: MainTableMethodRecord
    esp: MainTableMethodRecord


def print_to_latex(table: list[MainTableRecord]):
    for r in table:
        print(
            f"${r.etat}$ & {r.d} & {r.M}"
            f" & {r.baseline.add} & {r.baseline.pnr} & {r.baseline.p_min:.2f} & {r.baseline.p_med:.2f} & {r.baseline.p_max:.2f} & {r.baseline.F:.2f}"
            f" & {r.waring.add} & {r.waring.pnr} & {r.waring.p_min:.2f} & {r.waring.p_med:.2f} & {r.waring.p_max:.2f} & {r.waring.F:.2f}"
            f" & {r.esp.add} & {r.esp.pnr} & {r.esp.p_min:.2f} & {r.esp.p_med:.2f} & {r.esp.p_max:.2f} & {r.esp.F:.2f}"
        )


def maintable2dataframe(table: list[MainTableRecord]) -> pd.DataFrame:
    return pd.json_normalize([dataclass_asdict(r) for r in table])


def main_table(num_decompositions=1) -> list[MainTableRecord]:
    res = []

    for state in benchmark_states:
        # baseline
        base_mx_fid = 0
        base_ps = []
        for _, p, fid in optimal_preparation(
            state.state, 1, num_decompositions=num_decompositions, optimize_prob=False
        ):
            base_ps.append(p)
            base_mx_fid = max(base_mx_fid, fid)

        # waring
        waring_mx_fid = 0
        waring_ps = []
        waring_photons = None
        for W, p, fid in waring_preparation(
            state.state, [2, 3, 4, 5, 6], num_decompositions=num_decompositions
        ):
            waring_ps.append(p)
            waring_mx_fid = max(waring_mx_fid, fid)
            assert waring_photons is None or waring_photons == W.shape[0]
            waring_photons = W.shape[0]

        # ESP
        esp_mx_fid = 0
        esp_ps = []
        esp_rank = None
        for U, p, fid in optimal_preparation(
            state.state,
            state.extra_photons,
            num_decompositions=num_decompositions,
            optimize_prob=True,
        ):
            esp_ps.append(p)
            esp_mx_fid = max(esp_mx_fid, fid)
            assert esp_rank is None or esp_rank == U.shape[0]
            esp_rank = U.shape[0]

        d = state.num_photons
        M = state.num_modes

        res.append(
            MainTableRecord(
                etat=state.name,
                d=d,
                M=M,
                baseline=MainTableMethodRecord(
                    add=d + 1,
                    pnr=1,
                    p_min=min(base_ps),
                    p_med=np.median(base_ps),
                    p_max=max(base_ps),
                    F=base_mx_fid,
                ),
                waring=MainTableMethodRecord(
                    add=waring_photons,
                    pnr=waring_photons - d,
                    p_min=min(waring_ps),
                    p_med=np.median(waring_ps),
                    p_max=max(waring_ps),
                    F=waring_mx_fid,
                ),
                esp=MainTableMethodRecord(
                    add=esp_rank,
                    pnr=esp_rank - d,
                    p_min=min(esp_ps),
                    p_med=np.median(esp_ps),
                    p_max=max(esp_ps),
                    F=esp_mx_fid,
                ),
            )
        )

    return res


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-n",
        "--num_decompositions",
        type=int,
        default=25,
        help="Number of decompositions",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("main_table.csv"), help="Output file"
    )
    args = parser.parse_args()
    df = maintable2dataframe(main_table(args.num_decompositions))
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
