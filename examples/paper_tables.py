import sympy as sp
from photon_catalysis.utils import kets_to_state_dict
from photon_catalysis.optimal_preparation import optimal_preparation
from photon_catalysis.waring_preparation import waring_preparation

from dataclasses import dataclass

import logging
import os

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
        print(f'${r.etat}$ & {r.d} & {r.M}'
              f' & {r.baseline.add} & {r.baseline.pnr} & {r.baseline.p_min:.2f} & {r.baseline.p_med:.2f} & {r.baseline.p_max:.2f} & {r.baseline.F:.2f}'
              f' & {r.waring.add} & {r.waring.pnr} & {r.waring.p_min:.2f} & {r.waring.p_med:.2f} & {r.waring.p_max:.2f} & {r.waring.F:.2f}'
              f' & {r.esp.add} & {r.esp.pnr} & {r.esp.p_min:.2f} & {r.esp.p_med:.2f} & {r.esp.p_max:.2f} & {r.esp.F:.2f}')


def median(vs):
    if len(vs) % 2 == 0:
        return ( vs[len(vs) // 2] + vs[len(vs) // 2 - 1] ) / 2
    else:
        return vs[len(vs) // 2]

def main_table(num_decompositions=25) -> list[MainTableRecord]:
    res = []

    for state_name, state in all_states.items():
        # baseline
        base_mx_fid = 0
        base_ps = []
        for _, p, fid in optimal_preparation(state, 1, num_decompositions=num_decompositions, optimize_prob=False):
            base_ps.append(p)
            base_mx_fid = max(base_mx_fid, fid)

        # waring
        waring_mx_fid = 0
        waring_ps = []
        waring_photons = None
        for W, p, fid in waring_preparation(state, [2, 3, 4, 5, 6], num_decompositions=num_decompositions):
            waring_ps.append(p)
            waring_mx_fid = max(waring_mx_fid, fid)
            assert( waring_photons is None or waring_photons == W.shape[0] )
            waring_photons = W.shape[0]

        # ESP
        esp_mx_fid = 0
        esp_ps = []
        esp_rank = None
        for U, p, fid in optimal_preparation(state, expected_extra_photons[state_name], num_decompositions=num_decompositions, optimize_prob=True):
            esp_ps.append(p)
            esp_mx_fid = max(esp_mx_fid, fid)
            assert( esp_rank is None or esp_rank == U.shape[0] )
            esp_rank = U.shape[0]


        d = sum(list(state.keys())[0])
        M = len(list(state.keys())[0])

        res.append(MainTableRecord(
            etat = state_name,
            d = d,
            M = M,
            baseline = MainTableMethodRecord(add=d + 1, pnr=1, p_min=min(base_ps), p_med=median(base_ps), p_max=max(base_ps), F=base_mx_fid),
            waring = MainTableMethodRecord(add=waring_photons, pnr=waring_photons - d, p_min=min(waring_ps), p_med=median(waring_ps), p_max=max(waring_ps), F=waring_mx_fid),
            esp = MainTableMethodRecord(add=esp_rank, pnr=esp_rank - d, p_min=min(esp_ps), p_med=median(esp_ps), p_max=max(esp_ps), F=esp_mx_fid)
        ))

        break

    return res



if __name__ == '__main__':
    print_to_latex(main_table())
