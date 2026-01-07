"""
This module contains the definition of all the states used for benchmarking
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import sympy as sp

from photon_catalysis.utils import StateDict, kets_to_state_dict


@dataclass
class BenchmarkState:
    name: str
    state: StateDict
    extra_photons: int
    waring_rank: int

    @property
    def num_photons(self) -> int:
        return sum(next(iter(self.state.keys())))


# Create all benchmark states
benchmark_states = [
    BenchmarkState(
        name="psi_1",
        state=kets_to_state_dict([(2, 0, 0), (0, 2, 0), (0, 0, 2)]),
        extra_photons=1,
        waring_rank=3,  # (3-1)*2 = 4
    ),
    BenchmarkState(
        name="psi_2",
        state=kets_to_state_dict([(3, 0, 0), (0, 3, 0), (0, 0, 3)]),
        extra_photons=1,
        waring_rank=3,  # (3-1)*3 = 6
    ),
    BenchmarkState(
        name="psi_3",
        state=kets_to_state_dict([(4, 0, 0), (0, 4, 0), (0, 0, 4)]),
        extra_photons=2,
        waring_rank=3,  # (3-1)*4 = 8
    ),
    BenchmarkState(
        name="psi_4",
        state=kets_to_state_dict(
            [(2, 0, 0, 0), (0, 2, 0, 0), (0, 0, 2, 0), (0, 0, 0, 2)]
        ),
        extra_photons=2,
        waring_rank=4,  # (4-1)*2 = 6
    ),
    BenchmarkState(
        name="psi_5",
        state=kets_to_state_dict(
            [(0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (1, 0, 2), (2, 1, 0)]
        ),
        extra_photons=1,
        waring_rank=3,  # (3-1)*3 = 6
    ),
    BenchmarkState(
        name="psi_6",
        state=kets_to_state_dict([(1, 1, 0), (1, 0, 1), (0, 1, 1)]),
        extra_photons=1,
        waring_rank=3,  # (3-1)*2 = 4
    ),
    BenchmarkState(
        name="psi_7",
        state=kets_to_state_dict([(2, 2, 0), (2, 0, 2), (0, 2, 2)]),
        extra_photons=1,
        waring_rank=6,  # (6-1)*4 = 20
    ),
    BenchmarkState(
        name="psi_8",
        state=kets_to_state_dict([(2, 0, 0, 0), (0, 1, 1, 0), (0, 0, 0, 2)]),
        extra_photons=2,
        waring_rank=4,  # (4-1)*2 = 6
    ),
    BenchmarkState(
        name="psi_9",
        state=kets_to_state_dict(
            [(3, 0, 0, 0), (0, 2, 1, 0), (0, 1, 2, 0), (0, 0, 0, 3)]
        ),
        extra_photons=2,
        waring_rank=4,  # (4-1)*3 = 9
    ),
    BenchmarkState(
        name="psi_10",
        state=kets_to_state_dict([(0, 4, 0), (1, 2, 1), (2, 0, 2)]),
        extra_photons=2,
        waring_rank=6,  # (6-1)*4 = 20
    ),
    BenchmarkState(
        name="R4",
        state=kets_to_state_dict([(3, 0, 0), (0, 3, 0), (0, 0, 3), (1, 1, 1)]),
        extra_photons=1,
        waring_rank=4,  # (4-1)*3 = 9
    ),
    BenchmarkState(
        name="R5",
        state=kets_to_state_dict([(2, 1, 0), (0, 2, 1)]),
        extra_photons=1,
        waring_rank=5,  # (5-1)*3 = 12
    ),
    BenchmarkState(
        name="R2",
        state={
            (3, 0, 0): sp.sqrt(13) / 13,
            (1, 2, 0): sp.sqrt(39) / 13,
            (1, 1, 1): sp.sqrt(78) / 13,
            (1, 0, 2): sp.sqrt(39) / 13,
        },
        extra_photons=1,
        waring_rank=2,  # (2-1)*3 = 3
    ),
    BenchmarkState(
        name="K3",
        state={
            (3, 0, 0, 0): 1,
            (2, 1, 0, 0): 1,
            (2, 0, 1, 0): 1,
            (2, 0, 0, 1): 1,
            (1, 1, 1, 0): -1,
            (1, 1, 0, 1): -1,
            (1, 0, 1, 1): -1,
            (0, 1, 1, 1): -1,
        },
        extra_photons=2,
        waring_rank=5,  # (5-1)*3 = 12
    ),
]

# Optional: Create a dictionary for quick lookup by name
benchmark_states_dict = {state.name: state for state in benchmark_states}
