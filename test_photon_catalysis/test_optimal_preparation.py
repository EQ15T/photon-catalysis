import pytest

from photon_catalysis.benchmark_states import benchmark_states
from photon_catalysis.optimal_preparation import optimal_preparation


@pytest.mark.parametrize("benchmark_state", benchmark_states, ids=lambda s: s.name)
def test_optimal_preparation(benchmark_state):
    mx_fid = 0
    for _, _, fid in optimal_preparation(
        benchmark_state.state, benchmark_state.extra_photons, num_decompositions=2
    ):
        mx_fid = max(mx_fid, fid)
    assert mx_fid > 0.999
