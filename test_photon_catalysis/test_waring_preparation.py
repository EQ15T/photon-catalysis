import pytest

from photon_catalysis.benchmark_states import benchmark_states
from photon_catalysis.utils import infidelity, state_dict_to_array
from photon_catalysis.waring_preparation import projection_prob, waring_preparation


@pytest.mark.parametrize("benchmark_state", benchmark_states, ids=lambda s: s.name)
def test_waring_preparation(benchmark_state):
    mx_fid = 0
    for W, _, fid in waring_preparation(benchmark_state.state, [2, 3, 4, 5, 6], 2):
        mx_fid = max(mx_fid, fid)
        assert benchmark_state.waring_rank * benchmark_state.num_photons == W.shape[0]
    assert mx_fid > 0.999
