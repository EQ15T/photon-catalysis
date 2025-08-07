import pytest

from test_photon_catalysis.states import all_states
from photon_catalysis.optimal_preparation import optimal_preparation


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

def get_input_dict():
    r = {}
    for k, v in all_states.items():
        r[k] = (v, expected_extra_photons[k])
    return r

@pytest.mark.parametrize('state_name,tup', get_input_dict().items())
def test_optimal_preparation(state_name, tup):
    state, extra_photons = tup
    mx_fid = 0
    for _, _, fid in optimal_preparation(state, extra_photons, num_decompositions=2):
        mx_fid = max(mx_fid, fid)
    assert mx_fid > 0.999
