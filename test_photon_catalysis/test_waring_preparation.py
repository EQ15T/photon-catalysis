import pytest

from test_photon_catalysis.states import all_states
from photon_catalysis.waring_preparation import waring_preparation, projection_prob
from photon_catalysis.utils import state_dict_to_array, infidelity


expected_extra_photons = {
    'psi_1': 4,
    'psi_2': 6,
    'psi_3': 8,
    'psi_4': 6,
    'psi_5': 6,
    'psi_6': 4,
    'psi_7': 20,
    'psi_8': 6,
    'psi_9': 9,
    'psi_10': 20,
    'R2': 3,
    'R4': 9,
    'R5': 12,
    'K3': 12
}
assert(expected_extra_photons.keys() == all_states.keys())

def get_input_dict():
    r = {}
    for k, v in all_states.items():
        r[k] = (v, expected_extra_photons[k])
    return r

@pytest.mark.parametrize('state_name,tup', get_input_dict().items())
def test_waring_preparation(state_name, tup):
    state, extra_photons = tup
    degree = sum(list(state.keys())[0])
    mx_fid = 0
    for W, _, fid in waring_preparation(state, [2, 3, 4, 5, 6], 2):
        mx_fid = max(mx_fid, fid)
        assert extra_photons == W.shape[0] - degree
    assert mx_fid > 0.999
