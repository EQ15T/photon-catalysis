import sympy as sp
from photon_catalysis.utils import kets_to_state_dict


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