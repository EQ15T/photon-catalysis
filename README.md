# photon-catalysis

**Code for "Photon catalysis for general multimode multi-photon quantum state preparation"**

Andrei Aralov, Émilie Gillet, Viet Nguyen, Andrea Cosentino, Mattia Walschaers, and Massimo Frigerio
[arXiv:2507.19397](https://arxiv.org/abs/2507.19397)

---

## Installation

Simply run this line to install the package using PIP:
```shell
pip install -e git+https://github.com/EQ15T/photon-catalysis.git#egg=photon_catalysis
```

For development:
```shell
git clone https://github.com/EQ15T/photon-catalysis.git
cd photon-catalysis
pip install -e .
```

Optionally, to run the examples simulating the boson-sampling implementation of our schemes, the [Perceval](https://github.com/Quandela/Perceval) library from Quandela needs to be installed, using the following command:
```shell
pip install -e git+https://github.com/EQ15T/photon-catalysis.git#egg=photon_catalysis[boson_sampling]
```

Or:

```shell
pip install -e .[boson_sampling]
```

## Examples

### Basic usage

Using this software, one could obtain the set of linear forms in creation operators, such that their product, upon conditioning, corresponds to the desired core state. For exmaple, for the state

$$\ket{\psi} \propto \ket{2000} + \ket{0200} + \ket{0020} + \ket{0002}$$

the following code could be used:
```python
from photon_catalysis.utils import kets_to_state_dict
from photon_catalysis.optimal_preparation import optimal_preparation

extra_photons = 2
state = kets_to_state_dict([(2, 0, 0, 0), (0, 2, 0, 0), (0, 0, 2, 0), (0, 0, 0, 2)])
for W, prob, fid in optimal_preparation(state, extra_photons, num_decompositions=10):
    print(W)
```

Rows of `W` define the linear forms, `extra_photons` is the number of catalysis photons.
[test_photon_catalysis](test_photon_catalysis) contains tests that could serve as more complete usage examples.

### Boson sampling

The [boson_sampling.py](examples/boson_sampling.py) script from the [examples](examples) directory illustrates how to implement state preparation using
an approximation of photon addition, as a boson sampling scheme.

The script can be run from the command line, and simulates the Boson sampling scheme with different values of the photon addition beam-splitter reflexivity, to plot how this parameter impacts fidelity and probability of success.
