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

Optionally, to run the examples simulating the boson sampling implementation of our schemes, the [Perceval](https://github.com/Quandela/Perceval) library from Quandela needs to be installed, using the following command:
```shell
pip install -e git+https://github.com/EQ15T/photon-catalysis.git#egg=photon_catalysis[boson_sampling]
```

Or:

```shell
pip install -e .[boson_sampling]
```

Similarly, the examples simulating the gaussian boson sampling implementation requires the [StrawberryFields](https://github.com/XanaduAI/strawberryfields) library from Xanadu:

```shell
pip install -e git+https://github.com/EQ15T/photon-catalysis.git#egg=photon_catalysis[gaussian_boson_sampling]
```

Or:

```shell
pip install -e .[gaussian_boson_sampling]
```

Note that both can be simultaneously installed with the ```[boson_sampling,gaussian_boson_sampling]``` argument.

---

## Examples

### Basic usage

Using this software, one could obtain the set of linear forms in creation operators, such that their product, upon conditioning, corresponds to the desired core state. For example, for the state

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

### Boson sampling implementation (DV)

A Perceval circuit (and the accompanying input state and post-selection rules) can then be generated, and simulated, with:

```python
circuit = StatePreparationCircuit(W, state)
pcvl_circuit, input_state, post_select = circuit.to_perceval(
    photon_addition_r=0.7, decompose_unitaries=False
)
simulation = pcvl.Simulator(pcvl.SLOSBackend())
simulation.set_circuit(pcvl_circuit)
simulation.set_postselection(post_select)
final_state = simulation.evolve(input_state)
print(final_state)
```

---

## Additional scripts

The [photon_addition_performance_bs.py](examples/photon_addition_performance_bs.py) script from the [examples](examples) directory illustrates, on selected states, the trade-off between fidelity and probability of success resulting from the non-ideal implementation of photon addition with a beam-splitter, in the (DV) boson sampling setting. The script can be run from the command line, and outputs data and plots to the ```results``` directory.

The [photon_addition_performance_sqz.py](examples/photon_addition_performance_sqz.py) script is its counterpart for the Gaussian boson sampling scheme, simulating the imperfect photon addition achieved by two-mode squeezing and PNR detection. Note that this simulation is slow and the script can take hours to run and complete.
