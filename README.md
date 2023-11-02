
# Code and Data for "High-performance repetition cat code using fast noisy operations"

The code source for the paper "High-performance repetition cat code using fast noisy operations" is available in this repository. The code is written in Python and uses the [QuTiP](https://qutip.org/) framework for physical simulations (CNOT gate) and [PyMatching](https://pypi.org/project/PyMatching/) for logical simulations. 

## Directory structure

The code is organized as follows:

- `qsim`: contains the core code for the simulations.
  - `qsim/physical_gate`: contains the code for the physical simulations.
  - `qsim/logical_gate`: contains the code for the logical simulations.
  - `qsim/basis`: contains the code for the Fock basis and Shifted Fock Basis.
  - `qsim/utils`: contains useful functions and classes (tomography, error correction graphs, error models, fits for threshold curves).
- `notebooks`: contains the Jupyter notebooks used to generate the figures of the paper.
- `tests`: contains the tests for the core code.
- `data`: contains data of logical gates used to generate the figures of the paper.
- `experiments`: contains the code for running experiments based on the core code.
- `requirements.txt`: contains the list of the required Python packages.
- `setup.py`: contains the setup file for the package.
- `LICENSE`: contains the license of the code.
- `README.md`: contains the README file of the code.


## List of figures and where to find them

- [Figure 2a](notebooks/cnot.ipynb)
- [Figure 2b](notebooks/leakage_cnot_after_convergence.ipynb)
- [Figure 3](notebooks/phenomenological.ipynb)
- [Figure 4a](notebooks/optimized_gate_time.ipynb)
- [Figure 4b](notebooks/fixed_cnot_gate_time.ipynb)
- [Figure 5a](notebooks/optimized_gate_time.ipynb)
- [Figure 5b](notebooks/fixed_cnot_gate_time.ipynb)
- [Figure 6](notebooks/cnot.ipynb)
- [Figure 7b](notebooks/data_bitflips.ipynb)
- [Figure 7c](notebooks/correlations.ipynb)
- [Figure 7d](notebooks/correlations.ipynb)
- [Figure 7e](notebooks/correlations.ipynb)
- [Figure 9](notebooks/QEC_parity_measurement_threshold_and_overhead.ipynb)
- [Figure 10](notebooks/QEC_parity_measurement_threshold_and_overhead.ipynb)
- [Figure 11](notebooks/phenomenological.ipynb)
- [Figure 12a](notebooks/phenomenological.ipynb)
- [Figure 12b](notebooks/phenomenological.ipynb)
- [Figure 12c](notebooks/phenomenological.ipynb)
