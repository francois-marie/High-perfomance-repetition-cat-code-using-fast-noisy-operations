from itertools import product
from time import time
from typing import Any, Dict, List, Optional

import numpy as np
from qutip import Qobj
from qutip import basis as basis_state
from qutip import fock, qeye, tensor
from qutip.qip.operations import snot

from qsim.basis.sfb import SFB, SFBNonOrthonormal
from qsim.physical_gate.cnotidle import (
    CNOT12Nothing3SFBPhaseFlips,
    CNOT13Nothing2SFBPhaseFlips,
)
from qsim.physical_gate.pgate import ThreeQubitPGate


class ParityMeasurementGate(ThreeQubitPGate):
    def __init__(
        self,
        nbar: int,
        k1: float,
        k2: float,
        k1a: float,
        k2a: float,
        k1b: float,
        k2b: float,
        gate_time: float,
        truncature: int,
        initial_state: Optional[Qobj] = None,
        initial_state_name: str = 'unset',
        H: Optional[List[float]] = None,
        D: Optional[List[float]] = None,
        rho: Optional[Qobj] = None,
        num_tslots_pertime: int = 1000,
        N_ancilla: Optional[int] = None,
        N_b: Optional[int] = None,
    ):
        super().__init__(
            nbar,
            k1,
            k2,
            k1a,
            k2a,
            k1b,
            k2b,
            gate_time,
            truncature,
            initial_state,
            initial_state_name,
            H,
            D,
            rho,
            num_tslots_pertime,
            N_ancilla=N_ancilla,
            N_b=N_b,
        )
        params = {
            'nbar': self.nbar,
            'k1': self.k1,
            'k2': self.k2,
            'k1a': self.k1a,
            'k2a': self.k2a,
            'k1b': self.k1b,
            'k2b': self.k2b,
            'gate_time': self.gate_time,
            'truncature': self.truncature,
            'N_ancilla': self.N_ancilla,
            'N_b': self.N_b,
        }

        self.cnot12 = CNOT12Nothing3SFBPhaseFlips(**params)
        self.cnot13 = CNOT13Nothing2SFBPhaseFlips(**params)

    def _simulate(self) -> Dict[str, Any]:
        self.basis = SFB(
            nbar=self.nbar,
            d=self.truncature,
            d_ancilla=self.N_ancilla,
            d_b=self.N_b,
        )
        # default initial state
        if self.initial_state is None:
            self.initial_state = tensor(
                SFBNonOrthonormal(
                    nbar=self.nbar, d=self.N_ancilla
                ).data.evencat,
                SFBNonOrthonormal(
                    nbar=self.nbar, d=self.truncature
                ).data.evencat,
                SFBNonOrthonormal(nbar=self.nbar, d=self.N_b).data.evencat,
            )
        self.rho = self.initial_state

        for gate in [
            self.cnot12,
            self.cnot13,
        ]:
            # one qubit idle for preparation
            print(type(gate).__name__)
            gate.initial_state = self.rho
            gate.simulate()
            self.rho = gate.rho

    def transition_matrices(self):
        print('>>> CNOT transition matrix')
        t_start = time()
        plus = (basis_state(2, 0) + basis_state(2, 1)).unit()
        minus = (basis_state(2, 0) - basis_state(2, 1)).unit()
        initial_states = list(
            product(
                [plus, minus],
                [fock(self.N_ancilla, i) for i in range(self.N_ancilla)],
                [plus, minus],
                [fock(self.truncature, i) for i in range(self.truncature)],
                [plus, minus],
                [fock(self.N_b, i) for i in range(self.truncature)],
            )
        )
        dims = 2 * self.N_ancilla * 2 * self.truncature * 2 * self.N_b
        self.transition_matrix = np.zeros(
            (dims, dims),
            dtype=float,
        )
        hadamard = snot(1)
        self.H2 = tensor(
            hadamard,
            qeye(self.N_ancilla),
            hadamard,
            qeye(self.truncature),
            hadamard,
            qeye(self.N_b),
        )
        for j, (qa, ga, qd, gd, qb, gb) in enumerate(initial_states):
            psi0 = tensor(qa, ga, qd, gd, qb, gb)
            self.initial_state = psi0
            self.simulate()
            self.rho = self.H2 * self.rho * self.H2
            self.transition_matrix[:, j] = self.rho.diag()

        # Remove <0 numerical errors
        self.transition_matrix[self.transition_matrix < 0] = 0
        self.verb('Parity Measurement transition_matrices')
        self.verb(self.transition_matrix)
        print(
            f'Elapsed time Parity Measurement t matrix: {time() - t_start:.2f}s'
        )


if __name__ == '__main__':
    N_1 = 2  # truncature
    N_2 = 2  # ancilla
    N_3 = 2  # mode_b

    params = {
        'nbar': 4,
        'k1': 1e-3,
        'k2': 1,
        'k1a': 1e-3,
        'k2a': 5,
        'k1b': 1e-3,
        'k2b': 1,
        'gate_time': 1,
        'truncature': N_1,
        'N_ancilla': N_2,
        'N_b': N_3,
    }

    params.update({'k1a': params['k2a'] * params['k1']})
    pm = ParityMeasurementGate(**params)
    pm.simulate()
    pm.transition_matrices()
