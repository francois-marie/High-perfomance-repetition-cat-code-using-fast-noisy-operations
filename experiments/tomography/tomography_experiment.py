import multiprocessing as mp
import sys
from abc import abstractmethod
from typing import Optional

import numpy as np

from experiments.experiment import Experiment
from qsim.basis.basis import Basis
from qsim.basis.sfb import SFB
from qsim.physical_gate.cnot import CNOTSFB
from qsim.physical_gate.pgate import Gate
from qsim.utils.tomography import (
    CNOT_process_tomography,
    build_one_qubit_error_model,
    build_one_qubit_output,
    build_two_qubit_output,
    build_two_qubits_error_model,
    one_qubit_process_matrix,
)


class Tomography(Experiment):
    def __init__(
        self,
        N: int,
        gate: Optional[Gate] = None,
        basis: Optional[Basis] = None,
    ):
        self.N = N
        self.gate = gate
        self.basis = basis
        self.generate_cardinal_states()

    @abstractmethod
    def generate_cardinal_states(self):
        pass

    def single_simulation(self, index: int):
        self.gate.simulate(
            initial_state=self.cardinal_states[index],
            truncature=self.N,
            initial_state_name=self.cardinal_names[index],
        )

    def get_data(self):
        final_cardinal_states = []
        self.gate.basis = self.basis
        self.gate.truncature = self.N
        self.gate.num_tslots_pertime = 1000
        self.generate_cardinal_states()

        for name in self.cardinal_names:
            self.gate.initial_state_name = name

            data = self.gate._get_data()
            final_cardinal_states.append(data)

        try:
            rhop = self.build_output(
                *[
                    self.basis.to_code_space(state)
                    for state in final_cardinal_states
                ]
            )
            self.chi2error = self.build_process_tomography(rhop)
            self.data = self.build_error_model(self.chi2error)
        except:
            self.data = {}

    @abstractmethod
    def build_output(self):
        pass

    @abstractmethod
    def build_error_model(self, *args):
        pass

    @abstractmethod
    def build_process_tomography(self, rhop):
        pass

    def run_locally(self):
        self.generate_cardinal_states()
        pool = mp.Pool(len(self.cardinal_states))
        pool.starmap(
            self.single_simulation,
            [[i] for i in range(len(self.cardinal_states))],
        )


class OneQubitTomography(Tomography):
    def generate_cardinal_states(self):
        (
            self.cardinal_states,
            self.cardinal_names,
        ) = self.basis.tomography_one_qubit()

    def build_output(self, *args):
        return build_one_qubit_output(*args)

    def build_error_model(self, *args):
        return build_one_qubit_error_model(*args)

    def build_process_tomography(self, rhop):
        return one_qubit_process_matrix(rhop)



class TwoQubitTomography(Tomography):
    def generate_cardinal_states(self):
        (
            self.cardinal_states,
            self.cardinal_names,
        ) = self.basis.tomography_two_qubits()

    def build_output(self, *args):
        return build_two_qubit_output(*args)

    def build_error_model(self, *args):
        return build_two_qubits_error_model(*args)


class CNOTToomgraphy(TwoQubitTomography):
    def build_process_tomography(self, rhop):
        return CNOT_process_tomography(rhop)


if __name__ == "__main__":
    nbar = 4
    k2a_l = np.logspace(0, 2, 10)
    # k2a = k2a_l[int(sys.argv[1])]
    k2a = 1
    k1 = 1e-3
    k1a = k1 * k2a
    k2 = 1
    # index = int(sys.argv[2])

    N = 10
    # tomo.single_simulation(index=index)
    res = {}

    for k2a in k2a_l:
        gate_time = 1 / k2a

        cnot = CNOTSFB(
            nbar=nbar, k1=k1, k2=k2, k2a=k2a, k1a=k1a, gate_time=gate_time
        )
        basis = SFB(nbar=nbar, d=N)
        tomo = CNOTToomgraphy(N=10, gate=cnot, basis=basis)
        tomo.get_data()
        res[k2a] = tomo.data
    print(res)
