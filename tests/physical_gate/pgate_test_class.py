from typing import Callable, Type

import pytest
from qutip import Qobj, isequal, ket2dm

from qsim.basis.basis import Basis
from qsim.physical_gate import PGate
from qsim.physical_gate.pgate import (
    OneQubitPGate,
    ThreeQubitPGate,
    TwoQubitPGate,
)


class PGateTestClass:
    gate_cls: Type[PGate]
    basis_cls: Type[Basis]

    @pytest.fixture(autouse=True)
    def setup(self, target_state, nbar, truncature, tol):
        self.basis = self.basis_cls(nbar, truncature, truncature)
        self.target_state = target_state(self.basis)
        self.tol = tol

    # fmt: off
    @pytest.fixture(autouse=True)
    def setup_gate(self, initial_state, nbar, k1, k2, gate_time, truncature,
                   num_tslots_pertime):
        self.gate = self.gate_cls(nbar, k1, k2, gate_time, truncature,
                                  initial_state=initial_state(self.basis),
                                  num_tslots_pertime=num_tslots_pertime)
    # fmt: on

    def test_simulate(self):
        self.gate.simulate()


class OneQubitPGateTestClass(PGateTestClass):
    gate_cls: Type[OneQubitPGate]

    def test_simulate(self):
        super().test_simulate()
        assert self.gate.rho.dims == [
            self.basis.data.hilbert_space_dims,
            self.basis.data.hilbert_space_dims,
        ]
        assert isequal(ket2dm(self.target_state), self.gate.rho, tol=self.tol)


class TwoQubitPGateTestClass(PGateTestClass):
    gate_cls: Type[TwoQubitPGate]

    # fmt: off
    @pytest.fixture(autouse=True)
    def setup_gate(self, initial_state, target_state, nbar, k1, k2, k1a, k2a,
                   gate_time, truncature, num_tslots_pertime, N_ancilla):
        self.gate = self.gate_cls(nbar, k1, k2, k1a, k2a, gate_time, truncature,
                                  initial_state=initial_state(self.basis),
                                  num_tslots_pertime=num_tslots_pertime, N_ancilla=N_ancilla)
    # fmt: on

    def test_simulate(self):
        super().test_simulate()
        assert self.gate.rho.dims == [
            self.basis.ancilla.hilbert_space_dims
            + self.basis.data.hilbert_space_dims,
            self.basis.ancilla.hilbert_space_dims
            + self.basis.data.hilbert_space_dims,
        ]
        assert isequal(ket2dm(self.target_state), self.gate.rho, tol=self.tol)


class ThreeQubitPGateTestClass(PGateTestClass):
    gate_cls: Type[ThreeQubitPGate]

    @pytest.fixture(autouse=True)
    def setup(self, target_state, nbar, truncature, N_ancilla, N_b, tol):
        self.basis = self.basis_cls(nbar, truncature, N_ancilla, N_b)
        self.target_state = target_state(self.basis)
        self.tol = tol

    # fmt: off
    @pytest.fixture(autouse=True)
    def setup_gate(self, initial_state, target_state, nbar, k1, k2, k1a, k2a,k1b, k2b,
                   gate_time, truncature, num_tslots_pertime, N_ancilla, N_b):
        self.gate = self.gate_cls(nbar, k1, k2, k1a, k2a, k1b, k2b, gate_time, truncature,
                                  initial_state=initial_state(self.basis),
                                  num_tslots_pertime=num_tslots_pertime, N_ancilla=N_ancilla, N_b=N_b)
    # fmt: on

    def test_simulate(self):
        super().test_simulate()
        assert self.gate.rho.dims == [
            self.basis.ancilla.hilbert_space_dims
            + self.basis.data.hilbert_space_dims
            + self.basis.mode_b.hilbert_space_dims,
            self.basis.ancilla.hilbert_space_dims
            + self.basis.data.hilbert_space_dims
            + self.basis.mode_b.hilbert_space_dims,
        ]
        assert isequal(ket2dm(self.target_state), self.gate.rho, tol=self.tol)
