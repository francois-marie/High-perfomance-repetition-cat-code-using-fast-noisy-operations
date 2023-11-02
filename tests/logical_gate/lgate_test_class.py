from typing import Type

import pytest

from qsim.logical_gate import LGate


class LGateTestClass:
    gate_cls: Type[LGate]

    @pytest.fixture(autouse=True)
    def setup_gate(self, distance, physical_gates):
        self.gate = self.gate_cls(
            distance=distance, physical_gates=physical_gates
        )

    def test_simulate(self):
        self.gate.simulate()
