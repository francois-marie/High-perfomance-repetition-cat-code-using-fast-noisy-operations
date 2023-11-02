from typing import Callable

import pytest
from qutip import Qobj, tensor

from qsim.basis.basis import Basis
from qsim.basis.fock import Fock
from qsim.basis.sfb import SFB
from qsim.physical_gate.cnot import (
    CNOTSFB,
    CNOTFockFull,
    CNOTFockQutip,
    CNOTFockRotatedFrame,
    CNOTSFBPhaseFlips,
)
from tests.physical_gate.pgate_test_class import TwoQubitPGateTestClass


def state(label_a: str, label_b: str) -> Callable[[Basis], Qobj]:
    def state_getter(basis):
        return tensor(
            getattr(getattr(basis, 'ancilla'), label_a),
            getattr(getattr(basis, 'data'), label_b),
        )

    state_getter.__name__ = f'|{label_a},{label_b}⟩'
    return state_getter


test_states = [
    (state('zero', 'zero'), state('zero', 'zero')),
    (state('zero', 'one'), state('zero', 'one')),
    (state('zero', 'evencat'), state('zero', 'evencat')),
    (state('zero', 'oddcat'), state('zero', 'oddcat')),
    (state('one', 'zero'), state('one', 'one')),
    (state('one', 'one'), state('one', 'zero')),
    (state('one', 'evencat'), state('one', 'evencat')),
    (state('one', 'oddcat'), state('one', 'oddcat')),
    (state('evencat', 'evencat'), state('evencat', 'evencat')),
    (state('evencat', 'oddcat'), state('oddcat', 'oddcat')),
    (state('oddcat', 'evencat'), state('oddcat', 'evencat')),
    (state('oddcat', 'oddcat'), state('evencat', 'oddcat')),
]


@pytest.mark.parametrize('nbar', [4])
@pytest.mark.parametrize('k1', [1e-8])
@pytest.mark.parametrize('k2', [1.0])
@pytest.mark.parametrize('k1a', [1e-8])
@pytest.mark.parametrize('k2a', [1.0])
@pytest.mark.parametrize('gate_time', [4.0])
@pytest.mark.parametrize('num_tslots_pertime', [1000])
@pytest.mark.parametrize('tol', [5e-2])
@pytest.mark.parametrize('initial_state, target_state', test_states)
class CNOTGateTestClass(TwoQubitPGateTestClass):
    pass


@pytest.mark.slow
@pytest.mark.parametrize('truncature', [5])
class TestClassCNOTSFB(CNOTGateTestClass):
    gate_cls = CNOTSFB
    basis_cls = SFB


# @pytest.mark.slow
@pytest.mark.parametrize('truncature', [5])
class TestClassCNOTSFBPhaseFlips(CNOTGateTestClass):
    gate_cls = CNOTSFBPhaseFlips
    basis_cls = SFB


