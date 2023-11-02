from typing import Callable

import pytest
from qutip import Qobj, tensor

from qsim.basis.basis import Basis
from qsim.basis.sfb import SFB
from qsim.physical_gate.cnotidle import (
    CNOT12Idle3SFBPhaseFlips,
    CNOT13Idle2SFBPhaseFlips,
)
from tests.physical_gate.pgate_test_class import ThreeQubitPGateTestClass


def state(label_a: str, label_d: str, label_b: str) -> Callable[[Basis], Qobj]:
    def state_getter(basis):
        return tensor(
            getattr(getattr(basis, 'ancilla'), label_a),
            getattr(getattr(basis, 'data'), label_d),
            getattr(getattr(basis, 'mode_b'), label_b),
        )

    state_getter.__name__ = f'|{label_d},{label_a},{label_b}⟩'
    return state_getter


# CNOT between modes 1 & 2
test_states12 = [
    (state('zero', 'zero', 'zero'), state('zero', 'zero', 'zero')),
    (state('zero', 'one', 'one'), state('zero', 'one', 'one')),
    (state('zero', 'evencat', 'zero'), state('zero', 'evencat', 'zero')),
    (state('zero', 'oddcat', 'oddcat'), state('zero', 'oddcat', 'oddcat')),
    (state('one', 'zero', 'zero'), state('one', 'one', 'zero')),
    (state('one', 'one', 'one'), state('one', 'zero', 'one')),
]

# CNOT between modes 1 & 3
test_states13 = [
    (state('zero', 'zero', 'zero'), state('zero', 'zero', 'zero')),
    (state('zero', 'one', 'one'), state('zero', 'one', 'one')),
    (state('zero', 'evencat', 'zero'), state('zero', 'evencat', 'zero')),
    (state('zero', 'oddcat', 'oddcat'), state('zero', 'oddcat', 'oddcat')),
    (state('one', 'zero', 'zero'), state('one', 'zero', 'one')),
    (state('one', 'one', 'one'), state('one', 'one', 'zero')),
]


@pytest.mark.parametrize('nbar', [8])
@pytest.mark.parametrize('k1', [1e-8])
@pytest.mark.parametrize('k2', [1.0])
@pytest.mark.parametrize('k1a', [1e-8])
@pytest.mark.parametrize('k2a', [1.0])
@pytest.mark.parametrize('k1b', [1e-8])
@pytest.mark.parametrize('k2b', [1.0])
@pytest.mark.parametrize('gate_time', [2.0])
@pytest.mark.parametrize('num_tslots_pertime', [1000])
@pytest.mark.parametrize('tol', [5e-2])
class CNOTIdleGateTestClass(ThreeQubitPGateTestClass):
    pass


N_ancilla = 2
N_target = 2
N_idle = 2


@pytest.mark.parametrize('initial_state, target_state', test_states12)
@pytest.mark.parametrize('truncature', [N_target])
@pytest.mark.parametrize('N_ancilla', [N_ancilla])
@pytest.mark.parametrize('N_b', [N_idle])
class TestClassCNOT12Idle3GateSFBPhaseFlips(CNOTIdleGateTestClass):
    gate_cls = CNOT12Idle3SFBPhaseFlips
    basis_cls = SFB


@pytest.mark.parametrize('initial_state, target_state', test_states13)
@pytest.mark.parametrize('truncature', [N_idle])
@pytest.mark.parametrize('N_ancilla', [N_ancilla])
@pytest.mark.parametrize('N_b', [N_target])
class TestClassCNOT13Idle2GateSFBPhaseFlips(CNOTIdleGateTestClass):
    gate_cls = CNOT13Idle2SFBPhaseFlips
    basis_cls = SFB
