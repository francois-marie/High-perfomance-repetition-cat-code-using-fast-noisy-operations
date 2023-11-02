import pytest

from qsim.basis.fock import Fock
from qsim.basis.sfb import SFB
from qsim.physical_gate.idle import (
    IdleGateFock,
    IdleGateSFB,
    IdleGateSFBPhaseFlips,
)
from tests.physical_gate.pgate_test_class import OneQubitPGateTestClass, state


@pytest.mark.parametrize('nbar', [4])
@pytest.mark.parametrize('k1', [1e-8])
@pytest.mark.parametrize('k2', [1.0])
@pytest.mark.parametrize('gate_time', [2.0])
@pytest.mark.parametrize('num_tslots_pertime', [1000])
@pytest.mark.parametrize('tol', [1e-4])
@pytest.mark.parametrize(
    'initial_state, target_state',
    [(state('zero'), state('zero')), (state('one'), state('one'))],
)
class IdleGateTestClass(OneQubitPGateTestClass):
    pass


@pytest.mark.parametrize('truncature', [10])
class TestClassIdleGateSFB(IdleGateTestClass):
    gate_cls = IdleGateSFB
    basis_cls = SFB


@pytest.mark.parametrize('truncature', [10])
class TestClassIdleGateSFBPhaseFlips(IdleGateTestClass):
    gate_cls = IdleGateSFBPhaseFlips
    basis_cls = SFB


@pytest.mark.parametrize('truncature', [20])
class TestClassIdleGateFock(IdleGateTestClass):
    gate_cls = IdleGateFock
    basis_cls = Fock
