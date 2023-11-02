import pytest

from qsim.logical_gate.lidle import LIdleGatePhenomenological
from tests.logical_gate.phenomenological_test_class import (
    PhenomenologicalTestClass,
)


@pytest.mark.parametrize('num_rounds', [3])
class TestClassLIdleGatePhenomenological(PhenomenologicalTestClass):
    gate_cls = LIdleGatePhenomenological
