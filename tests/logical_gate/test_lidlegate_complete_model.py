import pytest

from qsim.logical_gate.lidle import (
    LIdleGateCompleteModel,
    LIdleGateCompleteModelPheno,
)
from tests.logical_gate.lidlegate_complete_model_pheno_test_class import (
    LIdleGateCompleteModelPhenoTestClass,
)
from tests.logical_gate.lidlegate_complete_model_test_class import (
    LIdleGateCompleteModelTestClass,
)


class TestLIdleGateCompleteModel:
    pass


@pytest.mark.parametrize('num_rounds', [3])
class TestClassLIdleGateCompleteModel(LIdleGateCompleteModelTestClass):
    gate_cls = LIdleGateCompleteModel


@pytest.mark.parametrize('num_rounds', [3])
class TestClassLIdleGateCompleteModelPheno(
    LIdleGateCompleteModelPhenoTestClass
):
    gate_cls = LIdleGateCompleteModelPheno
