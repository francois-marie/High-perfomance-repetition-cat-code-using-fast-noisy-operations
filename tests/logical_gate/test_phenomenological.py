import pytest

from qsim.logical_gate.phenomenological import (
    CheckMatrixPhenomenological,
    FilterPhenomenological,
)
from tests.logical_gate.phenomenological_test_class import (
    PhenomenologicalTestClass,
)


class TestPhenomenological:
    pass


@pytest.mark.slow
@pytest.mark.parametrize('num_rounds', [3])
class TestClassFilterPhenomenological(PhenomenologicalTestClass):
    gate_cls = FilterPhenomenological


@pytest.mark.parametrize('num_rounds', [4])
class TestClassCheckMatrixPhenomenological(PhenomenologicalTestClass):
    gate_cls = CheckMatrixPhenomenological
