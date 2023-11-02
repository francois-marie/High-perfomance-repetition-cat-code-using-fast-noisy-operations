from math import sqrt

import pytest

from tests.logical_gate.lgate_test_class import LGateTestClass


@pytest.mark.parametrize('p', [1e-1])
@pytest.mark.parametrize('q', [1e-1])
@pytest.mark.parametrize('distance', [3])
@pytest.mark.parametrize('logic_fail_max', [1_000])
class PhenomenologicalTestClass(LGateTestClass):
    @pytest.fixture(autouse=True)
    def setup_gate(
        self, distance, p: float, q: float, num_rounds: int, logic_fail_max: int
    ):
        self.gate = self.gate_cls(
            distance=distance,
            p=p,
            q=q,
            num_rounds=num_rounds,
            logic_fail_max=logic_fail_max,
        )

    def test_simulate(self, n: int = 2):
        assert any(self.assert_std() for _ in range(n))

    def assert_std(self):
        super().test_simulate()
        p_l = 1 - self.gate.results["state"]["outcome"]["Z"]
        num_trajectories = self.gate.results['state']['num_trajectories']['Z']
        std = sqrt(p_l * (1 - p_l) / num_trajectories)
        return p_l - 1.96 * std < 1.42e-1 < p_l + 1.96 * std
