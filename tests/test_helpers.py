from typing import Dict

from qsim.helpers import all_simulations, gen_params_simulate, simulate
from qsim.logical_gate.lgate import LGate
from qsim.utils.error_model import ErrorModel


class FakeLGate(LGate):
    def __init__(
        self,
        distance: int,
        physical_gates: Dict[str, ErrorModel],
        outcome: float,
        num_trajectories: int,
    ):
        super().__init__(distance, physical_gates)
        self._outcome = outcome
        self._num_trajectories = num_trajectories

    def _simulate_Z(self):
        self.outcome['Z'] = self._outcome
        self.num_trajectories['Z'] = self._num_trajectories

    def _save_name(self):
        return f'FakeLGate'


class TestHelpers:
    def test_lgate_simulate(self, tmpdir):
        simulate(
            FakeLGate,
            [
                {
                    'distance': 0,
                    'physical_gates': {},
                    'outcome': 0.0,
                    'num_trajectories': 3,
                }
            ],
            db_path=tmpdir / 'FakeLGate.json',
        ).join()

        simulate(
            FakeLGate,
            [
                {
                    'distance': 0,
                    'physical_gates': {},
                    'outcome': 1.0,
                    'num_trajectories': 7,
                }
            ],
            db_path=tmpdir / 'FakeLGate.json',
        ).join()

        sim = all_simulations(str(tmpdir / 'FakeLGate.json'))
        assert len(sim) == 1
        assert sim[0]['state'] == {
            'num_trajectories': {'X': 0, 'Y': 0, 'Z': 10},
            'outcome': {'X': None, 'Y': None, 'Z': 0.7},
        }

    def test_lgate_simulate_overwrite(self, tmpdir):
        simulate(
            FakeLGate,
            [
                {
                    'distance': 0,
                    'physical_gates': {},
                    'outcome': 0.0,
                    'num_trajectories': 3,
                }
            ],
            db_path=tmpdir / 'FakeLGate.json',
        ).join()

        simulate(
            FakeLGate,
            [
                {
                    'distance': 0,
                    'physical_gates': {},
                    'outcome': 1.0,
                    'num_trajectories': 7,
                }
            ],
            db_path=tmpdir / 'FakeLGate.json',
            overwrite_logical=True,
        ).join()

        sim = all_simulations(str(tmpdir / 'FakeLGate.json'))
        assert len(sim) == 1
        assert sim[0]['state'] == {
            'num_trajectories': {'X': 0, 'Y': 0, 'Z': 7},
            'outcome': {'X': None, 'Y': None, 'Z': 1.0},
        }

    def test_gen_params_simulate(self):
        assert gen_params_simulate(distance_l=range(3, 8, 2)) == [
            {'distance': 3},
            {'distance': 5},
            {'distance': 7},
        ]
        assert gen_params_simulate(
            distance_l=range(3, 8, 2), num_rounds_l=range(4, 7, 2)
        ) == [
            {'distance': 3, 'num_rounds': 4},
            {'distance': 3, 'num_rounds': 6},
            {'distance': 5, 'num_rounds': 4},
            {'distance': 5, 'num_rounds': 6},
            {'distance': 7, 'num_rounds': 4},
            {'distance': 7, 'num_rounds': 6},
        ]
