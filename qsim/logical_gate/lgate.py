import datetime
from abc import abstractmethod
from typing import Any, Dict, Iterable

from qsim.gate import Gate
from qsim.utils.error_model import ErrorModel


class LGate(Gate):
    """A logical gate."""

    def __init__(self, distance: int, physical_gates: Dict[str, ErrorModel]):
        super().__init__()

        self.distance = distance
        self.physical_gates = physical_gates

        self.outcome: Dict[str, Any] = {'X': None, 'Y': None, 'Z': None}
        """This dict has three keys, one for each kind of error. Each value
        contains the outcome for the specified error, or `None` if it has not
        been simulated."""

        self.num_trajectories: Dict[str, int] = {'X': 0, 'Y': 0, 'Z': 0}
        """This dict has three keys, one for each kind of error. Each value
        contains the number of simulated trajectories."""

    def _simulate(self):
        self._simulate_X()
        self._simulate_Y()
        self._simulate_Z()

    def _simulate_X(self):
        """Implement the simulation of the X error of the gate."""

    def _simulate_Y(self):
        """Implement the simulation of the Y error of the gate."""

    @abstractmethod
    def _simulate_Z(self):
        """Implement the simulation of the Z error of the gate."""

    def _get_results(self) -> Dict[str, Any]:
        return {
            'version': self.version,
            'class_name': self.class_name,
            'datetime': datetime.datetime.now(),
            'elapsed_time': self.elapsed_time,
            'identifier': f'ZL_{self._save_name()}',
            'distance': self.distance,
            'physical_gates': {
                k: v.get_dict() for k, v in self.physical_gates.items()
            },
            'state': {
                'outcome': self.outcome,
                'num_trajectories': self.num_trajectories,
            },
        }

    @staticmethod
    def get_non_key_attributes() -> Iterable[str]:
        return ('version', 'class_name', 'datetime', 'elapsed_time', 'state')

    @abstractmethod
    def _save_name(self):
        pass

    def __repr__(self):
        return f'{type(self).__name__}({(self.distance, self.physical_gates)})'

    def __str__(self):
        return (
            f'LGate : {type(self).__name__} \n\t Distance :'
            f' {self.distance} \n\t Physical Gates :'
            f' {self.print_physical_gates()} \n'
        )

    def print_physical_gates(self):
        res = '\n'
        for k, v in self.physical_gates.items():
            res += f'\t{k}: '
            res += f'\t\t{v.__str__()}\n'
        return res


class OneQubitLGate(LGate):
    pass


class TwoQubitLGate(LGate):
    pass


class ThreeQubitLGate(LGate):
    pass
