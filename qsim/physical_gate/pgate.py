import datetime
from math import sqrt
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from qutip import Qobj

from qsim.gate import Gate


class PGate(Gate):
    """A physical gate."""

    def __init__(
        self,
        nbar: int,
        k1: float,
        k2: float,
        gate_time: float,
        truncature: int,
        kphi: float=0,
        initial_state: Optional[Qobj] = None,
        initial_state_name: str = 'unset',
        H: Optional[List[float]] = None,
        D: Optional[List[float]] = None,
        rho: Optional[Qobj] = None,
        num_tslots_pertime: int = 1000,
    ):
        """
        Args:
            nbar: Number of photons.
            k1: todo
            k2: todo
        """
        super().__init__()

        self.nbar = nbar
        self.k1 = k1
        self.k2 = k2
        self.gate_time = gate_time
        self.truncature = truncature
        self.kphi = kphi
        self.initial_state = initial_state
        self.initial_state_name = initial_state_name
        self.H = H
        self.D = D
        self.rho = rho
        self.num_tslots_pertime = num_tslots_pertime

    def _pre_simulate(self):
        super()._pre_simulate()
        self.alpha = sqrt(self.nbar)
        self.basis = None

        # time
        self.num_tslots = int(self.num_tslots_pertime * self.gate_time)
        self.dt = 1 / self.num_tslots_pertime * 1 / self.max_frequency()
        self.times = np.linspace(
            0, self.gate_time, int(self.gate_time // self.dt)
        )

    def _get_results(
        self,
    ) -> Dict[str, Any]:  # pylint: disable=protected-access
        return {
            'version': self.version,
            'class_name': self.class_name,
            'datetime': datetime.datetime.now(),
            'elapsed_time': self.elapsed_time,
            'nbar': self.nbar,
            'k1': self.k1,
            'k2': self.k2,
            'gate_time': self.gate_time,
            'truncature': self.truncature,
            'kphi': self.kphi,
            'num_tslots_pertime': self.num_tslots_pertime,
            'initial_state_name': self.initial_state_name,
            'state': self.rho,
        }

    @staticmethod
    def get_non_key_attributes() -> Iterable[str]:
        return ('version', 'class_name', 'datetime', 'elapsed_time', 'state')

    def transition_matrices(self):
        pass

    def max_frequency(self) -> float:
        return max(self.k1, self.k2)

    def __repr__(self):
        return (
            f'{type(self).__name__}'
            f'({(self.nbar, self.k1, self.k2, self.gate_time)})'
        )

    def __str__(self):
        return (
            f'Gate : {type(self).__name__} \n Initial state :'
            f' {self.initial_state_name} \n (nbar, T, k2, k1, kphi) \n'
            f' {(self.nbar, self.gate_time, self.k2, self.k1, self.kphi)} \n'
        )


class OneQubitPGate(PGate):
    pass


class TwoQubitPGate(PGate):
    def __init__(
        self,
        nbar: int,
        k1: float,
        k2: float,
        k1a: float,
        k2a: float,
        gate_time: float,
        truncature: int,
        kphi: float=0,
        kphia: float=0,
        initial_state: Optional[Qobj] = None,
        initial_state_name: str = 'unset',
        H: Optional[List[float]] = None,
        D: Optional[List[float]] = None,
        rho: Optional[Qobj] = None,
        num_tslots_pertime: int = 1000,
        N_ancilla: Optional[int] = None,
    ):
        super().__init__(
            nbar,
            k1,
            k2,
            gate_time,
            truncature,
            kphi,
            initial_state,
            initial_state_name,
            H,
            D,
            rho,
            num_tslots_pertime,
        )
        self.k1a = k1a
        self.k2a = k2a
        self.kphia = kphia
        self.N_ancilla = truncature if N_ancilla is None else N_ancilla

    def _get_results(
        self,
    ) -> Dict[str, Any]:  # pylint: disable=protected-access
        return {
            **super()._get_results(),
            'k2a': self.k2a,
            'k1a': self.k1a,
            'kphia': self.kphia,
            'N_ancilla': self.N_ancilla,
        }

    def max_frequency(self) -> float:
        return max(self.k1, self.k2, self.k2a, self.k1a, self.kphi, self.kphia)

    def __repr__(self):
        return (
            f'{type(self).__name__}'
            f'({(self.nbar, self.k1, self.k2, self.kphi, self.gate_time)}, '
            f'{self.k2a, self.k1a, self.kphia})'
        )

    def __str__(self):
        return super().__str__() + f'\n k2a: {self.k2a}\n k1a: {self.k1a}\n kphia: {self.kphia}'


class ThreeQubitPGate(TwoQubitPGate):
    def __init__(
        self,
        nbar: int,
        k1: float,
        k2: float,
        k1a: float,
        k2a: float,
        k1b: float,
        k2b: float,
        gate_time: float,
        truncature: int,
        kphi: float=0,
        kphia: float=0,
        kphib: float=0,
        initial_state: Optional[Qobj] = None,
        initial_state_name: str = 'unset',
        H: Optional[List[float]] = None,
        D: Optional[List[float]] = None,
        rho: Optional[Qobj] = None,
        num_tslots_pertime: int = 1000,
        N_ancilla: Optional[int] = None,
        N_b: Optional[int] = None,
    ):
        super().__init__(
            nbar,
            k1,
            k2,
            k1a,
            k2a,
            gate_time,
            truncature,
            kphi,
            kphia,
            initial_state,
            initial_state_name,
            H,
            D,
            rho,
            num_tslots_pertime,
            N_ancilla,
        )
        self.k1b = k1b
        self.k2b = k2b
        self.kphib = kphib
        self.N_b = truncature if N_b is None else N_b

    def _get_results(
        self,
    ) -> Dict[str, Any]:  # pylint: disable=protected-access
        return {
            **super()._get_results(),
            'k2b': self.k2b,
            'k1b': self.k1b,
            'kphib': self.kphib,
            'N_b': self.N_b,
        }

    def max_frequency(self) -> float:
        return max(self.k1, self.k2, self.k2a, self.k1a, self.k2b, self.k1b, self.kphi, self.kphia, self.kphib)

    def __repr__(self):
        return (
            f'{type(self).__name__}'
            f'({(self.nbar, self.k1, self.k2, self.kphi, self.gate_time)}, '
            f'{self.k2a, self.k1a, self.kphia}), '
            f'{self.k2b, self.k1b, self.kphib})'
        )

    def __str__(self):
        return '\n'.join(
            [super().__str__(), f'k2b: {self.k2b}', f'k1b: {self.k1b}', f'kphib: {self.kphib}']
        )
