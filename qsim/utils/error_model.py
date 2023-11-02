from abc import ABC, abstractmethod
from importlib.machinery import SourcelessFileLoader
from typing import Optional

import numpy as np
from qutip.operators import qeye
from qutip.states import fock, ket2dm
from qutip.tensor import tensor

from experiments.repeated_cnots.correlations import (
    PhaseFlipsCorrelationReducedReducedModel,
    PhaseFlipsCorrelationSFB,
    full_simulation,
)
from qsim.physical_gate.cnot import CNOTSFBPhaseFlips
from qsim.utils.utils import combine, exponentiel_proba


class ErrorModel(ABC):
    def __init__(self) -> None:
        self.pI = None
        self.pX = None
        self.pY = None
        self.pZ = None

    @abstractmethod
    def get_dict(self):
        pass

    def __str__(self, only_non_zero_items=True):
        if only_non_zero_items:
            return dict(
                filter(lambda val: val[1] != 0, self.get_dict().items())
            )
        return self.get_dict()

    def get_error(self):
        p_l = list(self.get_dict().values())
        return np.random.choice(len(p_l), p=p_l)


class OneQubitErrorModel(ErrorModel):
    def __init__(
        self, pI: float = 0.0, pX: float = 0.0, pY: float = 0.0, pZ: float = 0.0
    ):
        self.pI = pI
        self.pX = pX
        self.pY = pY
        self.pZ = pZ
        sum_p = sum([v for (k, v) in self.get_dict().items() if k != 'pI'])
        if sum_p + self.pI != 1.0:
            if self.pI == 0.0:
                self.pI = 1 - sum_p
            else:
                raise ValueError('The probabilities should add up to 1.')

    def get_dict(self):
        return {'pI': self.pI, 'pX': self.pX, 'pY': self.pY, 'pZ': self.pZ}

    def get_stim_pauli_channel(self):
        return (
            self.pX,
            self.pY,
            self.pZ,
        )


class Depolarize1(OneQubitErrorModel):
    def __init__(self, p: float):
        super().__init__(1 - p, p / 3, p / 3, p / 3)



class TwoQubitErrorModel(ErrorModel):
    def __init__(
        self,
        pII: float = 0.0,
        pIX: float = 0.0,
        pIY: float = 0.0,
        pIZ: float = 0.0,
        pXI: float = 0.0,
        pXX: float = 0.0,
        pXY: float = 0.0,
        pXZ: float = 0.0,
        pYI: float = 0.0,
        pYX: float = 0.0,
        pYY: float = 0.0,
        pYZ: float = 0.0,
        pZI: float = 0.0,
        pZX: float = 0.0,
        pZY: float = 0.0,
        pZZ: float = 0.0,
    ):
        self.pII = pII
        self.pIX = pIX
        self.pIY = pIY
        self.pIZ = pIZ
        self.pXI = pXI
        self.pXX = pXX
        self.pXY = pXY
        self.pXZ = pXZ
        self.pYI = pYI
        self.pYX = pYX
        self.pYY = pYY
        self.pYZ = pYZ
        self.pZI = pZI
        self.pZX = pZX
        self.pZY = pZY
        self.pZZ = pZZ
        sum_p = sum([v for (k, v) in self.get_dict().items() if k != 'pII'])
        if sum_p + self.pII != 1.0:
            if self.pII == 0.0:
                self.pII = 1 - sum_p
            else:
                raise ValueError('The probabilities should add up to 1.')

    def get_dict(self):
        return {
            'pII': self.pII,
            'pIX': self.pIX,
            'pIY': self.pIY,
            'pIZ': self.pIZ,
            'pXI': self.pXI,
            'pXX': self.pXX,
            'pXY': self.pXY,
            'pXZ': self.pXZ,
            'pYI': self.pYI,
            'pYX': self.pYX,
            'pYY': self.pYY,
            'pYZ': self.pYZ,
            'pZI': self.pZI,
            'pZX': self.pZX,
            'pZY': self.pZY,
            'pZZ': self.pZZ,
        }

    def get_stim_pauli_channel(self):
        return (
            self.pIX,
            self.pIY,
            self.pIZ,
            self.pXI,
            self.pXX,
            self.pXY,
            self.pXZ,
            self.pYI,
            self.pYX,
            self.pYY,
            self.pYZ,
            self.pZI,
            self.pZX,
            self.pZY,
            self.pZZ,
        )



class CNOTCodeCapacity(TwoQubitErrorModel):
    def __init__(self, p_data: float):
        super().__init__(pZZ=p_data)


class CNOTPheno(TwoQubitErrorModel):
    def __init__(self, p_data: float, p_meas: float):
        super().__init__(pZZ=p_data, pZI=p_meas)


class CNOTAnalyticalAsymmetric(TwoQubitErrorModel):
    def __init__(
        self,
        nbar: int,
        k1d: float,
        k1a: float,
        k2d: float,
        k2a: int,
        gate_time: float,
        **kwargs,
    ):
        self.nbar = nbar
        self.k1d = k1d
        self.k2d = k2d
        self.gate_time = gate_time
        self.k2a = k2a
        self.k1a = k1a
        one_photon_loss_ancilla = exponentiel_proba(nbar * k1a * gate_time)
        pZI = combine(
            [
                np.pi**2 / (64 * nbar * (k2a + k2d) / 2 * gate_time),
                one_photon_loss_ancilla,
            ]
        )
        pZZ = exponentiel_proba(0.5 * nbar * k1d * gate_time)
        pIZ = exponentiel_proba(0.5 * nbar * k1d * gate_time)
        super().__init__(pZI=pZI, pZZ=pZZ, pIZ=pIZ)


class CNOTAnalytical(CNOTAnalyticalAsymmetric):
    def __init__(
        self,
        nbar: int,
        k1: float,
        k2: float,
        gate_time: float,
        **kwargs,
    ):
        super().__init__(
            nbar=nbar,
            k1d=k1,
            k1a=k1,
            k2d=k2,
            k2a=k2,
            gate_time=gate_time,
            **kwargs,
        )


class CNOTPRAp3p(TwoQubitErrorModel):
    def __init__(self, p: float):
        super().__init__(pII=1 - 4 * p, pZI=3 * p, pIZ=p / 2, pZZ=p / 2)


class CNOTPRAErrorModel(TwoQubitErrorModel):
    def __init__(self, nbar: int, k1: float, gate_time: float, **kwargs):
        # Error models
        p = exponentiel_proba(nbar * k1 * gate_time)

        p_CX_Z1 = (
            1 - 1 / 2 / np.pi / nbar / gate_time
        ) * p + 1 / 2 / np.pi / nbar / gate_time * (1 - p)
        p_CX_Z2 = p / 2
        p_CX_Z1Z2 = p / 2
        super().__init__(pZI=p_CX_Z1, pIZ=p_CX_Z2, pZZ=p_CX_Z1Z2)


class CNOTSimuErrorModel(TwoQubitErrorModel):
    def __init__(
        self,
        nbar: int,
        k1: float,
        gate_time: float,
        k2d: float,
        k2a: float,
        N_data: int,
        N_ancilla: int,
        **kwargs,
    ):
        cnot = CNOTSFBPhaseFlips(
            nbar=nbar,
            k2=k2d,
            k2a=k2a,
            k1=k1,
            k1a=k1 * k2a,
            gate_time=gate_time,
            truncature=N_data,
            N_ancilla=N_ancilla,
        )
        cnot.simulate()
        plus = ket2dm((fock(2, 0) + fock(2, 1)).unit())
        minus = ket2dm((fock(2, 0) - fock(2, 1)).unit())
        # Pplusplus = tensor(plus, qeye(N_ancilla), plus, qeye(N_data))
        Pplusminus = tensor(plus, qeye(N_ancilla), minus, qeye(N_data))
        Pminusplus = tensor(minus, qeye(N_ancilla), plus, qeye(N_data))
        Pminusminus = tensor(minus, qeye(N_ancilla), minus, qeye(N_data))

        # fid = np.real((cnot.rho * Pplusplus).tr())
        p_CX_Z1 = np.real((cnot.rho * Pminusplus).tr())
        p_CX_Z2 = np.real((cnot.rho * Pplusminus).tr())
        p_CX_Z1Z2 = np.real((cnot.rho * Pminusminus).tr())

        super().__init__(pZI=p_CX_Z1, pIZ=p_CX_Z2, pZZ=p_CX_Z1Z2)


class CNOTReducedModelNumericalErrorFromCorrelations(CNOTAnalyticalAsymmetric):
    def __init__(
        self,
        nbar: int,
        k1d: float,
        k1a: float,
        k2d: float,
        k2a: int,
        gate_time: float,
        truncature: int,
    ):
        super().__init__(
            nbar=nbar, k2d=k2d, k2a=k2a, k1d=k1d, k1a=k1a, gate_time=gate_time
        )
        self.truncature = truncature
        corrRRMnI = PhaseFlipsCorrelationReducedReducedModel(
            nbar=nbar,
            k1=k1d,
            k2=k2d,
            gate_time=gate_time,
            k2a=k2a,
            k1a=k1a,
            N=truncature,
            k2a_l=[k2a],
            N_ancilla=0,
        )
        one_photon_loss_ancilla = exponentiel_proba(nbar * k1a * gate_time)
        try:
            _, pZ1_l = corrRRMnI.get_phaseflip_data()
        except FileNotFoundError:
            full_simulation(
                k=5,
                nbar=nbar,
                k2a=k2a,
                N=truncature,
                exp=PhaseFlipsCorrelationReducedReducedModel,
                N_ancilla=0,
                force_sim=False,
                # force_sim=True,
            )
            _, pZ1_l = corrRRMnI.get_phaseflip_data()

        print('PZ1 non adiabatic from repeated_cnots')
        print(pZ1_l)
        self.pZI = combine([pZ1_l[-1], one_photon_loss_ancilla])


class CNOTCompleteModelNumericalErrorFromCorrelations(CNOTAnalyticalAsymmetric):
    def __init__(
        self,
        nbar: int,
        k1d: float,
        k1a: float,
        k2d: float,
        k2a: int,
        gate_time: float,
        truncature: int,
        N_ancilla: int,
    ):
        super().__init__(
            nbar=nbar, k2d=k2d, k2a=k2a, k1d=k1d, k1a=k1a, gate_time=gate_time
        )
        self.truncature = truncature
        corrSFB = PhaseFlipsCorrelationSFB(
            nbar=nbar,
            k1=k1d,
            k2=k2d,
            gate_time=gate_time,
            k2a=k2a,
            k1a=k1a,
            N=truncature,
            k2a_l=[k2a],
            N_ancilla=N_ancilla,
        )
        one_photon_loss_ancilla = exponentiel_proba(nbar * k1a * gate_time)
        try:
            _, pZ1_l = corrSFB.get_phaseflip_data()
        except FileNotFoundError:
            full_simulation(
                k=2,
                nbar=nbar,
                k2a=k2a,
                N=truncature,
                exp=PhaseFlipsCorrelationSFB,
                N_ancilla=N_ancilla,
                force_sim=False,
                # force_sim=True,
                use_mp=False,
            )
            _, pZ1_l = corrSFB.get_phaseflip_data()

        print('PZ1 non adiabatic from repeated_cnots')
        print(pZ1_l)
        print(f'{one_photon_loss_ancilla=}')
        self.pZI = combine([pZ1_l[-1], one_photon_loss_ancilla])



if __name__ == '__main__':
    params = {
        'nbar': 8,
        'distance': 7,
        'k1d': 1e-3,
        'k1': 1e-3,
        'k1a': 1e-3,
        'k2d': 1,
        'k2a': 1,
        'k2': 1,
        'gate_time': 1,
        'logic_fail_max': 1000,
        'N_data': 7,
        'N_ancilla': 7,
    }

    for EM_class in [CNOTSimuErrorModel, CNOTAnalyticalAsymmetric]:
        print(EM_class.__name__)
        print(EM_class(**params).get_dict())
