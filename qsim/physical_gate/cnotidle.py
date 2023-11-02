from typing import List, Optional

import numpy as np
from qutip import Options, Qobj
from qutip import basis as basis_state
from qutip import (
    destroy,
    fock_dm,
    isket,
    ket2dm,
    mesolve,
    num,
    qeye,
    sigmam,
    sigmap,
    sigmax,
    sigmaz,
    tensor,
)
from qutip.qip.operations import snot

from qsim.basis.sfb import SFB, SFBNonOrthonormal
from qsim.physical_gate.cnot import CNOTSFBReducedModel
from qsim.physical_gate.pgate import ThreeQubitPGate


class CNOTSFBReducedModelQubitBitParityMeasurementFirst(CNOTSFBReducedModel):
    def _simulate(self):
        if self.initial_state is None:
            self.initial_state = tensor(
                (basis_state(2, 0) + basis_state(2, 1)) / np.sqrt(2),
                basis_state(self.truncature, 0),
                basis_state(self.truncature, 0),
            )

        if isket(self.initial_state):
            self.initial_state = ket2dm(self.initial_state)

        if self.H is None:
            H = [
                [
                    tensor(
                        qeye(2), qeye(self.truncature), qeye(self.truncature)
                    ),
                    lambda x, args: 0,
                ]
            ]
        else:
            H = list(self.H)

        r1 = 4 * self.k2 * self.alpha**2
        L2d = [
            np.sqrt(r1)
            * tensor(fock_dm(2, 0), sigmap(), qeye(self.truncature)),
            [
                np.sqrt(r1)
                * tensor(fock_dm(2, 1), sigmap(), qeye(self.truncature)),
                np.exp(2j * np.pi * self.times / self.gate_time),
            ],
        ]
        r2 = np.pi**2 / 16 / self.alpha**2 / self.k2a / self.gate_time**2
        LH = [np.sqrt(r2) * tensor(sigmaz(), sigmam(), qeye(self.truncature))]
        c_ops = L2d + LH

        D = self.D or c_ops

        results = mesolve(
            H=H,
            rho0=self.initial_state,
            tlist=self.times,
            c_ops=D,
            e_ops=[
                tensor(
                    ket2dm(basis_state(2, 0) - basis_state(2, 1)) / 2,
                    qeye(self.truncature),
                    qeye(self.truncature),
                ),
                tensor(qeye(2), num(self.truncature), qeye(self.truncature)),
            ],
            args=None,
            options=Options(method='bdf', store_final_state=True),
            progress_bar=True if self.verbose else None,
            _safe_mode=True,
        )
        self.rho = results.final_state
        self.expect = results.expect
        return results


class CNOTSFBReducedModelQubitBitParityMeasurementSecond(CNOTSFBReducedModel):
    def _simulate(self):
        if self.initial_state is None:
            self.initial_state = tensor(
                (basis_state(2, 0) + basis_state(2, 1)) / np.sqrt(2),
                basis_state(self.truncature, 0),
                basis_state(self.truncature, 0),
            )

        if isket(self.initial_state):
            self.initial_state = ket2dm(self.initial_state)

        if self.H is None:
            H = [
                [
                    tensor(
                        qeye(2), qeye(self.truncature), qeye(self.truncature)
                    ),
                    lambda x, args: 0,
                ]
            ]
        else:
            H = list(self.H)

        r1 = 4 * self.k2 * self.alpha**2
        L2d = [
            np.sqrt(r1)
            * tensor(
                fock_dm(2, 0),
                qeye(self.truncature),
                sigmap(),
            ),
            [
                np.sqrt(r1)
                * tensor(
                    fock_dm(2, 1),
                    qeye(self.truncature),
                    sigmap(),
                ),
                np.exp(2j * np.pi * self.times / self.gate_time),
            ],
        ]
        r2 = np.pi**2 / 16 / self.alpha**2 / self.k2a / self.gate_time**2
        LH = [np.sqrt(r2) * tensor(sigmaz(), qeye(self.truncature), sigmam())]

        c_ops = L2d + LH

        D = self.D or c_ops

        results = mesolve(
            H=H,
            rho0=self.initial_state,
            tlist=self.times,
            c_ops=D,
            e_ops=[
                tensor(
                    ket2dm(basis_state(2, 0) - basis_state(2, 1)) / 2,
                    qeye(self.truncature),
                    qeye(self.truncature),
                ),
                tensor(qeye(2), qeye(self.truncature), num(self.truncature)),
            ],
            args=None,
            options=Options(method='bdf', store_final_state=True),
            progress_bar=True if self.verbose else None,
            _safe_mode=True,
        )
        self.rho = results.final_state
        self.expect = results.expect
        return results


class CNOT12Idle3SFBPhaseFlips(ThreeQubitPGate):
    """Simulation of the appproximated CNOT gate in the shifted fock basis.
        we do a CNOT between modes 1st (control) and 2nd (target) and the third mode does an Idle gate.
        Size : 2 * self.N_ancilla * 2 * self.truncature * 2 * self.N_b
    Args:
        ThreeQubitPGate (PGate): _description_
    """

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
        initial_state: Optional[Qobj] = None,
        initial_state_name: str = 'unset',
        H: Optional[List[float]] = None,
        D: Optional[List[float]] = None,
        rho: Optional[Qobj] = None,
        num_tslots_pertime: int = 1000,
        N_ancilla: Optional[int] = None,
        N_b: Optional[int] = None,
    ):
        if N_ancilla is None:
            N_ancilla = truncature
        if N_b is None:
            N_b = truncature
        self.N_ancilla = N_ancilla
        self.N_b = N_b
        if initial_state is None:
            initial_state = tensor(
                SFBNonOrthonormal(nbar=nbar, d=N_ancilla).data.evencat,
                SFBNonOrthonormal(nbar=nbar, d=truncature).data.evencat,
                SFBNonOrthonormal(nbar=nbar, d=N_b).data.evencat,
            )
            initial_state_name = '+++'
        super().__init__(
            nbar,
            k1,
            k2,
            k1a,
            k2a,
            k1b,
            k2b,
            gate_time,
            truncature,
            initial_state,
            initial_state_name,
            H,
            D,
            rho,
            num_tslots_pertime,
            N_ancilla=N_ancilla,
            N_b=N_b,
        )

    def _pre_simulate(self):
        super()._pre_simulate()
        self.b_a = None
        self.b_d = None
        self.b_b = None
        self.alpha = None
        self.L1a = None
        self.L1d = None
        self.L2a = None
        self.L2d = None
        self.H = None
        self.H2 = None
        self.U = None

    def populate_class_parameters(self):
        if self.b_a is None:
            self.b_a = destroy(self.N_ancilla)
        if self.b_d is None:
            self.b_d = destroy(self.truncature)
        if self.b_b is None:
            self.b_b = destroy(self.N_b)
        if self.alpha is None:
            self.alpha = np.sqrt(self.nbar)

        # Two-photon dissipation
        if self.L2a is None:
            self.L2a = np.sqrt(self.k2a) * tensor(
                qeye(2),
                self.b_a**2 + 2 * self.alpha * self.b_a,
                qeye(2),
                qeye(self.truncature),
                qeye(2),
                qeye(self.N_b),
            )
        if self.L2d is None:
            self.L2d = [
                np.sqrt(self.k2)
                * tensor(
                    fock_dm(2, 0),
                    qeye(self.N_ancilla),
                    qeye(2),
                    self.b_d**2 + 2 * self.alpha * self.b_d,
                    qeye(2),
                    qeye(self.N_b),
                ),
                [
                    np.sqrt(self.k2)
                    * tensor(
                        fock_dm(2, 1),
                        qeye(self.N_ancilla),
                        qeye(2),
                        self.b_d**2 + 2 * self.alpha * self.b_d,
                        qeye(2),
                        qeye(self.N_b),
                    ),
                    np.exp(2j * np.pi * self.times / self.gate_time),
                ],
                [
                    np.sqrt(self.k2)
                    * tensor(
                        sigmaz(),
                        self.b_a,
                        qeye(2),
                        qeye(self.truncature),
                        qeye(2),
                        qeye(self.N_b),
                    ),
                    self.alpha
                    / 2
                    * (np.exp(2j * np.pi * self.times / self.gate_time) - 1),
                ],
            ]
        self.L2b = np.sqrt(self.k2b) * tensor(
            qeye(2),
            qeye(self.N_ancilla),
            qeye(2),
            qeye(self.truncature),
            qeye(2),
            self.b_b**2 + 2 * self.alpha * self.b_b,
        )

        # Single photon loss
        if self.L1a is None:
            self.L1a = np.sqrt(self.k1a) * tensor(
                sigmaz(),
                self.b_a + self.alpha,
                qeye(2),
                qeye(self.truncature),
                qeye(2),
                qeye(self.N_b),
            )
        if self.L1d is None:
            self.L1d = [
                np.sqrt(self.k1)
                * tensor(
                    fock_dm(2, 0),
                    qeye(self.N_ancilla),
                    sigmaz(),
                    self.b_d + self.alpha,
                    qeye(2),
                    qeye(self.N_b),
                ),
                [
                    np.sqrt(self.k1)
                    * tensor(
                        fock_dm(2, 1),
                        qeye(self.N_ancilla),
                        sigmaz(),
                        self.b_d + self.alpha,
                        qeye(2),
                        qeye(self.N_b),
                    ),
                    np.exp(1j * np.pi * self.times / self.gate_time),
                ],
            ]
        self.L1b = np.sqrt(self.k1b) * tensor(
            qeye(2),
            qeye(self.N_ancilla),
            qeye(2),
            qeye(self.truncature),
            sigmaz(),
            self.b_b + self.alpha,
        )

        # Feedforward Hamiltonian
        if self.H is None:
            self.H = (
                np.pi
                / 4
                / self.gate_time
                / self.alpha
                * tensor(
                    sigmaz(),
                    self.b_a + self.b_a.dag(),
                    qeye(2),
                    self.b_d.dag() * self.b_d
                    + self.alpha * (self.b_d + self.b_d.dag()),
                    qeye(2),
                    qeye(self.N_b),
                )
            )
        hadamard = snot(1)
        if self.H2 is None:
            self.H2 = tensor(
                hadamard,
                qeye(self.N_ancilla),
                hadamard,
                qeye(self.truncature),
                qeye(2),
                qeye(self.N_b),
            )
        if self.U is None:
            self.U = tensor(
                fock_dm(2, 0),
                qeye(self.N_ancilla),
                qeye(2),
                qeye(self.truncature),
                qeye(2),
                qeye(self.N_b),
            ) + np.exp(-1j * np.pi * self.nbar) * tensor(
                fock_dm(2, 1),
                qeye(self.N_ancilla),
                sigmax(),
                qeye(self.truncature),
                qeye(2),
                qeye(self.N_b),
            )

    def _simulate(self):
        print('start simulate')
        if self.N_ancilla is None:
            self.N_ancilla = self.truncature
        if self.N_b is None:
            self.N_b = self.truncature
        self.populate_class_parameters()
        if isket(self.initial_state):
            self.initial_state = ket2dm(self.initial_state)
        # self.basis = SFB(nbar=self.nbar, d=self.truncature)

        results = mesolve(
            H=self.H,
            rho0=self.U * self.initial_state * self.U.dag(),
            tlist=self.times,
            c_ops=[self.L2a, self.L2d, self.L1a, self.L1d, self.L2b, self.L1b],
            e_ops=[
                tensor(
                    qeye(2),
                    qeye(self.N_ancilla),
                    qeye(2),
                    num(self.truncature),
                    qeye(2),
                    qeye(self.N_b),
                ),
                tensor(
                    qeye(2),
                    num(self.N_ancilla),
                    qeye(2),
                    qeye(self.truncature),
                    qeye(2),
                    qeye(self.N_b),
                ),
            ],
            options=Options(method='bdf', store_final_state=True),
            progress_bar=True if self.verbose else None,
            _safe_mode=True,
        )
        rhof = results.final_state
        self.expect = results.expect
        self.rho = rhof
        print('end simulate')


class CNOT12Nothing3SFBPhaseFlips(CNOT12Idle3SFBPhaseFlips):
    """Simulation of the appproximated CNOT gate in the shifted fock basis.
        we do a CNOT between modes 1st (control) and 2nd (target) and the third mode does nothing.
        Size : 2 * self.N_ancilla * 2 * self.truncature * 2 * self.N_b
    Args:
        ThreeQubitPGate (PGate): _description_
    """

    def _simulate(self):
        print('start simulate')
        if self.N_ancilla is None:
            self.N_ancilla = self.truncature
        if self.N_b is None:
            self.N_b = self.truncature
        self.populate_class_parameters()
        if isket(self.initial_state):
            self.initial_state = ket2dm(self.initial_state)
        # self.basis = SFB(nbar=self.nbar, d=self.truncature)

        results = mesolve(
            H=self.H,
            rho0=self.U * self.initial_state * self.U.dag(),
            tlist=self.times,
            c_ops=[self.L2a, self.L2d, self.L1a, self.L1d],
            e_ops=[
                tensor(
                    qeye(2),
                    qeye(self.N_ancilla),
                    qeye(2),
                    num(self.truncature),
                    qeye(2),
                    qeye(self.N_b),
                ),
                tensor(
                    qeye(2),
                    num(self.N_ancilla),
                    qeye(2),
                    qeye(self.truncature),
                    qeye(2),
                    qeye(self.N_b),
                ),
            ],
            options=Options(method='bdf', store_final_state=True),
            progress_bar=True if self.verbose else None,
            _safe_mode=True,
        )
        rhof = results.final_state
        self.expect = results.expect
        self.rho = rhof
        print('end simulate')


class CNOT13Idle2SFBPhaseFlips(ThreeQubitPGate):
    """Simulation of the appproximated CNOT gate in the shifted fock basis.
        we do a CNOT between modes 1st (control) and 3rd (target) and the second mode does an Idle gate.
        Size : 2 * self.N_ancilla * 2 * self.truncature * 2 * self.N_b
    Args:
        ThreeQubitPGate (PGate): _description_
    """

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
        initial_state: Optional[Qobj] = None,
        initial_state_name: str = 'unset',
        H: Optional[List[float]] = None,
        D: Optional[List[float]] = None,
        rho: Optional[Qobj] = None,
        num_tslots_pertime: int = 1000,
        N_ancilla: Optional[int] = None,
        N_b: Optional[int] = None,
    ):
        if N_ancilla is None:
            N_ancilla = truncature
        if N_b is None:
            N_b = truncature
        self.N_ancilla = N_ancilla
        self.N_b = N_b
        if initial_state is None:
            initial_state = tensor(
                SFBNonOrthonormal(nbar=nbar, d=truncature).data.evencat,
                SFBNonOrthonormal(nbar=nbar, d=N_ancilla).data.evencat,
                SFBNonOrthonormal(nbar=nbar, d=N_b).data.evencat,
            )
            initial_state_name = '+++'
        super().__init__(
            nbar,
            k1,
            k2,
            k1a,
            k2a,
            k1b,
            k2b,
            gate_time,
            truncature,
            initial_state,
            initial_state_name,
            H,
            D,
            rho,
            num_tslots_pertime,
            N_ancilla=N_ancilla,
            N_b=N_b,
        )

    def _pre_simulate(self):
        super()._pre_simulate()
        self.b_a = None
        self.b_d = None
        self.b_b = None
        self.alpha = None
        self.L1a = None
        self.L1d = None
        self.L2a = None
        self.L2d = None
        self.H = None
        self.H2 = None
        self.U = None

    def populate_class_parameters(self):
        if self.b_a is None:
            self.b_a = destroy(self.N_ancilla)
        if self.b_d is None:
            self.b_d = destroy(self.truncature)
        if self.b_b is None:
            self.b_b = destroy(self.N_b)
        if self.alpha is None:
            self.alpha = np.sqrt(self.nbar)

        # Two-photon dissipation
        if self.L2a is None:
            self.L2a = np.sqrt(self.k2a) * tensor(
                qeye(2),
                self.b_a**2 + 2 * self.alpha * self.b_a,
                qeye(2),
                qeye(self.truncature),
                qeye(2),
                qeye(self.N_b),
            )
        if self.L2d is None:
            self.L2d = [
                np.sqrt(self.k2)
                * tensor(
                    fock_dm(2, 0),
                    qeye(self.N_ancilla),
                    qeye(2),
                    qeye(self.truncature),
                    qeye(2),
                    self.b_d**2 + 2 * self.alpha * self.b_d,
                ),
                [
                    np.sqrt(self.k2)
                    * tensor(
                        fock_dm(2, 1),
                        qeye(self.N_ancilla),
                        qeye(2),
                        qeye(self.truncature),
                        qeye(2),
                        self.b_b**2 + 2 * self.alpha * self.b_b,
                    ),
                    np.exp(2j * np.pi * self.times / self.gate_time),
                ],
                [
                    np.sqrt(self.k2)
                    * tensor(
                        sigmaz(),
                        self.b_a,
                        qeye(2),
                        qeye(self.truncature),
                        qeye(2),
                        qeye(self.N_b),
                    ),
                    self.alpha
                    / 2
                    * (np.exp(2j * np.pi * self.times / self.gate_time) - 1),
                ],
            ]
        self.L2b = np.sqrt(self.k2b) * tensor(
            qeye(2),
            qeye(self.N_ancilla),
            qeye(2),
            self.b_d**2 + 2 * self.alpha * self.b_d,
            qeye(2),
            qeye(self.N_b),
        )

        # Single photon loss
        if self.L1a is None:
            self.L1a = np.sqrt(self.k1a) * tensor(
                sigmaz(),
                self.b_a + self.alpha,
                qeye(2),
                qeye(self.truncature),
                qeye(2),
                qeye(self.N_b),
            )
        if self.L1d is None:
            self.L1d = [
                np.sqrt(self.k1)
                * tensor(
                    fock_dm(2, 0),
                    qeye(self.N_ancilla),
                    qeye(2),
                    qeye(self.truncature),
                    sigmaz(),
                    self.b_b + self.alpha,
                ),
                [
                    np.sqrt(self.k1)
                    * tensor(
                        fock_dm(2, 1),
                        qeye(self.N_ancilla),
                        qeye(2),
                        qeye(self.truncature),
                        sigmaz(),
                        self.b_b + self.alpha,
                    ),
                    np.exp(1j * np.pi * self.times / self.gate_time),
                ],
            ]
        self.L1b = np.sqrt(self.k1b) * tensor(
            qeye(2),
            qeye(self.N_ancilla),
            sigmaz(),
            self.b_d + self.alpha,
            qeye(2),
            qeye(self.N_b),
        )

        # Feedforward Hamiltonian
        if self.H is None:
            self.H = (
                np.pi
                / 4
                / self.gate_time
                / self.alpha
                * tensor(
                    sigmaz(),
                    self.b_a + self.b_a.dag(),
                    qeye(2),
                    qeye(self.truncature),
                    qeye(2),
                    self.b_b.dag() * self.b_b
                    + self.alpha * (self.b_b + self.b_b.dag()),
                )
            )
        hadamard = snot(1)
        if self.H2 is None:
            self.H2 = tensor(
                hadamard,
                qeye(self.N_ancilla),
                qeye(2),
                qeye(self.truncature),
                hadamard,
                qeye(self.N_b),
            )
        if self.U is None:
            self.U = tensor(
                fock_dm(2, 0),
                qeye(self.N_ancilla),
                qeye(2),
                qeye(self.truncature),
                qeye(2),
                qeye(self.N_b),
            ) + np.exp(-1j * np.pi * self.nbar) * tensor(
                fock_dm(2, 1),
                qeye(self.N_ancilla),
                qeye(2),
                qeye(self.truncature),
                sigmax(),
                qeye(self.N_b),
            )

    def _simulate(self):
        print('start simulate')
        self.basis = SFB(
            nbar=self.nbar,
            d=self.truncature,
            d_ancilla=self.N_ancilla,
            d_b=self.N_b,
        )
        if self.N_ancilla is None:
            self.N_ancilla = self.truncature
        if self.N_b is None:
            self.N_b = self.truncature
        self.populate_class_parameters()
        if isket(self.initial_state):
            self.initial_state = ket2dm(self.initial_state)
        # self.basis = SFB(nbar=self.nbar, d=self.truncature)
        results = mesolve(
            H=self.H,
            rho0=self.U * self.initial_state * self.U.dag(),
            tlist=self.times,
            c_ops=[self.L2a, self.L2d, self.L1a, self.L1d, self.L2b, self.L1b],
            e_ops=[
                tensor(
                    qeye(2),
                    qeye(self.N_ancilla),
                    qeye(2),
                    num(self.truncature),
                    qeye(2),
                    qeye(self.N_b),
                ),
                tensor(
                    qeye(2),
                    num(self.N_ancilla),
                    qeye(2),
                    qeye(self.truncature),
                    qeye(2),
                    qeye(self.N_b),
                ),
            ],
            options=Options(method='bdf', store_final_state=True),
            progress_bar=True if self.verbose else None,
            _safe_mode=True,
        )
        rhof = results.final_state
        self.expect = results.expect
        self.rho = rhof
        print('end simulate')


class CNOT13Nothing2SFBPhaseFlips(CNOT13Idle2SFBPhaseFlips):
    """Simulation of the appproximated CNOT gate in the shifted fock basis.
        we do a CNOT between modes 1st (control) and 3rd (target) and the second mode does an Idle gate.
        Size : 2 * self.N_ancilla * 2 * self.truncature * 2 * self.N_b
    Args:
        ThreeQubitPGate (PGate): _description_
    """

    def _simulate(self):
        print('start simulate')
        self.basis = SFB(
            nbar=self.nbar,
            d=self.truncature,
            d_ancilla=self.N_ancilla,
            d_b=self.N_b,
        )
        if self.N_ancilla is None:
            self.N_ancilla = self.truncature
        if self.N_b is None:
            self.N_b = self.truncature
        self.populate_class_parameters()
        if isket(self.initial_state):
            self.initial_state = ket2dm(self.initial_state)
        # self.basis = SFB(nbar=self.nbar, d=self.truncature)
        results = mesolve(
            H=self.H,
            rho0=self.U * self.initial_state * self.U.dag(),
            tlist=self.times,
            c_ops=[self.L2a, self.L2d, self.L1a, self.L1d],
            e_ops=[
                tensor(
                    qeye(2),
                    qeye(self.N_ancilla),
                    qeye(2),
                    num(self.truncature),
                    qeye(2),
                    qeye(self.N_b),
                ),
                tensor(
                    qeye(2),
                    num(self.N_ancilla),
                    qeye(2),
                    qeye(self.truncature),
                    qeye(2),
                    qeye(self.N_b),
                ),
            ],
            options=Options(method='bdf', store_final_state=True),
            progress_bar=True if self.verbose else None,
            _safe_mode=True,
        )
        rhof = results.final_state
        self.expect = results.expect
        self.rho = rhof
        print('end simulate')
