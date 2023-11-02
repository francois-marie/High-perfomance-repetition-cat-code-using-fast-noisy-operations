from abc import ABC, abstractmethod
from functools import cached_property, reduce
from math import sqrt
from typing import Dict, List, Optional, TypedDict

import numpy as np
from qutip import Qobj, isket, ket2dm, num, qeye, tensor


class QubitOperators:
    """Gathers the operators of a qubit defined by its decomposition into
    subsystems (Fock or SFB) and the truncature used for the gauge
    mode and the photon population.

    Its properties include states (coherents states used for the encoding),
    operators (annihilation, number, identity, projector to the code space).
    """

    def __init__(self, d: int, hilbert_space_dims: List[int], alpha: float):
        """
        Args:
            d (int): The dimension of the harmonic oscillator (entire
                qubit in the case of Fock and only gauge mode in the
                SFB)
            hilbert_space_dims (List[int]): Decomposition of the qubit
                into several subsystems (only one subsystem in the Fock
                basis and two subsystems in the SFB: one 2 level qubit
                and one gauge)
            alpha (float): sqrt of the mean number of photons

        Example:
            >>> data = QubitOperators(5, [5], 2) # Fock basis
            >>> data = QubitOperators(5, [2, 5], 2) # SFB
        """
        self.d = d
        self.hilbert_space_dims = hilbert_space_dims
        self.a = Qobj(dims=self.dims)
        self.zero = Qobj(dims=self.dims)
        self.one = Qobj(dims=self.dims)
        self.evencat = Qobj(dims=self.dims)
        self.oddcat = Qobj(dims=self.dims)
        self.dims_np = reduce(lambda x, y: x * y, self.hilbert_space_dims)
        self.num_op = num(self.dims_np)
        self.num_op.dims = self.dims
        self.alpha = alpha

    @cached_property
    def adag(self) -> Qobj:
        return self.a.dag()

    @cached_property
    def ada(self) -> Qobj:
        return self.adag * self.a

    @cached_property
    def I(self) -> Qobj:  # noqa: E743
        return qeye(self.hilbert_space_dims)

    @cached_property
    def proj_code_space(self) -> Qobj:
        return ket2dm(self.evencat) + ket2dm(self.oddcat)

    @cached_property
    def dims(self) -> List[List[int]]:
        return [self.hilbert_space_dims, self.hilbert_space_dims]

    @cached_property
    def ancilla_dm(self) -> Qobj:
        return self.proj_code_space / 2

    @cached_property
    def Jz(self) -> Qobj:
        return (1j * np.pi * self.num_op).expm()


class BasisError(Exception):
    """Base exception for any exception coming from `Basis`."""


class UndefinedAncillaError(BasisError):
    """Exception raised when trying to use an undefined ancilla."""


class QubitDimensions(TypedDict):
    data: int
    ancilla: int


class Basis(ABC):  # pylint: disable=too-many-public-methods
    """Describes multiple qubits with their operators written in a
    particular basis. The possible basis include Fock, SFB and non
    orthonormal SFB.

    Given multiple truncatures (d, d_ancilla), it will create one qubit
    per truncature as a QubitOperators instance.

    Having them as parameters allows defining multi qubit operators used
    in master equations (dissipators, hamiltonians).
    """

    QUBIT_OPERATORS_CLS = QubitOperators

    def __init__(
        self,
        nbar: int,
        d: int,
        d_ancilla: Optional[int] = None,
        d_b: Optional[int] = None,
        verbose: bool = False,
    ):
        """Possible keys for qubit_dimensions: ['data', 'ancilla']"""
        self.nbar = nbar
        self.alpha = sqrt(nbar)

        self.data = self.QUBIT_OPERATORS_CLS(
            d, self.hilbert_space_dims(d), self.alpha
        )
        self.ancilla = (
            None
            if d_ancilla is None
            else self.QUBIT_OPERATORS_CLS(
                d_ancilla,
                self.hilbert_space_dims(d_ancilla),
                self.alpha,
            )
        )
        self.mode_b = (
            None
            if d_b is None
            else self.QUBIT_OPERATORS_CLS(
                d_b,
                self.hilbert_space_dims(d_b),
                self.alpha,
            )
        )

        self.verbose = verbose

    def _check_ancilla(self):
        if self.ancilla is None:
            raise UndefinedAncillaError('the basis contains no ancilla qubit')

    @abstractmethod
    def hilbert_space_dims(self, d: int) -> List[int]:
        pass

    @abstractmethod
    def to_code_space(self, rho: Qobj) -> Qobj:
        pass

    def Hz(self, eps_Z: float) -> Qobj:
        return eps_Z * (self.data.a + self.data.adag)

    def two_photon_pumping(self, k2: float) -> Qobj:
        return np.sqrt(k2) * (self.data.a**2 - self.alpha**2 * self.data.I)

    def two_photon_pumping_data(self, k2: float) -> Qobj:
        return np.sqrt(k2) * tensor(
            self.ancilla.I,
            self.data.a**2 - self.alpha**2 * self.data.I,
        )

    def two_photon_pumping_ancilla(self, k2a: float) -> Qobj:
        return np.sqrt(k2a) * tensor(
            self.ancilla.a**2 - self.alpha**2 * self.ancilla.I,
            self.data.I,
        )

    def one_photon_loss(self, k1: float) -> Qobj:
        return np.sqrt(k1) * self.data.a

    def one_photon_loss_data(self, k1: float) -> Qobj:
        return tensor(self.ancilla.I, np.sqrt(k1) * self.data.a)

    def one_photon_loss_ancilla(self, k1a: float) -> Qobj:
        return tensor(
            np.sqrt(k1a) * self.ancilla.a,
            self.data.I,
        )

    def thermic_photon(self, kphi: float) -> Qobj:
        return np.sqrt(kphi) * self.data.a.dag() * self.data.a

    def Hx(self, gate_time: float) -> Qobj:
        return -np.pi / gate_time * self.data.ada

    def X_diss(self, k2: float, dt: float, t: float, gate_time: float) -> Qobj:
        return np.sqrt(k2 * dt) * (
            self.data.a * self.data.a
            - np.exp(2j * np.pi * t / gate_time) * self.nbar * self.data.I
        )

    def Xalpha(self):
        return (
            self.data.zero * self.data.one.dag()
            + self.data.one * self.data.zero.dag()
        )

    def CNOTalpha(self):
        self._check_ancilla()
        return tensor(ket2dm(self.ancilla.zero), self.data.I) + tensor(
            ket2dm(self.ancilla.one), self.Xalpha
        )

    def HCNOT(self, gate_time: float) -> Qobj:
        self._check_ancilla()
        return Qobj(
            np.pi
            / 4
            / gate_time
            / self.alpha
            * tensor(
                self.ancilla.a
                + self.ancilla.adag
                - 2 * self.alpha * self.ancilla.I,
                self.data.ada - self.alpha**2 * self.data.I,
            )
        )

    def HCNOTPerfect(self, gate_time: float) -> Qobj:
        self._check_ancilla()
        return Qobj(
            -np.pi
            / gate_time
            * tensor(
                ket2dm(self.ancilla.one),
                self.data.ada - self.alpha**2 * self.data.I,
            )
        )

    @staticmethod
    def Hadamard() -> Qobj:
        return 1 / np.sqrt(2) * Qobj([[1, 1], [1, -1]])

    def leakage(self, rho: Qobj) -> Qobj:
        if isket(rho):
            rho = ket2dm(rho)
        if self.ancilla is None:
            return 1 - np.real(np.trace(self.data.proj_code_space * rho))
        return 1 - np.real(
            np.trace(
                tensor(
                    self.ancilla.proj_code_space,
                    self.data.proj_code_space,
                )
                * rho
            )
        )

    def generate_psi(
        self, c00: float, c01: float, c10: float, c11: float
    ) -> Qobj:
        """returns the state |psi> = c00 |00>
        + c01 |01> + c10 |10> + c11 |11>"""
        self._check_ancilla()
        return (
            c00 * tensor(self.ancilla.zero, self.data.zero)
            + c01 * tensor(self.ancilla.zero, self.data.one)
            + c10 * tensor(self.ancilla.one, self.data.zero)
            + c11 * tensor(self.ancilla.one, self.data.one)
        )

    def tomography_one_qubit(self) -> Dict[str, Qobj]:
        return {
            'k0': ket2dm(self.data.zero),
            'k1': ket2dm(self.data.one),
            'P': ket2dm(((self.data.zero + self.data.one) / np.sqrt(2))),
            'M': ket2dm(((self.data.zero + 1.0j * self.data.one) / np.sqrt(2))),
        }

    def tomography_two_qubits(self) -> Dict[str, Qobj]:
        return {
            'k0': self.generate_psi(1, 0, 0, 0),
            'k1': self.generate_psi(0, 1, 0, 0),
            'k2': self.generate_psi(0, 0, 1, 0),
            'k3': self.generate_psi(0, 0, 0, 1),
            'P01': self.generate_psi(1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0),
            'P02': self.generate_psi(1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0),
            'P03': self.generate_psi(1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)),
            'P12': self.generate_psi(0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0),
            'P13': self.generate_psi(0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)),
            'P23': self.generate_psi(0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)),
            'M01': self.generate_psi(1 / np.sqrt(2), 1j / np.sqrt(2), 0, 0),
            'M02': self.generate_psi(1 / np.sqrt(2), 0, 1j / np.sqrt(2), 0),
            'M03': self.generate_psi(1 / np.sqrt(2), 0, 0, 1j / np.sqrt(2)),
            'M12': self.generate_psi(0, 1 / np.sqrt(2), 1j / np.sqrt(2), 0),
            'M13': self.generate_psi(0, 1 / np.sqrt(2), 0, 1j / np.sqrt(2)),
            'M23': self.generate_psi(0, 0, 1 / np.sqrt(2), 1j / np.sqrt(2)),
        }
