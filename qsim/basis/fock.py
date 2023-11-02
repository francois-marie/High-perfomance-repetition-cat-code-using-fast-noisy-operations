from functools import cached_property
from itertools import product
from typing import List, Tuple

import numpy as np
from qutip import (
    Qobj,
    coherent,
    destroy,
    isket,
    ket2dm,
    plot_fock_distribution,
    qdiags,
    qeye,
    tensor,
)
from scipy.special import factorial2, iv

from qsim.basis.basis import Basis, QubitOperators


class QubitOperatorsFock(QubitOperators):
    def __init__(self, d: int, hilbert_space_dims: List[int], alpha: float):
        super().__init__(d, hilbert_space_dims, alpha)
        self.a = destroy(d)
        coha = coherent(self.d, self.alpha)
        cohma = coherent(self.d, -self.alpha)
        self.evencat = (coha + cohma).unit()
        self.oddcat = (coha - cohma).unit()
        self.zero = (self.evencat + self.oddcat) / np.sqrt(2)
        self.one = (self.evencat - self.oddcat) / np.sqrt(2)

    # invariants
    @cached_property
    def destroy(self) -> Qobj:
        return Qobj(np.diag(np.sqrt(np.arange(1, self.d)), 1))

    @cached_property
    def Jpp(self) -> Qobj:
        return Qobj(
            np.fromfunction(
                # set diagonal and odd row element to 1 and other to 0
                np.vectorize(lambda i, j: float(i == j and not i % 2)),
                (self.d, self.d),
            )
        )

    @cached_property
    def Jmm(self) -> Qobj:
        return qeye(self.d) - self.Jpp

    def positive(self, q: int) -> Qobj:
        return Qobj(
            np.diag(
                factorial2(np.arange(-1, self.d - 1, 1))
                / factorial2(np.arange(2 * q, self.d + 2 * q, 1))
            )
        )

    def negative(self, q: int) -> Qobj:
        return Qobj(
            np.diag(
                factorial2(np.arange(0, self.d, 1))
                / factorial2(np.arange(-1 - 2 * q, -1 + self.d - 2 * q, 1))
            )
        )

    def Jpmq(self, q: int) -> Qobj:
        if q >= 0:
            return self.positive(q) * self.Jpp * self.destroy ** (2 * q + 1)
        return (
            self.Jpp * self.destroy.trans() ** (-2 * q - 1) * self.negative(q)
        )

    def cpm(self, rho: Qobj) -> np.complex128:
        if isket(rho):
            rho = ket2dm(rho)
        return np.trace(self.Jpm.dag() * rho)

    def jx(self, rho: Qobj) -> np.complex128:
        if isket(rho):
            rho = ket2dm(rho)
        return np.trace(self.Jx * rho)

    @cached_property
    def Jpm(self) -> Qobj:
        Jpm = Qobj(np.array(np.zeros((self.d, self.d)), dtype=complex))
        for q in np.arange(-self.d, self.d + 1, 1):
            Jpm += (
                (-1) ** abs(q)
                / (2 * q + 1)
                * iv(q, abs(self.alpha) ** 2)
                * self.Jpmq(q)
            )
        Jpm *= np.sqrt(
            (2 * abs(self.alpha) ** 2) / (np.sinh(2 * abs(self.alpha) ** 2))
        )
        return Qobj(Jpm, dims=[[self.d], [self.d]])

    @cached_property
    def Ji(self) -> Qobj:
        return qeye(self.d) / np.sqrt(2)

    @cached_property
    def Jx(self) -> Qobj:
        return (self.Jpm + self.Jpm.dag()) / np.sqrt(2)

    @cached_property
    def Jy(self) -> Qobj:
        return 1.0j * ((-self.Jpm + self.Jpm.dag())) / np.sqrt(2)

    @cached_property
    def Jz(self) -> Qobj:
        return (self.Jpp - self.Jmm) / np.sqrt(2)

    def bitflip_proba(self, rho: Qobj) -> float:
        return np.real(1 - self.jx(rho)) / 2

    def invar_coeff(
        self, state: Qobj
    ) -> Tuple[np.complex128, np.complex128, np.complex128]:
        if isket(state):
            state = ket2dm(state)
        cpp = np.trace((self.Jpp.dag() * state))
        cmm = np.trace((self.Jmm.dag() * state))
        cpm = np.trace((self.Jpm.dag() * state))
        return cpp, cmm, cpm

    def rho_infinite(
        self, cpp: np.complex128, cmm: np.complex128, cpm: np.complex128
    ) -> Qobj:
        # alpha = self.alpha
        even_dm = ket2dm(self.evencat)
        odd_dm = ket2dm(self.oddcat)
        even_odd_dm = self.evencat * self.oddcat.dag()
        return (
            cpp * even_dm
            + cmm * odd_dm
            + cpm * even_odd_dm
            + cpm.conj() * even_odd_dm.dag()
        ).unit()


# pylint: disable=R0904
class Fock(Basis):
    """Describes multiple qubits with their operators written in Fock
    basis"""

    QUBIT_OPERATORS_CLS = QubitOperatorsFock

    def hilbert_space_dims(self, d) -> List[int]:
        return [d]

    def print_info_px_jx(self):
        states = {
            '0': self.data.zero,
            '1': self.data.one,
            '+': self.data.evencat,
            '-': self.data.oddcat,
        }
        for name, state in states.items():
            print(f'state: {name}')
            print('Px: ', self.data.bitflip_proba(state))
            print('jx: ', self.data.jx(state))
            cpm = self.data.cpm(state)
            print('cpm : ', cpm)
            print('cpm+cpm*: ', cpm + cpm.conj())

        states_ancilla = {
            '0': self.ancilla.zero,
            '1': self.ancilla.one,
            '+': self.ancilla.evencat,
            '-': self.ancilla.oddcat,
        }

        states_prod = product(states_ancilla.keys(), states.keys())
        for name1, name2 in states_prod:
            print(f'{name1}, {name2}')
            rho = tensor(states[name1], states[name2])
            rho = ket2dm(rho)
            print(f'Px, Oth {self.ancilla.bitflip_proba(rho.ptrace(0))}')
            print(f'Px, 1th {self.data.bitflip_proba(rho.ptrace(1))}')
            CNOT_rho_CNOT = self.CNOTalpha * rho * self.CNOTalpha
            print('after cnot')
            print(
                f'Px, Oth {self.ancilla.bitflip_proba(CNOT_rho_CNOT.ptrace(0))}'
            )
            print(f'Px, 1th {self.data.bitflip_proba(CNOT_rho_CNOT.ptrace(1))}')

    def to_code_space(self, rho: Qobj) -> Qobj:
        Jpp = self.data.Jpp
        Jpm = self.data.Jpm
        Jmm = self.data.Jmm

        length = len(rho.dims[0])

        if length == 1:
            rho_code = np.array(np.zeros((2, 2)), dtype=complex)
            rho_code[0, 0] = np.trace(Jpp * rho)
            rho_code[0, 1] = np.trace(Jpm.dag() * rho)
            rho_code[1, 0] = rho_code[0, 1].conj()
            rho_code[1, 1] = np.trace(Jmm * rho)
            rho_code = Qobj(rho_code)

            Hadamard = self.Hadamard()
            return Hadamard * rho_code * Hadamard
        Jpp_ancilla = self.ancilla.Jpp
        Jpm_ancilla = self.ancilla.Jpm
        Jmm_ancilla = self.ancilla.Jmm
        if length == 2:
            rho_code = np.array(np.zeros((4, 4)), dtype=complex)
            rho_code[0, 0] = np.trace(np.kron(Jpp_ancilla, Jpp) * rho)
            rho_code[0, 1] = np.trace(np.kron(Jpp_ancilla, Jpm.dag()) * rho)
            rho_code[0, 2] = np.trace(np.kron(Jpm_ancilla.dag(), Jpp) * rho)
            rho_code[0, 3] = np.trace(
                np.kron(Jpm_ancilla.dag(), Jpm.dag()) * rho
            )

            rho_code[1, 0] = rho_code[0, 1].conj()
            rho_code[1, 1] = np.trace(np.kron(Jpp_ancilla, Jmm) * rho)
            rho_code[1, 2] = np.trace(np.kron(Jpm_ancilla.dag(), Jpm) * rho)
            rho_code[1, 3] = np.trace(np.kron(Jpm_ancilla.dag(), Jmm) * rho)

            rho_code[2, 0] = rho_code[0, 2].conj()
            rho_code[2, 1] = rho_code[1, 2].conj()
            rho_code[2, 2] = np.trace(np.kron(Jmm_ancilla, Jpp) * rho)
            rho_code[2, 3] = np.trace(np.kron(Jmm_ancilla, Jpm.dag()) * rho)

            rho_code[3, 0] = rho_code[0, 3].conj()
            rho_code[3, 1] = rho_code[1, 3].conj()
            rho_code[3, 2] = rho_code[2, 3].conj()
            rho_code[3, 3] = np.trace(np.kron(Jmm_ancilla, Jmm) * rho)
            rho_code = Qobj(rho_code)

            Hadamard = self.Hadamard()
            return (
                tensor(Hadamard, Hadamard)
                * rho_code
                * tensor(Hadamard, Hadamard)
            )
        if length == 3:
            return None

        raise ValueError('wrong format for the input state rho')

    def thermal_state(self, nth: float) -> Qobj:
        c_p_init = []
        for n in range(self.data.d):
            c_p_init.append(nth**n / (1 + nth) ** (n + 1))

        gauge = qdiags(c_p_init, offsets=0).unit()
        if self.verbose:
            _, ax = plot_fock_distribution(gauge)
            ax.set_yscale('log')
            ax.set_ylim(1e-8, 2)
        return gauge
