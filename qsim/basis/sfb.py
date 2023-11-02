from typing import List

import numpy as np
from numpy.linalg import cholesky
from qutip import Qobj
from qutip import basis as basis_state
from qutip import destroy, qeye, sigmaz, tensor
from qutip.qip.operations import snot
from scipy.special import factorial, genlaguerre

from qsim.basis.basis import Basis, QubitOperators


class QubitOperatorsSFB(QubitOperators):
    def __init__(self, d: int, hilbert_space_dims: List[int], alpha: float):
        super().__init__(d, hilbert_space_dims, alpha)
        # 1. Define qubit and local identity operators
        # 1.1. Qubit operators
        # sx = sigmax()
        # sy = sigmay()
        sz = sigmaz()
        si = qeye(2)
        # sp = sigmap()
        # sm = sigmam()
        self.su = 0.5 * (si + sz)
        self.sd = 0.5 * (si - sz)
        had = snot(1)
        # 1.2. SFB modes
        self.b = destroy(self.d)
        self.id_gauge = qeye(self.d)

        # 2. Define States
        vac_gauge = basis_state(d, 0)
        self.zero = tensor(basis_state(2, 0), vac_gauge)
        self.one = tensor(basis_state(2, 1), vac_gauge)
        self.evencat = (self.zero + self.one) / np.sqrt(2)
        self.oddcat = (self.zero - self.one) / np.sqrt(2)

        # 3. Orthonormalization matrices
        D = D_alpha(alpha=self.alpha, d=self.d)
        Gp = self.id_gauge + D
        Gm = self.id_gauge - D
        self.cp_inv = Qobj(cholesky(Gp)).dag()
        self.cm_inv = Qobj(cholesky(Gm)).dag()
        self.cp = self.cp_inv.inv()
        self.cm = self.cm_inv.inv()
        self.c = (
            tensor(had, self.id_gauge)
            * (tensor(self.su, self.cp) + tensor(self.sd, self.cm))
            * tensor(had, self.id_gauge)
        )
        self.c_inv = self.c.inv()
        self.a = (
            self.c_inv
            * tensor(sz, self.b + self.alpha * self.id_gauge)
            * self.c
        )

    def proj_sfb_to_fock(self, N: int) -> Qobj:
        d, alpha = self.d, self.alpha
        Dalpha = D_alpha(alpha, d=self.d)
        Dmalpha = D_alpha(-alpha, d=self.d)
        alternate_sign = Qobj(np.diag([(-1) ** n for n in range(0, d)]))

        return (
            1
            / np.sqrt(2)
            * Qobj(
                np.block(
                    [
                        Dalpha + Dmalpha * alternate_sign,
                        Dalpha - Dmalpha * alternate_sign,
                    ]
                ),
                dims=[[N], [2, d]],
            )
        )

    def proj_osfb_to_fock(self, rho_osfb):
        N = 40
        Proj_SFB_to_Fock = self.proj_sfb_to_fock(N)
        had = snot(1)
        rho_fock = (
            Proj_SFB_to_Fock
            * self.c
            * tensor(had, self.id_gauge)
            * rho_osfb
            * tensor(had, self.id_gauge)
            * self.c_inv
            * Proj_SFB_to_Fock.dag()
        )
        return rho_fock


# pylint: disable=invalid-name
def D_alpha(alpha: float, d: int):
    def D_ij_inside(i: int, j: int) -> float:  # pylint: disable=invalid-name
        if i < j:
            # pylint: disable=arguments-out-of-order
            return D_ij(j, i, alpha)  # D is symmetric
        return D_ij(i, j, alpha)

    return np.fromfunction(np.vectorize(D_ij_inside), (d, d))


# 3. Orthonormalization matrices
def D_ij(i: int, j: int, alpha: float) -> float:  # pylint: disable=invalid-name
    if i < j:
        # pylint: disable=arguments-out-of-order
        return D_ij(j, i, alpha)  # D is symmetric
    return (
        (-1) ** i
        * np.exp(-2 * alpha**2)
        * np.sqrt(factorial(j) / factorial(i))
        * genlaguerre(j, i - j)(4 * alpha**2)
        * (2 * alpha) ** (i - j)
    )


class SFB(Basis):
    """Describes multiple qubits with their operators written in SFB.
    A cat qubit is decomposed into two subsystems: a two level qubit
    and a gauge (harmonic oscillator)."""

    QUBIT_OPERATORS_CLS = QubitOperatorsSFB

    def hilbert_space_dims(self, d) -> List[int]:
        return [2, d]

    def to_code_space(self, rho: Qobj) -> Qobj:
        length = len(rho.dims[0])
        if length == 2:
            return rho.ptrace(0)
        if length == 4:
            return rho.ptrace((0, 2))
        if length == 6:
            return rho.ptrace((0, 2, 4))

        raise ValueError('wrong format for the input state rho')


class QubitOperatorsSFBNonOrthonormal(QubitOperators):
    def __init__(self, d: int, hilbert_space_dims: List[int], alpha: float):
        super().__init__(d, hilbert_space_dims, alpha)
        self.b = destroy(self.d)
        self.id_gauge = qeye(self.d)

        # 2. Define States
        vac_gauge = basis_state(self.d, 0)
        self.zero = tensor(basis_state(2, 0), vac_gauge)
        self.one = tensor(basis_state(2, 1), vac_gauge)
        self.evencat = (self.zero + self.one) / np.sqrt(2)
        self.oddcat = (self.zero - self.one) / np.sqrt(2)

        self.a = tensor(sigmaz(), self.b + self.alpha * self.id_gauge)


class SFBNonOrthonormal(SFB):
    """Describes multiple qubits with their operators written in non
    orthonormal SFB"""

    QUBIT_OPERATORS_CLS = QubitOperatorsSFBNonOrthonormal
