# pylint: disable=too-many-lines
from itertools import product
from time import time
from typing import List, Optional

import numpy as np
from numpy import exp, kron, sqrt
from numpy.linalg import eigh
from qutip import Options, Qobj
from qutip import basis as basis_state
from qutip import (
    create,
    destroy,
    fock,
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
from scipy.linalg import expm

from qsim.basis.fock import Fock
from qsim.basis.sfb import SFB, SFBNonOrthonormal
from qsim.physical_gate.pgate import TwoQubitPGate
from qsim.utils.utils import exponentiel_proba, proj_code_space_sfb


class CNOT(TwoQubitPGate):
    pass


class CNOTFock(CNOT):
    """Simulation of the CNOT gate in the fock basis."""

    def _simulate(self):
        if isket(self.initial_state):
            self.initial_state = ket2dm(self.initial_state)
        self.rho = self.initial_state
        self.basis = Fock(
            nbar=self.nbar, d=self.truncature, d_ancilla=self.N_ancilla
        )
        self.p2pp_data = self.basis.two_photon_pumping(self.k2)
        self.p2pp_ancilla = self.basis.two_photon_pumping(self.k2a)
        self.lc = tensor(self.p2pp_ancilla, self.basis.data.I)

        # L1 = sqrt(k1) a
        self.l1c = tensor(
            self.basis.one_photon_loss(self.k1), self.basis.data.I
        )
        self.l1t = tensor(
            self.basis.ancilla.I, self.basis.one_photon_loss(self.k1)
        )

class CNOTFockFull(CNOTFock):
    """Simulation of the CNOT gate in the fock basis in the unrotated frame."""

    def _simulate(self):
        super()._simulate()
        # H = pi / (4 alpha T) (a1+a1d - 2 alpha) (a2a2d - alpha**2)
        self.h = self.basis.HCNOT(self.gate_time)
        # Lc = nsqrt(k2a) (a1**2 - alpha**2)
        # Lt = Lta + f(t) Ltb
        # Lta = a2**2 - apha**2
        # Ltb = alpha/2 (a1 - alpha)
        # f(t) = (exp(2i pi t / T) - 1)
        self.lta, self.ltb = self.L_target(self.basis)

        # build first Kraus map
        ldl = self.lc.dag() * self.lc

        ltaa = self.lta.dag() * self.lta
        ltbb = self.ltb.dag() * self.ltb
        ltab = self.lta.dag() * self.ltb
        ltba = self.ltb.dag() * self.lta
        II = tensor(self.basis.ancilla.I, self.basis.data.I)
        l1cc = (
            self.k1 * tensor(self.basis.ancilla.ada, self.basis.data.I)
            if self.k1 != 0
            else Qobj(
                np.zeros((self.truncature**2, self.truncature**2)),
                dims=[
                    [self.truncature, self.truncature],
                    [self.truncature, self.truncature],
                ],
            )
        )
        l1tt = (
            self.k1 * tensor(self.basis.ancilla.I, self.basis.data.ada)
            if self.k1 != 0
            else Qobj(
                np.zeros((self.truncature**2, self.truncature**2)),
                dims=[
                    [self.truncature, self.truncature],
                    [self.truncature, self.truncature],
                ],
            )
        )

        # rho(t+dt) = M0 rho(t) M0d + sum{dt*L rho Ld} + O(dt**2)
        # M0_init: time indep part of M0
        # M0_init = 1 - i dt H - 0.5*(Lcd*Lc + Ltad*Lta + L1cd*L1c + L1td*L1t)
        M0_init = (
            II
            - 1j * self.dt * self.h
            - 0.5 * self.dt * (ldl + ltaa + l1cc + l1tt)
        )
        M0_dag_init = (
            II
            + 1j * self.dt * self.h
            - 0.5 * self.dt * (ldl + ltaa + l1cc + l1tt)
        )

        coeffs_l = [self.f(t) for t in self.times]
        rho = self.rho

        for t_idx in range(len(self.times)):
            coeffs = coeffs_l[t_idx]
            Lt_remaining = sum(
                a * b for (a, b) in zip(coeffs, [ltbb, ltab, ltba])
            )
            M0 = M0_init - 0.5 * self.dt * Lt_remaining
            M0_dag = M0_dag_init - 0.5 * self.dt * Lt_remaining

            # iterate
            rho_tmp = M0 * rho * M0_dag

            # sum{dt*L rho Ld} = dt * (Lc rho Lcd + L1c rho L1cd +
            #                          L1t rho L1td +
            #                          (Lta + f(t)Ltb) rho (Ltad + f*(t)Ltbd))
            rho_tmp += self.dt * self.lc * rho * self.lc.dag()
            if self.k1 != 0:
                rho_tmp += self.dt * self.l1c * rho * self.l1c.dag()
                rho_tmp += self.dt * self.l1t * rho * self.l1t.dag()

            rho_tmp += (
                self.dt
                * (self.lta + coeffs[1] * self.ltb)
                * rho
                * (self.lta.dag() + coeffs[2] * self.ltb.dag())
            )

            rho = rho_tmp / np.trace(rho_tmp)
        self.rho = rho
        return rho

    def L_target(self, basis):
        A = np.sqrt(self.k2) * tensor(basis.ancilla.I, self.p2pp_data)
        B = (
            np.sqrt(self.k2)
            * self.alpha
            / 2
            * tensor(
                basis.ancilla.a - self.alpha * basis.ancilla.I, basis.data.I
            )
        )
        return A, B

    def f(self, t):
        f = np.exp(2.0j * np.pi * t / self.gate_time) - 1
        return [np.abs(f) ** 2, f, f.conjugate()]


class CNOTSFB(CNOT):
    """Simulation of the CNOT gate in the shifted fock basis."""

    def _simulate(self):
        if isket(self.initial_state):
            self.initial_state = ket2dm(self.initial_state)
        rho = self.initial_state

        self.U = tensor(
            fock_dm(2, 0),
            qeye(self.N_ancilla),
            qeye(2),
            qeye(self.truncature),
        ) + np.exp(-1j * np.pi * self.nbar) * tensor(
            fock_dm(2, 1),
            qeye(self.N_ancilla),
            sigmax(),
            qeye(self.truncature),
        )
        rho = self.U * rho * self.U.dag()

        self.basis = SFB(
            nbar=self.nbar, d=self.truncature, d_ancilla=self.N_ancilla
        )
        phi_t = np.pi * self.times / self.gate_time

        sz = sigmaz()
        si = qeye(2)
        sp = sigmap()
        sm = sigmam()
        su = 0.5 * (si + sz)
        sd = 0.5 * (si - sz)

        #  Define operators in the rotating frame
        # -----------------------------------------------------

        # Time-dependent target rotating terms
        # HR = adag_t a_t - self.alpha^2
        # RP = exp(i phi_t HR)
        # RM = exp(-i phi_t HR)
        # To compute these time-dependent terms, we diagonalize HR in the
        # OSFB, using
        # adag * a --> self.basis.c_inv * G_inv * [Z x (adag +
        # self.alpha)] * G * [Z * (a + self.alpha)] * c
        HR = (
            self.basis.data.c_inv
            * tensor(
                sz, self.basis.data.b + self.alpha * self.basis.data.id_gauge
            )
            * self.basis.data.c
        )
        HR = HR.dag() * HR - self.alpha**2 * tensor(
            si, self.basis.data.id_gauge
        )
        HR_eigv, HR_U = eigh(HR)

        # Dissipator on control mode
        # C = a_c^2 - self.alpha^2
        #   = C0 x I + CP x RP + CM x RM

        gauge_b_sqd_plus_2_alpha_b = (
            self.basis.ancilla.b**2 + 2 * self.alpha * self.basis.ancilla.b
        )

        dissip_control_gauge_mode_first = (
            self.basis.ancilla.cp_inv
            * gauge_b_sqd_plus_2_alpha_b
            * self.basis.ancilla.cp
        )
        dissipcontrol_gaugemodesecond = (
            self.basis.ancilla.cm_inv
            * gauge_b_sqd_plus_2_alpha_b
            * self.basis.ancilla.cm
        )
        C0 = 0.5 * tensor(
            si,
            dissip_control_gauge_mode_first + dissipcontrol_gaugemodesecond,
        )
        CP = 0.5 * tensor(
            sp,
            dissip_control_gauge_mode_first - dissipcontrol_gaugemodesecond,
        )
        CM = 0.5 * tensor(
            sm,
            dissip_control_gauge_mode_first - dissipcontrol_gaugemodesecond,
        )

        # Dissipator on target mode
        # A = a_t^2 - self.alpha^2 + 0.5 self.alpha
        # [exp(2 i phi_t) - 1] (a_c - self.alpha)
        #   = - self.alpha^2 I x I + A0 x AR + exp(2 i phi_t)
        # A1 x AR + [exp(2 i phi_t) - 1] A2 x I
        #   + [exp(2 i phi_t) - 1] AP x RP + [exp(2 i phi_t) - 1] AM x RM
        gauge_b_plus_alpha = (
            self.basis.ancilla.b + self.alpha * self.basis.ancilla.id_gauge
        )
        cm_inv_gauge_b_plus_alpha_cp = (
            self.basis.ancilla.cm_inv
            * gauge_b_plus_alpha
            * self.basis.ancilla.cp
        )
        cp_inv_gauge_b_plus_alpha_cm = (
            self.basis.ancilla.cp_inv
            * gauge_b_plus_alpha
            * self.basis.ancilla.cm
        )
        A0 = tensor(su, self.basis.ancilla.id_gauge)
        A1 = tensor(sd, self.basis.ancilla.id_gauge)
        A2 = (
            0.25
            * self.alpha
            * tensor(
                sz,
                cm_inv_gauge_b_plus_alpha_cp + cp_inv_gauge_b_plus_alpha_cm,
            )
            - 0.5 * self.alpha**2 * self.basis.ancilla.I
        )
        AP = (
            0.25
            * self.alpha
            * tensor(
                sp,
                cm_inv_gauge_b_plus_alpha_cp - cp_inv_gauge_b_plus_alpha_cm,
            )
        )
        AM = (
            0.25
            * self.alpha
            * tensor(
                sm,
                cp_inv_gauge_b_plus_alpha_cm - cm_inv_gauge_b_plus_alpha_cp,
            )
        )
        gauge_b_sqd_plus_2_alpha_b_data = (
            self.basis.data.b**2 + 2 * self.alpha * self.basis.data.b
        )
        AR = (
            self.basis.data.c_inv
            * tensor(
                si,
                gauge_b_sqd_plus_2_alpha_b_data
                + self.alpha**2 * self.basis.data.id_gauge,
            )
            * self.basis.data.c
        )

        # Feedforward Hamiltonian (minus perfect Hamiltonian)
        # H_ff = eps_Z(t) (a_c + adag_c - 2 self.alpha)
        # (adag_t a_t - self.alpha^2)
        #     = eps_Z(t) [H0 x HR + HP x (HR RP) + HM x (HR RM)]
        H0 = 0.5 * tensor(
            sz,
            cp_inv_gauge_b_plus_alpha_cm
            + cm_inv_gauge_b_plus_alpha_cp
            - 2 * self.alpha * self.basis.ancilla.id_gauge,
        )
        HP = 0.5 * tensor(
            sp,
            cm_inv_gauge_b_plus_alpha_cp - cp_inv_gauge_b_plus_alpha_cm,
        )
        HM = 0.5 * tensor(
            sm,
            cp_inv_gauge_b_plus_alpha_cm - cm_inv_gauge_b_plus_alpha_cp,
        )
        H0 = H0 + H0.dag()
        HP, HM = HP + HM.dag(), HM + HP.dag()

        # Perfect Hamtilonian
        # PF = - 4 self.alpha eps_Z(t) |1><1| x HR
        # PF = -4 * self.alpha * tensor(sd, self.basis.id_gauge)

        # Single photon loss operators on control
        # LC = sqrt(kappa_c) a_c
        #    = sqrt(kappa_c) [LC0 x I + LCP x RP + LCM x RM]
        LC0 = 0.5 * tensor(
            sz,
            cp_inv_gauge_b_plus_alpha_cm + cm_inv_gauge_b_plus_alpha_cp,
        )
        LCP = 0.5 * tensor(
            sp,
            cm_inv_gauge_b_plus_alpha_cp - cp_inv_gauge_b_plus_alpha_cm,
        )
        LCM = 0.5 * tensor(
            sm,
            cp_inv_gauge_b_plus_alpha_cm - cm_inv_gauge_b_plus_alpha_cp,
        )

        # Single photon loss operators on target
        # LA = sqrt(kappa_t) a_t
        #    = sqrt(kappa_t) [(A0 + exp(i phi_t) A1) x LA0]
        gauge_b_plus_alpha_data = (
            self.basis.data.b + self.alpha * self.basis.data.id_gauge
        )
        LA0 = (
            self.basis.data.c_inv
            * tensor(sz, gauge_b_plus_alpha_data)
            * self.basis.data.c
        )

        # Dephasing operators on control
        # LCDC = sqrt(kappa_phic) adag_c a_c
        #      = sqrt(kappa_phic) ?

        #  Function for Hamiltonian and loss operators definition
        # -----------------------------------------------------
        # Convert static operators to dense numpy arrays
        ID = tensor(self.basis.ancilla.I, self.basis.data.I).full()

        C0_ = tensor(C0, self.basis.data.I).full()
        CP_ = tensor(CP).full()
        CM_ = tensor(CM).full()

        A0_ = (
            -(self.alpha**2) * tensor(self.basis.ancilla.I, self.basis.data.I)
            + tensor(A0, AR)
            - tensor(A2, self.basis.data.I)
        ).full()
        A1_ = (tensor(A1, AR) + tensor(A2, self.basis.data.I)).full()
        AP_ = tensor(AP).full()
        AM_ = tensor(AM).full()

        H0_ = tensor(H0, HR).full()
        HP_ = tensor(HP).full()
        HM_ = tensor(HM).full()
        # PF_ = tensor(PF, HR).full()
        LC0_ = tensor(LC0, self.basis.data.I).full()
        LCP_ = tensor(LCP).full()
        LCM_ = tensor(LCM).full()

        LA0_ = tensor(A0, LA0).full()
        LA1_ = tensor(A1, LA0).full()

        # define time dependent hamiltonian and loss operators functions
        def rotators(phi):
            """
            Return rotators RP and RM at time t
            """
            RP = HR_U.dot(np.diag(exp(1j * phi * HR_eigv))).dot(HR_U.T)
            RM = RP.T.conj()
            return RP, RM

        def hamil(RP, RM):
            """
            Return Hamiltonian at time t
            """
            return (
                np.pi
                / (4 * self.alpha * self.gate_time)
                * (H0_ + kron(HP_, HR * RP) + kron(HM_, HR * RM))
            )

        def target_dissip_t(RP, RM, kap_2, phi):
            """
            Return target dissipator at time t
            """
            return sqrt(kap_2) * (
                A0_
                + exp(2j * phi) * A1_
                + (exp(2j * phi) - 1) * (kron(AP_, RP) + kron(AM_, RM))
            )

        def control_dissip_t(RP, RM, kap_2):
            """
            Return control disipator at time t
            """
            return sqrt(kap_2) * (C0_ + kron(CP_, RP) + kron(CM_, RM))

        def a_c(kap_c, RP, RM):
            """
            Return single photon loss basis on control at time t
            """
            return sqrt(kap_c) * (LC0_ + kron(LCP_, RP) + kron(LCM_, RM))

        def a_t(kap_t, phi):
            """
            Return single photon loss basis on target at time t
            """
            return sqrt(kap_t) * (LA0_ + exp(1j * phi) * LA1_)

        leakage_l = []
        proj_data = proj_code_space_sfb(self.truncature)
        for t_idx in range(len(self.times)):
            phi = phi_t[t_idx]
            # build Hamiltonian and dissipators
            RP, RM = rotators(phi)
            H = hamil(RP, RM)
            L_ops = [
                target_dissip_t(RP, RM, self.k2, phi),
                control_dissip_t(RP, RM, self.k2a),
            ]
            # if self.k1a != 0:
            if self.k1a != 0:
                # L_ops += [a_c(self.k1a, RP, RM)]
                L_ops += [a_c(self.k1a, RP, RM)]
            if self.k1 != 0:
                L_ops += [a_t(self.k1, phi)]
            Ld_ops = [L_op.T.conj() for L_op in L_ops]
            # build first Kraus map
            M0 = ID - 1j * self.dt * H
            M0_dag = ID + 1j * self.dt * H

            for L_op, Ld_op in zip(L_ops, Ld_ops):
                LdL = Ld_op.dot(L_op)
                M0 += -0.5 * self.dt * LdL
                M0_dag += -0.5 * self.dt * LdL
            # iterate
            rho_tmp = M0.dot(rho).dot(M0_dag)
            for L_op, Ld_op in zip(L_ops, Ld_ops):
                rho_tmp += self.dt * L_op.dot(rho).dot(Ld_op)
            rho = rho_tmp / np.real(rho_tmp.trace())
            leakage = 1 - np.real(
                np.trace(rho * tensor(qeye([2, self.N_ancilla]), proj_data))
            )
            leakage_l.append(leakage)

        self.leakage_l = leakage_l
        # Store final state and results
        self.rho = Qobj(
            rho,
            dims=[
                [2, self.basis.ancilla.d, 2, self.basis.data.d],
                [2, self.basis.ancilla.d, 2, self.basis.data.d],
            ],
        )


class CNOTSFBPhaseFlips(CNOT):
    """Simulation of the appproximated CNOT gate in the shifted fock basis."""

    def __init__(
        self,
        nbar: int,
        k1: float,
        k2: float,
        k1a: float,
        k2a: float,
        gate_time: float,
        truncature: int,
        initial_state: Optional[Qobj] = None,
        initial_state_name: str = 'unset',
        H: Optional[List[float]] = None,
        D: Optional[List[float]] = None,
        rho: Optional[Qobj] = None,
        num_tslots_pertime: int = 1000,
        N_ancilla: Optional[int] = None,
    ):
        if N_ancilla is None:
            N_ancilla = truncature
        self.N_ancilla = N_ancilla
        if initial_state is None:
            control = SFBNonOrthonormal(
                nbar=nbar, d=truncature, d_ancilla=self.N_ancilla
            ).ancilla.evencat
            target = SFBNonOrthonormal(
                nbar=nbar, d=truncature, d_ancilla=self.N_ancilla
            ).data.evencat
            initial_state = tensor(control, target)
            initial_state_name = '++'
        super().__init__(
            nbar,
            k1,
            k2,
            k1a,
            k2a,
            gate_time,
            truncature,
            initial_state,
            initial_state_name,
            H,
            D,
            rho,
            num_tslots_pertime,
        )
        self.N_ancilla = N_ancilla

    def _pre_simulate(self):
        super()._pre_simulate()
        self.b_a = None
        self.b_d = None
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
        if self.alpha is None:
            self.alpha = np.sqrt(self.nbar)

        # Two-photon dissipation
        if self.L2a is None:
            self.L2a = np.sqrt(self.k2a) * tensor(
                qeye(2),
                self.b_a**2 + 2 * self.alpha * self.b_a,
                qeye(2),
                qeye(self.truncature),
            )
        if self.L2d is None:
            self.L2d = [
                np.sqrt(self.k2)
                * tensor(
                    fock_dm(2, 0),
                    qeye(self.N_ancilla),
                    qeye(2),
                    self.b_d**2 + 2 * self.alpha * self.b_d,
                ),
                [
                    np.sqrt(self.k2)
                    * tensor(
                        fock_dm(2, 1),
                        qeye(self.N_ancilla),
                        qeye(2),
                        self.b_d**2 + 2 * self.alpha * self.b_d,
                    ),
                    np.exp(2j * np.pi * self.times / self.gate_time),
                ],
                [
                    np.sqrt(self.k2)
                    * tensor(
                        sigmaz(), self.b_a, qeye(2), qeye(self.truncature)
                    ),
                    self.alpha
                    / 2
                    * (np.exp(2j * np.pi * self.times / self.gate_time) - 1),
                ],
            ]
        # Single photon loss
        if self.L1a is None:
            self.L1a = np.sqrt(self.k1a) * tensor(
                sigmaz(), self.b_a + self.alpha, qeye(2), qeye(self.truncature)
            )
        if self.L1d is None:
            self.L1d = [
                np.sqrt(self.k1)
                * tensor(
                    fock_dm(2, 0),
                    qeye(self.N_ancilla),
                    sigmaz(),
                    self.b_d + self.alpha,
                ),
                [
                    np.sqrt(self.k1)
                    * tensor(
                        fock_dm(2, 1),
                        qeye(self.N_ancilla),
                        sigmaz(),
                        self.b_d + self.alpha,
                    ),
                    np.exp(1j * np.pi * self.times / self.gate_time),
                ],
            ]
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
                )
            )
        hadamard = snot(1)
        if self.H2 is None:
            self.H2 = tensor(
                hadamard, qeye(self.N_ancilla), hadamard, qeye(self.truncature)
            )
        if self.U is None:
            self.U = tensor(
                fock_dm(2, 0),
                qeye(self.N_ancilla),
                qeye(2),
                qeye(self.truncature),
            ) + np.exp(-1j * np.pi * self.nbar) * tensor(
                fock_dm(2, 1),
                qeye(self.N_ancilla),
                sigmax(),
                qeye(self.truncature),
            )

    def _simulate(self):
        print('start simulate')
        if self.N_ancilla is None:
            self.N_ancilla = self.truncature
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
                    qeye(2), qeye(self.N_ancilla), qeye(2), num(self.truncature)
                ),
                tensor(
                    qeye(2), num(self.N_ancilla), qeye(2), qeye(self.truncature)
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

    def transition_matrices(self):
        print('>>> CNOT transition matrix')
        t_start = time()
        plus = (basis_state(2, 0) + basis_state(2, 1)).unit()
        minus = (basis_state(2, 0) - basis_state(2, 1)).unit()
        initial_states = list(
            product(
                [plus, minus],
                [fock(self.N_ancilla, i) for i in range(self.N_ancilla)],
                [plus, minus],
                [fock(self.truncature, i) for i in range(self.truncature)],
            )
        )
        self.transition_matrix = np.zeros(
            (
                2 * self.N_ancilla * 2 * self.truncature,
                2 * self.N_ancilla * 2 * self.truncature,
            ),
            dtype=float,
        )
        for j, _ in enumerate(initial_states):
            qa, ga, qd, gd = initial_states[j]
            psi0 = tensor(qa, ga, qd, gd)
            self.initial_state = psi0
            self.simulate()
            self.rho = self.H2 * self.rho * self.H2
            self.transition_matrix[:, j] = self.rho.diag()

        # Remove <0 numerical errors
        self.transition_matrix[self.transition_matrix < 0] = 0
        self.verb('CNOT transition_matrices')
        self.verb(self.transition_matrix)
        print(f'Elapsed time CNOT t matrix: {time() - t_start:.2f}s')



class CNOTSFBReducedModel(CNOT):
    def _pre_simulate(self):
        super()._pre_simulate()
        self.get_g1()
        self.get_g2()
        self.get_nth()

    def _simulate(self):
        if self.initial_state is None:
            self.initial_state = tensor(
                (basis_state(2, 0) + basis_state(2, 1)) / np.sqrt(2),
                basis_state(self.truncature, 0),
            )

        if isket(self.initial_state):
            self.initial_state = ket2dm(self.initial_state)

        if self.H is None:
            H = [[tensor(qeye(2), qeye(self.truncature)), lambda x, args: 0]]
        else:
            H = list(self.H)
        c_ops = [
            np.sqrt(0.5 * self.k2)
            * tensor(
                sigmaz(),
                destroy(self.truncature) * destroy(self.truncature)
                + 2 * np.sqrt(self.nbar) * destroy(self.truncature),
            ),
            np.sqrt(0.5 * self.k2)
            * tensor(
                qeye(2),
                destroy(self.truncature) * destroy(self.truncature)
                + 2 * np.sqrt(self.nbar) * destroy(self.truncature),
            ),
            np.sqrt(
                np.pi**2
                / 16
                / self.nbar**2
                / self.k2a
                / self.gate_time**2
            )
            * tensor(
                sigmaz(),
                num(self.truncature)
                + self.alpha
                * (destroy(self.truncature) + create(self.truncature)),
            ),
        ]

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
                ),
                tensor(qeye(2), num(self.truncature)),
            ],
            args=None,
            options=Options(method='bdf', store_final_state=True),
            progress_bar=True if self.verbose else None,
            _safe_mode=True,
        )
        self.rho = results.final_state
        self.expect = results.expect
        return results

    def get_g1(self) -> None:
        self.g1 = 2 * self.nbar * self.k2

    def get_g2(self) -> None:
        self.g2 = np.pi**2 / (
            16 * self.nbar * self.k2a * (self.gate_time) ** 2
        )

    def get_nth(self) -> None:
        self.nth = self.g2 / (2 * self.g1 - self.g2)

    def get_pZI(self, t: float) -> float:
        return get_pzi(t=t, g1=self.g1, g2=self.g2)

    def num_gauge(self, t: float) -> float:
        self.get_nth()
        return self.nth * (1 - np.exp(-(2 * self.g1 - self.g2) * t))

    def get_R1(self) -> None:
        self.R1 = np.zeros((self.truncature, self.truncature))
        for n in range(self.truncature):
            self.R1[n, n] = (
                -4 * self.g1 * n - self.g2 * (n + 1)
                if n < self.truncature - 1
                else -4 * self.g1 * n
            )
            if n < self.truncature - 1:
                self.R1[n, n + 1] = (
                    4 * self.g1 * (n + 1)
                )  # n: index of the line
                self.R1[n + 1, n] = self.g2 * (n + 1)  # n+1: index of the line

    def get_R2(self) -> None:
        self.R2 = np.zeros((self.truncature, self.truncature))
        for n in range(self.truncature):
            self.R2[n, n] = (
                -2 * self.g1 * n - self.g2 * (n + 1)
                if n < self.truncature - 1
                else -2 * self.g1 * n
            )
            if n < self.truncature - 1:
                self.R2[n + 1, n] = -self.g2 * (n + 1)  # n+1: index of the line

    def thermal_c_p(self):
        self.get_nth()
        c_p_init = []
        for n in range(self.truncature):
            c_p_init.append(self.nth**n / (1 + self.nth) ** (n + 1))

        c_p_init = c_p_init / np.sum(c_p_init)
        return c_p_init

    def c_p_m(self, t: float, c_p_init=None, c_m_init=None):
        if c_m_init is None:
            c_m_init = np.zeros((self.truncature, 1))
        if c_p_init is None:
            c_p_init = np.zeros((self.truncature, 1))
            c_p_init[0] = 1
        self.get_R1()
        self.get_R2()

        er1, er2 = expm(self.R1 * t), expm(self.R2 * t)
        # sum s_t = 1
        s_t = er1.dot(c_p_init + c_m_init)
        d_t = er2.dot(c_p_init - c_m_init)
        c_p_t = 0.5 * (s_t + d_t)
        c_m_t = 0.5 * (s_t - d_t)
        return c_p_t, c_m_t

    def get_pZ1_resolution(self, t: float, c_p_init=None, c_m_init=None):
        _, c_m_t = self.c_p_m(t=t, c_p_init=c_p_init, c_m_init=c_m_init)
        return np.sum(c_m_t)

    def get_pZ1_thermal(self):
        basis = Fock(nbar=self.nbar, d=self.truncature)
        self.get_nth()
        self.initial_state = tensor(
            ket2dm((basis_state(2, 0) + basis_state(2, 1)) / np.sqrt(2)),
            basis.thermal_state(self.nth),
        )
        self.simulate()
        target_state = tensor(
            ket2dm((basis_state(2, 0) - basis_state(2, 1)) / np.sqrt(2)),
            qeye(self.truncature),
        )

        return np.real(np.trace(target_state * self.rho))

    def transition_matrices(self):
        print('>>> CNOT transition matrix')
        t_start = time()
        plus = (basis_state(2, 0) + basis_state(2, 1)).unit()
        minus = (basis_state(2, 0) - basis_state(2, 1)).unit()
        initial_states = list(
            product(
                [plus, minus],
                [fock(self.truncature, i) for i in range(self.truncature)],
            )
        )
        self.transition_matrix = np.zeros(
            (
                2 * self.truncature,
                2 * self.truncature,
            ),
            dtype=float,
        )
        h = tensor(snot(1), qeye(self.truncature))
        for j, (qa, gd) in enumerate(initial_states):
            psi0 = tensor(qa, gd)
            self.initial_state = psi0
            self.simulate()
            self.rho = h * self.rho * h
            self.transition_matrix[:, j] = self.rho.diag()

        # Remove <0 numerical errors
        self.transition_matrix[self.transition_matrix < 0] = 0
        self.verb('CNOT transition_matrices')
        self.verb(self.transition_matrix)

        print(f'Elapsed time CNOT t matrix: {time() - t_start:.2f}s')


def get_pzi(t: float, g1: float, g2: float) -> float:
    nth = g2 / (2 * g1 - g2)
    return exponentiel_proba(
        g2 * t / 2
        + (1 - np.exp(-(2 * g1 - g2) * t)) * (nth / 2 * (1 - t * g2))  # +0
        # (1 - np.exp(-(2*self.g1-self.g2)*t)) * (self.nth *(1-t*self.g2)) # +TS
    )


class CNOTSFBReducedReducedModel(CNOTSFBReducedModel):
    def _simulate(self):
        if self.initial_state is None:
            self.initial_state = tensor(
                (basis_state(2, 0) + basis_state(2, 1)) / np.sqrt(2),
                basis_state(self.truncature, 0),
            )

        if isket(self.initial_state):
            self.initial_state = ket2dm(self.initial_state)

        if self.H is None:
            H = [[tensor(qeye(2), qeye(self.truncature)), lambda x, args: 0]]
        else:
            H = list(self.H)
        c_ops = [
            np.sqrt(0.5 * self.k2)
            * tensor(
                sigmaz(),
                2 * np.sqrt(self.nbar) * destroy(self.truncature),
            ),
            np.sqrt(0.5 * self.k2)
            * tensor(
                qeye(2),
                2 * np.sqrt(self.nbar) * destroy(self.truncature),
            ),
            np.sqrt(
                np.pi**2
                / 16
                / self.nbar**2
                / self.k2a
                / self.gate_time**2
            )
            * tensor(
                sigmaz(),
                self.alpha * create(self.truncature),
            ),
        ]

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
                ),
                tensor(qeye(2), num(self.truncature)),
            ],
            args=None,
            options=Options(method='bdf', store_final_state=True),
            progress_bar=True if self.verbose else None,
            _safe_mode=True,
        )
        self.rho = results.final_state
        self.expect = results.expect
        return results

class CNOTFockQutip(CNOT):
    def _simulate(self):
        super()._simulate()
        rho0 = self.initial_state
        alpha = np.sqrt(self.nbar)

        a1 = tensor(destroy(self.N_ancilla), qeye(self.truncature))
        a2 = tensor(qeye(self.N_ancilla), destroy(self.truncature))

        L1c = np.sqrt(self.k1a) * a1
        L1t = np.sqrt(self.k1) * a2

        # Lphic = np.sqrt(kphi) * a1.dag() * a1
        # Lphit = np.sqrt(kphi) * a2.dag() * a2

        L2c = [np.sqrt(self.k2a) * (a1**2 - self.nbar)]

        L2t = [
            np.sqrt(self.k2) * (a2**2 - 1 / 2 * alpha * (a1 + alpha)),
            [
                np.sqrt(self.k2) * 1 / 2 * alpha * (a1 - alpha),
                np.exp(2j * np.pi * self.times / self.gate_time),
            ],
        ]

        # L2t_perf = [
        #    np.sqrt(k2) * (a2**2 - nbar*tensor(
        #     coherent_dm(N, alpha), qeye(N))
        # ),
        #    [np.sqrt(k2) * nbar * tensor(
        # coherent_dm(N, -alpha), qeye(N)), - np.exp(2j*np.pi*times/T
        #                                            )]
        # ]

        H_CX = (
            np.pi
            / (4 * alpha * self.gate_time)
            * (a1 + a1.dag() - 2 * alpha)
            * (a2.dag() * a2 - self.nbar)
        )
        # H_perf = -np.pi/T * tensor(coherent_dm(N, -alpha), num(N)-nbar)

        if rho0.isket:
            rho0 = ket2dm(rho0)

        rhof = mesolve(
            H_CX,
            rho0,
            self.times,
            c_ops=[L2c, L2t, L1c, L1t],
            # e_ops=[0 * tensor(qeye(self.N_ancilla), qeye(self.truncature))],
            # options=Options(store_final_state=True, rtol=1e-10, atol=1e-12),
            options=Options(rtol=1e-10, atol=1e-12),
            progress_bar=True if self.verbose else None,
            # ).final_state
        )
        self.rho = rhof
        print('end simulate')
