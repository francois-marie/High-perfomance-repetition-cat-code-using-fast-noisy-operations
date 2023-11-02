from itertools import product
from time import time

import numpy as np
from qutip import Options
from qutip import basis as basis_state
from qutip import destroy, fock, isket, ket2dm, mesolve, qeye, sigmaz, tensor
from qutip.qip.operations import snot

from qsim.basis.fock import Fock
from qsim.basis.sfb import SFB, SFBNonOrthonormal
from qsim.physical_gate.pgate import OneQubitPGate, TwoQubitPGate


class IdleGateFake(OneQubitPGate):
    def _simulate(self):
        if isket(self.initial_state):
            self.initial_state = ket2dm(self.initial_state)
        self.rho = self.initial_state

    def transition_matrices(self):
        self.transition_matrix = np.identity(2 * self.truncature, dtype=float)


class IdleGate(OneQubitPGate):
    """Simulation of the Idle gate in the Fock basis."""

    def _simulate(self):
        if self.initial_state is None:
            self.initial_state = SFBNonOrthonormal(
                nbar=self.nbar, d=self.truncature
            ).data.evencat
            self.initial_state_name = '+'
        if isket(self.initial_state):
            self.initial_state = ket2dm(self.initial_state)
        rho = self.initial_state
        if self.H is None:
            H = [[qeye(self.basis.data.hilbert_space_dims), lambda x, args: 0]]
        else:
            H = list(self.H)

        c_ops = [self.basis.two_photon_pumping(self.k2)]
        if self.k1 != 0.0:
            c_ops += [self.basis.one_photon_loss(self.k1)]
        if self.kphi != 0.0:
            c_ops += [self.basis.thermic_photon(self.kphi)]
        D = self.D or c_ops
        # proj_data = proj_code_space_sfb(self.truncature) if type(self.basis).__name__ == 'SFB' else
        results = mesolve(
            H=H,
            rho0=rho,
            tlist=self.times,
            c_ops=D,
            args=None,
            e_ops=[
                self.basis.data.num_op,
                # proj_data
            ],
            options=Options(method='bdf', store_final_state=True),
            progress_bar=True if self.verbose else None,
            _safe_mode=True,
        )
        self.rho = results.final_state
        self.results_mesolve = results


class IdleGateFock(IdleGate):
    """Simulation of the Idle gate in the Fock basis."""

    def _simulate(self):
        self.basis = Fock(nbar=self.nbar, d=self.truncature)
        super()._simulate()


class IdleGateSFB(IdleGate):
    """Simulation of the Idle gate in the Shifted Fock basis."""

    def _simulate(self):
        self.basis = SFB(nbar=self.nbar, d=self.truncature)
        super()._simulate()


class IdleGateSFBPhaseFlips(IdleGate):
    """Simulation of the approximated Idle gate for the phase flips
    in the Shifted Fock basis."""

    def populate_class_parameters(self):
        self.N_ancilla = 2
        if self.b_d is None:
            self.b_d = destroy(self.truncature)
        if self.alpha is None:
            self.alpha = np.sqrt(self.nbar)

        # Two-photon dissipation
        if self.L2d is None:
            self.L2d = np.sqrt(self.k2) * tensor(
                qeye(2), self.b_d**2 + 2 * self.alpha * self.b_d
            )

        # Single photon loss
        if self.L1d is None:
            self.L1d = np.sqrt(self.k1) * tensor(
                sigmaz(), self.b_d + self.alpha
            )
        self.hadamard = snot(1)
        if self.initial_state is None:
            self.initial_state = SFBNonOrthonormal(
                nbar=self.nbar, d=self.truncature
            ).data.evencat
            self.initial_state_name = '+'

    def _simulate(self):
        self.b_d = None
        self.alpha = None
        self.L1d = None
        self.L2d = None
        self.populate_class_parameters()
        rhof = mesolve(
            0 * self.L2d,
            rho0=self.initial_state,
            tlist=self.times,
            c_ops=[self.L2d, self.L1d],
            options=Options(store_final_state=True),
        ).final_state
        self.rho = rhof

    def transition_matrices(self):
        print('>>> Idle transition matrix')
        t_start = time()
        self.transition_matrix = np.zeros(
            (2 * self.truncature, 2 * self.truncature), dtype=float
        )
        plus = (basis_state(2, 0) + basis_state(2, 1)).unit()
        minus = (basis_state(2, 0) - basis_state(2, 1)).unit()
        initial_states = list(
            product(
                [plus, minus],
                [fock(self.truncature, i) for i in range(self.truncature)],
            )
        )
        for j, _ in enumerate(initial_states):
            qd, gd = initial_states[j]
            self.initial_state = tensor(qd, gd)
            self.simulate()
            self.rho = (
                tensor(self.hadamard, qeye(self.truncature))
                * self.rho
                * tensor(self.hadamard, qeye(self.truncature))
            )
            self.transition_matrix[:, j] = self.rho.diag()

        # Remove <0 numerical errors
        self.transition_matrix[self.transition_matrix < 0] = 0
        # make the sum of the distribution equal to one
        self.transition_matrix[0] = np.ones(
            self.transition_matrix[0].shape
        ) - self.transition_matrix[1:].sum(axis=0)

        self.verb('idle transition_matrices')
        self.verb(self.transition_matrix)
        print(f'Elapsed time Idle t matrix: {time() - t_start:.2f}s')
        # self.idle_qobj = Qobj(self.transition_matrix,
        # dims=[[2, self.truncature], [2, self.truncature]])


class IdleGateSFBPhaseFlipsFirstOrder(IdleGateSFBPhaseFlips):
    """Simulation of the approximated Idle gate for the phase flips
    in the Shifted Fock basis."""

    def _simulate(self):
        self.populate_class_parameters()
        if isket(self.initial_state):
            self.initial_state = ket2dm(self.initial_state)
        rhof = first_order(
            dt=self.dt,
            initial_state=self.initial_state,
            times=self.times,
            H=None,
            c_ops=[self.L2d, self.L1d],
        )
        self.rho = rhof


class IdleGateSFBReducedModel(IdleGateSFB):
    def _simulate(self):
        self.b_d = destroy(self.truncature)
        self.L = np.sqrt(self.k2) * (self.b_d**2 + 2 * self.alpha * self.b_d)
        rhof = mesolve(
            0 * self.L,
            rho0=self.initial_state,
            tlist=self.times,
            c_ops=[self.L],
            options=Options(store_final_state=True),
        ).final_state
        self.rho = rhof

    def transition_matrices(self):
        print('>>> Idle transition matrix')
        t_start = time()
        self.transition_matrix = np.zeros(
            (self.truncature, self.truncature), dtype=float
        )
        initial_states = [
            fock(self.truncature, i) for i in range(self.truncature)
        ]
        for j, _ in enumerate(initial_states):
            gd = initial_states[j]
            self.initial_state = gd
            self.simulate()
            self.rho = qeye(self.truncature) * self.rho * qeye(self.truncature)
            self.transition_matrix[:, j] = self.rho.diag()

        # Remove <0 numerical errors
        self.transition_matrix[self.transition_matrix < 0] = 0
        # make the sum of the distribution equal to one
        self.transition_matrix[0] = np.ones(
            self.transition_matrix[0].shape
        ) - self.transition_matrix[1:].sum(axis=0)

        self.verb('idle transition_matrices')
        self.verb(self.transition_matrix)
        print(f'Elapsed time Idle t matrix: {time() - t_start:.2f}s')
        # self.idle_qobj = Qobj(self.transition_matrix,
        # dims=[[2, self.truncature], [2, self.truncature]])


class IdleGateSFBReducedModelNoQutip(IdleGateSFB):
    def _simulate(self):
        if self.initial_state is None:
            self.initial_state = basis_state(self.truncature, 0)
        if isket(self.initial_state):
            self.initial_state = ket2dm(self.initial_state)
        self.b_d = destroy(self.truncature)
        self.L = np.sqrt(self.k2) * (self.b_d**2 + 2 * self.alpha * self.b_d)
        c_ops = [self.L]
        I = qeye(self.truncature)
        # rho(t+dt) = M0 rho(t) M0d + sum{dt*L rho Ld} + O(dt**2)
        # M0_init: time indep part of M0
        # M0_init = 1 - i dt H - 0.5*(Lcd*Lc + Ltad*Lta + L1cd*L1c + L1td*L1t)
        # H=0
        M0_init = I - 0.5 * self.dt * sum(i.dag() * i for i in c_ops)
        # states = []
        self.rho = self.initial_state
        rho = self.rho
        for _ in range(len(self.times)):
            # iterate
            rho_tmp = M0_init * rho * M0_init.dag()

            # sum{dt*L rho Ld} = dt * (Lc rho Lcd + L1c rho L1cd +
            #                          L1t rho L1td +
            #                          (Lta + f(t)Ltb) rho (Ltad + f*(t)Ltbd))
            for l in c_ops:
                rho_tmp += self.dt * l * rho * l.dag()
            rho = rho_tmp / np.trace(rho_tmp)
        self.rho = rho
        return rho


class IdleGateSFBReducedReducedModel(IdleGateSFBReducedModel):
    def _simulate(self):
        self.b_d = destroy(self.truncature)
        self.L = np.sqrt(self.k2) * (2 * self.alpha * self.b_d)
        rhof = mesolve(
            0 * self.L,
            rho0=self.initial_state,
            tlist=self.times,
            c_ops=[self.L],
            options=Options(store_final_state=True),
        ).final_state
        self.rho = rhof

if __name__ == '__main__':
    params = {
        'nbar': 8,
        'k1': 1e-3,
        # 'k1': 1e-3,
        # 'k1a': 1e-3,
        'k2': 1,
        # 'k2': 1,
        'gate_time': 1,
        'truncature': 2,
    }
    idle = IdleGateSFBPhaseFlips(**params)
    print(idle)
    idle.simulate()
