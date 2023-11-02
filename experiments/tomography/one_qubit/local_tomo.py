import matplotlib.pyplot as plt
# tomography
from qsim.utils.tomography import *
from abc import ABC

from qsim.basis.fock import Fock
from qsim.basis.sfb import SFB
from qsim.basis.basis import Basis
from qsim.helpers import simulate, search_simulation, _default_db_path

from qsim.physical_gate.z_gate import ZGateFock, ZGateSFB, ZGate
from qsim.physical_gate.idle import IdleGateFock
import numpy as np
import qutip as qt

from pprint import pprint


def z_gate_time(alpha, eps_Z, *args, **kwargs):
    return np.pi / (eps_Z * 4 * alpha)


def z_gate_pz_theo(nbar, k1, k2, gate_time, eps_Z, *args, **kwargs):
    return nbar * k1 * gate_time + eps_Z ** 2 * gate_time / (k2 * nbar)


class ZErrors(ABC):
    basis_class = Basis
    gate_class = ZGate
    chi = np.empty((4, 4))
    error_model = {}
    U = np.array([[1, 0], [0, -1]])

    def __init__(self, N: int, nbar: float = 4, k1: float = 1e-3, k2: float = 1,
                 eps_Z: float = 5e-2, kphi=0, *ars, **kwargs):
        self.nbar = nbar
        self.alpha = np.sqrt(nbar)
        self.basis = self.basis_class(nbar=nbar, d=N)
        self.N = N
        self.k1 = k1
        self.k2 = k2
        self.eps_Z = eps_Z
        self.kphi=kphi
        self.dims = [self.N]
        self.gate_time = z_gate_time(self.alpha, self.eps_Z)
        path = str(_default_db_path(self.gate_class))
        paths = path.split('.')
        self.final_path = paths[
                              0] + f'_{self.nbar}_{type(self.basis).__name__}_{self.N}' + \
                          paths[1]

    def gen_physical_parameters(self):
        return {
            'k1': self.k1,
            'k2': self.k2,
            'kphi': self.kphi,
            'num_tslots_pertime': 1e3,
            'eps_Z': self.eps_Z,
            'gate_time': self.gate_time,
            'nbar': self.nbar,
        }

    def compute_error_model(self):
        physical_params = self.gen_physical_parameters()
        pprint(physical_params)

        cardinal_states_one_qubit = self.basis.tomography_one_qubit()

        # one_qubit_params_fock = gen_params(cardinal_states_one_qubit_fock, N=N_fock)
        Z_params = gen_params(cardinal_states_one_qubit, N=self.N,
                              physical_params=physical_params)

        process = simulate(gate_class=self.gate_class, params=Z_params,
                           db_path=self.final_path,
                           overwrite_logical=True,
                           n_proc=4)
        process.join()

        final_states = {k: None for k in cardinal_states_one_qubit.keys()}
        for initial_state_name, initial_state in cardinal_states_one_qubit.items():
            res = search_simulation(
                self.final_path,
                **{**physical_params,
                   **{'initial_state_name': initial_state_name}}
            )
            rho = np.array(res[0]['state'], dtype=complex)
            rho = rho[0] + 1.0j * rho[1]
            rho = qt.Qobj(rho, dims=[self.dims, self.dims])
            final_states[initial_state_name] = rho

        qubits = [
            self.U @ self.basis.to_code_space(final_states[k]).full() @ self.U
            # self.basis.to_code_space(cardinal_states_one_qubit[k]).full()
            for k
            in ['k0', 'k1', 'P', 'M']]

        # eta = 1e-2
        # X = np.array([[0, 1], [1, 0]])
        # Y = np.array([[0, -1j], [1j, 0]])
        # for i in range(4):
        #     # qubits[i] = (1 - eta) * qubits[i] + eta * X @ qubits[i] @ X
        #     qubits[i] = (1 - eta) * qubits[i] + eta * Y @ qubits[i] @ Y
        # print(qubits)
        #
        rhop = build_one_qubit_output(
            *qubits
        )
        self.chi = one_qubit_process_matrix(rhop)
        self.error_model = build_one_qubit_error_model(self.chi)

    def plot_chi(self):
        plotting_one_qubit_process_matrix(self.chi,
                                          title=f'Tomography {type(self.basis).__name__} {self.gate_class.__name__}')

    def z_gate_pz_theo(self):
        return z_gate_pz_theo(nbar=self.nbar, k1=self.k1, k2=self.k2,
                              gate_time=self.gate_time, eps_Z=self.eps_Z)


class IErrorsFock(ZErrors):
    basis_class = Fock
    gate_class = IdleGateFock
    U = np.eye(2)

    def gen_physical_parameters(self):
        return {
            'k1': self.k1,
            'k2': self.k2,
            'kphi': self.kphi,
            'num_tslots_pertime': 1e3,
            'gate_time': self.gate_time,
            'nbar': self.nbar,
        }


class ZErrorsFock(ZErrors):
    basis_class = Fock
    gate_class = ZGateFock


class ZErrorsSFB(ZErrors):
    basis_class = SFB
    gate_class = ZGateSFB

    def __init__(self, N: int, nbar: float = 4, k1: float = 1e-3, k2: float = 1,
                 eps_Z: float = 5e-2, *ars, **kwargs):
        super().__init__(N, nbar, k1, k2, eps_Z, *ars, **kwargs)
        self.dims = [2, self.N]


def evolution_SFB_errors_XY():
    N_l = range(5, 19)
    pX_l = []
    pY_l = []
    # for N in N_l:
    #     print('N=', N, end='\r')
    #     basis_errors = ZErrorsSFB(N=N)
    #     basis_errors.compute_error_model()
    #     pX_l.append(basis_errors.chi[1][1])
    #     pY_l.append(basis_errors.chi[2][2])
    #
    # np.save('real_X_SFB.npy', np.real(pX_l))
    # np.save('real_Y_SFB.npy', np.real(pY_l))
    # np.save('imag_X_SFB.npy', np.imag(pX_l))
    # np.save('image_Y_SFB.npy', np.imag(pY_l))
    #
    pX_l_real = np.load('real_X_SFB.npy')
    pY_l_real = np.load('real_Y_SFB.npy')
    pX_l_imag = np.load('imag_X_SFB.npy')
    pY_l_imag = np.load('image_Y_SFB.npy')

    fig, axs = plt.subplots(1, 2)
    for i, p_l in enumerate([pX_l_real, pY_l_real]):
        axs[0].plot(
            N_l,
            # np.abs(p_l),
            p_l,
            'o',
            label='X' if i == 0 else 'Y',
        )
    # axs[0].set_yscale('log')
    for i, p_l in enumerate([pX_l_imag, pY_l_imag]):
        axs[1].plot(
            N_l,
            p_l,
            'o',
            label='X' if i == 0 else 'Y',
        )
    axs[0].set_title('Real Part')
    axs[1].set_title('Imag Part')
    axs[0].set_ylabel('Value')
    axs[0].set_xlabel('N SFB')
    axs[1].set_xlabel('N SFB')
    plt.show()


if __name__ == '__main__':
    # evolution_SFB_errors_XY()

    basis_errors = IErrorsFock(N=60, k1=1e-2, kphi=0)
    basis_errors.compute_error_model()
    chi = basis_errors.chi
    em = basis_errors.error_model

    basis_errors.plot_chi()
    print('diagonal of chi')
    print(np.diag(chi))
    chi[chi < 0] = 0
    print('Simulated error model')
    pprint(em)
    print(
        'Predicted error model:'
        f' pZ={basis_errors.z_gate_pz_theo()}'
    )
