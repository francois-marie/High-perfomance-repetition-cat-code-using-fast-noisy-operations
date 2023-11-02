import sys
from abc import abstractmethod
from itertools import product
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from qutip import Qobj, basis, ket2dm, qeye, tensor

from experiments.experiment import Experiment
from qsim import qsim_repo_path
from qsim.basis.basis import Basis
from qsim.basis.sfb import SFB
from qsim.physical_gate.cnot import CNOT, CNOTSFB
from qsim.physical_gate.idle import IdleGate, IdleGateSFB


class LeakageCNOT(Experiment):
    title = ''

    def __init__(
        self,
        nbar_list: List[int],
        gate_time_list: List[float],
        k2d=1,
        k2a=1,
        k1d=0,
        k1a=0,
        N=10,
        verbose: bool = False,
        idle: Optional[IdleGate] = None,
        cnot: Optional[CNOT] = None,
        basis: Optional[Basis] = None,
        target_state: Optional[Qobj] = None,
    ):
        super().__init__(verbose)
        if cnot is None:
            cnot = CNOTSFB(
                nbar=nbar_list[0],
                k1=k1d,
                k2=k2d,
                k1a=k1a,
                k2a=k2a,
                truncature=N,
                initial_state_name='Id0',
                gate_time=1.0,
            )

        if idle is None:
            idle = IdleGateSFB(
                nbar=nbar_list[0],
                k1=k1d,
                k2=k2d,
                truncature=N,
                initial_state_name='Id',
                gate_time=1.0,
            )

        if basis is None:
            basis = SFB(nbar=nbar_list[0], d=N, d_ancilla=N)

        self.nbar_list = nbar_list
        self.gate_time_list = gate_time_list
        self.k1 = k1d
        self.k2 = k2d
        self.k2a = k2a
        self.k1a = k1a
        self.N = N
        self.idle = idle
        self.cnot = cnot
        self.basis = basis
        self.target_state = target_state
        self.circuit_path = (
            f'{qsim_repo_path}/experiments/'
            'leakage_cnot_after_convergence/circuit.png'
        )

    @abstractmethod
    def populate_experiment(self, nbar: int, gate_time: float):
        pass

    @abstractmethod
    def _target_bitflip_state(self) -> None:
        pass

    @abstractmethod
    def get_target_bitflip(self) -> None:
        pass

    @abstractmethod
    def get_target_leakage_pre_idle(self) -> None:
        pass

    @abstractmethod
    def get_target_leakage_post_idle(self) -> None:
        pass

    @abstractmethod
    def get_initial_state_idle(self) -> None:
        pass

    def single_simulation(self, nbar: int, gate_time: float) -> None:
        self.populate_experiment(nbar=nbar, gate_time=gate_time)
        # rho_i = Id x 0
        self.initial_state = tensor(
            ket2dm(self.basis.ancilla.zero) + ket2dm(self.basis.ancilla.one),
            ket2dm(self.basis.data.zero),
        )

        self.cnot.simulate()
        # self.cnot._write_db()

        self.get_target_bitflip()
        self.get_target_leakage_pre_idle()

        initial_state = self.get_initial_state_idle()

        self.idle.simulate()
        # self.idle._write_db()

        self.get_target_leakage_post_idle()

    def get_data(self):
        self.data = {}
        for nbar, gate_time in product(self.nbar_list, self.gate_time_list):
            self.data[(nbar, gate_time)] = {}
            self.populate_experiment(nbar=nbar, gate_time=gate_time)
            self.cnot.initial_state_name = 'Id0'
            self.cnot.truncature = self.N
            self.cnot.num_tslots_pertime = 1000
            self.cnot.basis = self.basis

            self.idle.basis = self.basis

            #             initial_state = self.cnot._get_data().ptrace((2, 3))
            #             self.idle.simulate(
            #                 truncature=self.N,
            #                 initial_state=initial_state,
            #                 initial_state_name=f'LeakedTargetCNOT_{gate_time}',
            #             )
            #
            self.idle.initial_state_name = f'LeakedTargetCNOT_{gate_time}'
            self.idle.truncature = self.N
            self.idle.num_tslots_pertime = 1000
            self.idle.basis = self.basis

            self.get_target_bitflip()
            self.data[(nbar, gate_time)]['pIX'] = self.target_bitflip

            self.get_target_leakage_pre_idle()
            self.data[(nbar, gate_time)][
                'leakage_pre_idle'
            ] = self.target_leakage_pre_idle

            self.get_target_leakage_post_idle()
            self.data[(nbar, gate_time)][
                'leakage_post_idle'
            ] = self.target_leakage_post_idle

        if self.verbose:
            print(self.data)

    def plot(self, axs=None, marker_list=['o']):
        if axs is None:
            _, axs = plt.subplots(1, 3, figsize=(len(self.nbar_list) * 5, 6))

        for i in range(len(self.nbar_list)):
            nbar_target = self.nbar_list[i]
            absc = [
                gate_time
                for (nbar, gate_time) in self.data.keys()
                if self.data[(nbar, gate_time)] != {} and nbar == nbar_target
            ]
            ord = [
                self.data[(nbar, gate_time)]
                for (nbar, gate_time) in self.data.keys()
                if self.data[(nbar, gate_time)] != {} and nbar == nbar_target
            ]
            ordonnee_l = ['pIX', 'leakage_pre_idle', 'leakage_post_idle']
            label_l = [
                '$p_{\mathrm{IX}}$',
                '$p_{l_{\mathrm{1}}}$',
                '$p_{l_{\mathrm{2}}}$',
            ]
            for j, (ordonnee, label) in enumerate(zip(ordonnee_l, label_l)):
                print(f'{(ordonnee, label)=}')
                axs[i].plot(
                    absc,
                    [y[ordonnee] for y in ord],
                    marker_list[j % len(marker_list)],
                    label=label,
                    # ms=2 if marker_list[j % len(marker_list)] == 's' else 3,
                    ms=2 if marker_list[j % len(marker_list)] == 's' else 2.8,
                )
            axs[i].set_yscale('log')
            axs[i].set_xscale('log')
            # axs[i].set_xlabel('$\kappa_2 T$')
            axs[i].set_xlabel('$T$', fontsize=10)
            axs[i].set_title(f'$\overline{{n}} = {nbar_target}$')
            if i == 0:
                axs[i].legend(
                    # bbox_to_anchor=(1.05, 1), loc='upper left'
                )
            axs[0].set_ylabel('Leakage Error')

            axs[i].set_xlim(left=2e-2)
            axs[i].set_ylim((1e-10, 1))

            axs[i].set_xlabel('$T $ $[ \kappa_2^{-1} ]$')
        # plt.suptitle(self.title)
        plt.tight_layout()
        return axs


class LeakageCNOTSFB(LeakageCNOT):
    def _target_bitflip_state(self):
        # target bitflip
        self.target_state = tensor(
            qeye(2), qeye(self.N), ket2dm(basis(2, 1)), qeye(self.N)
        )

    def get_target_bitflip(self):
        self._target_bitflip_state()

        cnot_rho = self.cnot.get_data()
        if isinstance(cnot_rho, Qobj):
            self.target_bitflip = np.real(
                np.trace(self.target_state * cnot_rho)
            )
            if self.verbose:
                print('target bitflip')
                print(self.target_bitflip)
        else:
            print('Cnot attribute has no density matrix')
            self.target_bitflip = None

    def get_target_leakage_pre_idle(self):
        # target leakage pre idle
        cnot_rho = self.cnot.get_data()
        if isinstance(cnot_rho, Qobj):
            target_gauge = cnot_rho.ptrace(3)
            self.target_leakage_pre_idle = np.real(
                np.trace(target_gauge) - target_gauge[0, 0]
            )
            if self.verbose:
                print('target leakage before idle')
                print(self.target_leakage_pre_idle)
        else:
            self.target_leakage_pre_idle = None

    def get_target_leakage_post_idle(self):
        # target leakage post idle
        idle_rho = self.idle.get_data()
        if isinstance(idle_rho, Qobj):
            target_gauge = idle_rho.ptrace(1)
            self.target_leakage_post_idle = np.real(
                np.trace(target_gauge) - target_gauge[0, 0]
            )
            if self.verbose:
                print('target leakage after idle')
                print(self.target_leakage_post_idle)
        else:
            self.target_leakage_post_idle = None

    def get_initial_state_idle(self):
        return tensor(self.cnot.rho.ptrace((2, 3)))


class LeakageCNOTSFB_TCNOT_TIdle(LeakageCNOTSFB):
    title = '$T_{CNOT} = T_{Idle} = T$'

    def populate_experiment(self, nbar: int, gate_time: float):
        self.cnot = CNOTSFB(
            nbar=nbar,
            k1=self.k1,
            k2=self.k2,
            k2a=self.k2a,
            k1a=self.k1a,
            gate_time=gate_time,
        )
        self.idle = IdleGateSFB(
            nbar=nbar, k1=self.k1, k2=self.k2, gate_time=gate_time
        )
        self.basis = SFB(nbar=nbar, d=self.N)


class LeakageCNOTSFB_TCNOT(LeakageCNOTSFB):
    title = '$T_{CNOT} =  T$, $T_{Idle} = 1/\kappa_2$'

    def populate_experiment(self, nbar: int, gate_time: float):
        self.cnot = CNOTSFB(
            nbar=nbar,
            k1=self.k1,
            k2=self.k2,
            k2a=self.k2a,
            k1a=self.k1a,
            gate_time=gate_time,
        )
        self.idle = IdleGateSFB(
            nbar=nbar, k1=self.k1, k2=self.k2, gate_time=1 / self.k2
        )
        self.basis = SFB(nbar=nbar, d=self.N)


if __name__ == "__main__":
    # nbar = int(sys.argv[1])
    # ti = int(sys.argv[2])
    exp = LeakageCNOTSFB_TCNOT(
        nbar_list=[4, 6, 8],
        gate_time_list=list(np.logspace(-2, 1, 30)),
    )
    exp.get_data()
    print(exp.data)

    # run(nbar=nbar, ti=ti)
    # get_data()
