from abc import abstractmethod
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from qutip import Qobj, basis, fidelity, isket, ket2dm, qeye, tensor

from experiments.experiment import ExperimentPhysicalParameters
from qsim.basis.sfb import SFB
from qsim.physical_gate.cnot import CNOTSFB
from qsim.physical_gate.idle import IdleGateSFB


class RepeatedCNOT(ExperimentPhysicalParameters):
    def __init__(
        self,
        nbar: int,
        k2a: int,
        k2: float,
        k1: float,
        k1a: float,
        N: int,
        target: Qobj,
        initial_control: Qobj,
        initial_state_name_suffixe: str = 'Id0',
        gate_time: Optional[float] = None,
        nbar_l: Optional[list] = None,
        k2a_l: Optional[list] = None,
    ):
        if gate_time is None:
            gate_time = 1 / k2a
        super().__init__(nbar, k2a, k2, k1, k1a, gate_time, N)
        if isket(target):
            target = ket2dm(target)
        if isket(initial_control):
            initial_control = ket2dm(initial_control)
        self.initial_control = initial_control
        self.initial_state_name_suffixe = initial_state_name_suffixe
        self.target = target
        self.cnot = None
        self.idle = None
        self.basis = None
        self.populate()
        if nbar_l is None:
            self.nbar_l = range(2, 10)
        if k2a_l is None:
            self.k2a_l = range(1, 21)

    @abstractmethod
    def populate(self):
        pass

    def single_simulation(self):
        for index in range(self.k2a):
            print(f'>>>{index}')
            initial_state = tensor(self.initial_control, self.target)

            self.cnot.simulate(
                truncature=self.N,
                initial_state=initial_state,
                initial_state_name=(
                    f'k2aCNOTs{self.initial_state_name_suffixe}_{index}'
                ),
            )

            target = self.cnot.rho.ptrace((2, 3))
            print('target trace after ptrace : ')
            print(np.trace(target))
            self.idle.simulate(
                truncature=self.N,
                initial_state=target,
                initial_state_name=f'targetk2a_{self.k2a}_CNOTs{self.initial_state_name_suffixe}_{index}',
            )
            target = self.idle.rho

    def get_data(self):
        res_leakage_control = {}
        res_leakage_target = {}
        res_bitflip_target = {}
        res_phaseflip_control = {}
        res_fidelity_target = {}
        for nbar in self.nbar_l:
            res_leakage_control[nbar] = []
            res_leakage_target[nbar] = []
            res_bitflip_target[nbar] = []
            res_phaseflip_control[nbar] = []
            res_fidelity_target[nbar] = []

        for nbar in self.nbar_l:
            self.basis.nbar = nbar
            old_target = self.basis.evencat
            for k2a in self.k2a_l:
                gate_time = 1 / k2a

                # update
                self.idle.nbar = nbar
                self.idle.k2a = k2a
                self.idle.gate_time = gate_time
                self.idle.basis = basis
                self.idle.num_tslots_pertime = 1000
                self.cnot.nbar = nbar
                self.cnot.k2a = k2a
                self.cnot.gate_time = gate_time
                self.cnot.basis = basis
                self.cnot.truncature = N
                self.cnot.num_tslots_pertime = 1000

                partial_res_leakage_control = []
                partial_res_leakage_target = []
                partial_res_bitflip_target = []
                partial_res_phaseflip_control = []
                partial_res_fidelity_target = []
                for index in range(k2a):
                    idle.initial_state_name = f'targetk2a_{k2a}_CNOTs{initial_state_name_suffixe}_{index}'
                    target = idle._get_data()
                    cnot.initial_state_name = (
                        f'k2aCNOTs{initial_state_name_suffixe}_{index}'
                    )

                    control = cnot._get_data().ptrace((0, 1))
                    try:
                        print(target.dims)
                        partial_res_leakage_control.append(
                            basis.leakage(control)
                        )
                        partial_res_leakage_target.append(basis.leakage(target))
                        partial_res_bitflip_target.append(
                            np.trace(
                                target * tensor(ket2dm(basis(2, 1)), qeye(N))
                            )
                        )
                        partial_res_phaseflip_control.append(
                            np.trace(
                                control
                                * tensor(
                                    ket2dm(basis(2, 0) - basis(2, 1)) / 2,
                                    qeye(N),
                                )
                            )
                        )
                        partial_res_fidelity_target.append(
                            fidelity(old_target, target)
                        )

                    except:
                        print((nbar, k2a))
                        partial_res_leakage_control.append(0)
                        partial_res_leakage_target.append(0)
                        partial_res_bitflip_target.append(0)
                        partial_res_phaseflip_control.append(0)
                        partial_res_fidelity_target.append(0)
                    old_target = target
                res_leakage_control[nbar].append(partial_res_leakage_control)
                res_leakage_target[nbar].append(partial_res_leakage_target)
                res_bitflip_target[nbar].append(partial_res_bitflip_target)
                res_phaseflip_control[nbar].append(
                    partial_res_phaseflip_control
                )
                res_fidelity_target[nbar].append(partial_res_fidelity_target)

        self.res_leakage_control = res_leakage_control
        self.res_leakage_target = res_leakage_target
        self.res_bitflip_target = res_bitflip_target
        self.res_phaseflip_control = res_phaseflip_control
        self.res_fidelity_target = res_fidelity_target

    def plot(
        self,
        data: Dict[int, list],
        fid: bool = False,
        color_grad: str = 'linear',
    ):
        self.get_data()
        return plot_results(
            res=data,
            nbar_l=self.nbar,
            fid=fid,
            color_grad=color_grad,
        )


class RepeatedCNOTSFB(RepeatedCNOT):
    def populate(self):
        self.basis = SFB(nbar=self.nbar, d=self.N)
        self.cnot = CNOTSFB(
            nbar=self.nbar,
            k1=self.k1,
            k2=self.k2,
            k2a=self.k2a,
            gate_time=self.gate_time,
            k1a=self.k1a,
        )
        self.cnot.basis = self.basis
        self.idle = IdleGateSFB(
            nbar=self.nbar,
            k1=self.k1,
            k2=self.k2,
            gate_time=self.gate_time,
        )


def plot_results(
    res: dict,
    nbar_l=range(2, 10),
    fid: bool = False,
    color_grad: str = 'linear',
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(12, 8))

    abs = range(1, 21)
    for k2a in abs:
        for i in range(k2a):
            # if i == k2a-1:
            if color_grad == 'linear':
                alpha = (i + 1) / k2a
            elif color_grad == 'square':
                alpha = ((i + 1) / k2a) ** 2
            elif color_grad == '1':
                alpha = 1
            else:
                raise ValueError('Invalid color_grad')
            for nbar in nbar_l:
                if nbar % 2 == 0:
                    ax.scatter(
                        [k2a],
                        [1 - res[nbar][k2a - 1][i]]
                        if fid
                        else [res[nbar][k2a - 1][i]],
                        marker='o',
                        alpha=alpha,
                        label=f'$\overline{{n}} = {nbar}$'
                        if i == k2a - 1 and k2a == 1
                        else '',
                        s=3,
                    )
            ax.set_prop_cycle(None)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_xlabel(f'$\kappa_{{2, a}}/\kappa_{{2, d}}$')
    ax.set_ylabel(f'Target Bitflip')
    plt.tight_layout()
    return ax


if __name__ == '__main__':
    N = 10
    k2a_l = range(1, 21)
    k2 = 1
    k1 = 0
    k1a = 0
    nbar = 4
    k2a = 1
    gate_time = 1 / k2a
    basis = SFB(nbar=nbar, d=N)

    repeated_cnot = RepeatedCNOTSFB(
        nbar=nbar,
        k1=k1,
        k2=k2,
        gate_time=gate_time,
        k2a=k2a,
        k1a=k1a,
        N=N,
        target=basis.evencat,
        initial_control=basis.evencat,
    )

    repeated_cnot.single_simulation()
