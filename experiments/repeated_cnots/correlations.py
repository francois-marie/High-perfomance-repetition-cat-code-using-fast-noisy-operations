from __future__ import annotations

import pathlib
import sys
from abc import abstractmethod
from functools import reduce
from itertools import product
from multiprocessing import Pool, Queue

# from multiprocessing.dummy import Pool
from operator import mul
from os.path import exists
from typing import Any, Callable, Dict, List, Optional, Type

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from numpy import sqrt
from qutip import Qobj, basis, fock, fock_dm, ket2dm, qeye, tensor

from experiments.experiment import ExperimentPhysicalParameters
from qsim import data_directory_path
from qsim.basis.fock import Fock
from qsim.basis.sfb import SFB
from qsim.physical_gate.cnot import (
    CNOT,
    CNOTSFB,
    CNOTFockFull,
    CNOTFockQutip,
    CNOTSFBPhaseFlips,
    CNOTSFBReducedModel,
    CNOTSFBReducedReducedModel,
)
from qsim.physical_gate.idle import (
    IdleGate,
    IdleGateFake,
    IdleGateFock,
    IdleGateSFB,
    IdleGateSFBPhaseFlips,
    IdleGateSFBPhaseFlipsFirstOrder,
    IdleGateSFBReducedModel,
    IdleGateSFBReducedReducedModel,
)
from qsim.utils.utils import generate_ax_params


class BasePhaseFlipsCorrelation(ExperimentPhysicalParameters):
    cnot_gate_cls: Type[CNOT]
    idle_gate_cls: Type[IdleGate]

    def __init__(
        self,
        nbar: int = 4,
        k2: float = 1,
        k2a: float = 1.0,
        N: int = 10,
        gate_time: float = 1,
        k1: float = 0.0,
        k1a: float = 0.0,
        k2a_l: Optional[List[int]] = None,
        nbar_l: Optional[List[int]] = None,
        N_ancilla: Optional[int] = 0,
        num_tslots_pertime: Optional[int] = None,
        verbose: bool = False,
    ):
        super().__init__(nbar, k2a, k2, k1, k1a, gate_time, N, verbose)
        self.N_ancilla = N_ancilla
        self.alpha = sqrt(nbar)
        self.plus_code_space = None
        self.plus_ancilla_code_space = None
        self.proj_plus = None
        self.proj_minus = None
        self.id_data = tensor(qeye(2), qeye(self.N))
        self.data = {'': {'p': 1, 'rho': self.plus_code_space}}
        self.cnot = None
        self.idle = None
        if k2a_l is None:
            k2a_l = [1, 3, 6]
        self.k2a_l = k2a_l
        if nbar_l is None:
            nbar_l = [i for i in range(4, 11, 2)]
        self.nbar_l = nbar_l
        self.correlations = None
        if num_tslots_pertime is None:
            num_tslots_pertime = 10_000
        self.num_tslots_pertime = num_tslots_pertime
        self.plus_dm_qubit = ket2dm((basis(2, 0) + basis(2, 1)) / np.sqrt(2))
        self.minus_dm_qubit = ket2dm((basis(2, 0) - basis(2, 1)) / np.sqrt(2))
        qsim_path = pathlib.Path().absolute().parent.parent
        self.data_directory_path = qsim_path / 'data'
        self.cnot_name = self.cnot_gate_cls.__name__

    def reinit(self, **kwargs):
        for key, val in kwargs.items():
            try:
                setattr(self, key, val)
            except AttributeError:
                pass

    @abstractmethod
    def populate(self):
        pass

    @abstractmethod
    def _proj(self, rho):
        pass

    def _single_simulation(
        self,
        name: str,
        rho_i: Qobj,
    ):
        pass

    def single_simulation(
        self,
        name: str,
    ):
        print('begin routine')
        rho_i = self.data[name]['rho']
        print(rho_i.dims)
        self.populate()

        print('name: ', name)

        dict_plus, dict_minus = self._single_simulation(name=name, rho_i=rho_i)

        return {
            dict_plus['name']: {'p': dict_plus['p'], 'rho': dict_plus['rho']},
            dict_minus['name']: {
                'p': dict_minus['p'],
                'rho': dict_minus['rho'],
            },
        }

    def get_data(self):
        data_path = (
            f'{data_directory_path}/experiments/repeated_cnots/'
            f'correlations_{type(self).__name__}_{self.nbar}_'
            f'{self.k2a}_{self.N}_{self.N_ancilla}.npy'
        )
        if self.verbose:
            print(data_path)
        self.data = np.load(data_path, allow_pickle=True).item()

    def compute_correlations(self):
        if self.correlations is None:
            self.correlations = []
            for k2a in self.k2a_l:
                self.k2a = k2a
                self.get_data()
                max_length = max([len(i) for i in self.data.keys()])
                self.correlations.append(
                    [
                        correlations(1, k, self.data)
                        for k in range(2, max_length + 1)
                    ]
                )

    def compute_accroissement(self, k2a_min: int, k2a_max: int):
        self.compute_correlations()
        index_min = self.k2a_l.index(k2a_min)
        index_max = self.k2a_l.index(k2a_max)
        return [
            (
                np.log(self.correlations[index_min][i])
                - np.log(self.correlations[index_min][i + 1])
            )
            / (
                np.log(self.correlations[index_max][i])
                - np.log(self.correlations[index_max][i + 1])
            )
            for i in range(len(self.correlations[index_min]) - 1)
        ]

    def get_corr_data(self, k: int = 1):
        self.get_data()
        max_length = max([len(i) for i in self.data.keys()])
        x = range(0, max_length)
        y = [correlations(k, k + d, self.data) for d in x]
        return x, y

    def _plot(self, ax, marker='o:', k: int = 1):
        x, y = self.get_corr_data(k=k)
        if self.verbose:
            print(f'k2a={self.k2a}:')
            print(f'\t{y}')
        ax.plot(
            x[1:],
            # [correlations(k, k + d, self.data) for d in range(1, max_length)],
            y[1:],
            marker,
            label=(
                f'$\\kappa_{{2, a}} / \\kappa_{{2, d}} = {self.k2a}$,'
                f' $(N, N_a)=({self.N} ,{self.N_ancilla})$, '
                f' {self.cnot_name}'
            ),
        )

    def plot(self, k: int = 1, ax=None, marker='o:', legend=True):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(17, 7))
        if self.k2a_l is None:
            self.k2a_l = [1, 3, 6]
        for k2a in self.k2a_l:
            self.k2a = k2a
            try:
                self._plot(ax, k=k, marker=marker)
            except FileNotFoundError:
                pass
        generate_ax_params(
            ax,
            ylabel='Phase flips Correlation',
            xlabel='Index CNOT, $d$',
            xscale='linear',
            yscale='log',
            title=(
                f'$\\overline{{n}}={self.nbar}$ \n $p(Z_k \\cap Z_{{k+d}}) -'
                ' p(Z_k)p(Z_{k+d})$ \n $k=1$'
            ),
            legend=legend,
        )
        plt.tight_layout()
        return ax

    def get_phaseflip_data(self):
        self.get_data()
        max_length = max([len(i) for i in self.data.keys()])
        x = range(1, max_length)
        y = [proba_phase_flip(i, self.data) for i in range(1, max_length)]
        return x, y

    def plot_phase_flip(
        self, ax=None, marker=':o', legend=True, label_bool=True
    ):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(17, 7))

        if self.k2a_l is None:
            self.k2a_l = [1, 3, 6]
        for k2a in self.k2a_l:
            self.k2a = k2a
            try:
                x, y = self.get_phaseflip_data()
                x = [i / self.k2a for i in x]
                ax.plot(
                    x,
                    y,
                    marker,
                    label=(
                        f'$\\kappa_{{2, a}} / \\kappa_{{2, d}} = {self.k2a}$,'
                        f' $(N, N_a)=({self.N} ,{self.N_ancilla})$, '
                        f' {self.cnot_name}'
                    )
                    if label_bool is True
                    else '',
                )
            except FileNotFoundError:
                pass
        # generate_ax_params(
        #     ax,
        #     ylabel='pZ1',
        #     xlabel='Index CNOT',
        #     xscale='linear',
        #     yscale='log',
        #     title=f'$\\overline{{n}}={self.nbar}$',
        #     legend=legend,
        # )
        return ax

    def get_decorr_data(self):
        self.get_data()
        max_length = max([len(i) for i in self.data.keys()])
        print(max_length)
        # x = [i/(k2a+1) for i in range(1, max_length)]
        x = []
        y = []
        for k in range(1, max_length):
            if (
                correlations(1, k + 1, self.data) > 1e-14
                and correlations(1, k, self.data) > 1e-14
            ):
                y.append(
                    (
                        correlations(1, k + 1, self.data)
                        / correlations(1, k, self.data)
                    )
                    ** self.k2a
                )
                x.append(k)
        if self.verbose:
            print(x)
            print(y)
        return x, y

    def plot_decorr(self, ax=None, marker='o:', legend=True):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(17, 7))
        for k2a in self.k2a_l:
            x = []
            y = []
            try:
                self.k2a = k2a
                x, y = self.get_decorr_data()
                ax.plot(
                    x,
                    y,
                    marker,
                    label=(
                        f'$\\kappa_{{2, a}} / \\kappa_{{2, d}} = {self.k2a}$,'
                        f' $(N, N_a)=({self.N} ,{self.N_ancilla})$, '
                        f' {self.cnot_name}'
                    )  # if k2a == 1
                    # else f'$k2a/k2d = {self.k2a}$'
                    ,
                )
            except:
                pass

        xlabel = '$d$'
        generate_ax_params(
            ax,
            ylabel='Decorrelation rate',
            xlabel=xlabel,
            xscale='linear',
            title=f'$\\overline{{n}}={self.nbar}$',
            legend=legend,
        )
        plt.suptitle('(Corr(1, d)/Corr(1, d-1))**k2a')
        plt.tight_layout()
        return ax

    def get_nbar_slope(self, profondeur: int):
        nbar_i = self.nbar
        res = []
        for nbar in self.nbar_l:
            self.reinit(nbar=nbar)
            x, y = self.get_corr_data()
            max_length = max([len(i) for i in self.data.keys()])
            if profondeur > max_length:
                raise ValueError('The profondeur is too deep')
            res.append(y[profondeur])
        slope = 0
        j = 1
        while slope == 0 and j < len(self.nbar_l):
            if res[-j] > 0:
                if self.verbose:
                    print((res[-j], res[0], self.nbar_l[-j], self.nbar_l[0]))
                slope = (np.log(res[-j]) - np.log(res[0])) / (
                    self.nbar_l[-j] - self.nbar_l[0]
                )
                if self.verbose:
                    print('res -j: ', res[-j])
            else:
                j += 1
        if j == len(self.nbar_l):
            print(
                f'Could not compute slope: profondeur={profondeur},'
                f' k2a={self.k2a}'
            )

        self.slope = slope if slope != 0 else None
        if self.verbose:
            print('slope :', self.slope)
        self.reinit(nbar=nbar_i)

    def single_effective_corr(self, profondeur):
        self.get_nbar_slope(profondeur=profondeur)
        x, y = self.get_corr_data()
        max_length = max([len(i) for i in self.data.keys()])
        # print(nbar)
        if profondeur > max_length:
            raise ValueError('Profondeur too deep')
        if self.slope is None:
            return 0
        return y[profondeur] * np.exp(np.abs(self.slope) * self.nbar)

    def get_effective_corr_data(self):
        nbar_i = self.nbar
        max_length = 1e10
        for nbar in self.nbar_l:
            self.reinit(nbar=nbar)
            self.get_data()
            _max_length = max([len(i) for i in self.data.keys()])
            if max_length > _max_length:
                max_length = _max_length
        self.reinit(nbar=nbar_i)
        effective_corr = []
        for k in range(1, max_length):
            effective_corr.append(self.single_effective_corr(profondeur=k))
        return range(1, max_length), effective_corr

    def plot_effective_corr(self, ax=None, marker='o:', legend=True):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(17, 7))
        for k2a in self.k2a_l:
            x = []
            y = []
            self.reinit(k2a=k2a)
            if self.verbose:
                print(k2a)
            x, y = self.get_effective_corr_data()
            ax.plot(
                x,
                y,
                marker,
                label=(
                    f'$\\kappa_{{2, a}} / \\kappa_{{2, d}} = {self.k2a}$,'
                    f' $(N, N_a)=({self.N} ,{self.N_ancilla})$, '
                    f' {self.cnot_name} '
                )  # if k2a == 1
                # else f'$k2a/k2d = {self.k2a}$'
                ,
            )
            if self.verbose:
                print((x, y))

        xlabel = '$d$'
        generate_ax_params(
            ax,
            ylabel='Effective correlation',
            xlabel=xlabel,
            xscale='linear',
            title='$Corr(1, d) e^{{s \\overline{n}}}$ \n',
            legend=legend,
        )
        plt.tight_layout()
        return ax

    def get_nbar_slopes(self):
        nbar_i = self.nbar
        max_length = 1e10
        for nbar in self.nbar_l:
            self.reinit(nbar=nbar)
            self.get_data()
            _max_length = max([len(i) for i in self.data.keys()])
            if max_length > _max_length:
                max_length = _max_length
        self.reinit(nbar=nbar_i)
        res_slopes = []
        for k in range(1, max_length):
            self.get_nbar_slope(profondeur=k)
            res_slopes.append(-self.slope)
        return range(1, max_length), res_slopes

    def plot_slopes_nbar(self, ax=None, marker='o:', legend=True):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(17, 7))
        for k2a in self.k2a_l:
            x = []
            y = []
            self.reinit(k2a=k2a)
            if self.verbose:
                print(k2a)
            x, y = self.get_nbar_slopes()
            if self.verbose:
                print(x, y)
            ax.plot(
                x,
                y,
                marker,
                label=(
                    f'$\\kappa_{{2, a}} / \\kappa_{{2, d}} = {self.k2a}$,'
                    f' $(N, N_a)=({self.N} ,{self.N_ancilla})$, '
                    f' {self.cnot_name} '
                )  # if k2a == 1
                # else f'$k2a/k2d = {self.k2a}$'
                ,
            )
            if self.verbose:
                print((x, y))

        xlabel = '$d$'
        generate_ax_params(
            ax,
            ylabel='Slope ',
            xlabel=xlabel,
            xscale='linear',
            title='Evolution of the slope with the profondeur',
            legend=legend,
        )
        plt.tight_layout()
        return ax

    def get_proba_distribution(self, profondeur: int):
        self.get_data()
        self.proba_distribution = [
            proba_outcome(outcome=string, data=self.data)
            for string in [
                ''.join(item)
                for item in list(product(*[['+', '-']] * profondeur))
            ]
        ]
        self.proba_distribution.sort(reverse=True)
        return range(0, 2**profondeur), self.proba_distribution

    def plot_proba_distribution(self, profondeur: int, ax=None, legend=True):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(17, 7))
        x, y = self.get_proba_distribution(profondeur=profondeur)
        if self.verbose:
            print(x, y)
        ax.plot(
            x,
            y,
            '-',
            label=(
                f'$\\kappa_{{2, a}} / \\kappa_{{2, d}} = {self.k2a}$,'
                f' $(N, N_a)=({self.N} ,{self.N_ancilla})$, '
                f' {self.cnot_name} '
            )  # if k2a == 1
            # else f'$k2a/k2d = {self.k2a}$'
            ,
        )
        if self.verbose:
            print((x, y))

        xlabel = 'Ancilla results'
        generate_ax_params(
            ax,
            ylabel='Ordered Probability Distribution ',
            xlabel=xlabel,
            xscale='linear',
            title=f'prof, nbar, k2a: {profondeur}, {self.nbar}, {self.k2a}',
            legend=legend,
        )
        plt.tight_layout()
        return ax

    def proba_distribution_difference(
        self, profondeur: int, target_exp: BasePhaseFlipsCorrelation
    ):
        self.get_data()
        target_exp.get_data()
        return np.sum(
            [
                np.abs(
                    proba_outcome(outcome=string, data=self.data)
                    - proba_outcome(outcome=string, data=target_exp.data)
                )
                for string in [
                    ''.join(item)
                    for item in list(product(*[['+', '-']] * profondeur))
                ]
            ]
        )

    def plot_proba_distribution_difference(
        self, target_exp: BasePhaseFlipsCorrelation, ax=None, legend=True
    ):
        self.get_data()
        target_exp.get_data()
        max_profondeur = min(
            max([len(i) for i in self.data.keys()]),
            max([len(i) for i in target_exp.data.keys()]),
        )
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(17, 7))
        ax.plot(
            range(1, max_profondeur),
            [
                self.proba_distribution_difference(
                    profondeur=prof, target_exp=target_exp
                )
                for prof in range(1, max_profondeur)
            ],
            ':o',
            label=(
                f'$\\kappa_{{2, a}} / \\kappa_{{2, d}} = {self.k2a}$,'
                f' $(N, N_a)=({self.N} ,{self.N_ancilla})$, \n'
                f' {self.cnot_name} vs {target_exp.cnot_name} '
            ),
        )
        generate_ax_params(
            ax,
            ylabel='L1 norm',
            xlabel='Profondeur',
            xscale='linear',
            title=(
                f'$\\kappa_{{2, a}} / \\kappa_{{2, d}} = {self.k2a}$,'
                f' $\\overline{{n}}={self.nbar}$'
            ),
            legend=legend,
        )
        plt.tight_layout()
        return ax

    def get_ordered_strings(self, profondeur: int):
        strings = [
            ''.join(item) for item in list(product(*[['+', '-']] * profondeur))
        ]
        proba_list_unsorted = [
            proba_outcome(outcome=string, data=self.data)
            for string in [
                ''.join(item)
                for item in list(product(*[['+', '-']] * profondeur))
            ]
        ]
        _, proba_list = self.get_proba_distribution(profondeur=profondeur)

        return [
            strings[proba_list_unsorted.index(proba_list[i])]
            for i in range(len(strings))
        ]

    def get_cov_mat(self, vmin: float = 0.0, profondeur=None):
        self.get_data()
        if profondeur is None:
            try:
                profondeur = self.k2a
            except:
                profondeur = max([len(i) for i in self.data.keys()])

        self.cov_mat = np.array(
            [
                [
                    correlations(index1, index2, data=self.data)
                    for index1 in range(1, profondeur + 1)
                ]
                for index2 in range(1, profondeur + 1)
            ]
        )

        # values = [
        #     proba_outcome(outcome=st, data=self.data)
        #     * string_to_array(string=st)
        #     for st in get_strings_list(profondeur)
        # ]
        # X = np.stack(values, axis=1)
        # self.cov_mat = np.cov(X)
        self.cov_mat[self.cov_mat < vmin] = 0.0

    def get_distance_cov_mat(self, target_exp, method):
        self.get_cov_mat()
        target_exp.get_cov_mat()
        return method(self.cov_mat, target_exp.cov_mat)

    def plot_cov_mat(self, vmin: float = 0.0, legend: bool = True):
        self.get_cov_mat(vmin=vmin)
        fig, ax = plt.subplots(1, 1, figsize=(9, 8))
        im = ax.matshow(self.cov_mat, aspect='auto', norm=LogNorm())
        fig.colorbar(im)
        generate_ax_params(
            ax,
            ylabel='Index CNOT',
            xlabel='Index CNOT',
            xscale='linear',
            yscale='linear',
            title=(
                'Covariance matrix \n'
                f'{self.cnot_name} \n'
                f'$\\kappa_{{2, a}} / \\kappa_{{2, d}} = {self.k2a}$,'
                f' $\\overline{{n}}={self.nbar}$'
            ),
            legend=legend,
        )
        ax.grid()
        plt.tight_layout()
        return ax


def plot_cov_mat(cov_mat, title: str = ''):
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    im = ax.matshow(cov_mat, aspect='auto', norm=LogNorm())
    fig.colorbar(im)
    generate_ax_params(
        ax,
        ylabel='Index CNOT',
        xlabel='Index CNOT',
        xscale='linear',
        yscale='linear',
        title=f'Covariance matrix \n {title}',
    )
    ax.grid()
    plt.tight_layout()
    return ax


def string_outcome_to_bool(string: str = '+'):
    if string == '+':
        return 0
    elif string == '-':
        return 1
    else:
        return 0


def string_to_array(string: str = '+++'):
    return np.array(list(map(string_outcome_to_bool, string)))


def get_strings_list(prof: int):
    return [''.join(item) for item in list(product(*[['+', '-']] * prof))]


class PhaseFlipsCorrelation(BasePhaseFlipsCorrelation):
    """PhaseFlipsCorrelation based on simulations of CNOTs and Idles"""

    def proj(self, rho: Qobj, name: str):
        rho_plus = self.proj_plus * rho * self.proj_plus
        rho_minus = self.proj_minus * rho * self.proj_minus
        p_plus = np.real(np.trace(rho_plus))
        p_minus = np.real(np.trace(rho_minus))
        rho_plus = self._proj(rho_plus)
        rho_minus = self._proj(rho_minus)
        return (
            {
                'name': name + '+',
                'p': p_plus,
                'rho': rho_plus,
            },
            {
                'name': name + '-',
                'p': p_minus,
                'rho': rho_minus,
            },
        )

    def _single_simulation(
        self,
        name: str,
        rho_i: Qobj,
    ):
        self.idle = self.idle_gate_cls(
            nbar=self.nbar,
            k1=self.k1,
            k2=self.k2,
            gate_time=self.gate_time,
            truncature=self.N,
            initial_state=rho_i,
            initial_state_name=f'convergence_{name}',
            num_tslots_pertime=self.num_tslots_pertime,
        )
        print(self.idle)

        self.idle.simulate()
        rho = tensor(self.plus_ancilla_code_space, self.idle.rho)

        try:
            self.cnot = self.cnot_gate_cls(
                nbar=self.nbar,
                k1=self.k1,
                k2=self.k2,
                k2a=self.k2a,
                k1a=self.k1a,
                gate_time=self.gate_time,
                N_ancilla=self.N_ancilla,
                truncature=self.N,
                initial_state=rho,
                initial_state_name=f'{name}',
                num_tslots_pertime=self.num_tslots_pertime,
            )
            print('(N, N_a)')
            print(self.N, self.cnot.N_ancilla)
        except:
            self.cnot = self.cnot_gate_cls(
                nbar=self.nbar,
                k1=self.k1,
                k2=self.k2,
                k2a=self.k2a,
                k1a=self.k1a,
                gate_time=self.gate_time,
                truncature=self.N,
                initial_state=rho,
                initial_state_name=f'{name}',
                num_tslots_pertime=self.num_tslots_pertime,
            )

        print(self.cnot)
        self.cnot.simulate()
        return self.proj(rho=self.cnot.rho, name=name)


def full_simulation(
    k: int,
    nbar: int,
    k2a: int,
    N: int,
    N_ancilla: int,
    exp,
    force_sim=False,
    use_mp: int = 0,
):
    method = exp.__name__
    path = (
        f'{data_directory_path}/experiments/repeated_cnots/'
        f'correlations_{method}_{nbar}_{k2a}_{N}_{N_ancilla}.npy'
    )
    if exists(path) and not force_sim:
        data = np.load(
            path,
            allow_pickle=True,
        ).item()
    else:
        plus_code_space = exp.gen_plus_code_space(nbar=nbar, N=N)

        data = {'': {'p': 1, 'rho': plus_code_space}}
    for d in range(k):
        print('d=', d)
        if len([i for i in data.keys() if len(i) == d + 1]) == 2 ** (d + 1):
            pass
        else:
            if use_mp > 0:
                queue = Queue()
                for key in data.keys():
                    if len(key) == d:
                        queue.put(
                            (
                                key,
                                nbar,
                                k2a,
                                N,
                                N_ancilla,
                                exp,
                                data,
                            )
                        )
                queue.put(None)
                pool = Pool(use_mp)
                intermediate_data = pool.starmap(
                    routine,
                    iter(queue.get, None),
                )  # This will call q.get() until None is returned
            else:
                intermediate_data = []
                for key in data.keys():
                    if len(key) == d:
                        intermediate_data.append(
                            routine(
                                key,
                                nbar,
                                k2a,
                                N,
                                N_ancilla,
                                exp,
                                data,
                            )
                        )

            for single_data in intermediate_data:
                data.update(single_data)
            np.save(
                path,
                data,
            )
            print('final path')
            print(path)
            # pool.close()
            # queue.close()


def routine(
    name: str,
    nbar: int,
    k2a: int,
    N: int,
    N_ancilla: int,
    exp,
    data: Dict[str, Any],
):
    try:
        corr = exp(
            nbar=nbar, gate_time=1 / k2a, k2a=k2a, N=N, N_ancilla=N_ancilla
        )
    except:
        raise ValueError('Invalid method')
    corr.data = data
    return corr.single_simulation(
        name=name,
    )


class PhaseFlipsCorrelationSFB(PhaseFlipsCorrelation):
    cnot_gate_cls = CNOTSFBPhaseFlips
    idle_gate_cls = IdleGateSFBPhaseFlips

    def populate(self):
        self.basis = SFB(nbar=self.nbar, d=self.N)

        self.proj_plus = tensor(
            self.plus_dm_qubit, qeye(self.N_ancilla), self.id_data
        )
        self.proj_minus = tensor(
            self.minus_dm_qubit, qeye(self.N_ancilla), self.id_data
        )
        self.plus_plus_code_space = tensor(
            tensor(self.plus_dm_qubit, fock_dm(self.N_ancilla, 0)),
            ket2dm(self.basis.data.evencat),
        )
        self.plus_code_space = ket2dm(self.basis.data.evencat)
        self.plus_ancilla_code_space = tensor(
            self.plus_dm_qubit, fock_dm(self.N_ancilla, 0)
        )

    def _proj(self, rho: Qobj):
        return rho.ptrace((2, 3)) / np.real(np.trace(rho))

    @staticmethod
    def gen_plus_code_space(nbar: int, N: int):
        return ket2dm(SFB(nbar=nbar, d=N).data.evencat)


class PhaseFlipsCorrelationSFBNoIdle(PhaseFlipsCorrelationSFB):
    cnot_gate_cls = CNOTSFBPhaseFlips
    idle_gate_cls = IdleGateFake


class PhaseFlipsCorrelationSFBO(PhaseFlipsCorrelationSFB):
    cnot_gate_cls = CNOTSFB
    idle_gate_cls = IdleGateSFB


class PhaseFlipsCorrelationSFBONoIdle(PhaseFlipsCorrelationSFB):
    cnot_gate_cls = CNOTSFB
    idle_gate_cls = IdleGateFake


class PhaseFlipsCorrelationReducedModel(PhaseFlipsCorrelation):
    cnot_gate_cls = CNOTSFBReducedModel
    idle_gate_cls = IdleGateSFBReducedModel

    def populate(self):
        self.basis = SFB(nbar=self.nbar, d=self.N)
        self.proj_plus = tensor(self.plus_dm_qubit, qeye(self.N))
        self.proj_minus = tensor(self.minus_dm_qubit, qeye(self.N))
        self.plus_plus_code_space = tensor(
            self.plus_dm_qubit,
            fock_dm(self.N, 0),
        )
        self.plus_code_space = fock_dm(self.N, 0)
        self.plus_ancilla_code_space = self.plus_dm_qubit

    def _proj(self, rho):
        rho = rho.ptrace(1)
        return rho / np.real(np.trace(rho))

    @staticmethod
    def gen_plus_code_space(nbar: int, N: int):
        return fock_dm(N, 0)


class PhaseFlipsCorrelationReducedModelNoIdle(
    PhaseFlipsCorrelationReducedModel
):
    cnot_gate_cls = CNOTSFBReducedModel
    idle_gate_cls = IdleGateFake


class PhaseFlipsCorrelationReducedReducedModelNoIdle(
    PhaseFlipsCorrelationReducedModel
):
    cnot_gate_cls = CNOTSFBReducedReducedModel
    idle_gate_cls = IdleGateFake


class PhaseFlipsCorrelationReducedReducedModel(
    PhaseFlipsCorrelationReducedModel
):
    cnot_gate_cls = CNOTSFBReducedReducedModel
    idle_gate_cls = IdleGateSFBReducedReducedModel


class PhaseFlipsCorrelationFock(PhaseFlipsCorrelation):
    cnot_gate_cls = CNOTFockFull
    idle_gate_cls = IdleGateFock

    def populate(self):
        self.basis = Fock(nbar=self.nbar, d=self.N)

        self.proj_plus = tensor(ket2dm(self.basis.data.evencat), qeye(self.N))
        self.proj_minus = tensor(ket2dm(self.basis.data.oddcat), qeye(self.N))
        self.plus_plus_code_space = ket2dm(
            tensor(self.basis.data.evencat, self.basis.data.evencat)
        )
        self.plus_code_space = ket2dm(self.basis.data.evencat)
        self.plus_ancilla_code_space = self.plus_code_space

    def _proj(self, rho):
        target = rho.ptrace(1)
        return target / np.real(np.trace(target))

    @staticmethod
    def gen_plus_code_space(nbar: int, N: int):
        return ket2dm(Fock(nbar=nbar, d=N).data.evencat)


class PhaseFlipsCorrelationFockNoIdle(PhaseFlipsCorrelationFock):
    cnot_gate_cls = CNOTFockFull
    idle_gate_cls = IdleGateFake


class PhaseFlipsCorrelationFockQutipNoIdle(PhaseFlipsCorrelationFock):
    cnot_gate_cls = CNOTFockQutip
    idle_gate_cls = IdleGateFake


class PhaseFlipsCorrelationTransitionMatrices(BasePhaseFlipsCorrelation):
    cnot_gate_cls = CNOTSFBPhaseFlips
    idle_gate_cls = IdleGateSFBPhaseFlips

    """PhaseFlipsCorrelation based on simulations of CNOTs and Idles"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cnot_name = self.cnot_gate_cls.__name__ + 'TM'

    def populate(self):
        self.basis = SFB(nbar=self.nbar, d=self.N)
        self.plus_ancilla = tensor(basis(2, 0), fock(self.N_ancilla, 0))
        self.idle = self.idle_gate_cls(
            nbar=self.nbar,
            k1=self.k1,
            k2=self.k2,
            gate_time=self.gate_time,
            truncature=self.N,
            # initial_state=rho_i,
            # initial_state_name=f'convergence_{name}',
            num_tslots_pertime=self.num_tslots_pertime,
        )
        print(self.idle)
        path = (
            f'{self.data_directory_path}/experiments/repeated_cnots/'
            f'idle_tm_{self.idle_gate_cls.__name__}_{self.nbar}_{self.k2a}_{self.N}_{self.N_ancilla}.npy'
        )
        if exists(path):
            self.idle.transition_matrix = np.load(path)
        else:
            self.idle.transition_matrices()
            np.save(path, self.idle.transition_matrix)

        try:
            self.cnot = self.cnot_gate_cls(
                nbar=self.nbar,
                k1=self.k1,
                k2=self.k2,
                k2a=self.k2a,
                k1a=self.k1a,
                gate_time=self.gate_time,
                N_ancilla=self.N_ancilla,
                truncature=self.N,
                # initial_state=rho,
                # initial_state_name=f'{name}',
                num_tslots_pertime=self.num_tslots_pertime,
            )
            print('(N, N_a)')
            print(self.N, self.cnot.N_ancilla)
        except:
            self.cnot = self.cnot_gate_cls(
                nbar=self.nbar,
                k1=self.k1,
                k2=self.k2,
                k2a=self.k2a,
                k1a=self.k1a,
                gate_time=self.gate_time,
                truncature=self.N,
                # initial_state=rho,
                # initial_state_name=f'{name}',
                num_tslots_pertime=self.num_tslots_pertime,
            )
        path = (
            f'{self.data_directory_path}/experiments/repeated_cnots/'
            f'cnot_tm_{self.cnot_name}_{self.nbar}_{self.k2a}_{self.N}_{self.N_ancilla}.npy'
        )
        if exists(path):
            self.cnot.transition_matrix = np.load(path)
        else:
            self.cnot.transition_matrices()
            np.save(path, self.cnot.transition_matrix)

    @staticmethod
    def gen_plus_code_space(nbar: int, N: int):
        return tensor(basis(2, 0), fock(N, 0))

    def proj(self, rho: Qobj, name: str):
        p_plus = np.sum(rho[: rho.shape[0] // 2])
        p_minus = np.sum(rho[rho.shape[0] // 2 :])
        rho_plus = np.sum(
            rho[: rho.shape[0] // 2].reshape(self.N_ancilla, 2 * self.N, 1),
            axis=0,
        )
        rho_plus = rho_plus / np.sum(rho_plus)
        rho_plus = Qobj(rho_plus, dims=[[2, self.N], [1, 1]])
        rho_minus = np.sum(
            rho[rho.shape[0] // 2 :].reshape(self.N_ancilla, 2 * self.N, 1),
            axis=0,
        )
        rho_minus = rho_minus / np.sum(rho_minus)
        rho_minus = Qobj(rho_minus, dims=[[2, self.N], [1, 1]])
        return (
            {
                'name': name + '+',
                'p': p_plus,
                'rho': rho_plus,
            },
            {
                'name': name + '-',
                'p': p_minus,
                'rho': rho_minus,
            },
        )

    def _proj(self):
        pass

    def _single_simulation(
        self,
        name: str,
        rho_i: Qobj,
    ):
        post_idle = Qobj(
            self.idle.transition_matrix @ rho_i.data, dims=[[2, self.N], [1, 1]]
        )
        full_system = tensor(self.plus_ancilla, post_idle)
        res = self.cnot.transition_matrix @ full_system.data
        res = res / np.sum(res)
        print(np.sum(res))
        print(self.cnot)
        return self.proj(rho=res, name=name)


class PhaseFlipsCorrelationTransitionMatricesReducedModel(
    PhaseFlipsCorrelationTransitionMatrices
):
    cnot_gate_cls = CNOTSFBReducedModel
    idle_gate_cls = IdleGateSFBReducedModel

    def populate(self):
        super().populate()
        self.plus_ancilla = basis(2, 0)

    @staticmethod
    def gen_plus_code_space(nbar: int, N: int):
        return tensor(fock(N, 0))

    def proj(self, rho: Qobj, name: str):
        p_plus = np.sum(rho[: rho.shape[0] // 2])
        p_minus = np.sum(rho[rho.shape[0] // 2 :])
        rho_plus = rho[: rho.shape[0] // 2]
        rho_plus = rho_plus / np.sum(rho_plus)
        rho_plus = Qobj(rho_plus, dims=[[self.N], [1]])
        rho_minus = rho[rho.shape[0] // 2 :]
        rho_minus = rho_minus / np.sum(rho_minus)
        rho_minus = Qobj(rho_minus, dims=[[self.N], [1]])
        return (
            {
                'name': name + '+',
                'p': p_plus,
                'rho': rho_plus,
            },
            {
                'name': name + '-',
                'p': p_minus,
                'rho': rho_minus,
            },
        )

    def _proj(self):
        pass

    def _single_simulation(
        self,
        name: str,
        rho_i: Qobj,
    ):
        post_idle = Qobj(
            self.idle.transition_matrix @ rho_i.data, dims=[[self.N], [1]]
        )
        full_system = tensor(self.plus_ancilla, post_idle)
        res = self.cnot.transition_matrix @ full_system.data
        res = res / np.sum(res)
        print(np.sum(res))
        print(self.cnot)
        return self.proj(rho=res, name=name)


class PhaseFlipsCorrelationTransitionMatricesNoIdle(
    PhaseFlipsCorrelationTransitionMatrices
):
    cnot_gate_cls = CNOTSFBPhaseFlips
    idle_gate_cls = IdleGateFake


def proba_outcome(outcome: str, data):
    return reduce(
        mul,
        [data[outcome[:i]]['p'] for i in range(1, len(outcome) + 1)],
        1,
    )


def proba_phase_flip(index: int, data, verbose=False):
    """Compute the proba of phase flip at index i: proba that the ith measurement error fails
    """
    if verbose:
        print('index of proba phase flip : ', index)
    if index > max([len(key) for key in data.keys()]):
        raise ValueError('Index too big')

    if index == 0:
        return 0

    prefix_list = [
        ''.join(item) for item in list(product(*[['+', '-']] * (index - 1)))
    ]
    if verbose:
        print(f'prefix list: {prefix_list}')
    s = 0
    for prefix in prefix_list:
        if verbose:
            print(prefix)
        # Using reduce(f, iterable[, initializer])
        s_inter = proba_outcome(f'{prefix}-', data)
        if verbose:
            print('intermediate sum: ', s_inter)
        s += s_inter
    return s


def proba_two_phase_flips(index_A: int, index_B: int, data, verbose=False):
    index_m = min(index_A, index_B)
    index_M = max(index_A, index_B)
    if index_B > max([len(key) for key in data.keys()]):
        raise ValueError('Index too big')

    if index_m == index_M:
        return proba_phase_flip(index=index_m, data=data)

    prefix_list = [
        ''.join(item) for item in list(product(*[['+', '-']] * (index_M - 1)))
    ]
    s = 0
    for prefix in prefix_list:
        if verbose:
            print(prefix)
        if prefix[index_m - 1] == '-':
            s_inter = proba_outcome(f'{prefix}-', data)
            s += s_inter
    return s


def correlations(index_A: int, index_B: int, data):
    index_m = min(index_A, index_B)
    index_M = max(index_A, index_B)

    return proba_two_phase_flips(
        index_m, index_M, data, verbose=False
    ) - proba_phase_flip(index_m, data) * proba_phase_flip(index_M, data)


def get_prefix_list(index: int) -> list:
    return [
        "".join(item) for item in list(product(*[["+", "-"]] * index))
    ]  # list of all strings of size index


def get_maj_counts(index: int) -> list:
    prefix_list = get_prefix_list(index=index)
    counts = [
        prefix.count("-") for prefix in prefix_list
    ]  # number of minus in each string
    to_add = [count > index / 2 for count in counts]  # 1 if majority else 0
    return prefix_list, to_add


def get_majority_vote(data: dict, index: int, compute_proba: Callable) -> list:
    try:
        prefix_list, to_add = get_maj_counts(index=index)
        probs = [
            compute_proba(outcome=outcome, data=data) * added
            for (outcome, added) in zip(prefix_list, to_add)
        ]
        return reduce(lambda a, b: a + b, probs)
    except Exception as e:
        print(e)
        return 0


def get_majority_vote_correlated(data: dict, index: int) -> float:
    return get_majority_vote(
        data=data, index=index, compute_proba=proba_outcome
    )


def uncorrelated_proba(outcome: str, data):
    res = 1
    for i, char in enumerate(outcome):
        p = proba_phase_flip(i + 1, data)
        res *= p if char == "-" else 1 - p
    return res


def get_majority_vote_uncorrelated(data: dict, index: int) -> float:
    return get_majority_vote(
        data=data, index=index, compute_proba=uncorrelated_proba
    )


if __name__ == "__main__":
    methods_list = [
        # 0
        PhaseFlipsCorrelationSFB,
        # 1
        PhaseFlipsCorrelationReducedModel,
        # 2
        PhaseFlipsCorrelationFock,
        # 3
        PhaseFlipsCorrelationSFBNoQutip,
        PhaseFlipsCorrelationSFBPerfectDissip,
        PhaseFlipsCorrelationTransitionMatrices,
        PhaseFlipsCorrelationSFBO,
        # 14
        PhaseFlipsCorrelationSFBNoIdle,
        # 15
        PhaseFlipsCorrelationFockNoIdle,
        PhaseFlipsCorrelationReducedModelNoIdle,
        PhaseFlipsCorrelationTransitionMatricesReducedModel,
        PhaseFlipsCorrelationReducedReducedModelNoIdle,
        # 32
        PhaseFlipsCorrelationReducedReducedModel,
        # 33
        PhaseFlipsCorrelationFockQutipNoIdle,
        # 34
        PhaseFlipsCorrelationTransitionMatricesNoIdle,
    ]
    k2a_l = range(1, 12, 2)
    nbar_l = range(4, 11, 2)
    index = int(sys.argv[1])
    # simu_index = int(sys.argv[1])
    # N_l = [i for i in range(15, 21)]
    # N = N_l[simu_index]
    params = list(product(nbar_l, k2a_l))
    nbar, k2a = params[index]
    N = 2
    N_ancilla = 2

    # for nbar, k2a in params:
    # k_max = k2a + 1
    nbar, k2a = params[index]
    print(f'{nbar=}')
    print(f'{k2a=}')
    k_max = k2a + 1
    for method in [14]:
        #     for nbar in [4]:
        #             # for k2a in [20]:
        Exp = methods_list[method]
        print(Exp.__name__)
        full_simulation(
            k=k_max,
            nbar=nbar,
            k2a=k2a,
            N=N,
            exp=Exp,
            # N_ancilla=N_ancilla,
            N_ancilla=N_ancilla,
            force_sim=False,
            # force_sim=True,
            use_mp=48,
        )
