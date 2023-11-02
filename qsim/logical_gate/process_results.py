# pylint: disable=too-many-lines
# import pdb
from abc import ABC, abstractmethod
from functools import cached_property
from itertools import product
from os.path import exists
from typing import Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from qutip.qobj import Qobj
from scipy.optimize import curve_fit

from qsim.basis.sfb import SFB
from qsim.helpers import all_simulations, data_path
from qsim.logical_gate.lgate import LGate
from qsim.logical_gate.lidle import (
    LIdleGateCompleteModelAsym,
    LIdleGateCompleteModelAsymParityMeasurement,
    LIdleGateCompleteModelAsymReduced,
    LIdleGatePRA,
    LIdleGatePRA_intermediary,
    LIdleGatePRA_with_reconv,
    LIdleGatePRALRU,
)
from qsim.physical_gate.cnot import CNOTSFB
from qsim.physical_gate.pgate import PGate
from qsim.utils.logical_fit import FitAllAtOnce
from qsim.utils.tomography import (
    CNOT_process_tomography,
    build_two_qubit_output,
    build_two_qubits_error_model,
    factorized_CNOT_process_tomography,
)
from qsim.utils.utils import (
    PRA_p_to_k1,
    compute_pXL,
    compute_pXL_per_cycle,
    compute_pZL,
    compute_pZL_per_cycle,
    error_bar,
    filter_df,
    log_pX_fit_function,
    pX_fit_PRA,
    sigma_pZL,
)


def load_df(name: str, columns: list, compute: Callable) -> pd.DataFrame:
    # print(name)
    # print(columns)
    try:
        arr = np.load(f'{name}', allow_pickle=True)
        # print(arr)
        try:
            return pd.DataFrame(
                arr,
                columns=columns,
            )
        except ValueError as e:
            print(e)
            print(arr[0])
            return pd.DataFrame()
    except FileNotFoundError as e:
        print(e)
        return compute()


def loading_results(
    date_stamp: str,
    GATE: Union[PGate, LGate],
    distance_l: Optional[list] = None,
    suffix: str = '',
    prefix: str = '',
) -> list:
    if distance_l is None:
        distance_l = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    # pdb.set_trace()
    if issubclass(GATE, PGate):
        LOCAL_PATH = (
            # f'{data_path}/cleps/physical_gate/{date_stamp}/{GATE.__name__}'
            f'{data_path}/physical_gate/{GATE.__name__}'
        )
    elif issubclass(GATE, LGate):
        # LOCAL_PATH = f'{data_path}/logical_gate/{GATE.__name__}'
        LOCAL_PATH = (
            f'{data_path}/cleps/logical_gate/{date_stamp}/{GATE.__name__}'
        )
    else:
        raise ValueError('unsupported gate class')

    res = []
    if distance_l != []:
        for distance in distance_l:
            # print(distance)
            path = f'{LOCAL_PATH}{prefix}_{distance}{suffix}.json'
            # if exists(path):
            #     pass
            # else:
            #     path = f'{LOCAL_PATH}_distance_{distance}{suffix}.json'
            res += all_simulations(path)
        return res
    print(f'{LOCAL_PATH}{suffix}.json')
    return all_simulations(f'{LOCAL_PATH}{suffix}.json')


def px_format_df(
    df: pd.DataFrame,
    truncature_l=None,
    nbar_l=None,
    k2a_l=None,
    k1_l=None,
    remove_deterministic_phase=True,
    name: str = 'PRA',
) -> pd.DataFrame:
    if truncature_l is None:
        truncature_l = [7, 8, 9]
    if nbar_l is None:
        nbar_l = [4, 5, 6, 7, 8]
    if k2a_l is None:
        k2a_l = [5, 10, 15, 20, 25, 30]
    if k1_l is None:
        k1_l = [1e-2, 1e-3, 1e-4, 1e-5]
    df['state'] = df['state'].apply(lambda x: np.array(x, dtype=complex))
    df['state'] = df['state'].apply(lambda x: x[0] + 1.0j * x[1])
    df['rho'] = df.apply(
        lambda x: Qobj(
            x['state'],
            dims=[
                [2, x['N_ancilla'], 2, x['truncature']],
                [2, x['N_ancilla'], 2, x['truncature']],
            ],
        ),
        axis=1,
    )
    res_fit = []
    for truncature in truncature_l:
        for nbar in nbar_l:
            Phi = np.array([[1, 0], [0, 0]]) + np.exp(
                1j * np.pi * nbar
            ) * np.array([[0, 0], [0, 1]])
            Phi = np.kron(Phi, np.eye(8))
            shifted_fock_basis = SFB(
                nbar=nbar, d=truncature, d_ancilla=truncature
            )
            cardinal_states_two_qubits_sfb = (
                shifted_fock_basis.tomography_two_qubits()
            )
            for k2a in k2a_l:
                for k1 in k1_l:
                    # try:
                    dfp = df[
                        (df['nbar'] == nbar)
                        & (df['truncature'] == truncature)
                        & (df['k2a'] == k2a)
                        & (df['k1'] == k1)
                    ].reset_index()  # make sure indexes
                    # pair with number of rows
                    rhop = build_two_qubit_output(
                        *[
                            shifted_fock_basis.to_code_space(
                                dfp[dfp['initial_state_name'] == state][
                                    'rho'
                                ].values[0]
                            )
                            for state in cardinal_states_two_qubits_sfb
                        ]
                    )
                    # rhop = pre_operator.dot(rhop).dot(
                    #  pre_operator.T.conj())
                    # Remove deterministic geometric phase
                    if remove_deterministic_phase:
                        rhop = Phi.dot(rhop).dot(Phi.T.conj())

                        chi = factorized_CNOT_process_tomography(rhop)
                    else:
                        chi = CNOT_process_tomography(rhop)

                    # plotting_two_qubit_process_matrix(chi)
                    chi[chi < 0] = 0
                    cnot_em = build_two_qubits_error_model(chi)
                    # print('Simulated error model')
                    # print(cnot_em)
                    cnot_em.update(
                        {
                            'k2a': k2a,
                            'nbar': nbar,
                            'truncature': truncature,
                            'k1': k1,
                            'Name': name,
                        }
                    )
                    res_fit.append(cnot_em)
                    # except:
                    #     pass
    dff = pd.DataFrame(res_fit)
    labels = [
        'pX1',
        'pX2',
        'pY1',
        'pY1X2',
        'pX1X2',
        'pZ1X2',
        'pY2',
        'pX1Y2',
        'pX1Z2',
        'pY1Y2',
        'pY1Z2',
        'pZ1Y2',
    ]
    dff['pX'] = dff[labels].sum(axis=1)
    return dff


def from_error_model_to_single_error(
    dff: pd.DataFrame, labels=None
) -> pd.DataFrame:
    res_error = []
    if labels is None:
        labels = [
            'pX1',
            'pX2',
            'pY1',
            'pY1X2',
            'pX1X2',
            'pZ1X2',
            'pY2',
            'pX1Y2',
            'pX1Z2',
            'pY1Y2',
            'pY1Z2',
            'pZ1Y2',
        ]
    for _, single_res in dff.iterrows():
        res_error += [
            {
                'k2a': single_res['k2a'],
                'nbar': single_res['nbar'],
                'truncature': single_res['truncature'],
                'k1': single_res['k1'],
                'error_label': label,
                'error_value': single_res[label],
            }
            for label in labels
        ]
    return pd.DataFrame(res_error)


def pz_update_res(results: list):
    for i, single_res in enumerate(results):
        single_res.update(
            {
                'id': i,
                'pZL': 1 - single_res['state']['outcome']['Z'],
                'N': single_res['state']['num_trajectories']['Z'],
            }
        )
        single_res.update(
            {
                'err_pZL': error_bar(
                    N=single_res['state']['num_trajectories']['Z'],
                    p=single_res['pZL'],
                )
            }
        )
        if 'k2a' not in single_res.keys():
            single_res['k2a'] = 1
    return results


def pz_fit_from_df(
    df: pd.DataFrame,
    name: str = 'PRA',
    FitClass=FitAllAtOnce,
    max_p_L=1e-3,
    rel_tol=0.50,
):
    try:
        nbar_l = sorted(list(set(df['nbar'].to_numpy())))
    except KeyError:
        nbar_l = [4]
        df['nbar'] = df.apply(lambda x: 4, axis=1)
    try:
        k2a_l = sorted(list(set(df['k2a'].to_numpy())))
    except KeyError:
        k2a_l = [1]
        df['k2a'] = df.apply(lambda x: 1, axis=1)
    keys_to_keep = ['nbar', 'k2a', 'k1d', 'distance', 'pZL', 'N', 'err_pZL']
    df_partial = df[keys_to_keep]

    nbar_l.sort()
    k2a_l.sort()

    res_fit = []
    for k2a in k2a_l:
        for nbar in nbar_l:
            single_df = df_partial[
                (df_partial['nbar'] == nbar) & (df_partial['k2a'] == k2a)
            ].reset_index()  # make sure indexes pair with number of rows
            fit = FitClass(max_p_L=max_p_L, rel_tol=rel_tol)
            for _, row in single_df.iterrows():
                # print(row['distance'], row['nbar'])
                fit.update(
                    p=row['k1d'],
                    distance=row['distance'],
                    pl=row['pZL'],
                    num_trajectories=row['N'],
                )
            fit.validity_regime()
            try:
                fit.get_fit_params()
                res_fit.append(
                    {
                        'k2a': k2a,
                        'nbar': nbar,
                        'pth': fit.pth,
                        'a': fit.a,
                        'pth_err': fit.pth_err,
                        'a_err': fit.a_err,
                        'Name': name,
                    }
                )
                try:
                    res_fit[-1].update(
                        {
                            'c': fit.c,
                            'c_err': fit.c_err,
                        }
                    )
                except AttributeError:
                    pass

            except ValueError:
                # ydata empty
                pass
    return pd.DataFrame(res_fit)


def fit_plot(tup, *args, **kwargs):
    return pX_fit_PRA(tup[0], tup[1], *args, **kwargs)


def px_fit_from_df(
    dfxfit: pd.DataFrame,
    name: str = 'PRA',
    k2a: int = 1,
    truncature: int = 7,
    log_pX_fit_function_model: Callable = log_pX_fit_function,
):
    res_fit_px = []
    dfxfit['log_pX'] = dfxfit.apply(lambda x: np.log(x['pX']), axis=1)
    x = np.array((dfxfit['nbar'].to_numpy(), dfxfit['k1'].to_numpy()))
    y = dfxfit['log_pX'].to_numpy()
    # pylint: disable=unbalanced-tuple-unpacking
    popt_pX, pcov_pX = curve_fit(
        # fit_function_real,
        log_pX_fit_function_model,
        x,
        y,
        bounds=(0, np.inf),
    )
    perr_pX = np.sqrt(np.diag(pcov_pX))
    fit_letters = ['a', 'b', 'c', 'd']
    res_opt = {'k2a': k2a, 'truncature': truncature}
    res_opt.update(dict(zip(fit_letters, popt_pX)))
    res_opt.update({f'{k}_err': v for k, v in zip(fit_letters, perr_pX)})
    res_opt.update({'Name': name})
    res_fit_px.append(res_opt)
    return res_fit_px


def plot_fitted_px(
    dfx: pd.DataFrame,
    dfxfit: pd.DataFrame,
    fit_plot_model: Callable = fit_plot,
):
    df_to_plot = dfx.copy()
    g = sns.relplot(
        data=dfx,
        x='nbar',
        y='pX',
        row='k2a',
        col='k1',
        kind='scatter',
        height=2.2,
        aspect=1.42,
    ).set(yscale='log', title='')
    # g.set_titles(
    #     # # row_template = '{row_name}',
    #     # row_template='$\\kappa_2^a / \\kappa_2^d = {row_name}$',
    #     # # col_template='$p_{{target}} = {new_col_name[col_name]}$'
    #     # col_template='$\\kappa_1 / \\kappa_2 = {col_name}$',
    # )
    for i, ax in enumerate(g.fig.axes):
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.grid(alpha=0.4)
    g.set_ylabels('CNOT bit-flip errors', clear_inner=False)
    # g.set_xlabels('$\\overline{n}$', clear_inner=False)
    g.set_xlabels('Number of photons $\\alpha^2$', clear_inner=False)

    axes = g.fig.axes
    fit_letters = ['a', 'b', 'c', 'd']
    for i, k2a in enumerate(sorted(list(set(df_to_plot['k2a'].to_numpy())))):
        df_to_plot_partial = df_to_plot[(df_to_plot['k2a'] == k2a)]
        for j, k1 in enumerate(
            sorted(list(set(df_to_plot_partial['k1'].to_numpy())))
        ):
            ax = axes[
                i * len(list(set(df_to_plot_partial['k1'].to_numpy()))) + j
            ]
            # ax.grid(which='minor', alpha=0.5)

            # ax.set_prop_cycle(None)
            df_to_plot_partial = df_to_plot_partial[
                (df_to_plot_partial['k1'] == k1)
            ]

            nbar_to_use = sorted(
                list(set(df_to_plot_partial['nbar'].to_numpy()))
            )
            ax.plot(
                nbar_to_use,
                [
                    fit_plot_model(
                        (nbar, k1),
                        *[
                            dfxfit[dfxfit['k2a'] == k2a][k].to_numpy()[0]
                            for k in fit_letters
                            if k in dfxfit.keys()
                        ],
                    )
                    for nbar in nbar_to_use
                ],
                # [fit_function_real((nbar, k1), *popt_pX)
                # for nbar in list(nbar_to_use)],
            )
            df_to_plot_partial = df_to_plot[(df_to_plot['k2a'] == k2a)]


def fitted_relplot(
    df: pd.DataFrame,
    df_fit: pd.DataFrame,
    name: str,
    max_p_L: float,
    rel_tol: float,
    distance_fit_l: Optional[List[int]] = None,
    rectangular: bool = True,
) -> sns.FacetGrid:
    df_to_plot = df.copy()

    if distance_fit_l is None:
        distance_fit_l = list(set(df.distance))
    g = sns.FacetGrid(
        df_to_plot,
        # row='nbar_regime',
        # col='nbar_index',
        row='k2a',
        col='nbar',
        hue='distance',
        height=1.8 if rectangular else 3,
        aspect=1.32 if rectangular else 1,
        legend_out=False,
    )
    g.map(plt.errorbar, 'k1d', 'pZL', 'err_pZL', marker='+', linestyle='').set(
        yscale='log', xscale='log', ylim=(df['pZL'].min(), 1)
    )

    g.set_titles(
        col_template='$|\\alpha|^2 = {col_name}$',
        row_template='$\\Theta = {row_name}$',
    )
    axes = g.fig.axes
    for i, ax in enumerate(g.fig.axes):
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
    k2a_l = sorted(list(set(df_to_plot['k2a'].to_numpy())))
    nbar_l = sorted(list(set(df_to_plot['nbar'].to_numpy())))
    for index_ax, ax in enumerate(axes):
        index_k2a = index_ax // len(nbar_l)
        index_nbar = index_ax % len(nbar_l)
        k2a = k2a_l[index_k2a]
        # k2a = 1
        # nbar = (index_ax * 2) + 10
        # ax.set_title(f'$\\alpha^2 = {nbar}$')
        nbar = nbar_l[index_nbar]
        df_to_plot_partial = df_to_plot[
            (df_to_plot['k2a'] == k2a) & (df_to_plot['nbar'] == nbar)
        ]
        # print(f'{nbar=}')
        nbar_string = f'${nbar}$'
        nbar_string = nbar

        ax.set_prop_cycle(None)

        df_data_for_fit = df_to_plot_partial[
            (df_to_plot_partial['pZL'] < max_p_L)
            & (
                df_to_plot_partial['err_pZL']
                < rel_tol * df_to_plot_partial['pZL']
            )
            & (df_to_plot_partial['distance'].isin(distance_fit_l))
        ].copy()
        if not df_data_for_fit.empty:
            df_data_for_fit.plot(
                ax=ax,
                x='k1d',
                y='pZL',
                marker='o',
                fillstyle='none',
                linestyle='',
                legend=False,
            )
        ax.set_prop_cycle(None)

        dfz = df_fit.copy()
        dfz = dfz[(dfz['k2a'] == k2a) & (dfz['nbar'] == nbar_string)]
        # print(f'{dfz=}')

        if not dfz.empty:
            try:
                a = dfz['a'].to_numpy()[0]
                pth = dfz['pth'].to_numpy()[0]
                try:
                    c = dfz['c'].to_numpy()[0]
                except KeyError:
                    c = 0.5

                for distance in sorted(
                    list(set(df_to_plot_partial['distance'].to_numpy()))
                ):
                    # print(f'{distance=}')
                    # print(f'{j=}')
                    # absci = np.sort(
                    #     df_to_plot_partial[
                    #         df_to_plot_partial['distance'] == distance
                    #     ]['k1d'].to_numpy()
                    # )[:-5]
                    pmax_fit = (
                        df_data_for_fit[
                            df_data_for_fit['distance'] == distance
                        ].k1d.max()
                        if distance > 5
                        else 2e-3
                    )
                    absci = np.logspace(-5, np.log10(pmax_fit), 5)
                    # absci = (
                    #     np.logspace(-5, np.log10(2e-3), 21)
                    #     if k2a == 1
                    #     else np.logspace(-5, np.log10(5e-3), 21)
                    # )
                    # absci = [
                    #     p
                    #     for p in absci
                    #     if compute_pZL(
                    #         a=a, distance=distance, k1=p, pth=pth, c=c
                    #     )
                    #     < 5e-3
                    # ]
                    ax.plot(
                        absci,
                        [
                            compute_pZL(
                                a=a, distance=distance, k1=k1, pth=pth, c=c
                            )
                            for k1 in absci
                        ],
                        ':',
                        lw=1,
                    )
            except IndexError:
                # no fitting parameter
                pass
        # get back all the nbar values
        df_to_plot_partial = df_to_plot[(df_to_plot['k2a'] == k2a)]
    for ax in axes:
        ax.grid(visible='True', which='major', alpha=0.6)
        ax.grid(visible='True', which='minor', alpha=0.3, linewidth=0.3)
        ax.set_xlim(8e-5, 1.2e-2)
    # g.set_ylabels('$p_{Z_L}$', clear_inner=False)
    g.add_legend(fancybox=True, framealpha=0.5, fontsize=7)
    g.set_xlabels('$\\eta$', clear_inner=False)
    # g.set_ylabels('$p_{Z_L}$', clear_inner=False)
    # g.set_ylabels('Logical $Z_L$ error probability', clear_inner=False)
    g.set_ylabels('$p_{\mathrm{Z_L}}$', clear_inner=False)
    # g.set_xlabels('$\\eta$', clear_inner=False)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.legend(loc='center left',bbox_to_anchor=(1,0.5))
    # plt.savefig(f'{name}_pzl.pdf')
    return g


def pZL_threshold(
    df: pd.DataFrame, nbar_l: Optional[list], k2a_l: Optional[list]
) -> sns.FacetGrid:
    # df = method.dffz
    if nbar_l is None:
        nbar_l_to_plot = list(set(df['nbar']))
    else:
        nbar_l_to_plot = nbar_l
    if k2a_l is None:
        k2a_l_toplot = list(set(df['k2a']))
    else:
        k2a_l_toplot = k2a_l
    df_plot = df[
        (df['k2a'].isin(k2a_l_toplot)) & (df['nbar'].isin(nbar_l_to_plot))
    ]
    df_plot['$\Theta$'] = df_plot['k2a']
    df_plot['$\overline{n}$'] = df_plot['nbar']
    df_plot['$(\kappa_1/\kappa_2)_{th}$'] = df_plot['pth']
    palette = sns.color_palette()[: len(nbar_l_to_plot)]
    g = sns.FacetGrid(
        data=df_plot,
        hue='$\overline{n}$',
        height=1.7,
        aspect=1.42,
        palette=palette,
    )
    g.map(
        plt.errorbar,
        '$\Theta$',
        '$(\kappa_1/\kappa_2)_{th}$',
        'pth_err',
        marker='o',
        linestyle='',
    ).set(
        # yscale='log'
        )

    for i, ax in enumerate(g.fig.axes):
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.grid(alpha=0.4)
        # for j, nbar in enumerate(sorted(list(set(df_plot.nbar)))):
        #     ax.axhline(
        #         y=np.log(2)/6 / nbar, linestyle='--', color=palette[j], linewidth=0.8
        #     )
    # g.add_legend()
    ax.legend(
        title='$|\\alpha|^2$',
        handletextpad=0.08,
        labelspacing=0.2,
        fancybox=True, framealpha=0.5,
        # bbox_to_anchor=(0.5, 0),
        loc="upper left",
        ncol=1
    )
    # g._legend.set_title("$|\\alpha|^2$")
    g.set_ylabels('$\\eta_{\mathrm{th}}(|\\alpha|^2, \\Theta)$', clear_inner=False)
    g.set_xlabels('$\\Theta$', clear_inner=False)
    g.set_titles(col_template='$\\kappa_2^a / \\kappa_2^d = {col_name}$')
    # plt.tight_layout()
    return g


def add_nbar_relplot(
    dfo: pd.DataFrame,
    ax: plt.Axes,
    value: Union[float, int],
    key: str = 'p_target',
    color='b',
):
    # print(f'{p_target=}')
    x, y, label = [], [], []
    # print(
    #     sorted(
    # list(set(dfo[dfo['p_target'] == p_target]['nbar'].to_numpy()))))
    nbar_to_check = sorted(list(set(dfo[dfo[key] == value]['nbar'].to_numpy())))
    x = [
        dfo[(dfo[key] == value) & (dfo['nbar'] == nbar)]['k1'].min()
        for nbar in nbar_to_check
    ]
    y = [
        dfo[(dfo[key] == value) & (dfo['nbar'] == nbar) & (dfo['k1'] == k1)][
            'd'
        ].to_numpy()[0]
        for k1, nbar in zip(x, nbar_to_check)
    ]
    label = [
        dfo[(dfo[key] == value) & (dfo['nbar'] == nbar) & (dfo['k1'] == k1)][
            'nbar'
        ].to_numpy()[0]
        for k1, nbar in zip(x, nbar_to_check)
    ]
    # print(f'{x=}')
    # print(f'{y=}')
    # print(f'{label=}')
    for k1, d, nbar in zip(x, y, label):
        if d < 100:
            ax.text(k1 ** (1.01), d + 3, f'{int(nbar)}', color=color)
            print(f'{d=}')
            print(f'{k1=}')


def plot_overhead(
    dfo: pd.DataFrame, style='p_target', col=None, hue='Name'
) -> sns.FacetGrid:
    sns.set_context('paper', rc={'lines.markersize': 4, 'lines.linewidth': 0.3})

    g = sns.relplot(
        data=dfo[dfo['d'] != 0],
        x='k1',
        y='d',
        col=col,
        # row='',
        # hue='p_target',
        style=style,
        hue=hue,
        # size='k2a',
        # kind='scatter',
        # markers=True,
        markers=True,
        kind='line',
        height=2.34,
        aspect=1.0,
        # size = 'k2a',
        # sizes = (markersize, markersize+1)
    ).set(xscale='log', ylim=(0, 100))
    g.set_titles(
        # row_template = '{row_name}',
        # col_template='$p_{{target}} = {new_col_name[col_name]}$'
        col_template='$\\epsilon_{{L}} = {col_name}$'
    )
    # g.set_ylabels('Number of data cat qubits', clear_inner=False)
    g.set_ylabels('Repetition code distance', clear_inner=False)
    g.set_xlabels('$\\kappa_1 / \\kappa_2$', clear_inner=False)
    # g.add_legend()

    # coordinates of lower left of bounding box
    # leg.set_bbox_to_anchor([0.5, 0.5])
    # leg._loc = 2  # if required you can set the loc
    return g


def p_func(pX: float, a_z: float, pth: float, c_z: float, d: int, k1: float):
    p_XL = compute_pXL(pX=pX, d=d)
    p_ZL = compute_pZL(a=a_z, distance=d, pth=pth, c=c_z, k1=k1)
    return p_XL, p_ZL


def p_func_per_cycle(
    pX: float, a_z: float, pth: float, c_z: float, d: int, k1: float
):
    p_XL = compute_pXL_per_cycle(pX=pX, d=d)
    p_ZL = compute_pZL_per_cycle(a=a_z, distance=d, pth=pth, c=c_z, k1=k1)
    return p_XL, p_ZL


def save_df(name: str, df: pd.DataFrame) -> None:
    try:
        np.save(f'{name}', df.to_numpy(), allow_pickle=True)
    except AttributeError:
        print(f'Missing {name} df')


def merge_same_keys_dfz(dfz: pd.DataFrame) -> pd.DataFrame:
    nbar_l = set(dfz.nbar)
    k2a_l = set(dfz.k2a)
    k1_l = set(dfz.k1d)
    distance_l = set(dfz.distance)
    df_res = pd.DataFrame()
    for nbar in nbar_l:
        for k2a in k2a_l:
            for k1 in k1_l:
                for d in distance_l:
                    filter_d = {
                        'nbar': nbar,
                        'k2a': k2a,
                        'k1d': k1,
                        'distance': d,
                    }
                    df_filered = filter_df(dfz, filter_df=filter_d).copy()
                    if len(df_filered) == 1:
                        df_res = pd.concat(
                            [df_res, df_filered], ignore_index=True
                        )
                    elif len(df_filered) == 0:
                        pass
                    else:
                        fails = 0
                        N = 0
                        elapsed_time = 0
                        for i, row in df_filered.iterrows():
                            fails += row['pZL'] * row['N']
                            N += row['N']
                            elapsed_time += row['elapsed_time']

                        df_single_res = df_filered.head(1)
                        df_single_res['pZL'] = fails / N
                        df_single_res['N'] = N
                        df_single_res['err_pZL'] = error_bar(N=N, p=fails / N)
                        df_single_res['elapsed_time'] = elapsed_time
                        df_res = pd.concat(
                            [df_res, df_single_res], ignore_index=True
                        )
    return df_res


def gen_dfo(
    nbar_l=[4, 6, 8, 10, 12, 14, 16],
    distance_l=[i for i in range(3, 51, 2)],
    k2a_l=[1, 5, 10, 15, 20, 25],
    k1d_l=np.logspace(-4, -2, 21),
    eps_target_l=[1e-5, 1e-7, 1e-10],
    dfx: Optional[pd.DataFrame] = None,
    dfz: Optional[pd.DataFrame] = None,
    dffx: Optional[pd.DataFrame] = None,
    dffz: Optional[pd.DataFrame] = None,
    lambda_pXL=lambda a, nbar, distance: a * np.exp(-2 * nbar) * distance,
) -> pd.DataFrame:
    names = ['nbar', 'k2a', 'k1d', 'distance']
    params_dict = [
        {name: val for (name, val) in zip(names, param)}
        for param in product(nbar_l, k2a_l, k1d_l, distance_l)
    ]
    df = pd.DataFrame(params_dict)

    for dff in [dffx, dffz]:
        if 'k2a' not in dffx.columns:
            dff['k2a'] = 1
        print(dff)

    def gen_pX(x):
        nbar = x['nbar']
        distance = x['distance']
        k2a = x['k2a']
        if nbar < 9 and dfx is not None:
            pX_per_cycle = (
                dfx[(dfx['nbar'] == nbar) & (dfx['k2a'] == k2a)].pX.values[0]
                * distance
            )
            fitx = False
            pX_per_cycle_err = 0
        else:
            a = dffx[(dffx['k2a'] == k2a)].a.values[0]
            a_err = dffx[(dffx['k2a'] == k2a)].a_err.values[0]
            pX_per_cycle = lambda_pXL(a, nbar, distance)
            pX_per_cycle_err = lambda_pXL(a_err, nbar, distance)
            fitx = True
        return pd.Series([pX_per_cycle, pX_per_cycle_err, fitx])

    def gen_pZ(x):
        # pZ
        filter_param = {name: x[name] for name in names}
        nbar = x['nbar']
        distance = x['distance']
        k2a = x['k2a']
        k1d = x['k1d']
        if dfz is not None:
            df_filtered = filter_df(filter_df=filter_param, df=dfz)
            if len(df_filtered) == 1 and df_filtered.pZL.values[0] > 0:
                pZL_per_cycle = df_filtered.pZL.values[0] / distance
                pZL_per_cycle_err = df_filtered.err_pZL.values[0] / distance
                fitz = False
        else:
            try:
                dffzp = dffz[(dffz['nbar'] == nbar) & (dffz['k2a'] == k2a)]
                a = dffzp.a.values[0]
                c = dffzp.c.values[0]
                pth = dffzp.pth.values[0]
                a_err = dffzp.a_err.values[0]
                c_err = dffzp.c_err.values[0]
                pth_err = dffzp.pth_err.values[0]
                pZL_per_cycle = a * (k1d / pth) ** (c * (distance + 1))
                pZL_per_cycle_err = sigma_pZL(
                    a, distance, k1d, pth, c, a_err, pth_err, c_err
                )
                fitz = True
            except IndexError:
                pZL_per_cycle, pZL_per_cycle_err, fitz = 1, 1, True
        return pd.Series([pZL_per_cycle, pZL_per_cycle_err, fitz])

    def gen_type(x):
        if x['Fitx']:
            if x['Fitz']:
                Type = 'Fits'
            else:
                Type = 'Fit X'
        else:
            if x['Fitz']:
                Type = 'Fit Z'
            else:
                Type = 'Simu'
        return Type

    print('gen pX...')
    df[['pX_per_cycle', 'pX_per_cycle_err', 'Fitx']] = df.apply(
        lambda x: gen_pX(x), axis=1
    )
    print('gen pZ...')
    df[['pZL_per_cycle', 'pZL_per_cycle_err', 'Fitz']] = df.apply(
        lambda x: gen_pZ(x), axis=1
    )
    print('gen epsL...')
    df['epsL_per_cycle'] = df['pX_per_cycle'] + df['pZL_per_cycle']
    df['epsL_per_cycle_err'] = np.sqrt(
        df['pX_per_cycle_err'] ** 2 + df['pZL_per_cycle_err'] ** 2
    )
    df['Type'] = df.apply(lambda x: gen_type(x), axis=1)

    df = df.sort_values(by=['epsL_per_cycle', 'k1d'])
    df_res = pd.DataFrame()

    for epsL_per_cycle_target in eps_target_l:
        for k2a in k2a_l:
            for k1d in k1d_l:
                df_single_res = (
                    df[
                        (df['k2a'] == k2a)
                        & (df['epsL_per_cycle'] <= epsL_per_cycle_target)
                        & (df['k1d'] == k1d)
                    ]
                    .sort_values(by=['nbar', 'distance'])
                    .head(1)
                )
                df_single_res['eps_target'] = df_single_res.apply(
                    lambda x: epsL_per_cycle_target, axis=1
                )
                df_res = pd.concat([df_res, df_single_res], ignore_index=True)
                # df_res.reset_index(1, inplace=True)

    def gen_label(x):
        power = int(np.log10(x["eps_target"]))
        return f'$10^{{{power}}}$'

    df_res['title_label'] = df_res.apply(lambda x: gen_label(x), axis=1)
    return df_res


# pylint: disable=too-many-public-methods
class Overhead(ABC):
    name: str = 'Overhead'
    truncature_X: int
    GATE_pZ = LIdleGatePRA_with_reconv
    GATE_pX = CNOTSFB
    FitClass = FitAllAtOnce
    date_stamp = '20220610'
    suffix: str = ''
    k1_l = [1e-2, 1e-3, 1e-4, 1e-5]
    k2a_l = [1]
    nbar_l = [4, 5, 6, 7, 8]
    distance_l = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    k1d_l = np.logspace(-5, -2, 31)[::-1]
    truncature_l = [7, 8, 9]
    p_target_l = [10 ** (-i) for i in [10, 12, 14]]
    max_p_L = 5e-3
    rel_tol = 0.30
    columns_dfz = [
        'version',
        'class_name',
        'datetime',
        'elapsed_time',
        'identifier',
        'distance',
        'physical_gates',
        'state',
        'num_rounds',
        'graph_weights',
        'nbar',
        'k1a',
        'k2a',
        'gate_time',
        'k1d',
        'k2d',
        'N_data',
        'N_ancilla',
        'idle_tr',
        'idle_pr',
        'CNOT',
        'id',
        'pZL',
        'N',
        'err_pZL',
    ]
    columns_dffz = [
        'k2a',
        'nbar',
        'pth',
        'a',
        'c',
        'pth_err',
        'a_err',
        'c_err',
        'Name',
    ]
    columns_dfx = [
        'fid',
        'pX2',
        'pY2',
        'pZ2',
        'pX1',
        'pX1X2',
        'pX1Y2',
        'pX1Z2',
        'pY1',
        'pY1X2',
        'pY1Y2',
        'pY1Z2',
        'pZ1',
        'pZ1X2',
        'pZ1Y2',
        'pZ1Z2',
        'k2a',
        'nbar',
        'truncature',
        'k1',
        'pX',
    ]
    columns_dffx = ['truncature', 'a', 'b', 'a_err', 'b_err', 'Name']
    columns_dfo = ['p_target', 'k2a', 'k1', 'd', 'nbar', 'pXL', 'pZL', 'Name']
    legend_pZ_fit = '$p_L = ad(p/p_{th})^{c(d+1)}$'

    def __init__(self, distance_l: Optional[list] = None) -> None:
        self.distance_l = distance_l
        # Z errors

        # X errors

    def loading_results(
        self,
    ):
        print(f'{self.GATE_pZ=}')
        print(f'{self.suffix=}')
        print(f'{self.date_stamp=}')
        print(f'{self.name=}')
        return loading_results(
            date_stamp=self.date_stamp,
            GATE=self.GATE_pZ,
            distance_l=self.distance_l,
            suffix=self.suffix,
        )

    def save_df(self):
        names = ['raw_pz', 'pz', 'raw_px', 'px', 'overhead']
        name = ''
        #  k2a nbar pth a c pth_err a_err c_err Name

        # fid 	pX2 	pY2 	pZ2 	pX1 	pX1X2 	pX1Y2 	pX1Z2 	pY1
        # pY1X2 	pY1Y2 	pY1Z2 	pZ1 	pZ1X2 	pZ1Y2 	pZ1Z2 	k2a 	nbar
        # truncature k1 pX Name

        # k2a truncature nbar a b c a_err b_err c_err Name
        try:
            dfs = [self.dfz, self.dffz, self.dfx, self.dffx, self.dfo]
            for name, df in zip(names, dfs):
                save_df(name=f'{self.name}_{name}_df.npy', df=df)
        except AttributeError:
            print(f'Missing {name} df')

    def load_df(self):
        print('dfx')
        self.dfx = self.load_dfx()
        # print(self.dfx)
        # self.dfsx = self.load_dfsx()
        print('dffx')
        self.dffx = self.load_dffx()

        print('dfz')
        self.dfz = self.load_dfz()
        print('dffz')
        self.dffz = self.load_dffz()

        print('dfo')
        self.dfo = self.load_dfo()

    def load_dfx(self):
        return load_df(
            name=f'{self.name}_raw_px_df.npy',
            columns=self.columns_dfx,
            compute=self.compute_dfx,
        )

    @abstractmethod
    def compute_dfx(self):
        pass

    def load_dfsx(self):
        return from_error_model_to_single_error(dff=self.dfx)

    def load_dffx(self):
        return load_df(
            name=f'{self.name}_px_df.npy',
            columns=self.columns_dffx,
            compute=self.compute_dffx,
        )

    @abstractmethod
    def log_pX_fit_function_model(
        self,
        tup,
        a: float,
        b: float = 0,
        c: float = 0,
        d: float = 2,
    ):
        pass

    # flake8: noqa: E501
    def compute_dffx(self):
        res_fit_px = []
        for k2a, truncature in product(self.k2a_l, self.truncature_l):
            try:
                dffp = self.dfx[
                    (self.dfx['k2a'] == k2a)
                    & (self.dfx['truncature'] == truncature)
                    # & (self.dfx['k1'] <= 1e-4)
                    # & (self.dfx["nbar"].isin([4, 5, 6, 7, 8]))
                ].reset_index()
                func = self.log_pX_fit_function_model
                res_fit_px += px_fit_from_df(
                    dfxfit=dffp,
                    name=self.name,
                    k2a=k2a,
                    truncature=truncature,
                    log_pX_fit_function_model=func,
                )
            except KeyError:
                pass
        print(res_fit_px)
        return pd.DataFrame(res_fit_px)

    def load_dfz(self):
        return load_df(
            name=f'{self.name}_raw_pz_df.npy',
            columns=self.columns_dfz,
            compute=self.compute_dfz,
        )

    def compute_dfz(self):
        return pd.DataFrame(pz_update_res(self.loading_results()))

    def load_dffz(self):
        return load_df(
            name=f'{self.name}_pz_df.npy',
            columns=self.columns_dffz,
            compute=self.compute_dffz,
        )

    def compute_dffz(self):
        return pz_fit_from_df(
            df=self.dfz,
            name=self.name,
            FitClass=self.FitClass,
            max_p_L=self.max_p_L,
            rel_tol=self.rel_tol,
        )

    def load_dfo(self):
        return load_df(
            name=f'{self.name}_overhead_df.npy',
            columns=self.columns_dfo,
            compute=self.compute_dfo,
        )

    @abstractmethod
    def fit_plot(self, tup, *args, **kwargs):
        pass

    def plot_fitted_px(self):
        dff = self.dfx
        dffitpx = self.dffx
        df_plot = dff[(dff['truncature'] == self.truncature_X)].reset_index()
        return plot_fitted_px(
            dfx=df_plot,
            dfxfit=dffitpx[(dffitpx['truncature'] == self.truncature_X)],
            fit_plot_model=self.fit_plot,
        )

    def fitted_relplot(
        self,
        distance_fit_l: Optional[List[int]] = None,
        rectangular: bool = True,
    ):
        return fitted_relplot(
            self.dfz,
            self.dffz,
            name=self.name,
            max_p_L=self.max_p_L,
            rel_tol=self.rel_tol,
            distance_fit_l=distance_fit_l,
            rectangular=rectangular,
        )

    def pZL_threshold(
        self, nbar_l: Optional[list] = None, k2a_l: Optional[list] = None
    ) -> sns.FacetGrid:
        return pZL_threshold(self.dffz, nbar_l=nbar_l, k2a_l=k2a_l)

    @abstractmethod
    def pX_from_df(
        self,
        nbar: int,
        k1: float,
        k2a: int = 1,
    ) -> float:
        pass
        # print(f'{nbar=}')
        # print(f'{truncature=}')
        # print(f'{k2a=}')
        # a = df_fitx[(df_fitx['k2a'] == k2a) & (
        #     df_fitx['truncature'] == truncature)]['a'].to_numpy()[0]
        # b = df_fitx[(df_fitx['k2a'] == k2a) & (
        #     df_fitx['truncature'] == truncature)]['b'].to100umpy()[0]
        # print(f'{a=}')
        # print(f'{b=}')
        # return pX_fit_PRA(nbar, k1, a=0, b=0, c=a, d=b)
        # return pX_fit_PRA(nbar, k1, a=0, b=0, c=a, d=2)
        # todo: implement pX_from_df for all child classes

    @abstractmethod
    @cached_property
    def legend_pX_fit(self):
        pass

    def get_overhead_fit_parameter(
        self, nbar: int, k1: float, k2a=1
    ) -> Tuple[float]:
        pX_df = self.pX_from_df(nbar=nbar, k1=k1, k2a=k2a)
        dfz = self.dffz
        # print(f'{nbar=}')
        # print(f'{k1=}')
        # print(f'{k2a=}')
        # print(dfz[(dfz['k2a'] == k2a) & (dfz['nbar'] == nbar)])
        # print(f'{k2a=}')
        # print(f'{nbar=}')
        a_z = dfz[(dfz['k2a'] == k2a) & (dfz['nbar'] == nbar)]['a'].to_numpy()[
            0
        ]
        pth = dfz[(dfz['k2a'] == k2a) & (dfz['nbar'] == nbar)][
            'pth'
        ].to_numpy()[0]
        c_z = dfz[(dfz['k2a'] == k2a) & (dfz['nbar'] == nbar)]['c'].to_numpy()[
            0
        ]
        return pX_df, a_z, pth, c_z

    def optimize_only_data(self, p_target, k1, k2a=1, nbar_input_l=None):
        d = 3
        nbar = 4
        if nbar_input_l is None:
            nbar_input_l = [4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20]

        # print((nbar, k1, k2a))
        pX, a_z, pth, c_z = self.get_overhead_fit_parameter(
            nbar=nbar, k1=k1, k2a=k2a
        )
        # print(pX, a_z, pth, c_z)
        p_XL, p_ZL = p_func_per_cycle(
            pX=pX, a_z=a_z, pth=pth, c_z=c_z, d=d, k1=k1
        )
        pXL_l = [p_XL]
        pZL_l = [p_ZL]
        d_l = [3]
        nbar_l = [nbar]
        d_max = 200
        counter = 0
        while p_XL + p_ZL > p_target and d < d_max:
            if p_XL > p_target / 2:
                nbar += 2
            if nbar > max(nbar_input_l):
                # the maximum allowed nbar does not allow small enough pXL
                return [0], [0], [0], [0]
            pX, a_z, pth, c_z = self.get_overhead_fit_parameter(
                nbar=nbar, k1=k1, k2a=k2a
            )
            p_XL, p_ZL = p_func_per_cycle(
                pX=pX, a_z=a_z, pth=pth, c_z=c_z, d=d, k1=k1
            )
            # print((p_XL, p_ZL))
            pXL_l.append(p_XL)
            pZL_l.append(p_ZL)
            d_l.append(d)
            nbar_l.append(nbar)
            if p_ZL > p_target / 2 and p_ZL > p_XL:
                d += 2
            p_XL, p_ZL = p_func_per_cycle(
                pX=pX, a_z=a_z, pth=pth, c_z=c_z, d=d, k1=k1
            )
            # print((p_XL, p_ZL))
            pXL_l.append(p_XL)
            pZL_l.append(p_ZL)
            d_l.append(d)
            nbar_l.append(nbar)
            if p_XL > 0.5:
                raise ValueError('Too big p_XL')

            if p_ZL > 0.5:
                print(f'\t{k1=}')
                print(f'\t{(d, nbar)=}')
                print(f'\t{(p_XL, p_ZL)=}')
                raise ValueError('Too big p_ZL')
            counter += 1
        to_add = True
        if d >= d_max:
            to_add = False
        return d_l, nbar_l, pXL_l, pZL_l, to_add

    def compute_dfo(
        self,
    ) -> pd.DataFrame:
        res_overhead = []
        for p_target in self.p_target_l:
            print(f'{p_target=}')
            for k2a in self.k2a_l:
                print(f'{k2a=}')
                for k1 in self.k1d_l:
                    # print(f'{k1=}')
                    try:
                        (
                            d_l,
                            nbar_l,
                            pXL_l,
                            pZL_l,
                            to_add,
                        ) = self.optimize_only_data(p_target, k1, k2a=k2a)
                        if not to_add:
                            raise ValueError()
                        d, nbar, pXL, pZL = (
                            d_l[-1],
                            nbar_l[-1],
                            pXL_l[-1],
                            pZL_l[-1],
                        )
                        res_overhead.append(
                            {
                                'p_target': p_target,
                                'k2a': k2a,
                                'k1': k1,
                                'd': d,
                                'nbar': nbar,
                                'pXL': pXL,
                                'pZL': pZL,
                                'Name': f'{self.name}',
                            }
                        )
                    except ValueError:
                        pass
                power = 1 / 10
                delta_k1 = 10**power
                while res_overhead[-1]['d'] < 100 and delta_k1 > (1 + 1e-2):
                    k1 = res_overhead[-1]['k1']
                    print(f'{delta_k1=}')
                    try:
                        k1 *= delta_k1
                        print(f'{k1=}')
                        (
                            d_l,
                            nbar_l,
                            pXL_l,
                            pZL_l,
                            to_add,
                        ) = self.optimize_only_data(p_target, k1, k2a=k2a)
                        if not to_add:
                            raise ValueError()
                        d, nbar, pXL, pZL = (
                            d_l[-1],
                            nbar_l[-1],
                            pXL_l[-1],
                            pZL_l[-1],
                        )
                        res_overhead.append(
                            {
                                'p_target': p_target,
                                'k2a': k2a,
                                'k1': k1,
                                'd': d,
                                'nbar': nbar,
                                'pXL': pXL,
                                'pZL': pZL,
                                'Name': self.name,
                            }
                        )

                    except ValueError:
                        power /= 2
                        delta_k1 = 10**power

        return pd.DataFrame(res_overhead)

    def merge_same_keys_dfz(self):
        return merge_same_keys_dfz(dfz=self.dfz)


class PRApX(Overhead):
    name = 'PRA'
    truncature_X = 10
    truncature_l = [10]
    columns_dffx = ['k2a', 'truncature', 'a', 'b', 'a_err', 'b_err', 'Name']
    columns_dfx = [
        'k2a',
        'k1',
        'nbar',
        'pX',
        'truncature',
        'Name',
    ]
    columns_dfz = [
        'version',
        'class_name',
        'datetime',
        'elapsed_time',
        'identifier',
        'distance',
        'physical_gates',
        'state',
        'num_rounds',
        'graph_weights',
        'nbar',
        'k1a',
        'k2a',
        'gate_time',
        'k1d',
        'k2d',
        'id',
        'pZL',
        'N',
        'err_pZL',
    ]

    def compute_dfx(self):
        results_PT = np.load('PT_results.npy', allow_pickle=True).item()
        timing = np.load('timing.npy', allow_pickle=True).item()

        lines2 = {}
        for delta in [0.0001, 0.001, 0.01, 0.1]:
            lines2[delta] = [], []

        print(lines2.keys())

        for key in sorted(timing):
            nbar, _, delta, _ = key
            lines2[delta][0].append(nbar)
            Chi2Error = results_PT[key][1]
            thresh = 1e-10
            C2E = Chi2Error

            RC2E = abs(np.real(C2E))
            indices = RC2E < thresh
            RC2E[indices] = 0
            IC2E = abs(np.imag(C2E))
            indices = IC2E < thresh
            IC2E[indices] = 0

            lines2[delta][1].append(
                1 - RC2E[0, 0] - RC2E[3, 3] - RC2E[12, 12] - RC2E[15, 15]
            )
        res = []
        for k1, (nbar_l, pX_l) in lines2.items():
            res += [
                {
                    'k2a': 1,
                    'k1': k1,
                    'nbar': nbar,
                    'pX': pX,
                    'truncature': self.truncature_X,
                    'Name': 'PRA',
                }
                for (nbar, pX) in zip(nbar_l, pX_l)
            ]
        return pd.DataFrame(res)

    def load_dfsx(self):
        self.dfsx = pd.DataFrame()

    def log_pX_fit_function_model(
        self,
        tup,
        a: float,
        b: float = 0,
        c: float = 0,
        d: float = 2,
    ):
        return log_pX_fit_function(tup, a=a, b=b, c=c, d=d)

    def compute_dffx(self):
        return pd.DataFrame(
            px_fit_from_df(
                dfxfit=self.dfx[
                    (self.dfx['nbar'].isin([4, 5, 6, 7, 8]))
                    & (self.dfx['k1'].isin([1e-3, 1e-4]))
                ],
                name='PRA',
                truncature=self.truncature_X,
                log_pX_fit_function_model=self.log_pX_fit_function_model,
            )
        )

    def pX_from_df(
        self,
        nbar: int,
        k1: float,
        k2a: int = 1,
    ):
        df_fitx = self.dffx
        truncature = self.truncature_X
        a = df_fitx[
            (df_fitx['k2a'] == k2a) & (df_fitx['truncature'] == truncature)
        ]['a'].to_numpy()[0]
        b = df_fitx[
            (df_fitx['k2a'] == k2a) & (df_fitx['truncature'] == truncature)
        ]['b'].to_numpy()[0]
        return pX_fit_PRA(nbar, k1, a=a, b=b, c=0, d=2)

    @cached_property
    def legend_pX_fit(self):
        return (
            '$p_X = e^{-2 \\overline{n}} (a \\sqrt{\\kappa_1 / \\kappa_2} + b'
            ' \\kappa_1 / \\kappa_2)$'
        )


class PRA(PRApX):
    GATE_pZ = LIdleGatePRA
    columns_dfz = [
        'version',
        'class_name',
        'datetime',
        'elapsed_time',
        'identifier',
        'distance',
        'physical_gates',
        'state',
        'num_rounds',
        'graph_weights',
        'p',
        'id',
        'pZL',
        'N',
        'err_pZL',
        'k2a',
        'k1d',
        'nbar',
    ]

    def compute_dfz(self):
        res = pz_update_res(all_simulations(self.GATE_pZ))
        for single_res in res:
            single_res.update({'k1d': PRA_p_to_k1(p=single_res['p'], k2=1)})
        return pd.DataFrame(res)

    def get_overhead_fit_parameter(self, nbar: int, k1: float, k2a=1):
        pX = self.pX_from_df(nbar=nbar, k1=k1, k2a=k2a)
        dfz = self.dffz
        # print(f'{k2a=}')
        # print(f'{nbar=}')
        a_z = dfz[(dfz['k2a'] == k2a)]['a'].to_numpy()[0]
        pth = dfz[(dfz['k2a'] == k2a)]['pth'].to_numpy()[0]
        c_z = dfz[(dfz['k2a'] == k2a)]['c'].to_numpy()[0]
        return pX, a_z, pth, c_z

    def fit_plot(self, tup, *args, **kwargs):
        return pX_fit_PRA(tup[0], tup[1], *args, **kwargs)


class PRALRU(PRA):
    name = 'PRA_LRU'
    GATE_pZ = LIdleGatePRALRU
    date_stamp = '20220628'

    def compute_dfz(self):
        res = pz_update_res(self.loading_results())
        for single_res in res:
            single_res.update({'k1d': PRA_p_to_k1(p=single_res['p'], k2=1)})
        return pd.DataFrame(res)


class PRAConv(PRApX):
    name = 'PRA_conv'


class Intermediary(PRA):
    name = 'Intermediary'
    GATE_pZ = LIdleGatePRA_intermediary


class FixedCNOT(Overhead):
    name = 'Fixed_CNOT'
    truncature_X = 10
    truncature_l = [10]
    columns_dfx = [
        'fid',
        'pX2',
        'pY2',
        'pZ2',
        'pX1',
        'pX1X2',
        'pX1Y2',
        'pX1Z2',
        'pY1',
        'pY1X2',
        'pY1Y2',
        'pY1Z2',
        'pZ1',
        'pZ1X2',
        'pZ1Y2',
        'pZ1Z2',
        'k2a',
        'nbar',
        'truncature',
        'k1',
        'Name',
        'pX',
    ]
    columns_dffx = [
        'k2a',
        'truncature',
        'a',
        'a_err',
        'Name',
    ]
    columns_dfz = [
        'nbar',
        'k2a',
        'k1d',
        'distance',
        'pZL',
        'N',
        'err_pZL',
        'Name',
    ]

    def fit_plot(self, tup, *args, **kwargs):
        return pX_fit_PRA(tup[0], tup[1], 0, 0, *args, **kwargs)

    def log_pX_fit_function_model(
        self,
        tup,
        a: float,
        b: float = 0,
        c: float = 0,
        d: float = 2,
    ):
        return log_pX_fit_function(tup, a=c, b=b, c=a, d=d)

    def compute_dfx(self):
        res = loading_results('20220330', GATE=self.GATE_pX, distance_l=[])
        res_border = loading_results(
            '20220401', GATE=self.GATE_pX, distance_l=[]
        )
        df = pd.DataFrame(res + res_border)
        return px_format_df(
            df=df,
            truncature_l=[self.truncature_X],
            k2a_l=self.k2a_l,
            remove_deterministic_phase=False,
            name=f'{self.name}',
        )

    def pX_from_df(
        self,
        nbar: int,
        k1: float,
        k2a: int = 1,
    ):
        df_fitx = self.dffx
        truncature = self.truncature_X
        a = df_fitx[
            (df_fitx['k2a'] == k2a) & (df_fitx['truncature'] == truncature)
        ]['a'].to_numpy()[0]
        return pX_fit_PRA(nbar, k1, a=0, b=0, c=a, d=2)

    @cached_property
    def legend_pX_fit(self) -> str:
        return '$p_X = a \\times e^{{-2 \\overline{{n}}}}$'

    def compute_dfz(self):
        data_name = f'{data_path}/../experiments/logical_gate/outcome.npy'
        data_dict = np.load(data_name, allow_pickle=True).item()

        new_data_name = (
            f'{data_path}/../experiments/logical_gate/outcome (1).npy'
        )
        new_data_dict = np.load(new_data_name, allow_pickle=True).item()
        data_dict.update(new_data_dict)
        return pd.DataFrame(
            [
                {
                    'nbar': nbar,
                    'k2a': 1,
                    'k1d': k1,
                    'distance': d,
                    'pZL': e,
                    'N': N,
                    'err_pZL': error_bar(N=N, p=e),
                    'Name': f'{self.name}',
                }
                for (d, k1, nbar, N), e in data_dict.items()
            ]
        )


class Asym(Overhead):
    name = 'Asym'
    truncature_X = 25  # 9
    truncature_l = [25]
    truncature_Z = 3
    k2a_l = [5, 10, 15, 20, 25]
    date_stamp = '20220602'
    GATE_pZ = LIdleGateCompleteModelAsym
    columns_dffx = [
        'k2a',
        'truncature',
        'a',
        # 'b',
        'a_err',
        # 'b_err',
        'Name',
    ]
    columns_dfx = ['nbar', 'k1', 'k2a', 'pX', 'truncature']
    old_columns_dfx = [
        'fid',
        'pX2',
        'pY2',
        'pZ2',
        'pX1',
        'pX1X2',
        'pX1Y2',
        'pX1Z2',
        'pY1',
        'pY1X2',
        'pY1Y2',
        'pY1Z2',
        'pZ1',
        'pZ1X2',
        'pZ1Y2',
        'pZ1Z2',
        'k2a',
        'nbar',
        'truncature',
        'k1',
        # 'Name',
        'pX',
    ]

    def pX_from_df(
        self,
        nbar: int,
        k1: float,
        k2a: int = 1,
    ):
        df_fitx = self.dffx
        truncature = self.truncature_X
        print(f'{truncature=}')
        a = df_fitx[
            (df_fitx['k2a'] == k2a) & (df_fitx['truncature'] == truncature)
        ]['a'].to_numpy()[0]
        # b = df_fitx[
        # (df_fitx['k2a'] == k2a) & (df_fitx['truncature'] == truncature)
        # ]['b'].to_numpy()[0]
        # print(f'{a=}')
        # print(f'{b=}')
        return pX_fit_PRA(nbar, k1, a=0, b=0, c=a, d=2)

    def log_pX_fit_function_model(
        self,
        tup,
        a: float,
        # b: float = 0,
        # c: float = 0,
        # d: float = 2,
    ):
        return log_pX_fit_function(tup, a=0, b=0, c=a, d=2)

    def fit_plot(self, tup, *args, **kwargs):
        return pX_fit_PRA(tup[0], tup[1], 0, 0, *args, **kwargs)

    def compute_dfx(self):
        res = loading_results(
            '20220602', GATE=self.GATE_pX, distance_l=self.k2a_l
        )
        print(f'{len(res)=}')
        res_big_nbar = loading_results(
            '20220613',
            GATE=self.GATE_pX,
            distance_l=self.k2a_l,
            suffix='_big_nbar',
        )
        print(f'{len(res_big_nbar)=}')
        res_missing_nbar = loading_results(
            '20220613',
            GATE=self.GATE_pX,
            distance_l=self.k2a_l,
            suffix='_missing_nbar6',
        )
        print(f'{len(res_missing_nbar)=}')
        df = pd.DataFrame(res + res_big_nbar + res_missing_nbar)
        return px_format_df(
            df=df,
            truncature_l=self.truncature_l,
            nbar_l=self.nbar_l,
            k2a_l=self.k2a_l,
            k1_l=self.k1_l,
            name=self.name,
        )

    @cached_property
    def legend_pX_fit(self):
        return '$p_X = a \\times e^{{-b \\overline{{n}}}}$'

    def update_pZL(self):
        new_method = Asym()
        new_method.date_stamp = '20220801'
        new_method.dfz = new_method.compute_dfz()
        keys_to_keep = ['nbar', 'k2a', 'k1d', 'distance', 'pZL', 'N', 'err_pZL']
        ndfp = new_method.dfz[keys_to_keep]

        dfp = self.dfz[keys_to_keep]
        self.dfz = pd.concat([dfp, ndfp], axis=0)
        self.dffz = self.compute_dffz()


class AsymRed(Asym):
    name = 'AsymRed'
    k2a_l = [1, 5, 10]
    nbar_l = [4, 6, 8, 10]
    date_stamp = '20220830'
    GATE_pZ = LIdleGateCompleteModelAsymReduced


class AsymParityMeasurement(Asym):
    name = 'AsymParityMeasurement'
    k2a_l = [1, 5, 10, 15, 20, 25]
    nbar_l = [4, 6, 8, 10, 12, 14, 16]
    date_stamp = '20220914'
    GATE_pZ = LIdleGateCompleteModelAsymParityMeasurement
    columns_dffz = [
        'k2a',
        'nbar',
        'pth',
        'a',
        'pth_err',
        'a_err',
        'Name',
        'c',
        'c_err',
    ]
    columns_dfo = [
        'nbar',
        'k2a',
        'k1d',
        'distance',
        'pX_per_cycle',
        'pX_per_cycle_err',
        'Fitx',
        'pZL_per_cycle',
        'pZL_per_cycle_err',
        'Fitz',
        'epsL_per_cycle',
        'epsL_per_cycle_err',
        'Type',
        'eps_target',
        'title_label',
    ]

    def gen_dfo(
        self,
        nbar_l=[4, 6, 8, 10, 12, 14, 16],
        distance_l=[i for i in range(3, 51, 2)],
        k2a_l=[1, 5, 10, 15, 20, 25],
        k1d_l=np.logspace(-4, -2, 21),
        eps_target_l=[1e-5, 1e-7, 1e-10],
    ) -> pd.DataFrame:
        return gen_dfo(
            nbar_l=nbar_l,
            distance_l=distance_l,
            k2a_l=k2a_l,
            k1d_l=k1d_l,
            eps_target_l=eps_target_l,
            dfx=self.dfx,
            dfz=self.dfz,
            dffx=self.dffx,
            dffz=self.dffz,
        )

    def plot_overhead(
        self,
        d_max=40,
        k2a_max=26,
        eps_target_l=[1e-5, 1e-7, 1e-10],
    ):
        df_res = self.dfo
        dffz = self.dffz

        df_res.sort_values(by=['k1d'])

        # palette = sns.color_palette()[:2]
        df_res_plotted = df_res[
            (df_res['distance'] <= d_max) & (df_res['k2a'] <= k2a_max)
        ]
        palette = sns.color_palette()[: len(set(df_res_plotted.k2a))]
        # style palette
        dash_list = sns._core.unique_dashes(
            df_res_plotted['k2a'].unique().size + 1
        )
        style = {key: dash_list[0] for key in df_res_plotted['k2a'].unique()}
        g = sns.relplot(
            data=df_res_plotted,
            x='k1d',
            y='distance',
            hue='k2a',
            col='title_label',
            # style='Type',
            kind='line',
            marker='o',
            markersize=4,
            markers=False,
            palette=palette,
            height=2,
            aspect=1.0,
            dashes=style,
            # linewidth = 0.3,
            linewidth=1,
            legend=False,
        ).set(
            xscale='log',
        )

        for i, (ax, eps_target) in enumerate(zip(g.fig.axes, eps_target_l)):
            texts = []
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.set_xlim(8e-5, 1.3e-2)
            ax.set_ylim(0, 45)

            ax.grid(which='major', alpha=0.5)
            # ax.grid(which='minor')
            nbar_set = set(
                df_res_plotted[df_res_plotted['eps_target'] == eps_target].nbar
            )
            # nbar_min = min(nbar_set)
            # nbar_max = max(nbar_set)
            # ax.text(0.25, 0.85, f'$\\overline{{n}} \in [{nbar_min}, {nbar_max}]$', transform = ax.transAxes)
            for i, k2a in enumerate(set(df_res_plotted.k2a)):
                print()
                ax.axvline(
                    x=dffz[
                        (dffz['k2a'] == k2a) & (dffz['nbar'] == max(nbar_set))
                    ].pth.to_numpy()[0],
                    color=palette[i],
                    linestyle='-',
                    linewidth=0.5,
                )

            df_res_p = df_res_plotted[(df_res['eps_target'] == eps_target)]
            for item, color in zip(df_res_p.groupby('k2a'), palette):
                for x, y, nbar, Type in item[1][
                    ['k1d', 'distance', 'nbar', 'Type']
                ].values:
                    # s = set(item[1]['k2a'])
                    # k2a = next(iter(s))
                    # if k2a == 1:
                    #     ax.text(x*0.95, y+2, f'${int(nbar)}$', color=color, fontsize=8)
                    # elif k2a == 15:
                    #     ax.text(x*0.95, y-5, f'${int(nbar)}$', color=color, fontsize=8)
                    # else:
                    # ax.text(x*1.01, y, f'${int(nbar)}$', color=color, fontsize=8)
                    # texts.append(ax.text(x, y, f'${int(nbar)}$', color=color, fontsize=6))
                    if Type == 'Simu':
                        ax.plot(x, y, 'o', color=color)
            ax.set_yticks([10 * i for i in range(5)], minor=False)
            ax.set_yticklabels([10 * i for i in range(5)], fontsize=8)
            # adjust_text(texts, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="-", lw=0.5))
        # g._legend.set_title("$\\kappa_2^a / \\kappa_2^d$")
        # g.fig.axes[2].legend()
        g.set_titles(
            # row_template = '{row_name}',
            # col_template='$p_{{target}} = {new_col_name[col_name]}$'
            col_template='$\\epsilon_{{L}} = ${col_name}'
        )

        for k2a in set(df_res_plotted.k2a):
            g.fig.axes[0].plot([], [], label=f'{k2a}', linewidth=1)
        g.fig.axes[0].legend(
            title='$\\Theta$',
            title_fontsize="10",
            fontsize='6.5',
            fancybox=True,
            framealpha=0.0,
            loc='upper left',
            bbox_to_anchor=(0, 1.04),
        )

        g.set_ylabels('Code distance', clear_inner=False)
        g.set_xlabels('$\\eta$', clear_inner=False)
        # plt.legend(loc = 2, bbox_to_anchor = (1,1))
        plt.savefig('overhead_asym.pdf')


if __name__ == '__main__':
    # GATE = LIdleGatePRA_with_reconv
    # res = loading_results(date_stamp='20220610', GATE=GATE)
    # pz_update_res(res)
    # df = pd.DataFrame(res)
    # dffit = pz_fit_from_df(df)
    # print(dffit)
    method = FixedCNOT()
    # method.load_df()
    method.dfz = method.load_dfz()
    method.dffz = method.load_dffz()
    method.dfx = pd.DataFrame()
    method.dffx = pd.DataFrame()
    method.dfo = pd.DataFrame()
    method.save_df()
