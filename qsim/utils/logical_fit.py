from abc import ABC, abstractmethod
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, leastsq

from qsim.utils.utils import error_bar, generate_plot_params

from qsim.utils.quantum_guidelines import global_setup, plot_setup

global_setup()


class LogicalFit(ABC):
    def __init__(self, display_errorbars=True, max_p_L=None, rel_tol=None):
        self.data = {}
        self.d_list = []
        self.p_list = []
        self.pth = None
        self.display_errorbars = display_errorbars
        if rel_tol is None:
            rel_tol = 5e-1
        if max_p_L is None:
            max_p_L = 1e-2
        self.max_p_L = max_p_L
        self.rel_tol = rel_tol

    def update(self, p: float, distance: int, pl: float, num_trajectories: int):
        self.data.update(
            {
                (p, distance): {
                    'pl': pl,
                    'num_trajectories': num_trajectories,
                }
            }
        )

    def get_d_list(self):
        self.d_list = list({key[1] for key in self.data.keys()})
        self.d_list.sort()

    def get_p_list(self):
        self.p_list = list({key[0] for key in self.data.keys()})
        self.p_list.sort()

    def get_infid_n_lists(self, d: int):
        self.get_p_list()
        infid, n_list, absc = [], [], []
        for p in self.p_list:
            try:
                infidelity = self.data[p, d]['pl']
                if infidelity > 0.0:
                    infid.append(infidelity)
                    n_list.append(self.data[p, d]['num_trajectories'])
                    absc.append(p)
            except KeyError:
                pass
        return infid, n_list, absc

    @abstractmethod
    def get_fit_params(self):
        pass

    def threshold(self):
        if self.pth is None:
            self.get_fit_params()
        return self.pth

    def plot_raw(self, ax, marker_list=['+']):
        self.get_d_list()
        self.get_p_list()
        for i, d in enumerate(self.d_list):
            infid, n_list, absc = self.get_infid_n_lists(d=d)
            if self.display_errorbars:
                ax.errorbar(
                    absc,
                    infid,
                    yerr=[
                        error_bar(N=n_list[i], p=infid[i])
                        for i in range(len(absc))
                    ],
                    marker=marker_list[i % len(marker_list)],
                    label=f'${d}$',
                    linestyle='',
                )
            else:
                ax.plot(
                    absc,
                    infid,
                    marker=marker_list[i % len(marker_list)],
                    label=f'$d={d}$',
                    linestyle='',
                )

    def plot(
        self, ax=None, title: str = '', xlim=None, ylim=None, marker_list=['+']
    ):
        # generate_plot_params()
        if ax is None:
            fig = plot_setup(aspect_ratio=1)
            ax = fig.add_subplot()
        self.get_d_list()
        self.get_p_list()
        # self.threshold()
        ax.set_xlabel('Error rate')
        self._plot(ax, title=title, marker_list=marker_list)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylabel('Logical Error Rate $p_{Z_L}$')
        # ax.grid()
        # ax.grid(which='minor', color='gray', alpha=0.1)
        ax.grid(which='major', color='gray', alpha=0.4)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend(frameon=False, title='Distance')
        return ax

    @abstractmethod
    def _plot(self, ax):
        pass

    def validity_regime(self):
        self.get_p_list()
        self.get_d_list()

        # self.data = {k: v for (k, v) in self.data.items() if self.cond(v=v)}
        # apply condition on each point of self.data
        self.data_used_in_fit = {
            k: v for (k, v) in self.data.items() if self.cond(v=v)
        }


        # # remove the points where increasing the distance
        # # does not decrease the logical error
        # for p in self.p_list:
        #     pl_list = []
        #     self.d_list_to_remove = []
        #     for d in self.d_list:
        #         try:
        #             pl_list.append(self.data[p, d]['pl'])
        #             self.d_list_to_remove.append(d)
        #         except KeyError:
        #             pass
        #     pl_ordered = pl_list.copy()
        #     pl_ordered.sort()
        #     if pl_list != pl_ordered[::-1]:
        #         # the Logical error rate does not decrease
        #         # as the distance of the code increases
        #         for d in self.d_list_to_remove:
        #             self.data.pop((p, d))

    def cond(self, v):
        return True



class FirstOrder(LogicalFit):
    """pL propto a (p/pth)**(d+1)/2"""

    def get_fitted_d_list(self):
        self.fitted_d_list = []
        self.get_d_list()
        for d in self.d_list:
            infid, _, absc = self.get_infid_n_lists(d=d)
            if len(absc) > 1:
                self.fitted_d_list.append(d)

    def single_fit_coeff(self, d: int):
        # log pL = (d+1)/2 log p
        # + log a - (d+1)/2 log pth -> b
        infid, n_list, absc = self.get_infid_n_lists(d=d)

        def linear_fit(x, b):
            return (d + 1) / 2 * x + b

        popt, pcov = curve_fit(
            linear_fit,
            np.log10(absc),
            np.log10(infid),
            sigma = [np.log10(error_bar(N=n, p=p)) for n, p in zip(n_list, infid)],
            maxfev=5000,
            bounds=(0, np.inf),
            # absolute_sigma=True,
        )

        # plt.plot(np.log10(absc), np.log10(infid), 'o')
        # plt.plot(np.log10(absc), linear_fit(np.log10(absc), *popt), label=f'popt={popt}')
        # plt.title(f'd={d}')
        # plt.legend()
        # plt.show()


        # return popt[0]
        return (popt, pcov)



    def get_fit_params(self):
        """
        return: pth, A
        """
        # -(d+1)/2 log pth + log a
        def affine_func(x, a, b):
            # a=pth, b=A
            return a * (x + 1) / 2 + b
            # return -np.log10(a) * (x + 1) / 2 + np.log10(b)

        self.get_fitted_d_list()
        # y = [self.single_fit_coeff(d) for d in self.fitted_d_list]
        y = []
        yerr = []
        for d in self.fitted_d_list:
            popt, pcov = self.single_fit_coeff(d=d)
            y.append(popt[0])
            perr = np.sqrt(np.diag(pcov))
            yerr.append(perr[0])


        popt, pcov = curve_fit(
            affine_func,
            self.fitted_d_list,
            y,
            sigma=yerr,
            # absolute_sigma=True,
        )
        perr = np.sqrt(np.diag(pcov))
        self.pth = 10 ** (-popt[0])
        self.a = 10 ** (popt[1])
        self.pth_err = perr[0] * np.log(10) * np.exp(-popt[0] * np.log(10))
        self.a_err =  perr[1] * np.log(10) * np.exp(-popt[1] * np.log(10))

        # self.pth = popt[0]
        # self.a = popt[1]
        # self.pth_err = perr[0]
        # self.a_err = perr[1]

    def _plot(self, ax, title: str = '', marker_list=['+']):
        self.plot_raw(ax=ax, marker_list=marker_list)
        ax.set_prop_cycle(None)
        self.validity_regime()
        # self.get_fit_params()
        self.threshold()
        for i, d in enumerate(self.d_list):
            _, _, absc = self.get_infid_n_lists(d=d)
            ax.plot(
                absc,
                [self.a * (p / self.pth) ** ((d + 1) / 2) for p in absc],
                # label=f'$d={d}$',
                marker='',  # no marker for the fit
                linestyle=':',
            )
        ax.set_title(
            '$p_L = a(\\frac{\\delta}{\\delta_{{th}}}'
            f' )^{{(d+1)/2}}$, \n $a={self.a:.2f},'
            f' \\delta_{{th}}={self.pth:.3f}$ \n {title}'
        )
        ax.set_xlabel(
            'Error Rate $\\delta = \\frac{1}{2 \\sqrt{\\pi}}'
            ' \\sqrt{\\frac{\\kappa_1}{\\kappa_2}}$'
        )
        ax.set_xscale('log')

    def cond(self, v):
        # remove points with too big error_bars
        if (
            error_bar(
                N=v['num_trajectories'],
                p=v['pl'],
            )
            / v['pl']
            > self.rel_tol
            or v['pl'] > self.max_p_L
        ):
            return False
        return True


class FirstOrderOver4(FirstOrder):
    """pL propto a (p/pth)**(d+1)/4"""

    def single_fit_coeff(self, d: int):
        # log pL = (d+1)/2 log p
        # + log a - (d+1)/2 log pth -> b
        infid, _, absc = self.get_infid_n_lists(d=d)

        def linear_fit(x, b):
            return (d + 1) / 4 * x + b

        popt, _ = curve_fit(linear_fit, np.log10(absc), np.log10(infid))
        return popt[0]

    def get_fit_params(self):
        """
        return: pth, A
        """
        # -(d+1)/4 log pth + log a
        def affine_func(x, a, b):
            return a * (x + 1) / 4 + b

        self.get_fitted_d_list()

        popt, _ = curve_fit(
            affine_func,
            self.fitted_d_list,
            [self.single_fit_coeff(d) for d in self.fitted_d_list],
        )
        self.pth = 10 ** (-popt[0])
        self.a = 10 ** (popt[1])

    def _plot(self, ax, title: str = '', marker_list=['+']):
        self.plot_raw(ax=ax, marker_list=marker_list)
        ax.set_prop_cycle(None)
        self.validity_regime()
        # self.get_fit_params()
        self.threshold()
        for i, d in enumerate(self.d_list):
            _, _, absc = self.get_infid_n_lists(d=d)
            ax.plot(
                absc,
                [self.a * (p / self.pth) ** ((d + 1) / 4) for p in absc],
                # label=f'$d={d}$',
                marker='',  # no marker for the fit
                linestyle=':',
            )
        ax.set_title(
            '$p_L = a(\\frac{\\delta}{\\delta_{{th}}}'
            f' )^{{(d+1)/4}}$, \n $a={self.a:.2f},'
            f' \\delta_{{th}}={self.pth:.3f}$ \n {title}'
        )
        ax.set_xlabel('Error Rate $\\delta = \\frac{\\kappa_1}{\\kappa_2}$')
        ax.set_xscale('log')


class FittedExponent(FirstOrder):
    """pL propto a (bp)**cd"""

    def single_fit_coeff(self, d: int):
        # log pL = cd log p
        # + log a + cd log b
        # c -> c
        # e -> log a + cd log b
        infid, _, absc = self.get_infid_n_lists(d=d)

        def linear_fit(x, c, e):
            return c * d * x + e

        popt, _ = curve_fit(linear_fit, np.log10(absc), np.log10(infid))
        return popt[0], popt[1]

    def get_fit_params(self):
        """
        return: pth, A
        """
        self.get_fitted_d_list()

        fits = [self.single_fit_coeff(d) for d in self.fitted_d_list]
        self.c = np.mean([fit[0] for fit in fits])
        # cd log b + log a

        def affine_func(x, A, B):
            return x * B + A

        popt, _ = curve_fit(
            affine_func,
            [
                c * d
                for c, d in zip([fit[0] for fit in fits], self.fitted_d_list)
            ],
            [fit[1] for fit in fits],
        )
        self.pth = 10 ** (-popt[1])
        self.a = 10 ** (popt[0])

    def _plot(self, ax, title: str = '', marker_list=['+']):
        self.plot_raw(ax, marker_list=marker_list)
        ax.set_prop_cycle(None)
        self.get_fit_params()
        for d in self.d_list:
            _, _, absc = self.get_infid_n_lists(d=d)
            ax.plot(
                absc,
                [self.a * (p / self.pth) ** (self.c * d) for p in absc],
                label=f'$d={d}$',
                marker='',  # no marker for the fit
                linestyle=':',
            )

        ax.set_title(
            f'$p_L = a(p/\\delta_{{th}})^{{cd}}$, \n $a={self.a:.2f},'
            f' \\delta_{{th}}={self.pth:.3f}, c={self.c:.2f}$ \n {title}'
        )


def rescaled_error(p: float, distance: int, A, B, C, D, pth, nu0, mu):
    # rescaled error rate
    x = (p - pth) * distance ** (1 / nu0)
    return A + B * x + C * x ** 2 + D * distance ** (-1 / mu)


class FirstOrderPhysicalParameters(FirstOrder):
    def __init__(
        self,
        nbar: int,
        k1d: float,
        k1a: float,
        k2d: float,
        k2a: int,
        gate_time: float,
    ):
        super().__init__()
        self.nbar = nbar
        self.k1d = k1d
        self.k2d = k2d
        self.gate_time = gate_time
        self.k2a = k2a
        self.k1a = k1a

    """pL propto a (p/pth)**(d+1)/2"""

    def _plot(self, ax, title: str = '', marker_list=['+']):
        super()._plot(ax=ax, title=title, marker_list=marker_list)
        ax.set_xlabel(
            'Error Rate $\\delta = \\overline{n} \\kappa_1/\\kappa_2$'
        )
        ax.set_title(ax.get_title() + f'\n $\\overline{{n}} = {self.nbar}$')



class FitAllAtOnce(FirstOrder):
    """pL propto ad(bp)**c(d+1)"""

    legend = '$p_L = ad(p/p_{th})^{c(d+1)}$'

    def fit_function(self, tup, a, b, c):
        # tup: (logp, d)
        return np.log(a) + c * (tup[1] + 1) * (tup[0] - np.log(b))

    def fit_plot(self, tup, *args, **kwargs):
        return tup[1] * np.exp(self.fit_function(tup, *args, **kwargs))

    def fit_plot_per_cycle(self, tup, *args, **kwargs):
        return np.exp(self.fit_function(tup, *args, **kwargs))

    def gen_x_y_fit(self):
        x = []
        logpl = []
        dl = []
        y = []
        yerr = []
        count = 0

        self.validity_regime()
        for p, d in self.data_used_in_fit.keys():
            e = self.data_used_in_fit[(p, d)]['pl']
            N = self.data_used_in_fit[(p, d)]['num_trajectories']
            yerr.append(error_bar(N=N, p=e) / e)
            # N = self.data[(p, d)]['num_trajectories']
            logpl.append(np.log(p))
            dl.append(d)
            if self.per_round:
                y.append(np.log(e / d))
            else:
                y.append(np.log(e))
            count += 1

        x = np.concatenate((logpl, dl))
        x.shape = (2, count)
        return x, y, yerr

    def get_fit_params(self):
        """
        return: pth, A
        """
        x, y, yerr = self.gen_x_y_fit()
        popt, pcov = curve_fit(
            self.fit_function,
            x,
            y,
            maxfev=5000,
            bounds=(0, np.inf),
            sigma=yerr,
            absolute_sigma=True,
        )
        self.popt = popt
        perr = np.sqrt(np.diag(pcov))
        self.perr = perr
        self.get_errors()

    def get_errors(self):
        self.a, self.pth, self.c = self.popt
        self.a_err, self.pth_err, self.c_err = self.perr

    def _plot(self, ax, title: str = '', marker_list=['+']):
        self.plot_raw(ax, marker_list=marker_list)
        ax.set_prop_cycle(None)
        self.validity_regime()
        self.get_fit_params()
        for d in self.d_list:
            _, _, absc = self.get_infid_n_lists(d=d)
            p_max = max(absc)
            # p_max = max(absc) if absc != [] else 1e-3
            # p_max = min(absc[5], 1e-3)
            absc = np.logspace(-5, np.log10(p_max))
            absc = [
                p
                for p in absc
                if self.fit_plot((np.log(p), d), *self.popt) < 1e-3
            ]
            ax.plot(
                absc,
                [self.fit_plot((np.log(p), d), *self.popt) for p in absc],
                # label=f'$d={d}$',
                marker='',  # no marker for the fit
                linestyle=':',
            )

        ax.set_title(f'{self.gen_title()}')

    def _plot_per_cycle(self, ax, title: str = '', marker_list=['+']):
        self.plot_raw_per_cycle(ax, marker_list=marker_list)
        ax.set_prop_cycle(None)
        self.validity_regime()
        self.get_fit_params()
        for d in self.d_list:
            _, _, absc = self.get_infid_n_lists(d=d)
            p_max = max(absc) if absc != [] else 1e-3
            absc = np.logspace(-5, np.log10(p_max))
            ax.plot(
                absc,
                [
                    self.fit_plot_per_cycle((np.log(p), d), *self.popt)
                    for p in absc
                ],
                # label=f'$d={d}$',
                marker='',  # no marker for the fit
                linestyle=':',
            )

        ax.set_title(f'{self.gen_title()}')

    def gen_eq(self):
        return f'$a={self.a:.3f}, p_{{th}}={self.pth:.5f}, c={self.c:.3f}$'

    def gen_title(self, title: str = ''):
        return (
            f'{type(self).__name__} \n{self.legend}, \n{self.gen_eq()}\n{title}'
        )