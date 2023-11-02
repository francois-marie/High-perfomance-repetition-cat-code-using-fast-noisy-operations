from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib.axes import Axes
from qutip import Qobj, fock, ket2dm, qeye, tensor
from scipy.sparse.linalg import expm


def unitary(H, dt):
    if type(H) == sp.csc.csc_matrix:
        return expm(-1j * H * dt)
    else:
        return expm(-1j * H.tocsc() * dt)


def generate_plot_params():
    plt.rcParams.update(
        {
            'text.usetex': True,
            'font.family': 'serif',
            'mathtext.fontset': 'cm',
            'mathtext.rm': 'serif',
            'font.weight': 'normal',
            'axes.labelweight': 'normal',
            'axes.linewidth': 1.5,
            'axes.titlepad': 10,
            'xtick.major.width': 1.5,
            'xtick.major.size': 10.0,
            'xtick.minor.size': 5.0,
            'xtick.direction': 'in',
            'xtick.major.pad': 10,
            'ytick.major.width': 1.5,
            'ytick.major.size': 10.0,
            'ytick.minor.size': 5.0,
            'ytick.direction': 'in',
            'font.size': 25,
            'figure.max_open_warning': 0,
            'legend.fontsize': 22,
            # 'axes.prop_cycle': cycler('color', plt.cm.Set1.colors)
            # 'image.cmap': 'viridis'
        }
    )


def generate_ax_params(
    ax: Axes,
    title: str = '$\\overline{{n}}, \\kappa_{{2, a}}$',
    xlabel: str = '$\\kappa_1/\\kappa_2$',
    ylabel: str = 'Logical Phase Flip $Z_L$ Error Probability $p_{Z_L}$',
    xscale: str = 'log',
    yscale: str = 'log',
    legend=True,
    color_map: str = 'gist_rainbow',
):
    ax.grid(which='minor', color='gray', alpha=0.1)
    ax.grid(which='major', color='gray', alpha=0.4)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # set_color(ax=ax, color_map=color_map)
    if legend:
        ax.legend()
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


def set_color(ax, color_map: str = 'gist_rainbow'):  # 'viridis'
    num_colors = len(ax.get_lines())
    print(num_colors)
    cm = plt.get_cmap(color_map)
    color = [cm(1.0 * i / num_colors) for i in range(num_colors)]
    ax.set_prop_cycle(color=color)
    for i in range(num_colors):
        ax.get_lines()[i].set_color(color[i])


def combine(list_p) -> float:
    """
    input:
    @list_p: list of probabilities for a given error

    output:
    @p : probability that an error occurred by combining the probabilities
    """
    if isinstance(list_p, (np.float64, float)):
        return list_p
    return reduce(lambda a, b: a * (1 - b) + (1 - a) * b, list_p)


def exponentiel_proba(p: float):
    """True expression of the error probability process valid for
    1 photon loss and 2 photon pumping non-adiabatic error

    Args:
        p (float): Physical first step error rate

    Returns:
        float: Exponential error probability
    """
    return (1 - np.exp(-2 * p)) / 2


def nonadiab_phaseflip(nbar: int, k2: float, gate_time: float):
    return exponentiel_proba(np.pi**2 / (64 * nbar * k2 * gate_time))


def one_photon_loss_phaseflip(nbar: int, k1: float, gate_time: float, **kwargs):
    return exponentiel_proba(nbar * k1 * gate_time)


def CNOT_control_phaseflip(
    nbar: int, k1: float, k2: float, gate_time: float, **kwargs
):
    return exponentiel_proba(
        combine(
            [
                one_photon_loss_phaseflip(
                    nbar=nbar, k1=k1, gate_time=gate_time
                ),
                nonadiab_phaseflip(nbar=nbar, k2=k2, gate_time=gate_time),
            ]
        )
    )


def CNOT_target_phaseflip(nbar: int, k1: float, gate_time: float, **kwargs):
    return exponentiel_proba(
        0.5 * one_photon_loss_phaseflip(nbar=nbar, k1=k1, gate_time=gate_time)
    )


def error_bar(N: int, p: float):
    # standard deviation (sqrt root of variance) / sqrt of size
    if p * (1 - p) / N < 0:
        print(f'{p=}')
        print(f'{N=}')
        raise ValueError(f'The error bar in negative but has to be positive')
    return 1.96 * np.sqrt(p * (1 - p) / N)


def optimal_gate_time(nbar: int, k1: float, k2: float, **kwargs) -> float:
    # cf PRA
    # return 1 / 2 / nbar / np.sqrt(np.pi) / np.sqrt(k1 * k2)
    # cf AWS
    return np.pi / 8 / nbar / np.sqrt(2) / np.sqrt(k1 * k2)



def get_eps_Z(theta: float, nbar: int, gate_time: float, **kwargs) -> float:
    return theta / (4 * np.sqrt(nbar) * gate_time)


def Z_gate_optimal_gate_time(
    nbar: int, k1: float, k2: float, theta: float, **kwargs
) -> float:
    return theta / (4 * nbar ** (3 / 2) * np.sqrt(k1 * k2))


def get_theta(eps_Z: float, nbar: int, gate_time: float, **kwargs) -> float:
    return eps_Z / (4 * np.sqrt(nbar) * gate_time)


def PRA_p_to_k1(p: float, k2: float = 1):
    return k2 * p**2 * (8 * np.sqrt(2) / np.pi) ** 2
    # return k2 * 4 * np.pi * p ** 2 # old PRA


def PRA_k1_to_p(k1: float, k2: float, **kwargs):
    # return 1 / 2 * np.sqrt(k1 / np.pi / k2) # old PRA
    return np.pi / 8 / np.sqrt(2) * np.sqrt(k1 / k2)


def proj_code_space_sfb(N: int):
    plus = tensor((fock(2, 0) + fock(2, 1)) / np.sqrt(2), fock(N, 0))
    minus = tensor((fock(2, 0) - fock(2, 1)) / np.sqrt(2), fock(N, 0))
    plus_dm = ket2dm(plus)
    minus_dm = ket2dm(minus)
    return plus_dm + minus_dm


def pX_fit_PRA(
    nbar: int,
    k1: float,
    a: float = 5.58,
    b: float = 1.68,
    c: float = 0,
    d: float = 2.0,
):
    # print(f'{a=}')
    # print(f'{b=}')
    # print(f'{c=}')
    # print(f'{d=}')
    return (a * np.sqrt(k1) + b * k1 + c) * np.exp(-d * nbar)


def log_pX_fit_function(
    tup, a: float = 5.58, b: float = 1.68, c: float = 0, d: float = 2.0
):
    return -d * tup[0] + np.log(a * np.sqrt(tup[1]) + b * tup[1] + c)


def compute_pZL(a: float, distance: int, k1: float, pth: float, c: float = 0.5):
    return a * distance * (k1 / pth) ** (c * (distance + 1))


def compute_pZL_per_cycle(
    a: float, distance: int, k1: float, pth: float, c: float
):
    return a * (k1 / pth) ** (c * (distance + 1))


def dpZL_dpth(a: float, distance: int, k1: float, pth: float, c: float):
    return a * (k1 / pth) ** (c * distance) * (-c * distance / pth)


def dpZL_dc(a: float, distance: int, k1: float, pth: float, c: float):
    return a * distance * (k1 / pth) ** (c * distance) * np.log(k1 / pth)


def sigma_pZL(
    a: float,
    distance: int,
    k1: float,
    pth: float,
    c: float,
    sigma_a: float,
    sigma_pth: float,
    sigma_c: float,
):
    return np.sqrt(
        sigma_a**2 * (k1 / pth) ** (2 * c * distance)
        + sigma_pth**2 * dpZL_dpth(a, distance, k1, pth, c) ** 2
        + sigma_c**2 * dpZL_dc(a, distance, k1, pth, c) ** 2
    )


def compute_pXL(pX: float, d: int):
    return combine([pX] * 2 * d * (d - 1))


def compute_pXL_per_cycle(pX: float, d: int):
    return combine([pX] * 2 * (d - 1))

