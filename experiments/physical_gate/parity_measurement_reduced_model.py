from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from qutip import Qobj
from qutip import basis as basis_state
from qutip import fock_dm, ket2dm, mesolve, qeye, sigmaz, tensor
from qutip.qip.operations import snot

from qsim.physical_gate.cnotidle import (
    CNOTSFBReducedModelQubitBitParityMeasurementFirst,
    CNOTSFBReducedModelQubitBitParityMeasurementSecond,
)


def parity_measurement(
    ancilla_name: str = '+',
    gauge_0: int = 0,
    gauge_1: int = 0,
    params: Optional[dict] = None,
) -> Qobj:
    if params is None:
        raise ValueError('Params empty')
    plus = ket2dm((basis_state(2, 0) + basis_state(2, 1)) / np.sqrt(2))
    minus = ket2dm((basis_state(2, 0) - basis_state(2, 1)) / np.sqrt(2))
    if ancilla_name == '+':
        ancilla = plus
    elif ancilla_name == '-':
        ancilla = minus
    else:
        raise ValueError('Wrong value for the ancilla: + or -')

    truncature = params['truncature']
    params.update(
        {
            'initial_state': tensor(
                ancilla,
                fock_dm(truncature, gauge_0),
                fock_dm(truncature, gauge_1),
            ),
            'initial_state_name': f'{ancilla_name}{str(gauge_0)}{str(gauge_1)}',
        }
    )
    cnot1 = CNOTSFBReducedModelQubitBitParityMeasurementFirst(**params)
    results1 = cnot1.simulate()
    res1 = cnot1.expect
    rho = results1['state']
    params.update(
        {
            'initial_state': rho,
            'initial_state_name': (
                f'CNOT{ancilla_name}{str(gauge_0)}{str(gauge_1)}'
            ),
        }
    )
    cnot2 = CNOTSFBReducedModelQubitBitParityMeasurementSecond(**params)
    results2 = cnot2.simulate()
    res2 = cnot2.expect
    rho = results2['state']
    return rho, res1, res2


def gen_proba(rho) -> dict:
    minus = ket2dm((basis_state(2, 0) - basis_state(2, 1)) / np.sqrt(2))
    plus = ket2dm((basis_state(2, 0) + basis_state(2, 1)) / np.sqrt(2))
    return {
        f'{name_ancilla}{str(gauge_0)}{str(gauge_1)}': np.real(
            np.trace(
                rho * tensor(ancilla, fock_dm(2, gauge_0), fock_dm(2, gauge_1))
            )
        )
        for name_ancilla, ancilla in zip(['+', '-'], [plus, minus])
        for gauge_0 in [0, 1]
        for gauge_1 in [0, 1]
    }


def ancilla_phase_flip_proba_gauged(
    res: dict, gauge_0: int = 0, gauge_1: int = 0, ancilla: str = '+'
) -> float:
    return sum(
        [
            res[f'{ancilla}{str(gauge_0)}{gauge_1}'][k]
            for k in res[f'{ancilla}{str(gauge_0)}{gauge_1}'].keys()
            if k[0] == '-'
        ]
    )


def ancilla_phase_flip_proba(res: dict) -> float:
    return sum(
        [
            ancilla_phase_flip_proba_gauged(res, gauge_0, gauge_1, ancilla)
            for gauge_0 in [0, 1]
            for gauge_1 in [0, 1]
            for ancilla in ['+', '-']
        ]
    )


def gen_res_parity_measurement(params=None):
    if params is None:
        params = params
    return {
        f'{ancilla}{str(gauge_0)}{str(gauge_1)}': gen_proba(
            parity_measurement(
                ancilla_name=ancilla,
                gauge_0=gauge_0,
                gauge_1=gauge_1,
                params=params,
            )[0]
        )
        for gauge_0 in [0, 1]
        for gauge_1 in [0, 1]
        for ancilla in ['+', '-']
    }


def update_systems_parity_measurement(
    res: dict, gauge_0: int, gauge_1: int, ancilla: int
) -> Tuple:
    """Update the two adjacent data systems and the ancilla system after
    one parity measurement counting only non adiabatic errors

    Args:
        res (dict): probability distribution
        gauge_0 (int): state of the gauge of top data
        gauge_1 (int): state of the gauge of bottom data
        ancilla (int): state of the ancilla

    Returns:
        Tuple: tuple consisting of ancilla and both data gauges.
    """
    ancilla_name = '+' if ancilla == 0 else '-'
    final_state = np.random.choice(
        list(res[f'{ancilla_name}{gauge_0}{gauge_1}'].keys()),
        1,
        p=list(res[f'{ancilla_name}{gauge_0}{gauge_1}'].values()),
    )[0]
    return (
        0 if final_state[0] == '+' else 1,
        int(final_state[1]),
        int(final_state[2]),
    )


def gen_cnot_onephotonloss(p_ancilla: float, p_data: float) -> str:
    return np.random.choice(
        ['I', 'Z1', 'Z2', 'Z1Z2'],
        1,
        p=[1 - p_ancilla - 2 * p_data, p_ancilla, p_data, p_data],
    )[0]


def update_systems_cnot_onephotonloss(error_str: str) -> Tuple[int]:
    """Updates the ancilla and data systems after drawing cnot errors from one photon loss.

    Args:
        error_str (str): Label of the error

    Returns:
        Tuple[int]: ancilla and data error syndromes.
    """
    error_dict = {
        'I': (0, 0),
        'Z1': (1, 0),
        'Z2': (0, 1),
        'Z1Z2': (1, 1),
    }
    return error_dict[error_str]


if __name__ == '__main__':
    p = snot() * basis_state(2, 0)
    m = snot() * basis_state(2, 1)
    gauge_0 = 0
    gauge_1 = 0
    k2 = 1
    nbar = 8
    k2a = 20 * k2
    gate_time = 1 / k2a
    k1 = 0
    k1a = k1 * k2a
    truncature = 2

    params = {
        "nbar": nbar,
        "k2": k2,
        "k1": k1,
        "k1a": k1a,
        "k2a": k2a,
        "gate_time": gate_time,
        "truncature": truncature,
    }

    r1 = 4 * k2 * nbar
    r2 = np.pi**2 / 16 / nbar / k2a / gate_time**2
    print(f'{r1=}')
    print(f'{r2=}')
    res = gen_res_parity_measurement(params=params)
    print(f'{res=}')
    # plt.figure(figsize=(15, 10))
    rho, res1, res2 = parity_measurement(
        gauge_0=0, gauge_1=0, ancilla_name='+', params=params
    )
    plt.axhline(y=r2 / r1, linestyle=':', label='Thermal pop')
    plt.plot(res1[1], label='$\\hat{n}$ gauge_0')
    plt.plot(res2[1], label='$\\hat{n}$ gauge_1')
    plt.plot(res1[0], label='P-, 1st')
    plt.plot(res2[0], label='P-, 2nd')
    plt.legend()
    plt.show()
    print(update_systems_parity_measurement(res, 1, 0, '+'))
