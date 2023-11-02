import os
import sys

import numpy as np
from qutip import tensor

from qsim.basis.fock import Fock
from qsim.basis.sfb import SFB
from qsim.helpers import all_simulations
from qsim.helpers import data_path as db_path
from qsim.helpers import (
    first_simulation,
    gen_params_simulate,
    search_simulation,
    simulate,
)
from qsim.physical_gate.cnot import (
    CNOTSFB,
    CNOTSFBPhaseFlips,
    CNOTSFBReducedModel,
    CNOTSFBReducedModelNoQutip,
)
from qsim.physical_gate.z_gate import ZGateFock
from qsim.utils.utils import (
    Z_gate_optimal_gate_time,
    get_eps_Z,
    optimal_gate_time,
)


def gen_Z_params(
    nbar_l=range(4, 9),
    k1_l=np.logspace(-4, -1, 61),
    k2_l=[1.0],
    truncature_l=[7],
    gate_time_func=Z_gate_optimal_gate_time,
    theta_l=[np.pi / 4, np.pi / 2, np.pi],
):
    params_no_state = gen_params_simulate(
        nbar_l=nbar_l,
        k1_l=k1_l,
        k2_l=k2_l,
        truncature_l=truncature_l,
        theta_l=theta_l,
    )
    params = []
    for param in params_no_state:
        nbar, truncature = param['nbar'], param['truncature']
        param.update(
            {
                'gate_time': gate_time_func(**param),
            }
        )
        param.update({'eps_Z': get_eps_Z(**param)})
        param.pop('theta')
        basis = Fock(nbar=nbar, d=truncature)
        cardinal_states_one_qubit_fock = basis.tomography_one_qubit()

        for k, v in cardinal_states_one_qubit_fock.items():
            param.update({'initial_state': v, 'initial_state_name': k})
            params.append(param.copy())

    return params


def gen_physical_params(
    nbar_l=range(8, 9, 2),
    k1_l=np.logspace(-4, -1, 61),
    k2_l=[1.0],
    k2a_l=np.logspace(0, 2, 21),
    truncature_l=[7],
    # N_ancilla_l=[7],
    gate_time_func=lambda x: 1 / x,
):
    params_no_state = gen_params_simulate(
        nbar_l=nbar_l,
        k1_l=k1_l,
        k2_l=k2_l,
        k2a_l=k2a_l,
        truncature_l=truncature_l,
        # N_ancilla_l=N_ancilla_l,
    )
    params = []

    for param in params_no_state:
        nbar, truncature = param['nbar'], param['truncature']
        param.update(
            {
                'gate_time': gate_time_func(param['k2a']),
                'k1a': param['k2a'] * param['k1'],
            }
        )
        param.update({'N_ancilla': param['truncature']})
        basis = SFB(nbar=nbar, d=truncature, d_ancilla=truncature)
        cardinal_states_two_qubits_sfb = basis.tomography_two_qubits()
        for k, v in cardinal_states_two_qubits_sfb.items():
            param.update({'initial_state': v, 'initial_state_name': k})
            params.append(param.copy())

    return params


def gen_physical_params_red_mod(
    nbar_l=range(4, 5, 2),
    k1_l=np.logspace(-4, -1, 1),
):
    params = []

    # k2a_l = np.logspace(0, 2, 21)
    k2d = 1
    k2a = 1 * k2d
    truncature = 10
    initial_state_name = '++'

    for nbar in nbar_l:
        for k1 in k1_l:
            # gate_time = 1 / k2a
            gate_time = optimal_gate_time(nbar=nbar, k1=k1, k2=k2d)
            params.append(
                {
                    'nbar': nbar,
                    'k1': k1,
                    'k2': k2d,
                    'k1a': k1,
                    'k2a': k2a,
                    'gate_time': gate_time,
                    'truncature': truncature,
                    'initial_state_name': initial_state_name,
                }
            )
    return params


class CNOTSFB_5(CNOTSFB):
    pass


class CNOTSFB_10(CNOTSFB):
    pass


class CNOTSFB_15(CNOTSFB):
    pass


class CNOTSFB_20(CNOTSFB):
    pass


class CNOTSFB_25(CNOTSFB):
    pass


class CNOTSFB_30(CNOTSFB):
    pass


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


if __name__ == "__main__":
    N_PROC_LOCAL = 10
    # try:
    #     N_PROC_CLEPS = int(os.environ['N_PROC_CLEPS'])
    #     print(f'{N_PROC_CLEPS=}')
    # except:
    #     print('not cleps env')
    #     N_PROC_CLEPS = N_PROC_LOCAL

    N_PROC_CLEPS = 60
    GATE = CNOTSFB
    k2a_l = [5, 10, 15, 20, 25]
    index = int(sys.argv[1])
    k2a = k2a_l[index]
    params = gen_physical_params(
        k1_l=[1e-4, 1e-5],
        k2a_l=[k2a],
        nbar_l=[6],
        truncature_l=[7, 8, 9],
    )

    # Gate = ZGateFock
    # params = gen_Z_params(
    #     k1_l=[1e-5, 1e-4, 1e-3, 1e-2],
    #     truncature_l=range(7, 28, 4),
    # )

    # params = params[:3]
    print(f'{len(params)=}')
    # p = simulate(Gate, params=params, n_proc=N_PROC_LOCAL)
    db_path /= 'physical_gate'
    db_path /= f'{GATE.__name__}_{k2a}_missing_nbar6.json'
    p = simulate(GATE, params=params, n_proc=N_PROC_CLEPS, db_path=db_path)
    p.join()

    for param in params:
        try:
            param.pop('initial_state')
            param.pop('k1a')
        except KeyError:
            pass
        print(param)
        res = search_simulation(GATE, db_path=db_path, **param)
        print(f'{len(res)=}')
        res = res[0]
        res.pop('state')
        print(res)

    # res = all_simulations(CNOTSFBPhaseFlips)
    # for i in res:
    #     i.pop('state')
    # for i in res:
    #     print(i)

    # it = first_simulation(CNOTSFBPhaseFlips)
    # print(it.keys())
    # print(it.items())


#     params = gen_physical_params_red_mod()
#
#     simulate(CNOTSFBReducedModel, params=params, n_proc=16)
#
#     simulate(CNOTSFBReducedModelNoQutip, params=params, n_proc=16)
