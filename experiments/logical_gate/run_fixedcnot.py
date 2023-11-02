import sys
from itertools import product
from time import time
from typing import List, Type

import numpy as np

from qsim.helpers import (
    data_path,
    gen_params_simulate,
    search_simulation,
    simulate,
)
from qsim.logical_gate.lidle import (
    LIdleGate,
    LIdleGateCompleteModel,
    LIdleGateCompleteModelAsym,
    LIdleGateCompleteModelAsymParityMeasurement,
    LIdleGateCompleteModelAsymReduced,
    LIdleGateCompleteModelPheno,
    LIdleGateCompleteModelPRA,
    LIdleGatePRA,
)
from qsim.utils.error_model import (
    CNOTCompleteModelNumericalErrorFromCorrelations,
)
from qsim.utils.utils import optimal_gate_time


def gen_fixedcnot_params(
    nbar_l=range(8, 9, 2),
    k1d_l=np.logspace(-4, -1, 61),
    k2d_l=[1.0],
    k2a_l=np.logspace(0, 2, 21),
    N_data_l=[7],
    N_ancilla_l=None,
    distance_l=range(3, 10, 2),
    **kwargs,
):
    if N_ancilla_l is None:
        params = gen_params_simulate(
            nbar_l=nbar_l,
            k1d_l=k1d_l,
            k2d_l=k2d_l,
            k2a_l=k2a_l,
            distance_l=distance_l,
            N_data_l=N_data_l,
            **kwargs,
        )
        for param in params:
            param.update({'N_ancilla': param['N_data']})

    else:
        params = gen_params_simulate(
            nbar_l=nbar_l,
            k1d_l=k1d_l,
            k2d_l=k2d_l,
            k2a_l=k2a_l,
            distance_l=distance_l,
            N_data_l=N_data_l,
            N_ancilla_l=N_ancilla_l,
            **kwargs,
        )
    for param in params:
        param.update(
            {
                'gate_time': 1 / param['k2a'],
                'num_rounds': param['distance'] * param['k2a'],
                'k1a': param['k2a'] * param['k1d'],
            }
        )

        param_em = param.copy()
        N_data = param_em.pop('N_data')
        param_em['truncature'] = N_data
        param_em.pop('distance')
        param_em.pop('num_rounds')

        # compute phase flips correlations
        # cnot_em = CNOTCompleteModelNumericalErrorFromCorrelations(**param_em)
    return params


def gen_fixedcnotpheno_params(
    nbar_l=range(8, 9, 2),
    k1d_l=np.logspace(-4, -1, 61),
    k2d_l=[1.0],
    k2a_l=np.logspace(0, 2, 21),
    N_data_l=[7],
    N_ancilla_l=[4],
    distance_l=range(3, 10, 2),
):
    # pheno
    p_l = np.concatenate(
        (np.logspace(-2, -1, 11), np.logspace(-1, np.log10(3e-1), 5))
    )[::-1]
    # PRA
    p_l = np.logspace(-3, -1, 21)[::-1]

    params = gen_params_simulate(
        nbar_l=nbar_l,
        k1d_l=k1d_l,
        k2d_l=k2d_l,
        k2a_l=k2a_l,
        distance_l=distance_l,
        N_data_l=N_data_l,
        N_ancilla_l=N_ancilla_l,
    )
    for param in params:
        param.update(
            {
                'gate_time': optimal_gate_time(
                    nbar=param['nbar'], k1=param['k1d'], k2=param['k2d']
                ),
                'num_rounds': param['distance'],
                'k1a': param['k2a'] * param['k1d'],
                'init_transition_matrix': False,
                'logic_fail_max': 500,
            }
        )
    return params


def main(
    Gate: Type[LIdleGate] = LIdleGateCompleteModel,
    nbar_l=range(8, 9, 2),
    k1d_l=np.logspace(-4, -1, 61),
    k2d_l=[1.0],
    k2a_l=np.logspace(0, 2, 21),
    N_data_l=[7],
    N_ancilla_l=[7],
    distance_l=range(3, 10, 2),
    num_proc=10,
):
    params = gen_fixedcnot_params(
        nbar_l=nbar_l,
        k1d_l=k1d_l,
        k2d_l=k2d_l,
        k2a_l=k2a_l,
        distance_l=distance_l,
        N_data_l=N_data_l,
        N_ancilla_l=N_ancilla_l,
    )
    print(params)
    t_start = time()
    p = simulate(Gate, params, n_proc=num_proc)
    p.join()
    print('time')
    print(f'{time() - t_start:.2f}s')


if __name__ == "__main__":
    # p = 2e-2
    # k1d = np.pi * 4 * p ** 2
    # k1d = 1e-2
    # k1d = 0
    # params = gen_fixedcnotpheno_params(
    #     nbar_l=[4],
    #     # k1d_l=[1e-2],
    #     k1d_l=[np.pi * 4 * p ** 2 for p in p_l],
    #     k2a_l=[1],
    #     distance_l=range(3, 8, 2),
    #     N_data_l=[3],
    #     N_ancilla_l=[2],
    # )

    # test_params = gen_fixedcnot_params(
    #     nbar_l=range(8, 9, 2),
    #     k1d_l=[5e-4],
    #     k2a_l=[1],
    #     distance_l=[5],
    #     N_data_l=[5],
    #     N_ancilla_l=[3],
    # )
    # params = gen_fixedcnot_params(
    #     nbar_l=[14, 20],
    #     k1d_l=np.logspace(-5, -2, 31)[::-1],
    #     k2a_l=[1],
    #     distance_l=[3, 5, 7, 13, 15, 17],
    #     N_data_l=[5],
    #     N_ancilla_l=[3],
    # )
    # params = gen_fixedcnot_params(
    #     nbar_l=[4],
    #     k1d_l=[1e-3],
    #     k2a_l=[1],
    #     distance_l=[3],
    #     N_data_l=[2],
    #     N_ancilla_l=[2],
    #     init_transition_matrix_l=[True],
    #     logic_fail_max_l=[100],
    # )

    t_start = time()

    # GATE = LIdleGateCompleteModelPheno
    # GATE = LIdleGateCompleteModelPRA
    # GATE = LIdleGateCompleteModel
    # GATE = LIdleGateCompleteModelAsym
    # GATE = LIdleGateCompleteModelAsymReduced
    GATE = LIdleGateCompleteModelAsymParityMeasurement

    # LOCAL TESTING
    # obj = GATE(
    #     **params[0],
    # )
    # print(obj)
    # # obj.simulate()

    # p = simulate(GATE, params=params, n_proc=10)
    # p.join()

    # CLUSTER PROD
    index = int(sys.argv[1])
    # distance_l = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    distance_l = [3, 5, 7, 9, 11]
    k1d_l = np.logspace(-4, -2, 21)[::-1]
    # nbar_l = [6, 8, 10, 12, 14, 16, 18, 20]
    nbar_l = [4, 6, 8, 10]
    k2a_l = [1, 5, 10]
    # params = list(product(nbar_l, distance_l))
    # param = params[index]
    distance = distance_l[index]

    # nbar, distance = param
    # print(f'{nbar=}')
    print(f'{distance=}')

    data_path /= 'logical_gate'
    data_path /= f'{GATE.__name__}_distance_{distance}.json'
    for partial_k1d_l in [k1d_l[:11], k1d_l[11:16], k1d_l[16:]]:
        for k2a in k2a_l:
            params = gen_fixedcnot_params(
                # nbar_l=[6, 8, 10, 12, 14, 16, 18, 20],
                nbar_l=nbar_l,
                k1d_l=partial_k1d_l,
                k2a_l=[k2a],
                # distance_l=[distance],
                distance_l=[distance],
                N_data_l=[3],
                N_ancilla_l=[2],
                init_transition_matrix_l=[True],
                logic_fail_max_l=[100],
            )

            p = simulate(
                GATE,
                params=params,
                n_proc=min(len(params), 40),
                db_path=data_path,
            )
            p.join()

    # print(obj.idle_pr.transition_matrix)
    # print(obj.cnot.transition_matrix)

    print('time')
    print(f'{time() - t_start:.2f}s')
