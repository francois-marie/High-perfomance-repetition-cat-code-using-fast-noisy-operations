import sys
from itertools import product
from multiprocessing import Pool

import numpy as np

from qsim.physical_gate.cnot import CNOTSFBPhaseFlips
from qsim.physical_gate.idle import IdleGateSFBPhaseFlips
from qsim.physical_gate.pgate import PGate
from qsim.utils.error_model import (
    CNOTCompleteModelNumericalErrorFromCorrelations,
)

cnot_gate_cls = CNOTSFBPhaseFlips
# idle_tr_gate_cls = IdleGateSFBPhaseFlips  # total reconvergence
# idle_pr_gate_cls = IdleGateSFBPhaseFlips  # partial reconvergence
# idle_pr_gate_cls_ancilla = IdleGateSFBPhaseFlips  # partial reconvergence

N_data_l = [3, 6, 7]
index = int(sys.argv[1])
N_data = N_data_l[index]


def load_or_save(file_name: str, gate: PGate) -> None:
    try:
        gate.transition_matrix = np.load(file_name)
    except FileNotFoundError:
        gate.transition_matrices()
        np.save(file_name, gate.transition_matrix)


def routine(nbar: int, k1: float, k2a: int):
    gate_time = 1 / k2a
    k1d = k1
    k2d = 1
    k1a = k1d * k2a
    N_ancilla = N_data

    param = {
        'nbar': nbar,
        'k1': k1,
        'k2a': k2a,
        'N_data': N_data,
        'N_ancilla': N_ancilla,
        'gate_time': gate_time,
        'k1d': k1d,
        'k2d': k2d,
        'k1a': k1a,
    }
    print(param)

    cnot = cnot_gate_cls(
        nbar=nbar,
        k1=k1d,
        k2=k2d,
        k2a=k2a,
        k1a=k1a,
        gate_time=gate_time,
        N_ancilla=N_ancilla,
        truncature=N_data,
    )

    # # partial reconvergence during parity_measurement cnot gate_time
    # idle_pr = idle_pr_gate_cls(
    #     nbar=nbar,
    #     k1=k1d,
    #     k2=k2d,
    #     gate_time=gate_time,
    #     truncature=N_data,
    # )
    # # total reconvergence after k2a rounds of error detection
    # idle_tr = idle_tr_gate_cls(
    #     nbar=nbar,
    #     k1=k1d,
    #     k2=k2d,
    #     gate_time=1 / k2d,
    #     truncature=N_data,
    # )

    # # partial reconvergence during parity_measurement cnot gate_time
    # idle_pr_ancilla = idle_pr_gate_cls_ancilla(
    #     nbar=nbar,
    #     k1=k1a,
    #     k2=k2a,
    #     gate_time=gate_time,
    #     truncature=N_ancilla,
    # )

    # gate_l = [cnot, idle_pr, idle_tr, idle_pr_ancilla]
    gate_l = [cnot]
    for gate in gate_l:
        print(type(gate).__name__)
        print(gate)
        print(f'{gate.truncature=}')

    # prefix = '../../data/experiments/qec/'
    prefix = ''
    cnot_file_name = f'{prefix}CNOT_{nbar}_{k2a}_{N_data}_{N_ancilla}_{k1d}.npy'
    # idle_tr_file_name = f'{prefix}Idle_tr_{nbar}_{k2d}_{N_data}_{k1d}.npy'
    # idle_pr_file_name = f'{prefix}Idle_pr_{nbar}_{k2a}_{N_data}_{k1d}.npy'
    # idle_pr_ancilla_file_name = (
    #     f'{prefix}Idle_pr_ancilla_{nbar}_{k2a}_{N_ancilla}_{k1a}.npy'
    # )
    # file_name_l = [
    #     cnot_file_name,
    #     idle_pr_file_name,
    #     idle_tr_file_name,
    #     idle_pr_ancilla_file_name,
    # ]
    file_name_l = [
        cnot_file_name,
    ]

    for file_name, gate in zip(file_name_l, gate_l):
        print(f'{file_name=}')
        load_or_save(file_name=file_name, gate=gate)


def test_coverage(
    nbar_l=[6, 8, 10, 12, 14, 16, 18, 20],
    k1_l=np.logspace(-4, -2, 21),
    k2a_l=[5, 10, 15, 20, 25],
    N_data_l=[3, 6, 7, 8, 9],
):
    params = list(product(nbar_l, k1_l, k2a_l, N_data_l))
    res_coverage = {k: 0 for k in params}
    # print(res_coverage)
    prefix = ''
    for k in res_coverage.keys():
        nbar, k1d, k2a, N_data = k
        N_ancilla = N_data
        cnot_file_name = (
            f'{prefix}CNOT_{nbar}_{k2a}_{N_data}_{N_ancilla}_{k1d}.npy'
        )
        try:
            np.load(cnot_file_name)
            res_coverage[k] = 1
        except FileNotFoundError:
            pass
    # print(res_coverage)
    for N_data_target in N_data_l:
        res = 0
        for k in res_coverage.keys():
            nbar, k1d, k2a, N_data = k
            if N_data == N_data_target and res_coverage[k] == 1:
                res += 1
        res /= len(nbar_l) * len(k1_l) * len(k2a_l)
        print(f'{N_data_target=} : {res*100:.2f}%')
    return res_coverage


def routine_correlations(nbar: int, k1: float, k2a: int):
    param_em = {
        'nbar': nbar,
        'k1': k1,
        'k2a': k2a,
    }
    param_em.update(
        {
            'k1d': k1,
            'k1a': k1 * k2a,
            'k2d': 1,
            'gate_time': 1 / k2a,
            'truncature': N_data,
            'N_ancilla': N_data,
        }
    )
    param_em.pop('k1')
    cnot_em = CNOTCompleteModelNumericalErrorFromCorrelations(**param_em)
    return cnot_em


if __name__ == '__main__':
    num_proc = 50

    nbar_l = [4, 6, 8, 10]
    # k1_l = np.logspace(-4, -2, 21)
    k1_l = [0]
    k2a_l = [2 * i + 1 for i in range(6)]
    params = list(product(nbar_l, k1_l, k2a_l))
    # print(f'{len(params)=}')

    pool = Pool(num_proc)
    res = pool.starmap(
        routine,
        # routine_correlations,
        params,
    )  # This will call q.get() until None is returned

    # test_coverage()
