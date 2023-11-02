import sys
from itertools import product

import numpy as np
import pandas as pd
from qutip import basis as qutipbasis
from qutip import ket2dm, qeye, tensor

from qsim.basis.fock import Fock
from qsim.basis.sfb import SFB
from qsim.helpers import data_path as db_path
from qsim.physical_gate.cnot import CNOTSFB, CNOTFockQutip
from qsim.physical_gate.idle import IdleGateFock, IdleIdleGateFock


def run_simu_2_cnots(index: int) -> None:
    k2a_l = [1, 5, 10, 15, 20, 25]
    print(f'{k2a_l=}')
    nbar_l = [2, 3, 4, 5, 6, 7, 8]
    nbar_N_l = []
    for nbar in nbar_l:
        N_max = 8 * nbar if nbar <= 6 else 5 * nbar
        nbar_N_l += [(nbar, N_max)]
        # nbar_N_l += [(nbar, N_max)]
    print(f'{nbar_l=}')
    # k1_l = [10**(-2)]
    k1_l = [0]
    print(f'{k1_l=}')

    params = list(product(nbar_N_l, k2a_l, k1_l))

    nbar_N_l = []
    for nbar in [2, 8]:
        N_max = 6 * nbar if nbar <= 6 else 5 * nbar
        nbar_N_l += [(nbar, N_max - 1), (nbar, N_max - 2), (nbar, N_max - 3)]
    params += list(product(nbar_N_l, [1, 5], k1_l))

    print(f'{len(params)=}')
    print(params)
    print(params[index])
    (nbar, N), k2a, k1d = params[index]

    k2d = 1
    k1a = k1d * k2a
    # gate_time = optimal_gate_time(nbar, k1d, k2d)
    gate_time = 1 / k2a

    N_data = N
    N_ancilla = N
    basis = Fock(nbar=nbar, d=N_data, d_ancilla=N_ancilla)
    # +0
    initial_state = tensor(
        ket2dm(basis.ancilla.evencat),
        ket2dm(basis.data.zero),
    )
    cnot = CNOTFockQutip(
        nbar=nbar,
        k2=k2d,
        k2a=k2a,
        k1=k1d,
        k1a=k1a,
        gate_time=gate_time,
        truncature=N_data,
        N_ancilla=N_ancilla,
        initial_state=initial_state,
        initial_state_name="+0"
        # kphi=1e-4,
    )

    idleidle = IdleIdleGateFock(
        nbar=nbar,
        k2=k2d,
        k2a=k2a,
        k1=k1d,
        k1a=k1a,
        gate_time=gate_time,
        truncature=N_data,
        N_ancilla=N_ancilla,
        initial_state=initial_state,
        initial_state_name="CNOT+0",
    )
    print(f'{k2a=}')
    rho = initial_state
    for i in range(k2a):
        print(f'{i=}')

        # prep
        idleidle.initial_state = rho
        idleidle.simulate()

        # CNOT
        print('1st')
        cnot.initial_state = idleidle.rho
        cnot.initial_state_name = 'CNOT+0'
        res = cnot.simulate()

        # reconvergence
        idleidle.initial_state = cnot.rho
        idleidle.simulate()

        # CNOT
        cnot.initial_state = idleidle.rho
        cnot.initial_state_name = 'CNOT+0'
        print('2nd')
        res = cnot.simulate()

        # measure
        idleidle.initial_state = cnot.rho
        idleidle.simulate()

        rho_data = idleidle.rho.ptrace(1)
        # Jx = basis.data.Jx
        # pX = np.real(np.trace(rho_data * tensor(ket2dm(qutipbasis(2, 1)), qeye(N))))
        pX = np.real(basis.data.bitflip_proba(rho_data))
        print(pX)

        rho = tensor(
            ket2dm(basis.ancilla.evencat),
            rho_data,
        )

        result = {
            'class': type(cnot).__name__,
            'index': i,
            'nbar': nbar,
            'k2a': k2a,
            'k1d': k1d,
            'N': N,
            'k2d': k2d,
            'k1a': k1a,
            'gate_time': gate_time,
            'pX': pX,
        }
        prefix = 'QEC_cycle_'
        index = 54
        np.save(
            f'{db_path}/experiments/physical_gate/{prefix}{index}_{i}.npy',
            result,
            allow_pickle=True,
        )
        d = np.load(
            f'{db_path}/experiments/physical_gate/{prefix}{index}_{i}.npy',
            allow_pickle=True,
        )
        print(d)

def run_simu_1_cnot(index: int) -> None:
    k2a_l = [1, 5, 10, 15, 20]
    print(f'{k2a_l=}')
    nbar_l = [8]
    nbar_N_l = [37]
    print(f'{nbar_l=}')
    k1_l = [0]
    print(f'{k1_l=}')

    params = list(product(nbar_l, nbar_N_l, k2a_l, k1_l))


    print(f'{len(params)=}')
    print(params)
    print(params[index])
    nbar, N, k2a, k1d = params[index]

    k2d = 1
    k1a = k1d * k2a
    # gate_time = optimal_gate_time(nbar, k1d, k2d)
    gate_time = 1 / k2a

    N_data = N
    N_ancilla = N
    basis = Fock(nbar=nbar, d=N_data, d_ancilla=N_ancilla)
    # +0
    # id
    identity = (ket2dm(basis.ancilla.evencat) + ket2dm(basis.ancilla.oddcat))/2
    initial_state = tensor(
        identity,
        ket2dm(basis.data.zero),
    )
    cnot = CNOTFockQutip(
        nbar=nbar,
        k2=k2d,
        k2a=k2a,
        k1=k1d,
        k1a=k1a,
        gate_time=gate_time,
        truncature=N_data,
        N_ancilla=N_ancilla,
        initial_state=initial_state,
        initial_state_name="id0"
        # kphi=1e-4,
    )

    idleidle = IdleIdleGateFock(
        nbar=nbar,
        k2=k2d,
        k2a=k2a,
        k1=k1d,
        k1a=k1a,
        gate_time=gate_time,
        truncature=N_data,
        N_ancilla=N_ancilla,
        initial_state=initial_state,
        initial_state_name="CNOTid0",
    )
    print(f'{k2a=}')
    rho = initial_state
    for i in range(k2a):
        print(f'{i=}')

        # prep
        idleidle.initial_state = rho
        idleidle.simulate()

        # CNOT
        print('1st')
        cnot.initial_state = idleidle.rho
        cnot.initial_state_name = 'CNOTid0'
        res = cnot.simulate()

        # measure
        idleidle.initial_state = cnot.rho
        idleidle.simulate()

        rho_data = idleidle.rho.ptrace(1)
        # Jx = basis.data.Jx
        # pX = np.real(np.trace(rho_data * tensor(ket2dm(qutipbasis(2, 1)), qeye(N))))
        pX = np.real(basis.data.bitflip_proba(rho_data))
        print(pX)

        rho = tensor(
            ket2dm(basis.ancilla.evencat),
            rho_data,
        )

        result = {
            'class': type(cnot).__name__,
            'index': i,
            'nbar': nbar,
            'k2a': k2a,
            'k1d': k1d,
            'N': N,
            'k2d': k2d,
            'k1a': k1a,
            'gate_time': gate_time,
            'pX': pX,
        }
        prefix = 'single_cnot_repeated_'
        np.save(
            f'{db_path}/experiments/physical_gate/{prefix}{index}_{i}.npy',
            result,
            allow_pickle=True,
        )
        d = np.load(
            f'{db_path}/experiments/physical_gate/{prefix}{index}_{i}.npy',
            allow_pickle=True,
        )
        print(d)


def gen_df(*args):
    res = []
    # prefix_l = ['', 'a', 'aa']
    # prefix_l = ['two_cnots_idle_']
    index_max_l = [100]
    for prefix, index_max in zip(args, index_max_l):
        print(prefix)
        for i in range(index_max):
            for step in range(25):
                try:
                    sres = np.load(
                        f'{db_path}/experiments/physical_gate/{prefix}{i}_{step}.npy',
                        allow_pickle=True,
                    ).item()
                    sres_class = {'class': 'CNOTFockQutip'}
                    print(sres)
                    sres_class.update(sres)
                    res.append(sres_class)
                except FileNotFoundError:
                    pass

        df = pd.DataFrame(res)
        print(f'{db_path}/experiments/physical_gate/{prefix}df.npy')
        np.save(
            f'{db_path}/experiments/physical_gate/{prefix}df.npy',
            df.to_numpy(),
            allow_pickle=True,
        )


if __name__ == "__main__":
    index = int(sys.argv[1])
    run_simu_1_cnot(index=index)
    # from multiprocessing import Queue, Pool
    # q = Queue()
    # for i in range(12):
    #     q.put((i))
    # q.put(None)
    # pool = Pool(5)
    # intermediate_data = pool.map(
    #     run_simu,
    #     iter(q.get, None),
    # )

    # for i in range(12):
    #     index = i
    #     run_simu(index=index)

    # gen_df()
