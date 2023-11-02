from time import time
from typing import Type

import numpy as np

from qsim.helpers import (
    first_simulation,
    gen_params_simulate,
    search_simulation,
    simulate,
)
from qsim.logical_gate.lidle import (
    LIdleGate,
    LIdleGatePhenomenological,
    LIdleGatePRA,
)
from qsim.logical_gate.phenomenological import (
    CheckMatrixPhenomenological,
    CheckMatrixPhenomenologicalold,
)
from qsim.utils.utils import error_bar


def main(
    Gate: Type[LIdleGate] = CheckMatrixPhenomenological,
    p_l=np.logspace(-2, -1, 11),
    num_rounds_l=np.arange(20, 101, 4),
    distance_l=range(3, 8, 2),
    q_l=None,
    num_proc=10,
):
    # num_rounds_l = np.linspace(10, 20, 11)
    if Gate.__name__ in [
        'CheckMatrixPhenomenological',
        'CheckMatrixPhenomenologicalold',
        'LIdleGatePhenomenological',
    ]:
        parameters = (
            gen_params_simulate(
                distance_l=distance_l, p_l=p_l, num_rounds_l=num_rounds_l
            )
            if q_l is None
            else gen_params_simulate(
                distance_l=distance_l,
                p_l=p_l,
                q_l=q_l,
                num_rounds_l=num_rounds_l,
            )
        )
    elif Gate.__name__ == 'LIdleGatePRA':
        parameters = gen_params_simulate(
            distance_l=distance_l, p_l=p_l, num_rounds_l=num_rounds_l
        )
    else:
        raise ValueError('Gate not implemented yet')

    t_start = time()
    p = simulate(Gate, parameters, n_proc=num_proc)
    p.join()
    print('time')
    print(f'{time() - t_start:.2f}s')


def main_search(
    Gate=CheckMatrixPhenomenological,
    p_l=np.logspace(-2, -1, 11),
    num_rounds_l=np.arange(20, 101, 4),
    distance_l=range(3, 8, 2),
    q_l=None,
):
    # num_rounds_l = [i for i in range(10, 21)]
    if Gate.__name__ in [
        'CheckMatrixPhenomenological',
        'CheckMatrixPhenomenologicalold',
        'LIdleGatePhenomenological',
    ]:
        parameters = (
            gen_params_simulate(
                distance_l=distance_l, p_l=p_l, num_rounds_l=num_rounds_l
            )
            if q_l is None
            else gen_params_simulate(
                distance_l=distance_l,
                p_l=p_l,
                q_l=q_l,
                num_rounds_l=num_rounds_l,
            )
        )
    elif Gate.__name__ == 'LIdleGatePRA':
        parameters = gen_params_simulate(
            distance_l=distance_l, p_l=p_l, num_rounds_l=num_rounds_l
        )
    else:
        raise ValueError('Gate not implemented yet')
    t_start = time()
    print(parameters)
    for param in parameters:
        print(param)
        res = search_simulation(Gate, **param)
        print(f'\t{len(res)=}')
        if len(res) == 1:
            res = res[0]
            # print(res)
            outcome = res['state']['outcome']['Z']
            num_trajectories = res['state']['num_trajectories']['Z']
            err = error_bar(N=num_trajectories, p=1 - outcome)
            print(f'p={1 - outcome} + /- {err}')
    print('time')
    print(f'{time() - t_start:.2f}s')


if __name__ == '__main__':
    p = 1e-2
    p_l = [p]
    distance = 3
    distance_l = [distance]
    num_rounds_l = [distance]

    gate_l = [LIdleGatePhenomenological]
    for Gate in gate_l:
        print(Gate.__name__)
        main_search(
            Gate=Gate,
            p_l=p_l,
            distance_l=distance_l,
            num_rounds_l=[distance + 1]
            if Gate.__name__ == 'CheckMatrixPhenomenological'
            else [distance],
        )
        it = first_simulation(Gate)
        print('keys of document')
        print(it.keys())
        print('first document of db')
        print(it.items())

    Gate = LIdleGatePhenomenological
    obj = Gate(
        p=p_l[0],
        q=p_l[0],
        distance=distance_l[0],
        num_rounds=num_rounds_l[0],
        logic_fail_max=500,
    )
    obj.simulate()

    # main(
    #     Gate=Gate,
    #     p_l=p_l,
    #     distance_l=distance_l,
    #     num_rounds_l=[
    #         distance + 1] if Gate.__name__ == 'CheckMatrixPhenomenological' else [distance]
    # )
    # main_search(
    #     Gate=Gate,
    #     p_l=p_l,
    #     distance_l=distance_l,
    #     num_rounds_l=[
    #         distance + 1] if Gate.__name__ == 'CheckMatrixPhenomenological' else [distance]
    # )
