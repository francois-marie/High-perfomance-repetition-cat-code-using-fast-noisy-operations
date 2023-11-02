import sys
from qsim.basis.sfb import SFB
from qsim.physical_gate.idle import IdleGateSFB
from qsim.helpers import simulate, _default_db_path


def run_gate(
    nbar: int,
    k2: float,
    k1: float,
    gate_time: float,
    N: int,
    state_index: int,
    Basis= SFB,
    Gate = IdleGateSFB,
    save:bool=False,
):
    basis = Basis(nbar=nbar, d=N)
    cardinal_states, cardinal_names = basis.tomography_one_qubit()
    if not save:
        gate = Gate(
            nbar=nbar,
            k1=k1,
            k2=k2,
            gate_time=gate_time,
        )
        gate.simulate(
            initial_state=cardinal_states[state_index],
            truncature=N,
            initial_state_name=cardinal_names[state_index],
        )
    else:
        path = _default_db_path(Gate)
        paths = path.split('.')
        final_path = paths[0]+f'_{state_index}'+paths[1]
        simulate(
            Gate,
            params=[
                {
                    'nbar': nbar,
                    'k1': k1,
                    'k2': k2,
                    'gate_time': gate_time,
                    'initial_state': cardinal_states[state_index],
                    'truncature': N,
                    'initial_state_name': cardinal_names[state_index],
                }
            ],
            db_path=final_path,
            n_proc=5,
        )


if __name__ == "__main__":
    nbar = int(sys.argv[1])
    k1i = int(sys.argv[2])
    state_index = int(sys.argv[3])
    k2 = 1
    gate_time = 1
    N = 10

    k1_l = [0, 1e-5, 1e-4, 1e-3]
    k1 = k1_l[k1i]
    run_gate(
        nbar=nbar,
        k2=k2,
        k1=k1,
        gate_time=gate_time,
        N=N,
        state_index=state_index,
    )
