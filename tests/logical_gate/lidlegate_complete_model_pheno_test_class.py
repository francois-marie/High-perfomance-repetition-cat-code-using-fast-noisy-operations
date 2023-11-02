import numpy as np
import pytest

from qsim.helpers import search_simulation
from qsim.logical_gate.lidle import (
    LIdleGatePhenomenological,
    perfect_CNOT_transition_matrix,
    reconvergence,
)
from qsim.utils.utils import exponentiel_proba, optimal_gate_time
from tests.logical_gate.lgate_test_class import LGateTestClass


# k1d chosen to be close to Phenomenological p=1e-2 q=1e-2
@pytest.mark.parametrize('k1d', [np.pi * 4 * 1e-2**2])
@pytest.mark.parametrize('k1a', [np.pi * 4 * 1e-2**2])
@pytest.mark.parametrize('nbar', [4])
@pytest.mark.parametrize('k2d', [1])
@pytest.mark.parametrize('k2a', [1])
@pytest.mark.parametrize('distance', [3])
@pytest.mark.parametrize('N_data', [5])
@pytest.mark.parametrize('N_ancilla', [2])
@pytest.mark.parametrize('logic_fail_max', [1000])
class LIdleGateCompleteModelPhenoTestClass(LGateTestClass):
    @pytest.fixture(autouse=True)
    def setup_gate(
        self,
        distance: int,
        k1d: float,
        k1a: float,
        nbar: int,
        num_rounds: int,
        k2a: int,
        k2d: int,
        N_data: int,
        N_ancilla: int,
        logic_fail_max: int,
    ):
        gate_time = optimal_gate_time(nbar=nbar, k1=k1d, k2=k2d)

        self.gate = self.gate_cls(
            distance=distance,
            k1d=k1d,
            nbar=nbar,
            k2a=k2a,
            k2d=k2d,
            N_data=N_data,
            N_ancilla=N_ancilla,
            logic_fail_max=logic_fail_max,
            k1a=k1a,
            gate_time=gate_time,
        )

    @pytest.mark.slow
    def test_simulate(self):
        super().test_simulate()
        p_l = 1 - self.gate.results["state"]["outcome"]["Z"]
        num_trajectories = self.gate.results['state']['num_trajectories']['Z']
        std = sqrt(p_l * (1 - p_l) / num_trajectories)
        # assert p_l - 1.96 * std < 1.90e-1 < p_l + 1.96 * std

    @pytest.mark.slow
    def test_reconvergence(self, N: int = 10_000):
        transition_matrix_l = [
            self.gate.idle_pr_ancilla.transition_matrix,
            self.gate.idle_pr.transition_matrix,
            self.gate.idle_tr.transition_matrix,
        ]
        dims_l = [self.gate.N_ancilla, self.gate.N_data, self.gate.N_data]
        target_p_l = [
            self.gate.one_photon_loss_ancilla,
            self.gate.one_photon_loss_data,
            self.gate.one_photon_loss_data_tr,
        ]

        for i, (tm, dim, p_theo) in enumerate(
            zip(transition_matrix_l, dims_l, target_p_l)
        ):
            print(f'{i=}')
            l = [0] * N
            l = reconvergence(tm, l)
            p_exp = sum(l) / dim / N
            # print(f'{p_exp}')
            std = sqrt(p_exp * (1 - p_exp) / N)
            # print(f'{1.96*std}')
            # print(f'{p_theo}')
            assert p_exp - 1.96 * std <= p_theo <= p_exp + 1.96 * std

    @pytest.mark.slow
    def test_single_mode_partial_reconvergence(self, N: int = 10_000):
        self.gate.data_qubit_state = [0] * N
        for index in range(N):
            self.gate.single_mode_partial_reconvergence(index=index)
        dim = self.gate.N_data
        p_exp = sum(self.gate.data_qubit_state) / dim / N
        print(f'{p_exp}')
        std = sqrt(p_exp * (1 - p_exp) / N)
        print(f'{1.96*std}')
        p_theo = self.gate.one_photon_loss_data
        print(f'{p_theo}')
        assert p_exp - 1.96 * std <= p_theo <= p_exp + 1.96 * std

    @pytest.mark.slow
    def test_apply_cnot_error(self, N: int = 10_000):
        syndrome = [0] * N
        self.gate.data_qubit_state = [0] * N
        for i in range(N):
            self.gate.apply_cnot_error(syndrome=syndrome, i=i, _cnot_round=0)

        p_exp_l = [0, 0, 0]
        for i in range(N):
            if (
                self.gate.data_qubit_state[i] == self.gate.N_data
                and syndrome[i] == self.gate.N_ancilla
            ):
                p_exp_l[2] += 1
            elif (
                self.gate.data_qubit_state[i] == self.gate.N_data
                and syndrome[i] == 0
            ):
                p_exp_l[1] += 1
            if (
                self.gate.data_qubit_state[i] == 0
                and syndrome[i] == self.gate.N_ancilla
            ):
                p_exp_l[0] += 1

        p_exp_l = [p / N for p in p_exp_l]
        p_theo_l = [
            self.gate.physical_gates['CNOT'].pZI,
            self.gate.physical_gates['CNOT'].pIZ,
            self.gate.physical_gates['CNOT'].pZZ,
        ]
        print(f'{p_exp_l}')
        print(f'{p_theo_l}')
        for i in range(3):
            std = sqrt(p_exp_l[i] * (1 - p_exp_l[i]) / N)
            assert (
                p_exp_l[i] - 1.96 * std
                <= p_theo_l[i]
                <= p_exp_l[i] + 1.96 * std
            )

    @pytest.mark.slow
    def test_graph_weights(self):
        labels = ['pvin', 'pvout', 'pdiag', 'phor', 'pbord', 'p_prepw']
        self.gate.generate_error_correction_graph()
        k1 = self.gate.k1d
        nbar = self.gate.nbar
        gate_time = self.gate.gate_time
        p = 1e-2
        distance = self.gate.distance
        res = search_simulation(
            LIdleGatePRA,
            **{'p': p, 'distance': distance, 'num_rounds': distance},
        )
        res = res[0]
        tol = 5 / 100
        for label in labels:
            # print(label)
            # print(self.gate.graph_weights[label])
            # print(res['graph_weights'][
            #     label
            # ])
            assert (
                self.gate.graph_weights[label] * (1 - tol)
                <= res['graph_weights'][label]
                <= self.gate.graph_weights[label] * (1 + tol)
            )

    def test_perfect_CNOT_transition_matrix(self):
        tm = self.gate.cnot.transition_matrix
        tm[tm > 0.9] = 1
        tm[tm < 0.1] = 0
        tm = tm.astype('int8')
        perfect_CNOT_tm = perfect_CNOT_transition_matrix(
            N_ancilla=self.gate.N_ancilla, N_data=self.gate.N_data
        )
        assert (perfect_CNOT_tm == tm).all()
