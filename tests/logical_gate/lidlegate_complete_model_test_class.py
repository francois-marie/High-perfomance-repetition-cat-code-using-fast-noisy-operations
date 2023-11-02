import numpy as np
import pytest

from qsim.helpers import search_simulation
from qsim.logical_gate.lidle import (
    LIdleGatePRA,
    perfect_CNOT_transition_matrix,
    qubit_phase_flip,
    reconvergence,
)
from qsim.utils.utils import exponentiel_proba, optimal_gate_time
from tests.logical_gate.lgate_test_class import LGateTestClass


def get_tol(p_exp: float, N: int) -> float:
    return 1.96 * np.sqrt(p_exp * (1 - p_exp) / N)


# k1d chosen to be close to PRA p=1e-2
@pytest.mark.parametrize('k1d', [np.pi * 4 * 5e-3**2])
@pytest.mark.parametrize('k1a', [np.pi * 4 * 5e-3**2])
@pytest.mark.parametrize('nbar', [4])
@pytest.mark.parametrize('k2d', [1])
@pytest.mark.parametrize('k2a', [1])
@pytest.mark.parametrize('distance', [5])
@pytest.mark.parametrize('N_data', [5])
@pytest.mark.parametrize('N_ancilla', [3])
@pytest.mark.parametrize('logic_fail_max', [1000])
class LIdleGateCompleteModelTestClass(LGateTestClass):
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
        # gate_time = optimal_gate_time(nbar=nbar, k1=k1d, k2=k2d)
        gate_time = 1

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
        std = np.sqrt(p_l * (1 - p_l) / num_trajectories)
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
            # self.gate.one_photon_loss_data_tr,
            0,
        ]

        for i, (tm, dim, p_theo) in enumerate(
            zip(transition_matrix_l, dims_l, target_p_l)
        ):
            print(f'{i=}')
            l = [0] * N
            l = reconvergence(tm, l)
            p_exp = sum(l) / dim / N
            # print(f'{p_exp}')
            std = np.sqrt(p_exp * (1 - p_exp) / N)
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
        std = np.sqrt(p_exp * (1 - p_exp) / N)
        print(f'{1.96*std}')
        p_theo = self.gate.one_photon_loss_data
        print(f'{p_theo}')
        assert p_exp - 1.96 * std <= p_theo <= p_exp + 1.96 * std

    # @pytest.mark.slow
    def test_simulate(self, n: int = 2):
        assert any(self.apply_cnot_error_test() for _ in range(n))

    def apply_cnot_error_test(self, N: int = 10_000):
        syndrome = [0] * N
        self.gate.data_qubit_state = [0] * N
        print(syndrome[:10])
        for i in range(N):
            self.gate.apply_cnot_error(syndrome=syndrome, i=i, _cnot_round=0)
            if i < 11:
                print(syndrome[:10])

        p_exp_l = [0, 0, 0]
        for i in range(N):
            if (
                self.gate.data_qubit_state[i] >= self.gate.N_data
                and syndrome[i] >= self.gate.N_ancilla
            ):
                p_exp_l[2] += 1
            elif (
                self.gate.data_qubit_state[i] >= self.gate.N_data
                and syndrome[i] < self.gate.N_ancilla
            ):
                p_exp_l[1] += 1
            elif (
                self.gate.data_qubit_state[i] < self.gate.N_data
                and syndrome[i] >= self.gate.N_ancilla
            ):
                p_exp_l[0] += 1
            elif (
                self.gate.data_qubit_state[i] < self.gate.N_data
                and syndrome[i] < self.gate.N_ancilla
            ):
                pass
            else:
                raise ValueError("wrong")

        p_exp_l = [p / N for p in p_exp_l]
        p_theo_l = [
            self.gate.physical_gates['CNOT'].pZI,
            self.gate.physical_gates['CNOT'].pIZ,
            self.gate.physical_gates['CNOT'].pZZ,
        ]
        print(f'{p_exp_l=}')
        print(f'{p_theo_l=}')
        return any(
            p_exp_l[i] - get_tol(p_exp=p_exp_l[i], N=N)
            <= p_theo_l[i]
            <= p_exp_l[i] + get_tol(p_exp=p_exp_l[i], N=N)
            for i in range(3)
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
        # print(tm)
        # print(tm[:, 0])
        # print(tm[0, :])
        threshold = 0.4
        tm[tm > threshold] = 1
        tm[tm < 1 - threshold] = 0
        tm = tm.astype('int8')
        # print(tm)
        # print(tm[:, 0])
        # print(tm[0, :])
        perfect_CNOT_tm = perfect_CNOT_transition_matrix(
            N_ancilla=self.gate.N_ancilla, N_data=self.gate.N_data
        )
        # print(perfect_CNOT_tm)
        # print(perfect_CNOT_tm[:, 0])
        # print(perfect_CNOT_tm[0, :])
        assert (perfect_CNOT_tm == perfect_CNOT_tm).all()

    # qubits

    def test_data_qubit_phase_flip(self):
        lidle = self.gate
        lidle.N_data = 10
        # qubits
        lidle.data_qubit_state = [0, 0, 0]
        lidle.data_qubit_phase_flip(index=0)
        assert lidle.data_qubit_state == [lidle.N_data, 0, 0]
        lidle.data_qubit_phase_flip(index=0)
        assert lidle.data_qubit_state == [0, 0, 0]

        lidle.data_qubit_state = [2, 10, 4]
        lidle.data_qubit_phase_flip(index=0)
        lidle.data_qubit_phase_flip(index=1)
        lidle.data_qubit_phase_flip(index=2)
        assert lidle.data_qubit_state == [12, 0, 14]

        lidle.data_qubit_state = [20, 0, 0]
        try:
            lidle.data_qubit_phase_flip(index=0)
        except ValueError:
            pass

    def test_get_data_qubit_state(self):
        lidle = self.gate
        lidle.N_data = 10
        lidle.data_qubit_state = [0, 14, 10]
        assert lidle.get_data_qubit_state(index=0) == 0
        assert lidle.get_data_qubit_state(index=1) == 1
        assert lidle.get_data_qubit_state(index=2) == 1
        lidle.distance = 3
        assert lidle.get_data_qubit() == [0, 1, 1]
        lidle.data_qubit_state = [0, 0, 0]
        assert lidle.get_data_qubit() == [0, 0, 0]

    def test_qubit_phase_flip(self):
        lidle = self.gate
        values = [1, 1, 1]
        assert qubit_phase_flip(index=0, values=values, mode_size=1) == [
            0,
            1,
            1,
        ]
        values = [1, 1, 0]
        assert qubit_phase_flip(index=2, values=values, mode_size=1) == [
            1,
            1,
            1,
        ]
        values = [2, 1, 0]
        try:
            qubit_phase_flip(index=0, values=values, mode_size=1)
        except ValueError:
            pass

    # gauges

    def test_get_data_gauge_state(self):
        lidle = self.gate
        lidle.N_data = 10
        lidle.distance = 3
        lidle.data_qubit_state = [0, 14, 10]
        assert lidle.get_data_gauge_state(index=0) == 0
        assert lidle.get_data_gauge_state(index=1) == 4
        assert lidle.get_data_gauge_state(index=2) == 0
        assert lidle.get_data_gauge() == [0, 4, 0]
        lidle.data_qubit_state = [0, 0, 0]
        assert lidle.get_data_gauge() == [0, 0, 0]

    def test_update_single_data_mode(self):
        lidle = self.gate
        lidle.N_data = 10
        lidle.data_qubit_state = [0, 14, 10]
        lidle.update_single_data_mode(gauge=0, index=0)
        assert lidle.data_qubit_state == [0, 14, 10]
        lidle.update_single_data_mode(gauge=0, index=1)
        assert lidle.data_qubit_state == [0, 10, 10]
        lidle.update_single_data_mode(gauge=0, index=2)
        assert lidle.data_qubit_state == [0, 10, 10]
        lidle.update_single_data_mode(gauge=5, index=2)
        assert lidle.data_qubit_state == [0, 10, 15]
        try:
            lidle.update_single_data_mode(gauge=lidle.N_data + 1, index=0)
        except ValueError:
            pass

    def test_update_data_modes_from_gauges(self):
        lidle = self.gate
        lidle.N_data = 10
        lidle.distance = 3
        lidle.data_qubit_state = [0, 14, 10]
        lidle.update_data_modes_from_gauges(gauges=[1, 2, 3])
        assert lidle.data_qubit_state == [1, 12, 13]

        # reconvergence

    def test_reconvergence(self):
        lidle = self.gate
        gauges = [0, 0, 0]
        new_gauges = reconvergence(
            transition_matrix=lidle.idle_pr.transition_matrix,
            data_qubit_state=gauges,
        )
        assert new_gauges == gauges
        # lidle.idle_pr.transition_matrix

    def test_get_syndrome_gauge_from_joint_state(self):
        lidle = self.gate
        lidle.N_ancilla = 10
        lidle.N_data = 10
        joint_state_l = [0, 12, 3, 22, 205]
        syndrome_gauge_l = [(0, 0), (0, 2), (0, 3), (1, 2), (10, 5)]
        for joint_state, syndrome_gauge in zip(joint_state_l, syndrome_gauge_l):
            assert (
                lidle.get_syndrome_gauge_from_joint_state(
                    joint_state=joint_state
                )
                == syndrome_gauge
            )
