# pylint: disable=too-many-lines
from abc import abstractmethod
from curses import KEY_B2
from functools import cached_property
from typing import Any, Dict, Optional, Tuple

import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt
from numpy.random import choice, rand
from pymatching import Matching

from experiments.physical_gate.parity_measurement_reduced_model import (
    gen_cnot_onephotonloss,
    gen_res_parity_measurement,
    update_systems_parity_measurement,
)
from qsim.logical_gate.lgate import OneQubitLGate
from qsim.physical_gate.cnot import (
    CNOTSFBPhaseFlips,
    CNOTSFBReducedReducedModel,
)
from qsim.physical_gate.idle import (
    IdleGateFake,
    IdleGateSFBPhaseFlips,
    IdleGateSFBReducedReducedModel,
)
from qsim.physical_gate.QEC_parity_measurement import ParityMeasurementGate
from qsim.utils.error_correction_graph import ErrorCorrectionGraph
from qsim.utils.error_model import (
    CNOTAnalyticalAsymmetric,
    CNOTCompleteModelNumericalErrorFromCorrelations,
    CNOTPRAp3p,
    CNOTReducedModelNumericalErrorFromCorrelations,
    ErrorModel,
    OneQubitErrorModel,
    TwoQubitErrorModel,
)
from qsim.utils.utils import (
    combine,
    error_bar,
    exponentiel_proba,
    one_photon_loss_phaseflip,
    optimal_gate_time,
)


class LIdleGate(OneQubitLGate):
    """Simulation of the idle_tr Logical gate."""

    def __init__(
        self,
        distance: int,
        physical_gates: Optional[Dict[str, ErrorModel]] = None,
        num_rounds=None,
        verbose=False,
        traj_max=500_000,
        logic_fail_max=1_000,
        debug=False,
    ):
        if num_rounds is None:
            num_rounds = distance
        if physical_gates is None:
            physical_gates = {
                'Preparation': OneQubitErrorModel(),
                'IdleGate': OneQubitErrorModel(),
                'MeasureX': OneQubitErrorModel(),
                'CNOT': TwoQubitErrorModel(),
                'IdleGateInterCNOTData': OneQubitErrorModel(),
                'IdleGateInterCNOTAncilla': OneQubitErrorModel(),
            }

        super().__init__(distance, physical_gates)
        self.num_rounds = num_rounds
        self.verbose = verbose
        self.error_correction_graph = None
        self.traj_max = traj_max
        self.logic_fail_max = logic_fail_max
        self.num_neighbours = None
        self.error_counts = []
        self.graph_weights = None
        self.pvin = None
        self.p0 = None
        self.pvout = None
        self.pdiag = None
        self.phor = None
        self.pbor = None
        self.p_prepw = None
        self.p3 = None
        self.sampler_error = {'debug': debug}

    def _get_results(self) -> Dict[str, Any]:
        return {
            **super()._get_results(),
            'num_rounds': self.num_rounds,
            'graph_weights': self.graph_weights,
        }

    def __str__(self):
        return super().__str__() + f'\tnum_rounds: {self.num_rounds} \n'

    def _generate_error_correction_graph(self):  # pylint: disable=invalid-name
        pass

    def _post_weights(self):
        pass

    def generate_error_correction_graph(self):
        self.weights_error_correction_graph()
        self._generate_error_correction_graph()
        self.graph_weights = {
            'pvin': self.pvin,
            'pvout': self.pvout,
            'pdiag': self.pdiag,
            'phor': self.phor,
            'pbord': self.pbord,
            'p_prepw': self.p_prepw,
        }
        self.error_correction_graph = ErrorCorrectionGraph(
            num_rows=self.num_rounds + 1,
            num_columns=self.distance - 1,
            graph_weights=self.graph_weights,
        )

        self.error_correction_graph.add_weight()
        self._post_weights()
        self.m = Matching(self.error_correction_graph.g)

    def reinit_graph_weights(self):
        self.pvin = None
        self.p0 = None
        self.pvout = None
        self.pdiag = None
        self.phor = None
        self.pbor = None
        self.p_prepw = None
        self.p3 = None

    def weights_error_correction_graph(self):
        # noise models
        # preparation
        p_prep = self.physical_gates['Preparation'].pZ
        self.p_prep = p_prep
        p_prepw = self.physical_gates['IdleGate'].pZ
        # measurement
        p_meas = self.physical_gates['MeasureX'].pZ
        self.p_meas = p_meas
        p_measw = self.physical_gates['IdleGate'].pZ
        p_idle = self.physical_gates['IdleGate'].pZ
        p_idle_inter_CNOT_data = self.physical_gates['IdleGateInterCNOTData'].pZ
        self.p_idle_inter_CNOT_data = p_idle_inter_CNOT_data
        p_idle_inter_CNOT_ancilla = self.physical_gates[
            'IdleGateInterCNOTAncilla'
        ].pZ
        self.p_idle_inter_CNOT_ancilla = p_idle_inter_CNOT_ancilla
        self.p_idle = p_idle
        self.p_measw = p_measw
        # cnot
        p_CXw = self.physical_gates['IdleGate'].pZ
        self.p_CXw = p_CXw
        p_CX_Z2 = self.physical_gates['CNOT'].pIZ
        p_CX_Z1 = self.physical_gates['CNOT'].pZI
        p_CX_Z1Z2 = self.physical_gates['CNOT'].pZZ

        if self.p_prepw is None:
            self.p_prepw = p_prepw
        # compute effective error probabilities

        # vertical edges
        # vertical edges starting from first vertical edge
        # to second to last
        if self.pvin is None:
            self.pvin = combine([p_prepw, p_CX_Z1Z2])
        # vertical edges starting from second vertical edge
        # to last
        if self.pvout is None:
            self.pvout = combine([p_CX_Z2, p_measw])
        # diagonal edges
        if self.pdiag is None:
            self.pdiag = combine([p_CX_Z2, p_CX_Z1Z2, p_idle_inter_CNOT_data])

        # error on the top data qubit waiting
        # during first row of cnot
        # self.p0 = combine([p_prepw, p_CXw, p_idle])
        # only on sampler, not used on ECG
        self.p0 = combine([p_prepw, p_CXw])  # only on sampler, not used on ECG
        # error on the bottom data qubit waiting
        # during second row of cnot
        # self.p3 = combine([p_CXw, p_measw, p_idle])
        # only on sampler, not used on ECG
        self.p3 = combine([p_CXw, p_measw])  # only on sampler, not used on ECG
        self.phor = combine(
            [p_prep, p_CX_Z1, p_CX_Z1, p_meas, p_idle_inter_CNOT_ancilla]
        )
        self.pbord = combine([p_CXw, p_idle_inter_CNOT_data])
        # No p_idle on p0 & p3.

    def syndrome_to_detection_event(self, syndrome: np.ndarray):
        detection_events = np.copy(syndrome)
        detection_events[1:, :] = syndrome[1:, :] ^ syndrome[0:-1, :]
        self.detection_events = detection_events
        detection_event_list = []
        for x in detection_events:
            detection_event_list += list(x)
        detection_event_list.append(sum(detection_event_list) % 2)
        self.detection_event_list = detection_event_list

    def decode(self, syndrome: np.ndarray):
        self.syndrome_to_detection_event(syndrome=syndrome)
        self.error = self.m.decode(
            self.detection_event_list,
            num_neighbours=self.num_neighbours,
        )
        self.verb(f'Syndromes: {syndrome}')
        self.verb(f'error : {self.error}')

    def _simulate_Z(self):
        if self.error_correction_graph is None:
            self.generate_error_correction_graph()
        self.monte_carlo()

    @abstractmethod
    def parity_measurement(self):
        pass

    @abstractmethod
    def perfect_parity_measurement(self):
        pass

    def sample(self):
        """Runs one instance of a quantum error correction
        scheme for a given number of rounds
        """
        if self.m is None:
            self.m = Matching(self.error_correction_graph.g)
        # re-seed the random number generator
        np.random.seed()
        # logical control qubits
        self.all_data_qubit_state_cumulative = []
        self.data_qubit_state = [0] * self.distance
        if self.verbose:
            print('initial logical control: ')
            print(self.data_qubit_state)
        self.error_count = [0, 0]
        self.time_index = 0
        syndrome_full = self.repeated_imperfect_parity_measurements()
        # last perfect measurement
        syndrome_full.append(self.perfect_parity_measurement())
        self.error_counts.append(self.error_count.copy())
        self.all_data_qubit_state_cumulative.append(
            self.data_qubit_state.copy()
        )
        self.all_data_qubit_state_cumulative = np.asarray(
            self.all_data_qubit_state_cumulative
        )
        # decode using pyMatching
        syndrome = np.asarray(syndrome_full)
        self.syndrome = syndrome
        self.decode(self.syndrome)
        for txt in [
            f'Final syndrome: {syndrome}',
            f'Final logical control: {self.data_qubit_state}',
            f'Final error: {self.error}',
        ]:
            self.verb(txt)

        if self.verbose:
            print(f'{self.data_qubit_state=}')
        return int(
            all(self.data_qubit_state == self.error)
        )  # int(data_qubit_state == error1)

    def repeated_imperfect_parity_measurements(
        self,
    ):  # pylint: disable=invalid-name
        # measurement results record
        syndrome_full = []
        self.verb('sample loop')
        # qec on controls
        # d rounds of imperfect QEC
        for index in range(self.num_rounds):
            syndrome1 = self.parity_measurement()
            syndrome_full.append(syndrome1.copy())
            self._repeated_imperfect_parity_measurements(index=index)
        return syndrome_full

    def _repeated_imperfect_parity_measurements(  # pylint: disable=invalid-name
        self, index: int
    ):
        pass

    def _monte_carlo_from_sample(self):
        return 1 - self.sample()

    def _monte_carlo_from_graph(self):
        noise, syndrome = self.m.add_noise()
        correction = self.m.decode(syndrome)
        return not np.allclose(correction, noise)

    def _monte_carlo(self):
        return self._monte_carlo_from_sample()

    def monte_carlo(
        self,
    ):
        if self.error_correction_graph is None:
            self.generate_error_correction_graph()
        if self.verbose:
            self.error_correction_graph.plot_graph()
        print(f'{self.graph_weights=}')

        num_trajectories = 0
        logical_failure = 0
        while (
            num_trajectories < self.traj_max
            and logical_failure < self.logic_fail_max
        ):
            # logical_failure += 1 - self.sample()
            logical_failure += self._monte_carlo()
            num_trajectories += 1
        outcome = 1 - logical_failure / num_trajectories
        self.outcome['Z'], self.num_trajectories['Z'] = (
            outcome,
            num_trajectories,
        )
        std = error_bar(N=num_trajectories, p=outcome)
        print(f'infidelity: {1-self.outcome["Z"]:.4f} +/- {std:.4f}')
        print(f'{self.num_trajectories["Z"]} trajectories')

    def gen_syndrome_l(self, num_shots: int) -> list:
        syndrome_l = []
        data_qubit_state_l = []
        for _ in range(num_shots):
            np.random.seed()
            # logical control qubits
            self.all_data_qubit_state_cumulative = []
            self.data_qubit_state = [0] * self.distance
            if self.verbose:
                print('initial logical control: ')
                print(self.data_qubit_state)
            self.error_count = [0, 0]
            self.time_index = 0
            syndrome_full = self.repeated_imperfect_parity_measurements()
            # last perfect measurement
            syndrome_full.append(self.perfect_parity_measurement())
            self.error_counts.append(self.error_count.copy())
            self.all_data_qubit_state_cumulative.append(
                self.data_qubit_state.copy()
            )
            self.all_data_qubit_state_cumulative = np.asarray(
                self.all_data_qubit_state_cumulative
            )
            # decode using pyMatching
            syndrome = np.asarray(syndrome_full)
            syndrome_l.append(syndrome)
            data_qubit_state_l.append(self.data_qubit_state)
        return syndrome_l, data_qubit_state_l

    def simulate_from_syndrome_l(
        self, syndrome_l: list, data_qubit_state_l: list
    ):
        if self.error_correction_graph is None:
            self.generate_error_correction_graph()
        if self.verbose:
            self.error_correction_graph.plot_graph()
        print(f'{self.graph_weights=}')

        self.data_qubit_state = [0] * self.distance
        self.all_data_qubit_state_cumulative = []
        self.error_count = [0, 0]
        self.time_index = 0
        num_trajectories = 0
        logical_failure = 0
        for syndrome, data_qubit_state in zip(syndrome_l, data_qubit_state_l):
            # logical_failure += 1 - self.sample()
            self.syndrome = syndrome
            self.data_qubit_state = data_qubit_state
            self.decode(self.syndrome)
            for txt in [
                f'Final syndrome: {syndrome}',
                f'Final logical control: {self.data_qubit_state}',
                f'Final error: {self.error}',
            ]:
                self.verb(txt)

            if self.verbose:
                print(f'{self.data_qubit_state=}')
            sample = int(
                all(self.data_qubit_state == self.error)
            )  # int(data_qubit_state == error1)
            logical_failure += 1 - sample
            num_trajectories += 1
        outcome = 1 - logical_failure / num_trajectories
        self.outcome['Z'], self.num_trajectories['Z'] = (
            outcome,
            num_trajectories,
        )
        std = error_bar(N=num_trajectories, p=outcome)
        print(f'infidelity: {1-self.outcome["Z"]:.4f} +/- {std:.4f}')
        print(f'{self.num_trajectories["Z"]} trajectories')


def add_sampler_error_key(
    sampler_error: dict, error_name: str, legend: str, value: float
):
    if sampler_error['debug']:
        if error_name not in sampler_error.keys():
            sampler_error.update(
                {error_name: {'legend': legend, 'value': value, 'spots': []}}
            )


def add_sampler_error_spot(
    sampler_error: dict, error_name: str, time_index: int, qubit_index: int
):
    if sampler_error['debug']:
        sampler_error[error_name]['spots'].append([time_index, qubit_index])


class LIdleGateErrorModel(LIdleGate):
    """Logical idle_tr Gate with error sampled from error models."""

    def parity_measurement(self) -> list:
        syndrome = []
        # first error on top
        # p0 : p_prepw + p_CXw
        self.data_qubit_state[-1] ^= rand() < self.p0
        add_sampler_error_key(
            sampler_error=self.sampler_error,
            error_name='p0',
            legend='$p_0 = p_{prepw} + p_{CXw}$',
            value=self.p0,
        )
        add_sampler_error_spot(
            sampler_error=self.sampler_error,
            error_name='p0',
            time_index=self.time_index,
            qubit_index=self.distance - 1,
        )
        self.data_qubit_state[-1] ^= rand() < self.p_idle_inter_CNOT_data
        add_sampler_error_key(
            sampler_error=self.sampler_error,
            error_name='p_idle_inter_CNOT_data',
            legend='$p_{idle}',
            value=self.p_idle_inter_CNOT_data,
        )
        add_sampler_error_spot(
            sampler_error=self.sampler_error,
            error_name='p_idle_inter_CNOT_data',
            time_index=self.time_index + 2,
            qubit_index=self.distance - 1,
        )

        # cnot errors
        pZ1, pZ2, pZ1Z2 = (
            self.physical_gates['CNOT'].pZI,
            self.physical_gates['CNOT'].pIZ,
            self.physical_gates['CNOT'].pZZ,
        )
        P_CNOT = (
            1 - pZ1 - pZ2 - pZ1Z2,
            pZ2,
            pZ1,
            pZ1Z2,
        )

        for i in range(self.distance - 1):
            # indices
            qubit1 = self.distance - 1 - i
            qubit2 = self.distance - 1 - i - 1

            # draw cnot errors
            CNOT_err_1 = choice(
                range(4), p=P_CNOT
            )  # Draw random CNOT error according to the noise model
            CNOT_err_2 = choice(
                range(4), p=P_CNOT
            )  # Draw random CNOT error according to the noise model

            # apply input errors
            # top
            if CNOT_err_1 == 3:  # Z1Z2
                self.data_qubit_state[qubit1] ^= 1
            add_sampler_error_key(
                sampler_error=self.sampler_error,
                error_name='pZZ',
                legend='$p_{ZZ}$',
                value=pZ1Z2,
            )
            add_sampler_error_spot(
                sampler_error=self.sampler_error,
                error_name='pZZ',
                time_index=self.time_index + 3,
                qubit_index=qubit1,
            )
            # bottom
            # self.pvin: p_prepw + p_CX_Z1Z2
            if rand() < self.p_prepw:  # p1 : p_prepw + p_CX_Z1Z2
                # print('error via prep wait')
                self.data_qubit_state[qubit2] ^= 1
            add_sampler_error_key(
                sampler_error=self.sampler_error,
                error_name='p_prepw',
                legend='$p_{prepw}$',
                value=self.p_prepw,
            )
            add_sampler_error_spot(
                sampler_error=self.sampler_error,
                error_name='p_prepw',
                time_index=self.time_index,
                qubit_index=qubit2,
            )
            if CNOT_err_2 == 3:  # Z1Z2
                self.data_qubit_state[qubit2] ^= 1
            add_sampler_error_key(
                sampler_error=self.sampler_error,
                error_name='pZZ',
                legend='$p_{Z_cZ_t}$',
                value=pZ1Z2,
            )
            add_sampler_error_spot(
                sampler_error=self.sampler_error,
                error_name='pZZ',
                time_index=self.time_index + 1,
                qubit_index=qubit2,
            )

            # apply CNOTs between data qubits and ancilla
            ancilla = (
                self.data_qubit_state[qubit1] ^ self.data_qubit_state[qubit2]
            )
            # apply output errors
            # pvout: p_CX_Z2, p_measw
            if rand() < self.p_measw:  # p2 : p_CX_Z2 + p_measw
                self.data_qubit_state[qubit1] ^= 1
                # print('error via meas wait')
            add_sampler_error_key(
                sampler_error=self.sampler_error,
                error_name='p_measw',
                legend='$p_{measw}$',
                value=self.p_measw,
            )
            add_sampler_error_spot(
                sampler_error=self.sampler_error,
                error_name='p_measw',
                time_index=self.time_index + 4,
                qubit_index=qubit1,
            )
            if rand() < self.p_idle_inter_CNOT_data:
                self.data_qubit_state[qubit2] ^= 1
            add_sampler_error_key(
                sampler_error=self.sampler_error,
                error_name='p_idle_inter_CNOT_data',
                legend='$p_{idle}',
                value=self.p_idle_inter_CNOT_data,
            )
            add_sampler_error_spot(
                sampler_error=self.sampler_error,
                error_name='p_idle_inter_CNOT_data',
                time_index=self.time_index + 2,
                qubit_index=qubit2,
            )

            if CNOT_err_1 == 1:  # Z2
                self.data_qubit_state[qubit1] ^= 1
            add_sampler_error_key(
                sampler_error=self.sampler_error,
                error_name='pIZ',
                legend='$p_{Z_t}',
                value=pZ2,
            )
            add_sampler_error_spot(
                sampler_error=self.sampler_error,
                error_name='pIZ',
                time_index=self.time_index + 3,
                qubit_index=qubit1,
            )

            if CNOT_err_2 == 1:  # Z2
                self.data_qubit_state[qubit2] ^= 1
            add_sampler_error_key(
                sampler_error=self.sampler_error,
                error_name='pIZ',
                legend='$p_{Z_t}',
                value=pZ2,
            )
            add_sampler_error_spot(
                sampler_error=self.sampler_error,
                error_name='pIZ',
                time_index=self.time_index + 1,
                qubit_index=qubit2,
            )

            # apply measurement error to 'real' measurement result
            # phor: [p_prep, p_CX_Z1, p_CX_Z1,
            # p_meas, p_idle_inter_CNOT_ancilla]

            p_ancilla = combine(
                [self.p_prep, self.p_meas, self.p_idle_inter_CNOT_ancilla]
            )
            meas_error = rand() < p_ancilla
            add_sampler_error_key(
                sampler_error=self.sampler_error,
                error_name='p_ancilla',
                legend='$p_{ancilla} = p_{prep}+ p_{meas}+ p_{idle}$',
                value=p_ancilla,
            )
            add_sampler_error_spot(
                sampler_error=self.sampler_error,
                error_name='p_ancilla',
                time_index=self.time_index + 4,
                qubit_index=(qubit1 + qubit2) / 2,
            )
            ancilla ^= meas_error
            if CNOT_err_1 == 2:  # Z1
                ancilla ^= 1
            add_sampler_error_key(
                sampler_error=self.sampler_error,
                error_name='pZ1',
                legend='$p_{Z_c}',
                value=pZ1,
            )
            add_sampler_error_spot(
                sampler_error=self.sampler_error,
                error_name='pZ1',
                time_index=self.time_index + 3,
                qubit_index=(qubit1 + qubit2) / 2,
            )
            if CNOT_err_2 == 2:  # Z1
                ancilla ^= 1
            add_sampler_error_key(
                sampler_error=self.sampler_error,
                error_name='pZ1',
                legend='$p_{Z_c}',
                value=pZ1,
            )
            add_sampler_error_spot(
                sampler_error=self.sampler_error,
                error_name='pZ1',
                time_index=self.time_index + 1,
                qubit_index=(qubit1 + qubit2) / 2,
            )
            syndrome.append(ancilla)
        # last error at the bottom
        if rand() < self.p3:  # p3 : p_CXw + p_measw
            self.data_qubit_state[0] ^= 1
        add_sampler_error_key(
            sampler_error=self.sampler_error,
            error_name='p3',
            legend='$p_3 = p_{CXw} + p_{measw}$',
            value=self.p3,
        )
        add_sampler_error_spot(
            sampler_error=self.sampler_error,
            error_name='p3',
            time_index=self.time_index + 4,
            qubit_index=0,
        )
        syndrome.reverse()
        self.all_data_qubit_state_cumulative.append(
            self.data_qubit_state.copy()
        )
        self.time_index += 5
        return syndrome


    def perfect_parity_measurement(self):
        """Perfect parity measurement"""
        syndrome = [
            self.data_qubit_state[i] ^ self.data_qubit_state[i + 1]
            for i in range(self.distance - 1)
        ]
        return syndrome

    def _save_name(self):
        if isinstance(self.physical_gates['CNOT'], CNOTAnalyticalAsymmetric):
            return (
                f'{self.physical_gates["CNOT"].nbar}_{self.distance}'
                f'_{self.num_rounds}'
                f'_{self.physical_gates["CNOT"].k2d}'
                f'_{self.physical_gates["CNOT"].k2a}'
                f'_{self.physical_gates["CNOT"].gate_time}_'
                f'{self.physical_gates["CNOT"].k1d}'
                f'_{self.physical_gates["CNOT"].k1a}'
                f'_{type(self.physical_gates["CNOT"]).__name__}'
            )
        return (
            f'{self.distance}_{self.num_rounds}'
            f'_{type(self.physical_gates["CNOT"]).__name__}'
        )

    def _generate_error_correction_graph(self):  # pylint: disable=invalid-name
        pass


class LIdleGatePhenomenological(LIdleGateErrorModel):
    def __init__(
        self,
        distance: int,
        p: float,
        q: float,
        physical_gates: Optional[Dict[str, ErrorModel]] = None,
        num_rounds=None,
        verbose=False,
        traj_max=500_000,
        logic_fail_max=5_000,
    ):
        if num_rounds is None:
            num_rounds = distance
        super().__init__(
            distance,
            physical_gates,
            num_rounds,
            verbose,
            traj_max,
            logic_fail_max,
        )
        self.p = p
        self.q = q
        self.physical_gates['IdleGate'].pZ = self.p
        self.physical_gates['MeasureX'].pZ = self.q

    def _generate_error_correction_graph(self):
        # remove weights to be consistent with pheno error model
        # we use pvin, p0, phor
        self.p_prepw = self.p
        self.p_measw = 0.0

        self.p0 = self.p
        self.pvout = 0
        self.p3 = 0
        self.pbord = 0
        self.pdiag = 0
        self.phor = self.q

    def _get_results(self) -> Dict[str, Any]:
        return {
            **super()._get_results(),
            'p': self.p,
            'q': self.q,
        }


class LIdleGatePRA(LIdleGateErrorModel):
    def __init__(
        self,
        distance: int,
        p: float,
        physical_gates: Optional[Dict[str, ErrorModel]] = None,
        num_rounds=None,
        verbose=False,
        traj_max=1_000_000,
        logic_fail_max=1_000,
    ):
        if num_rounds is None:
            num_rounds = distance
        super().__init__(
            distance,
            physical_gates,
            num_rounds,
            verbose,
            traj_max,
            logic_fail_max,
        )
        self.p = p
        self.physical_gates['IdleGate'].pZ = self.p
        self.physical_gates['Preparation'].pZ = self.p
        self.physical_gates['MeasureX'].pZ = self.p
        self.physical_gates['CNOT'] = CNOTPRAp3p(p=p)

    def _generate_error_correction_graph(self):
        pass

    def _get_results(self) -> Dict[str, Any]:
        return {
            **super()._get_results(),
            'p': self.p,
        }


# pylint: disable=invalid-name
class LIdleGatePRA_perfect_ancilla(LIdleGatePRA):
    def __init__(
        self,
        distance: int,
        p: float,
        physical_gates: Optional[Dict[str, ErrorModel]] = None,
        num_rounds=None,
        traj_max=500_000,
        logic_fail_max=50_000,
    ):
        if num_rounds is None:
            num_rounds = distance
        super().__init__(
            distance,
            p,
            physical_gates,
            num_rounds,
            traj_max,
            logic_fail_max,
        )
        self.physical_gates['IdleGate'].pZ = self.p
        self.physical_gates['Preparation'] = OneQubitErrorModel()
        self.physical_gates['MeasureX'] = OneQubitErrorModel()
        self.physical_gates['CNOT'] = CNOTPRAp3p(p=p)


class LIdleGateCircuitBased(LIdleGateErrorModel):
    def __init__(
        self,
        nbar: int,
        k1d: float,
        k1a: float,
        k2d: float,
        k2a: int,
        gate_time: float,
        distance: int,
        physical_gates: Optional[Dict[str, ErrorModel]] = None,
        num_rounds=None,
        traj_max: int = 500_000,
        logic_fail_max: int = 1_000,
        verbose=False,
        debug=False,
    ):
        if num_rounds is None:
            num_rounds = distance
        super().__init__(
            distance,
            physical_gates,
            num_rounds,
            verbose,
            traj_max,
            logic_fail_max,
            debug=debug,
        )
        self.nbar = nbar
        self.k1d = k1d
        self.k2d = k2d
        self.gate_time = gate_time
        self.k2a = k2a
        self.k1a = k1a
        one_photon_loss_data = exponentiel_proba(nbar * k1d * gate_time)
        one_photon_loss_ancilla = exponentiel_proba(nbar * k1a * gate_time)

        self.physical_gates['IdleGate'].pZ = one_photon_loss_data
        self.physical_gates['Preparation'].pZ = one_photon_loss_ancilla
        self.physical_gates['MeasureX'].pZ = one_photon_loss_ancilla
        self.physical_gates['CNOT'] = CNOTAnalyticalAsymmetric(
            nbar=nbar,
            k2d=k2d,
            k2a=k2a,
            k1d=k1d,
            k1a=k1a,
            gate_time=gate_time,
        )
        self.physical_gates['IdleGateInterCNOTData'].pZ = one_photon_loss_data
        self.physical_gates[
            'IdleGateInterCNOTAncilla'
        ].pZ = one_photon_loss_ancilla

    def _generate_error_correction_graph(self):
        pass

    def _get_results(self) -> Dict[str, Any]:
        return {
            **super()._get_results(),
            'nbar': self.nbar,
            'k1a': self.k1a,
            'k2a': self.k2a,
            'gate_time': self.gate_time,
            'k1d': self.k1d,
            'k2d': self.k2d,
        }

    def __str__(self):
        return (
            super().__str__()
            + 'Physical parameters: \n'
            + f'\tnbar: {self.nbar} \n'
            f'\tk1a: {self.k1a} \n'
            f'\tk2a: {self.k2a} \n'
            f'\tgate_time: {self.gate_time} \n'
            f'\tk1d: {self.k1d} \n'
            f'\tk2d: {self.k2d} \n'
        )


class LIdleGatePRA_with_reconv(LIdleGateCircuitBased):
    def __init__(
        self,
        nbar: int,
        k1: float,
        distance: int,
        physical_gates: Optional[Dict[str, ErrorModel]] = None,
        num_rounds=None,
        traj_max: int = 100_000_000,
        logic_fail_max: int = 1_000,
        verbose=False,
        debug=False,
    ):
        k1d = k1
        k1a = k1
        k2a = 1
        k2d = 1
        gate_time = 1
        opti_gate_time = optimal_gate_time(nbar=nbar, k1=k1, k2=1)

        super().__init__(
            nbar,
            k1d,
            k1a,
            k2d,
            k2a,
            gate_time,
            distance,
            physical_gates,
            num_rounds,
            traj_max,
            logic_fail_max,
            verbose,
            debug,
        )
        self.p_PRA = exponentiel_proba(p=nbar * k1d * opti_gate_time)
        self.physical_gates['CNOT'] = CNOTPRAp3p(p=self.p_PRA)

    def _generate_error_correction_graph(self):
        p_CXw = self.p_PRA
        p_prepw = self.physical_gates['IdleGate'].pZ
        p_measw = self.physical_gates['IdleGate'].pZ
        p_idle_inter_CNOT_data = self.physical_gates['IdleGateInterCNOTData'].pZ
        # remove weights to be consistent with error model
        self.p0 = combine([p_prepw, p_CXw])  # only on sampler, not used on ECG
        # error on the bottom data qubit waiting
        # during second row of cnot
        # self.p3 = combine([p_CXw, p_measw, p_idle])
        # only on sampler, not used on ECG
        self.p3 = combine([p_CXw, p_measw])  # only on sampler, not used on ECG
        self.pbord = combine([p_CXw, p_idle_inter_CNOT_data])

    def _monte_carlo(self):
        return self._monte_carlo_from_graph()


class LIdleGatePRA_intermediary(LIdleGateCircuitBased):
    def __init__(
        self,
        nbar: int,
        k1: float,
        distance: int,
        physical_gates: Optional[Dict[str, ErrorModel]] = None,
        num_rounds=None,
        traj_max: int = 100_000_000,
        logic_fail_max: int = 1_000,
        verbose=False,
        debug=False,
    ):
        k1d = k1
        k1a = k1
        k2a = 1
        k2d = 1
        opti_gate_time = optimal_gate_time(nbar=nbar, k1=k1, k2=1)
        gate_time = max(1, opti_gate_time)

        super().__init__(
            nbar,
            k1d,
            k1a,
            k2d,
            k2a,
            gate_time,
            distance,
            physical_gates,
            num_rounds,
            traj_max,
            logic_fail_max,
            verbose,
            debug,
        )
        self.p_PRA = exponentiel_proba(p=nbar * k1d * opti_gate_time)
        self.p_fixed = exponentiel_proba(p=nbar * k1d / k2d)

        self.physical_gates['CNOT'] = CNOTPRAp3p(p=self.p_PRA)
        self.physical_gates['IdleGateInterCNOTData'].pZ = self.p_fixed
        self.physical_gates['IdleGateInterCNOTAncilla'].pZ = self.p_fixed

    def _generate_error_correction_graph(self):
        p_CXw = self.p_PRA
        p_prepw = self.physical_gates['IdleGate'].pZ
        p_measw = self.physical_gates['IdleGate'].pZ
        p_idle_inter_CNOT_data = self.physical_gates['IdleGateInterCNOTData'].pZ
        # remove weights to be consistent with error model
        self.p0 = combine([p_prepw, p_CXw])  # only on sampler, not used on ECG
        # error on the bottom data qubit waiting
        # during second row of cnot
        # self.p3 = combine([p_CXw, p_measw, p_idle])
        # only on sampler, not used on ECG
        self.p3 = combine([p_CXw, p_measw])  # only on sampler, not used on ECG
        self.pbord = combine([p_CXw, p_idle_inter_CNOT_data])

    def _monte_carlo(self):
        return self._monte_carlo_from_graph()


class LIdleGatePRALRU(LIdleGatePRA):
    def __init__(
        self,
        distance: int,
        p: float,
        physical_gates: Optional[Dict[str, ErrorModel]] = None,
        num_rounds=None,
        verbose=False,
        traj_max=1_000_000,
        logic_fail_max=1_000,
    ):
        if num_rounds is None:
            num_rounds = distance
        super().__init__(
            distance,
            p,
            physical_gates,
            num_rounds,
            verbose,
            traj_max,
            logic_fail_max,
        )
        self.physical_gates['IdleGateInterCNOTData'].pZ = self.p
        self.physical_gates['IdleGateInterCNOTAncilla'].pZ = self.p

    def _generate_error_correction_graph(self):
        pass

    def _monte_carlo(self):
        return self._monte_carlo_from_graph()


class LIdleGateFastCNOT(LIdleGateCircuitBased):
    def __init__(
        self,
        nbar: int,
        k1d: float,
        k1a: float,
        k2d: float,
        k2a: int,
        gate_time: float,
        distance: int,
        physical_gates: Optional[Dict[str, ErrorModel]] = None,
        num_rounds=None,
        traj_max: int = 500_000,
        logic_fail_max: int = 1_000,
        verbose=False,
    ):
        if num_rounds is None:
            num_rounds = distance
        super().__init__(
            nbar,
            k1d,
            k1a,
            k2d,
            k2a,
            gate_time,
            distance,
            physical_gates,
            num_rounds,
            traj_max,
            logic_fail_max,
            verbose,
        )
        self.nbar = nbar
        self.k1d = k1d
        self.k2d = k2d
        self.gate_time = gate_time
        self.k2a = k2a
        self.k1a = k1a
        one_photon_loss_data = exponentiel_proba(nbar * k1d * gate_time)

        self.physical_gates['IdleGate'].pZ = one_photon_loss_data
        self.physical_gates['Preparation'] = OneQubitErrorModel()
        self.physical_gates['MeasureX'] = OneQubitErrorModel()
        self.physical_gates['CNOT'] = CNOTAnalyticalAsymmetric(
            nbar=nbar,
            k2d=k2d,
            k2a=k2a,
            k1d=k1d,
            k1a=k1a,
            gate_time=gate_time,
        )


class LIdleGateSimu(
    LIdleGateCircuitBased
):  # pylint: disable=too-many-public-methods
    """Logical idle_tr Gate with error sampled from simulations."""

    cnot_gate_cls = CNOTSFBPhaseFlips
    idle_tr_gate_cls = IdleGateSFBPhaseFlips  # total reconvergence
    idle_pr_gate_cls = IdleGateSFBPhaseFlips  # partial reconvergence
    idle_pr_gate_cls_ancilla = (
        IdleGateSFBPhaseFlips  # partial reconvergence on ancilla
    )

    def __init__(
        self,
        nbar: int,
        k1d: float,
        k1a: float,
        k2d: float,
        k2a: int,
        gate_time: float,
        distance: int,
        physical_gates: Optional[Dict[str, ErrorModel]] = None,
        num_rounds=None,
        N_data: int = 5,
        traj_max: int = 100_000_000,
        logic_fail_max: int = 500,
        N_ancilla=None,
        init_transition_matrix=True,
        debug=False,
    ):
        super().__init__(
            nbar=nbar,
            k2d=k2d,
            k2a=k2a,
            k1d=k1d,
            k1a=k1a,
            gate_time=gate_time,
            distance=distance,
            physical_gates=physical_gates,
            num_rounds=num_rounds,
            traj_max=traj_max,
            logic_fail_max=logic_fail_max,
        )
        self.N_data = N_data
        if N_ancilla is None:
            N_ancilla = 1
        self.N_ancilla = N_ancilla

        self.init_gates()

        one_photon_loss_data = exponentiel_proba(
            self.nbar * self.k1d * self.gate_time
        )
        one_photon_loss_ancilla = exponentiel_proba(
            self.nbar * self.k1a * self.gate_time
        )
        one_photon_loss_data_tr = exponentiel_proba(
            self.nbar * self.k1d / self.k2d
        )

        self.one_photon_loss_ancilla = one_photon_loss_ancilla
        self.one_photon_loss_data = one_photon_loss_data
        self.one_photon_loss_data_tr = one_photon_loss_data_tr
        self.p_CXZ1_one_photon = self.one_photon_loss_ancilla
        self.p_ppm = self.one_photon_loss_data
        self.populate_idle_gates()
        if init_transition_matrix:
            # boolean to avoid computing transition matrices if not needed
            self.populate_cnot_gate()
        self.populate_error_models()
        self.debug = debug
        if self.debug:
            self.system_state = {'ancilla': [], 'data': []}

    def init_gates(self):
        try:
            self.cnot = self.cnot_gate_cls(
                nbar=self.nbar,
                k1=self.k1d,
                k2=self.k2d,
                k2a=self.k2a,
                k1a=self.k1a,
                gate_time=self.gate_time,
                N_ancilla=self.N_ancilla,
                truncature=self.N_data,
            )
            print('(N, N_a)')
            print(self.N_data, self.cnot.N_ancilla)
        except TypeError:
            self.cnot = self.cnot_gate_cls(
                nbar=self.nbar,
                k1=self.k1d,
                k2=self.k2d,
                k2a=self.k2a,
                k1a=self.k1a,
                gate_time=self.gate_time,
                truncature=self.N_data,
            )
        print(self.cnot)

        # partial reconvergence during parity_measurement cnot gate_time
        self.idle_pr = self.idle_pr_gate_cls(
            nbar=self.nbar,
            k1=self.k1d,
            k2=self.k2d,
            gate_time=self.gate_time,
            truncature=self.N_data,
        )
        print(self.idle_pr)
        # total reconvergence after k2a rounds of error detection
        self.idle_tr = self.idle_tr_gate_cls(
            nbar=self.nbar,
            k1=self.k1d,
            k2=self.k2d,
            gate_time=1 / self.k2d,
            truncature=self.N_data,
        )
        print(self.idle_tr)
        try:
            # partial reconvergence during parity_measurement cnot gate_time
            self.idle_pr_ancilla = self.idle_pr_gate_cls_ancilla(
                nbar=self.nbar,
                k1=self.k1a,
                k2=self.k2a,
                gate_time=self.gate_time,
                truncature=self.N_ancilla,
            )
            print(self.idle_pr_ancilla)
        except TypeError:
            print('Idle gate idle_pr_gate_cls_ancilla is not defined')
            self.idle_pr_ancilla = None

    def add_system(self, sys):
        if self.debug:
            if len(sys) == self.distance - 1:
                self.system_state['ancilla'].append(sys)
            elif len(sys) == self.distance:
                self.system_state['data'].append(sys)
            else:
                print(f'{sys=}')
                print(f'{self.distance=}')

                raise ValueError('wrong dimension to add')

    def _post_weights(self):
        pass

    def populate_idle_gates(self):
        # prefix = '../../data/experiments/qec/'
        prefix = ''
        idle_tr_file_name = (
            f'{prefix}Idle_tr_{self.nbar}_{self.k2d}_'
            f'{self.N_data}_{self.k1d}.npy'
        )
        idle_pr_file_name = (
            f'{prefix}Idle_pr_{self.nbar}_{self.k2a}_'
            f'{self.N_data}_{self.k1d}.npy'
        )
        idle_pr_ancilla_file_name = (
            f'{prefix}Idle_pr_ancilla_{self.nbar}_{self.k2a}_'
            f'{self.N_ancilla}_{self.k1a}.npy'
        )

        file_name_l = [
            idle_tr_file_name,
            idle_pr_file_name,
            idle_pr_ancilla_file_name,
        ]

        for file_name, gate in zip(
            file_name_l,
            [self.idle_tr, self.idle_pr, self.idle_pr_ancilla],
        ):
            try:
                gate.transition_matrix = np.load(file_name)
            except FileNotFoundError:
                gate.transition_matrices()
                np.save(file_name, gate.transition_matrix)

    def populate_cnot_gate(self):
        # prefix = '../../data/experiments/qec/'
        prefix = ''
        cnot_file_name = (
            f'{prefix}CNOT_{self.nbar}_{self.k2a}_'
            f'{self.N_data}_{self.N_ancilla}_{self.k1d}.npy'
        )

        file_name_l = [
            cnot_file_name,
        ]

        for file_name, gate in zip(
            file_name_l,
            [self.cnot],
        ):
            try:
                gate.transition_matrix = np.load(file_name)
            except FileNotFoundError:
                gate.transition_matrices()
                np.save(file_name, gate.transition_matrix)

    def populate_error_models(self):
        self.physical_gates = {
            'Preparation': OneQubitErrorModel(pZ=self.one_photon_loss_ancilla),
            'IdleGate': OneQubitErrorModel(pZ=self.one_photon_loss_data),
            'MeasureX': OneQubitErrorModel(pZ=self.one_photon_loss_ancilla),
            'CNOT': CNOTAnalyticalAsymmetric(
                nbar=self.nbar,
                k2d=self.k2d,
                k2a=self.k2a,
                k1d=self.k1d,
                k1a=self.k1a,
                gate_time=self.gate_time,
            ),
            'IdleGateInterCNOTData': OneQubitErrorModel(
                pZ=self.one_photon_loss_data
            ),
            'IdleGateInterCNOTAncilla': OneQubitErrorModel(
                pZ=self.one_photon_loss_ancilla
            ),
        }

    def __str__(self):
        return (
            super().__str__()
            + f'\tN_data: {self.N_data} \n'
            f'\tN_ancilla: {self.N_ancilla} \n'
            f'\tdims qubit data: {self.qubit_data_dim} \n'
        )

    @cached_property
    def qubit_data_dim(self) -> int:
        return 2

    @abstractmethod
    def partial_reconvergence(self, **kwargs):
        pass

    @abstractmethod
    def partial_reconvergence_ancilla(self, syndrome: list, *args, **kwargs):
        pass

    @abstractmethod
    def single_mode_partial_reconvergence(  # pylint: disable=invalid-name
        self, index: int, *args, **kwargs
    ) -> None:
        pass

    @abstractmethod
    def total_reconvergence(self):
        pass

    def faulty_ancilla_preparation(self, syndrome):
        # faulty preparation but still in code space
        for anc, _ in enumerate(syndrome):
            if rand() < self.physical_gates['Preparation'].pZ:
                if self.verbose:
                    print('\t\tprep error on ancilla')
                    print(f'\t\t\t{syndrome=}')
                syndrome[anc] = self.N_ancilla
                if self.verbose:
                    print(f'\t\t\t{syndrome=}')
                self.error_count[1] += 1
        return syndrome

    @abstractmethod
    def apply_cnot_error(self, syndrome, i, _cnot_round):
        pass

    def get_error_sampler(self):
        self.error_sampler = {
            'one_photon_loss_ancilla': self.one_photon_loss_ancilla,
            'one_photon_loss_data': self.one_photon_loss_data,
            'one_photon_loss_data_tr': self.one_photon_loss_data_tr,
            'p_idle_inter_CNOT_ancilla': self.p_idle_inter_CNOT_ancilla,
            'p_idle_inter_CNOT_data': self.p_idle_inter_CNOT_data,
            'p_CXZ1_one_photon': self.p_CXZ1_one_photon,
            'p_ppm': self.p_ppm,
            'p_prep': self.physical_gates['Preparation'].pZ,
            'p_prepw': self.p_prepw,
            'p_CXw': self.p_CXw,
            'p_CX_IZ': self.physical_gates['CNOT'].pIZ,
            'p_CX_ZZ': self.physical_gates['CNOT'].pZZ,
            'p_meas': self.physical_gates['MeasureX'].pZ,
        }

    def cnot_rounds(self, syndrome):
        # CNOT ROUNDS
        if self.verbose:
            print('\tCNOT rounds')
            print(f'\t{self.data_qubit_state=}')
            print(f'\t{syndrome=}')
        # reconvergence of the bottom data qubit during first
        # round of CNOT
        self.single_mode_partial_reconvergence(index=-1, p=self.p_CXw)
        self.add_system(syndrome)
        self.add_system(self.data_qubit_state)

        # two rounds of CNOTs
        for _cnot_round in range(2):  # pylint: disable=too-many-nested-blocks
            for i in range(self.distance - 1):
                syndrome = self.apply_cnot_error(
                    syndrome, i=i, _cnot_round=_cnot_round
                )
            if _cnot_round == 0:
                # round of Idle gates
                self.partial_reconvergence(
                    p=self.p_idle_inter_CNOT_data,
                    flag_pheno=True,
                    flag_PRA=True,
                )
                syndrome = self.partial_reconvergence_ancilla(
                    syndrome=syndrome,
                    p=self.p_idle_inter_CNOT_ancilla,
                )
            self.add_system(syndrome)
            self.add_system(self.data_qubit_state)
        # reconvergence of the top data qubit during second
        # round of CNOT
        self.single_mode_partial_reconvergence(index=0, p=self.p_CXw)
        return syndrome

    def parity_measurement(self):  # pylint: disable=too-many-branches
        """Parity Measurement for the QEC circuit via transition matrices

        each system (ancilla and data) is composed of a qubit and a gauge
            the gauge of the ancilla (resp data) is of len N_ancilla
            (resp N_data).
            The state of a mode is represented by an integer which encodes
            the state
            of the qubit and of the gauge.

        A syndrome mode a_i can be represented by an int between 0 and
            2*N_ancilla-1
            between a_i = 0 and a_i= N_ancilla-1, the mode is: 0 tensor
            gauge in a_i
            between a_i = N_ancilla and a_i= 2*N_ancilla-1, the mode is:
            1 tensor gauge in a_i - N_ancilla

        A data mode l_i can be represented by an int between 0 and
            2*N_data-1
            between l_i = 0 and l_i= N_data-1, the mode is: 0 tensor
            gauge in l_i
            between l_i = N_data and l_i= 2*N_data-1, the mode is:
            1 tensor gauge in l_i - N_data

        Returns:
            syndrome (list): list of the syndrome results
        """
        # pdb.set_trace()
        # print('parity meas')
        if self.verbose:
            print('Parity Measurement')
        # ANCILLA PREPARATION
        if self.verbose:
            print('\tAncilla Preparation')
        syndrome = [0] * (self.distance - 1)  # preparation
        self.add_system(syndrome)
        self.add_system(self.data_qubit_state)
        syndrome = self.faulty_ancilla_preparation(syndrome)

        self.partial_reconvergence(
            p=self.p_prepw
        )  # data error during preparation of ancillas
        self.add_system(syndrome)
        self.add_system(self.data_qubit_state)

        syndrome = self.cnot_rounds(syndrome=syndrome)

        self.add_system(syndrome)
        self.add_system(self.data_qubit_state)

        # perfect ancilla measurement
        # - (perfect photon number parity measurement)
        if self.verbose:
            print(syndrome)
        syndrome = self.faulty_measurement(syndrome)
        return syndrome

    def faulty_measurement(self, syndrome: list) -> list:
        # faulty measurement
        syndrome = [syn // self.N_ancilla for syn in syndrome]
        if self.verbose:
            print('final syndrome when projecting')
            print(syndrome)
        # here syndrome is a list of booleans
        if self.verbose:
            print('\tMeasurement error')
        for anc in range(self.distance - 1):
            if rand() < self.physical_gates['MeasureX'].pZ:
                self.error_count[1] += 1
                syndrome[anc] ^= 1
        self.partial_reconvergence(p=self.p_measw, flag_pheno=True)
        self.add_system(syndrome)
        self.add_system(self.data_qubit_state)
        return syndrome

    def get_syndrome_gauge_from_joint_state(  # pylint: disable=invalid-name
        self, joint_state
    ):
        new_syndrome_mode = joint_state // (self.qubit_data_dim * self.N_data)
        new_data_mode = joint_state % (self.qubit_data_dim * self.N_data)
        new_data_gauge = new_data_mode % self.N_data
        # new_data_qubit = new_data_mode // self.N_data
        new_syndrome = new_syndrome_mode

        if new_syndrome > 2 * self.N_ancilla - 1 or new_syndrome < 0:
            raise ValueError(
                f'The syndrome value is {new_syndrome} but cannot be higher'
                f' than {2*self.N_ancilla-1} or lower than 0'
            )
        if (
            new_data_gauge > self.qubit_data_dim * self.N_data - 1
            or new_data_gauge < 0
        ):
            raise ValueError(
                f'The gauge value is {new_data_gauge} but cannot be higher than'
                f' {self.qubit_data_dim * self.N_data-1} or lower than 0'
            )
        print(f'{(new_syndrome, new_data_gauge)=}')
        return (new_syndrome, new_data_gauge)

    def data_qubit_phase_flip(self, index: int) -> None:
        if (
            self.data_qubit_state[index] >= self.N_data
            and self.data_qubit_state[index] < 2 * self.N_data
        ):
            self.data_qubit_state[index] -= self.N_data
        elif (
            self.data_qubit_state[index] >= 0
            and self.data_qubit_state[index] < self.N_data
        ):
            self.data_qubit_state[index] += self.N_data
        else:
            raise ValueError(
                f'data_qubit_state of index {index} is'
                f' {self.data_qubit_state[index]} but should be between 0 and'
                f' {2*self.N_data-1}'
            )

    def get_data_qubit_state(self, index: int) -> int:
        return self.data_qubit_state[index] // self.N_data

    def get_data_qubit(self) -> list:
        return [
            self.get_data_qubit_state(index) for index in range(self.distance)
        ]

    def get_data_gauge_state(self, index: int) -> int:
        return self.data_qubit_state[index] % self.N_data

    def get_data_gauge(self) -> list:
        return [
            self.get_data_gauge_state(index) for index in range(self.distance)
        ]

    def update_gauge_single_data_mode(self, gauge: int, index: int) -> None:
        if gauge > self.N_data:
            raise ValueError(
                f'The gauge is {gauge} but cannot be bigger than {self.N_data}'
            )
        self.data_qubit_state[index] = (
            self.get_data_qubit_state(index) * self.N_data + gauge
        )

    def update_data_modes_from_gauges(self, gauges: list) -> None:
        self.data_qubit_state = [
            self.get_data_qubit_state(index) * self.N_data + gauges[index]
            for index in range(self.distance)
        ]

    def update_single_data_mode(self, gauge: int, index: int) -> None:
        self.data_qubit_state[index] = (
            self.get_data_qubit_state(index) * self.N_data + gauge
        )

    def perfect_parity_measurement(self):
        # partial reconvergence as errors on data modes
        # self.partial_reconvergence(p=self.p_ppm)
        # final destructive measurement of the data
        final_state = [data // self.N_data for data in self.data_qubit_state]
        self.data_qubit_state = final_state

        # extract perfect final syndrome from
        # the data destructive measurement outcomes
        return [
            final_state[i] ^ final_state[i + 1]
            for i in range(len(final_state) - 1)
        ]

    def _save_name(self):
        return (
            f'{self.nbar}_{self.distance}_{self.num_rounds}'
            f'_{self.k2d}'
            f'_{self.k2a}_{self.gate_time}_{self.k1d}_{self.k1a}'
            f'_{type(self.physical_gates["CNOT"]).__name__}'
        )

    def _get_results(
        self,
    ) -> Dict[str, Any]:
        return {
            **super()._get_results(),
            'N_data': self.N_data,
            'N_ancilla': self.N_ancilla,
            'idle_tr': {
                'class_name': self.idle_tr.class_name,
                'nbar': self.idle_tr.nbar,
                'k1': self.idle_tr.k1,
                'k2': self.idle_tr.k2,
                'gate_time': self.idle_tr.gate_time,
                'truncature': self.idle_tr.truncature,
                'num_tslots_pertime': self.idle_tr.num_tslots_pertime,
            },
            'idle_pr': {
                'class_name': self.idle_pr.class_name,
                'nbar': self.idle_pr.nbar,
                'k1': self.idle_pr.k1,
                'k2': self.idle_pr.k2,
                'gate_time': self.idle_pr.gate_time,
                'truncature': self.idle_pr.truncature,
                'num_tslots_pertime': self.idle_pr.num_tslots_pertime,
            },
            'CNOT': {
                'class_name': self.cnot.class_name,
                'nbar': self.cnot.nbar,
                'k1': self.cnot.k1,
                'k2': self.cnot.k2,
                'k1a': self.cnot.k1a,
                'k2a': self.cnot.k2a,
                'gate_time': self.cnot.gate_time,
                'truncature': self.cnot.truncature,
                'num_tslots_pertime': self.cnot.num_tslots_pertime,
            },
        }


class LIdleGateSimuTotalConvergence(LIdleGateSimu):
    def _repeated_imperfect_parity_measurements(self, index: int):
        if (index + 1) % self.k2a == 0:
            # reconvergence of the data qubits
            # using num_rounds = d*k2a instead of two for loops
            # because the QEC graph is initialized from num_rounds
            self.total_reconvergence()


class LIdleGateCompleteModel(LIdleGateSimuTotalConvergence):
    """Logical idle Gate with error
    sampled from PhaseFlip approx of the CNOT."""

    cnot_gate_cls = CNOTSFBPhaseFlips
    idle_tr_gate_cls = IdleGateFake  # total reconvergence
    idle_pr_gate_cls = IdleGateSFBPhaseFlips  # partial reconvergence
    idle_pr_gate_cls_ancilla = IdleGateSFBPhaseFlips  # partial reconvergence

    def populate_error_models(self):
        print(f'{self.one_photon_loss_ancilla=}')
        self.physical_gates = {
            'Preparation': OneQubitErrorModel(pZ=self.one_photon_loss_ancilla),
            'IdleGate': OneQubitErrorModel(pZ=self.one_photon_loss_data),
            'MeasureX': OneQubitErrorModel(pZ=self.one_photon_loss_ancilla),
            'CNOT': CNOTCompleteModelNumericalErrorFromCorrelations(
                nbar=self.nbar,
                k2d=self.k2d,
                k2a=self.k2a,
                k1d=self.k1d,
                k1a=self.k1a,
                gate_time=self.gate_time,
                truncature=self.N_data,
                N_ancilla=self.N_ancilla,
            ),
            'IdleGateInterCNOTData': OneQubitErrorModel(
                pZ=self.one_photon_loss_data
            ),
            'IdleGateInterCNOTAncilla': OneQubitErrorModel(
                pZ=self.one_photon_loss_ancilla
            ),
        }
        # self.idle_tr.transition_matrix = perfect_transition_matrix(
        #     N=self.N_data
        # )

    def faulty_ancilla_preparation(self, syndrome):
        return reconvergence(self.idle_pr_ancilla.transition_matrix, syndrome)

    def partial_reconvergence(self, **kwargs):
        self.data_qubit_state = reconvergence(
            self.idle_pr.transition_matrix, self.data_qubit_state
        )

    def partial_reconvergence_ancilla(self, syndrome, *args, **kwargs):
        return reconvergence(self.idle_pr_ancilla.transition_matrix, syndrome)

    def single_mode_partial_reconvergence(  # pylint: disable=invalid-name
        self, index: int, *args, **kwargs
    ) -> None:
        transition_matrix = self.idle_pr.transition_matrix
        N = transition_matrix[:, 0].shape[0]
        self.data_qubit_state[index] = choice(
            range(N),
            p=transition_matrix[:, self.data_qubit_state[index]]
            / np.sum(transition_matrix[:, self.data_qubit_state[index]]),
        )

    def total_reconvergence(self):
        self.data_qubit_state = reconvergence(
            self.idle_tr.transition_matrix, self.data_qubit_state
        )

    def apply_cnot_error(self, syndrome, i: int, _cnot_round: int):
        # joint ancilla / data state
        # represented by an integer
        joint_state = (
            syndrome[i] * (self.qubit_data_dim * self.N_data)
            + self.data_qubit_state[i + _cnot_round]
        )
        # apply CNOT
        # joint_state becomes a probability distribution
        # roll dice
        # choose a final state for the ancilla data system
        joint_state = choice(
            range(2 * self.N_ancilla * self.qubit_data_dim * self.N_data),
            p=self.cnot.transition_matrix[:, joint_state]
            / np.sum(self.cnot.transition_matrix[:, joint_state]),
        )
        # separate states
        new_syndrome = joint_state // (self.qubit_data_dim * self.N_data)
        self.data_qubit_state[i + _cnot_round] = joint_state % (
            self.qubit_data_dim * self.N_data
        )
        if self.verbose:
            if new_syndrome != syndrome[i]:
                self.error_count[1] += 1
                print('\t\tAncilla error in CNOT')
                print(f'\t\t\t{syndrome=}')
        syndrome[i] = new_syndrome
        if self.verbose:
            print(f'\t\t\t{syndrome=}')

        return syndrome

    def perfect_parity_measurement(self):
        # final destructive measurement of the data
        # self.partial_reconvergence()
        final_state = [data // self.N_data for data in self.data_qubit_state]
        self.data_qubit_state = final_state

        # extract perfect final syndrome from
        # the data destructive measurement outcomes
        return [
            final_state[i] ^ final_state[i + 1]
            for i in range(len(final_state) - 1)
        ]


class LIdleGateCompleteModelPheno(LIdleGateCompleteModel):
    def populate_error_models(self):
        self.p = one_photon_loss_phaseflip(
            nbar=self.nbar, k1=self.k1d, gate_time=self.gate_time
        )
        self.q = one_photon_loss_phaseflip(
            nbar=self.nbar, k1=self.k1a, gate_time=self.gate_time
        )
        print(f'{self.p=}')
        print(f'{self.q=}')
        self.one_photon_loss_ancilla = self.q
        self.one_photon_loss_data = self.p
        print(f'{self.one_photon_loss_ancilla=}')
        self.physical_gates = {
            'Preparation': OneQubitErrorModel(),
            'IdleGate': OneQubitErrorModel(pZ=self.one_photon_loss_data),
            'MeasureX': OneQubitErrorModel(pZ=self.one_photon_loss_ancilla),
            # 'MeasureX': OneQubitErrorModel(),  # no error on sampler
            # 'CNOT': CNOTCompleteModelNumericalErrorFromCorrelations(
            # 'CNOT': CNOTAnalyticalAsymmetric(
            #     nbar=self.nbar,
            #     k2d=self.k2d,
            #     k2a=self.k2a,
            #     k1d=self.k1d,
            #     k1a=self.k1a,
            #     gate_time=self.gate_time,
            #     # truncature=self.N_data,
            #     # N_ancilla=self.N_ancilla,
            # ),
            'CNOT': TwoQubitErrorModel(),
            'IdleGateInterCNOTData': OneQubitErrorModel(
                # pZ=self.one_photon_loss_data
                pZ=0.0
            ),
            'IdleGateInterCNOTAncilla': OneQubitErrorModel(
                # pZ=self.one_photon_loss_ancilla
                pZ=0.0
            ),
        }
        self.one_photon_loss_data_tr = 0
        self.p_idle_inter_CNOT_ancilla = 0
        self.p_idle_inter_CNOT_data = 0
        self.idle_tr.transition_matrix = perfect_transition_matrix(
            N=self.N_data
        )
        print(self.idle_tr.transition_matrix)
        self.idle_pr.transition_matrix = pheno_transition_matrix(
            N=self.N_data, p=self.q
        )
        print(self.idle_pr.transition_matrix)
        self.idle_pr_ancilla.transition_matrix = perfect_transition_matrix(
            N=self.N_ancilla
        )
        print(self.idle_pr_ancilla.transition_matrix)
        self.cnot.transition_matrix = perfect_CNOT_transition_matrix(
            N_ancilla=self.N_ancilla, N_data=self.N_data
        )
        print(self.cnot.transition_matrix)
        res = self.print_physical_gates()
        print(res)
        # print(f'{self.cnot.transition_matrix=}')

    def _generate_error_correction_graph(self):
        # remove weights to be consistent with pheno error model
        # we use pvin, p0, phor
        self.p_prepw = self.p
        self.p_CXw = 0.0
        self.p_measw = 0.0
        self.p0 = self.p
        self.pvout = 0
        self.p3 = 0
        self.pbord = 0
        self.pdiag = 0
        self.phor = self.q

    def single_mode_partial_reconvergence(
        self, index: int, *args, **kwargs
    ) -> None:
        pass

    def partial_reconvergence(self, **kwargs):
        if ('flag_pheno', True) not in kwargs.items():
            super().partial_reconvergence()


class LIdleGateCompleteModelPRA(LIdleGateCompleteModel):
    def populate_error_models(self):
        self.p = one_photon_loss_phaseflip(
            nbar=self.nbar, k1=self.k1d, gate_time=self.gate_time
        )
        print(f'{self.p=}')
        self.one_photon_loss_ancilla = self.p
        self.one_photon_loss_data = self.p
        print(f'{self.one_photon_loss_ancilla=}')
        self.physical_gates = {
            # 'Preparation': OneQubitErrorModel(pZ=self.one_photon_loss_ancilla)
            'Preparation': OneQubitErrorModel(pZ=self.one_photon_loss_data),
            'IdleGate': OneQubitErrorModel(pZ=self.one_photon_loss_data),
            'MeasureX': OneQubitErrorModel(pZ=self.one_photon_loss_ancilla),
            # 'MeasureX': OneQubitErrorModel(),  # no error on sampler
            # 'CNOT': CNOTCompleteModelNumericalErrorFromCorrelations(
            # 'CNOT': CNOTAnalyticalAsymmetric(
            #     nbar=self.nbar,
            #     k2d=self.k2d,
            #     k2a=self.k2a,
            #     k1d=self.k1d,
            #     k1a=self.k1a,
            #     gate_time=self.gate_time,
            #     # truncature=self.N_data,
            #     # N_ancilla=self.N_ancilla,
            # ),
            'CNOT': CNOTPRAp3p(p=self.p),
            'IdleGateInterCNOTData': OneQubitErrorModel(
                # pZ=self.one_photon_loss_data
                pZ=0.0
            ),
            'IdleGateInterCNOTAncilla': OneQubitErrorModel(
                # pZ=self.one_photon_loss_ancilla
                pZ=0.0
            ),
        }
        self.one_photon_loss_data_tr = 0
        self.p_idle_inter_CNOT_ancilla = 0
        self.p_idle_inter_CNOT_data = 0
        self.idle_tr.transition_matrix = perfect_transition_matrix(
            N=self.N_data
        )
        print(self.idle_tr.transition_matrix)
        self.idle_pr.transition_matrix = pheno_transition_matrix(
            N=self.N_data, p=self.p
        )
        print(self.idle_pr.transition_matrix)
        self.idle_pr_ancilla.transition_matrix = perfect_transition_matrix(
            N=self.N_ancilla
        )
        print(self.idle_pr_ancilla.transition_matrix)
        self.cnot.transition_matrix = pra_CNOT_transition_matrix(
            N_ancilla=self.N_ancilla,
            N_data=self.N_data,
            pZ1=3 * self.p,
            pZ2=0.5 * self.p,
            pZ1Z2=0.5 * self.p,
        )
        print(self.cnot.transition_matrix)
        res = self.print_physical_gates()
        print(res)
        # print(f'{self.cnot.transition_matrix=}')

    def partial_reconvergence(self, **kwargs):
        if ('flag_PRA', True) not in kwargs.items():
            super().partial_reconvergence()


class LIdleGateCompleteModelAsym(LIdleGateCompleteModel):
    cnot_gate_cls = CNOTSFBPhaseFlips
    idle_tr_gate_cls = IdleGateSFBPhaseFlips  # total reconvergence
    idle_pr_gate_cls = IdleGateSFBPhaseFlips  # partial reconvergence
    idle_pr_gate_cls_ancilla = IdleGateSFBPhaseFlips  # partial reconvergence

    def _post_weights(self):
        p = exponentiel_proba(self.nbar * self.k1d / self.k2d)
        self.error_correction_graph.add_weight_data_reconvergence(
            p=p, k2a=self.k2a
        )

    def populate_error_models(self):
        print(f'{self.one_photon_loss_ancilla=}')
        self.physical_gates = {
            'Preparation': OneQubitErrorModel(pZ=self.one_photon_loss_ancilla),
            'IdleGate': OneQubitErrorModel(pZ=self.one_photon_loss_data),
            'MeasureX': OneQubitErrorModel(pZ=self.one_photon_loss_ancilla),
            'CNOT': CNOTCompleteModelNumericalErrorFromCorrelations(
                nbar=self.nbar,
                k2d=self.k2d,
                k2a=self.k2a,
                k1d=self.k1d,
                k1a=self.k1a,
                gate_time=self.gate_time,
                truncature=self.N_data,
                N_ancilla=self.N_ancilla,
            ),
            'IdleGateInterCNOTData': OneQubitErrorModel(
                pZ=self.one_photon_loss_data
            ),
            'IdleGateInterCNOTAncilla': OneQubitErrorModel(
                pZ=self.one_photon_loss_ancilla
            ),
        }


class LIdleGateCompleteModelAsymParityMeasurement(LIdleGateCompleteModelAsym):
    cnot_gate_cls = ParityMeasurementGate

    def init_gates(self):
        try:
            self.cnot = self.cnot_gate_cls(
                nbar=self.nbar,
                k1=self.k1d,
                k2=self.k2d,
                k2a=self.k2a,
                k1a=self.k1a,
                k1b=self.k1d,
                k2b=self.k2d,
                gate_time=self.gate_time,
                N_ancilla=self.N_ancilla,
                truncature=self.N_data,
                N_b=self.N_data,
            )
            print('(N, N_a)')
            print(self.N_data, self.cnot.N_ancilla)
        except TypeError:
            self.cnot = self.cnot_gate_cls(
                nbar=self.nbar,
                k1=self.k1d,
                k2=self.k2d,
                k2a=self.k2a,
                k1a=self.k1a,
                gate_time=self.gate_time,
                truncature=self.N_data,
            )
        print(self.cnot)

        # partial reconvergence during parity_measurement cnot gate_time
        self.idle_pr = self.idle_pr_gate_cls(
            nbar=self.nbar,
            k1=self.k1d,
            k2=self.k2d,
            gate_time=self.gate_time,
            truncature=self.N_data,
        )
        print(self.idle_pr)
        # total reconvergence after k2a rounds of error detection
        self.idle_tr = self.idle_tr_gate_cls(
            nbar=self.nbar,
            k1=self.k1d,
            k2=self.k2d,
            gate_time=1 / self.k2d,
            truncature=self.N_data,
        )
        print(self.idle_tr)
        try:
            # partial reconvergence during parity_measurement cnot gate_time
            self.idle_pr_ancilla = self.idle_pr_gate_cls_ancilla(
                nbar=self.nbar,
                k1=self.k1a,
                k2=self.k2a,
                gate_time=self.gate_time,
                truncature=self.N_ancilla,
            )
            print(self.idle_pr_ancilla)
        except TypeError:
            print('Idle gate idle_pr_gate_cls_ancilla is not defined')
            self.idle_pr_ancilla = None

    def populate_cnot_gate(self):
        # prefix = '../../data/experiments/qec/'
        prefix = 'parity_measurement_2_'
        cnot_file_name = (
            f'{prefix}CNOT_{self.nbar}_{self.k2a}_'
            f'{self.N_data}_{self.N_ancilla}_{self.k1d}.npy'
        )

        file_name_l = [
            cnot_file_name,
        ]

        for file_name, gate in zip(
            file_name_l,
            [self.cnot],
        ):
            try:
                gate.transition_matrix = np.load(file_name)
            except FileNotFoundError:
                gate.transition_matrices()
                np.save(file_name, gate.transition_matrix)

    def cnot_rounds(self, syndrome):
        if self.num_rounds != self.distance * self.k2a:
            raise ValueError(
                'The number of rounds is inconsistent with distance and'
                ' asymmetry'
            )
        # CNOT ROUNDS
        if self.verbose:
            print('\tCNOT rounds')
            print(f'\t{self.data_qubit_state=}')
            print(f'\t{syndrome=}')
        # reconvergence of the bottom data qubit during first
        # round of CNOT
        self.single_mode_partial_reconvergence(index=-1)
        self.add_system(syndrome)
        self.add_system(self.data_qubit_state)

        # reconvergence ancilla between cnots
        syndrome = self.partial_reconvergence_ancilla(syndrome=syndrome)

        # two rounds of CNOTs
        for i in range(self.distance - 1):
            index_ancilla = (
                self.distance - 2 - i
            )  # start from bottom of circuit
            # (cf single mode partial reconvergence)
            index_data_1 = index_ancilla
            index_data_2 = index_ancilla + 1

            # reconvergence of the bottom data qubit during idle
            # between CNOTs
            self.single_mode_partial_reconvergence(
                index=index_data_2
            )  # deals with phase flip caused by one photon loss and reconvergence of the gauge

            def run_pm(qa, q1, q2):
                joint_state = (
                    q2
                    + 2 * self.N_data * q1
                    + 2 * 2 * self.N_data * self.N_data * qa
                )
                joint_state = choice(
                    range(
                        2 * self.N_data * 2 * self.N_ancilla * 2 * self.N_data
                    ),
                    p=self.cnot.transition_matrix[:, joint_state]
                    / np.sum(self.cnot.transition_matrix[:, joint_state]),
                )
                qa = joint_state // (2 * self.N_data * 2 * self.N_data)
                q1 = (
                    joint_state - qa * (2 * self.N_data * 2 * self.N_data)
                ) // (2 * self.N_data)
                q2 = (
                    joint_state - qa * (2 * self.N_data * 2 * self.N_data)
                ) % (2 * self.N_data)
                return (qa, q1, q2)

            qa = syndrome[index_ancilla]
            q1 = self.data_qubit_state[index_data_1]
            q2 = self.data_qubit_state[index_data_2]

            (
                syndrome[index_ancilla],
                self.data_qubit_state[index_data_1],
                self.data_qubit_state[index_data_2],
            ) = run_pm(qa, q1, q2)

            self.add_system(syndrome)
            self.add_system(self.data_qubit_state)
        # reconvergence of the top data qubit during second
        # round of CNOT
        self.single_mode_partial_reconvergence(index=0, p=self.p_CXw)
        return syndrome


class LIdleGateCompleteModelAsymReduced(LIdleGateCompleteModelAsym):
    idle_pr_gate_cls_ancilla = IdleGateFake  # partial reconvergence

    def populate_error_models(self):
        print(f'{self.one_photon_loss_ancilla=}')
        self.physical_gates = {
            'Preparation': OneQubitErrorModel(pZ=self.one_photon_loss_ancilla),
            'IdleGate': OneQubitErrorModel(pZ=self.one_photon_loss_data),
            'MeasureX': OneQubitErrorModel(pZ=self.one_photon_loss_ancilla),
            'CNOT': CNOTAnalyticalAsymmetric(
                nbar=self.nbar,
                k2d=self.k2d,
                k2a=self.k2a,
                k1d=self.k1d,
                k1a=self.k1a,
                gate_time=self.gate_time,
                truncature=self.N_data,
                N_ancilla=self.N_ancilla,
            ),
            'IdleGateInterCNOTData': OneQubitErrorModel(
                pZ=self.one_photon_loss_data
            ),
            'IdleGateInterCNOTAncilla': OneQubitErrorModel(
                pZ=self.one_photon_loss_ancilla
            ),
        }
        self.res_parity_measurement = gen_res_parity_measurement(
            params={
                'nbar': self.nbar,
                'k2': self.k2d,
                'k1': self.k1d,
                'k1a': self.k1a,
                'k2a': self.k2a,
                'gate_time': self.gate_time,
                'truncature': self.N_data,
            }
        )

    def faulty_ancilla_preparation(self, syndrome):
        return syndrome

    def cnot_rounds(self, syndrome):
        # CNOT ROUNDS
        if self.verbose:
            print('\tCNOT rounds')
            print(f'\t{self.data_qubit_state=}')
            print(f'\t{syndrome=}')
        self.add_system(syndrome)
        self.add_system(self.data_qubit_state)

        # reconverger bottom qubit
        # reconvergence of the bottom data qubit during first
        # round of CNOT
        self.single_mode_partial_reconvergence(
            index=-1
        )  # deals with phase flip caused by one photon loss and reconvergence of the gauge

        # two rounds of CNOTs
        for i in range(self.distance - 1):
            index_ancilla = (
                self.distance - 2 - i
            )  # start from bottom of circuit
            # (cf single mode partial reconvergence)
            index_data_1 = index_ancilla
            index_data_2 = index_ancilla + 1

            # reconvergence of the bottom data qubit during idle
            # between CNOTs
            self.single_mode_partial_reconvergence(
                index=index_data_2
            )  # deals with phase flip caused by one photon loss and reconvergence of the gauge

            # one photon loss

            # drawing errors
            error_CNOT_1 = gen_cnot_onephotonloss(
                p_ancilla=self.one_photon_loss_ancilla,
                p_data=self.physical_gates['CNOT'].pIZ,
            )
            error_CNOT_2 = gen_cnot_onephotonloss(
                p_ancilla=self.one_photon_loss_ancilla,
                p_data=self.physical_gates['CNOT'].pIZ,
            )

            if error_CNOT_1 == 'Z1Z2':
                self.data_qubit_phase_flip(index_data_1)
            if error_CNOT_2 == 'Z1Z2':
                self.data_qubit_phase_flip(index_data_2)

            ancilla = syndrome[index_ancilla]
            if ancilla > self.N_ancilla * 2 - 1:
                print(f'{ancilla=}')

            # perfect cnot error propagation
            ancilla ^= self.get_data_qubit_state(
                index_data_1
            ) ^ self.get_data_qubit_state(index_data_2)

            p_ancilla = combine(
                [self.p_prep, self.p_meas, self.p_idle_inter_CNOT_ancilla]
            )
            ancilla ^= rand() < p_ancilla

            ancilla ^= error_CNOT_1 == 'Z1'
            ancilla ^= error_CNOT_2 == 'Z1'

            # non adiabatic errors of the two cnots
            (
                syndrome[index_ancilla],
                gauge_0,
                gauge_1,
            ) = update_systems_parity_measurement(
                res=self.res_parity_measurement,
                gauge_0=self.get_data_gauge_state(index_data_1),
                gauge_1=self.get_data_gauge_state(index_data_2),
                ancilla=syndrome[index_ancilla],
            )
            self.update_gauge_single_data_mode(
                gauge=gauge_0, index=index_data_1
            )
            self.update_gauge_single_data_mode(
                gauge=gauge_1, index=index_data_2
            )

            if error_CNOT_1 == 'Z2':
                self.data_qubit_phase_flip(index_data_1)
            if error_CNOT_2 == 'Z2':
                self.data_qubit_phase_flip(index_data_2)

            syndrome[index_ancilla] = ancilla
            if ancilla > self.N_ancilla * 2 - 1:
                print(f'{ancilla=}')

        self.add_system(syndrome)
        self.add_system(self.data_qubit_state)
        # reconvergence of the top data qubit during second
        # round of CNOT
        self.single_mode_partial_reconvergence(index=0)
        return syndrome

    def faulty_measurement(self, syndrome: list) -> list:
        # faulty measurement
        syndrome = [syn // self.N_ancilla for syn in syndrome]
        if self.verbose:
            print('final syndrome when projecting')
            print(syndrome)
        # here syndrome is a list of booleans
        if self.verbose:
            print('\tMeasurement error')
        # no error on syndrome here because measurement errors
        # have already been taken into account because the kraus
        # map of the non adiabatic errors perform the measurement.
        self.partial_reconvergence(p=self.p_measw, flag_pheno=True)
        self.add_system(syndrome)
        self.add_system(self.data_qubit_state)
        return syndrome


class LIdleGateCompleteModelSymmetrical(LIdleGateCompleteModel):
    def _repeated_imperfect_parity_measurements(self, index: int):
        pass


class LIdleGateReducedModel(LIdleGateSimuTotalConvergence):
    """Logical idle Gate with error sampled from Reduced Model of the CNOT."""

    cnot_gate_cls = CNOTSFBReducedReducedModel
    idle_tr_gate_cls = IdleGateSFBReducedReducedModel  # total reconvergence
    idle_pr_gate_cls = IdleGateSFBReducedReducedModel  # partial reconvergence
    idle_pr_gate_cls_ancilla = None  # partial reconvergence

    @cached_property
    def qubit_data_dim(self) -> int:
        return 1

    def populate_error_models(self):
        self.physical_gates = {
            'Preparation': OneQubitErrorModel(pZ=self.one_photon_loss_ancilla),
            'IdleGate': OneQubitErrorModel(pZ=self.one_photon_loss_data),
            'MeasureX': OneQubitErrorModel(pZ=self.one_photon_loss_ancilla),
            'CNOT': CNOTReducedModelNumericalErrorFromCorrelations(
                nbar=self.nbar,
                k2d=self.k2d,
                k2a=self.k2a,
                k1d=self.k1d,
                k1a=self.k1a,
                gate_time=self.gate_time,
                truncature=self.N_data,
            ),
        }

    def partial_reconvergence(self, **kwargs):
        p = kwargs['p']
        if p is None:
            p = self.physical_gates['IdleGate'].pZ
        # one photon loss on data qubit
        for index in range(self.distance):
            if rand() < p:
                self.error_count[0] += 1
                if self.verbose:
                    print('\t\tQubit error on data, PR')
                    print(f'\t\t\t{self.data_qubit_state=}')
                self.data_qubit_phase_flip(index=index)
                if self.verbose:
                    print(f'\t\t\t{self.data_qubit_state=}')
        # reconvergence of the gauge
        gauges = self.get_data_gauge()
        new_gauges = reconvergence(
            transition_matrix=self.idle_pr.transition_matrix,
            data_qubit_state=gauges,
        )
        self.update_data_modes_from_gauges(gauges=new_gauges)

    def partial_reconvergence_ancilla(self, syndrome, *args, **kwargs):
        p = self.p_idle_inter_CNOT_ancilla
        for anc in range(self.distance - 1):
            if rand() < p:
                self.error_count[1] += 1
                if self.verbose:
                    print('\t\tIdle error on ancilla')
                    print(f'\t\t\t{syndrome=}')
                syndrome[anc] = self.N_ancilla
                if self.verbose:
                    print(f'\t\t\t{syndrome=}')
        return syndrome

    def single_mode_partial_reconvergence(  # pylint: disable=invalid-name
        self, index: int, *args, **kwargs
    ) -> None:
        p = self.physical_gates['IdleGate'].pZ
        if rand() < p:
            self.error_count[0] += 1
            if self.verbose:
                print('\t\tQubit error on data, single mode PR')
                print(f'\t\t\t{self.data_qubit_state=}')
            self.data_qubit_phase_flip(index=index)
            if self.verbose:
                print(f'\t\t\t{self.data_qubit_state=}')
        gauge_bottom = self.get_data_gauge_state(index=index)
        new_gauge_bottom = choice(
            range(self.idle_pr.transition_matrix[:, 0].shape[0]),
            p=self.idle_pr.transition_matrix[:, gauge_bottom]
            / np.sum(self.idle_pr.transition_matrix[:, gauge_bottom]),
        )
        self.update_gauge_single_data_mode(gauge=new_gauge_bottom, index=index)

    def total_reconvergence(self):
        p = self.one_photon_loss_data_tr
        if self.verbose:
            print('\ttotal reconvergence')
        # one photon loss on data qubit
        for index in range(self.distance):
            if rand() < p:
                self.error_count[0] += 1
                if self.verbose:
                    print('\t\tQubit error on data, TR')
                    print(f'\t\t\t{self.data_qubit_state=}')
                self.data_qubit_phase_flip(index=index)
                if self.verbose:
                    print(f'\t\t\t{self.data_qubit_state=}')
        # reconvergence of the gauge
        gauges = self.get_data_gauge()
        new_gauges = reconvergence(
            transition_matrix=self.idle_tr.transition_matrix,
            data_qubit_state=gauges,
        )
        self.update_data_modes_from_gauges(gauges=new_gauges)

    def apply_cnot_error(self, syndrome, i: int, _cnot_round: int):
        # Non adiabatic error on ancilla qubit
        # gauge update on data
        # joint ancilla / data state
        # represented by an integer
        joint_state = (
            syndrome[i] * (self.qubit_data_dim * self.N_data)
            + self.data_qubit_state[i + _cnot_round]
        )
        # apply CNOT
        # joint_state becomes a probability distribution
        # roll dice
        # choose a final state for the ancilla data system
        joint_state = choice(
            range(2 * self.N_ancilla * self.qubit_data_dim * self.N_data),
            p=self.cnot.transition_matrix[:, joint_state]
            / np.sum(self.cnot.transition_matrix[:, joint_state]),
        )
        # separate states
        (
            new_syndrome,
            new_gauge,
        ) = self.get_syndrome_gauge_from_joint_state(joint_state=joint_state)
        if self.verbose:
            if new_syndrome != syndrome[i]:
                self.error_count[1] += 1
                print('\t\tAncilla error in CNOT')
                print(f'\t\t\t{syndrome=}')
        syndrome[i] = new_syndrome
        self.verb(f'\t\t\t{syndrome=}')

        self.update_gauge_single_data_mode(
            gauge=new_gauge, index=i + _cnot_round
        )
        # single photon loss on ancilla
        if rand() < self.p_CXZ1_one_photon:
            self.error_count[1] += 1
            if self.verbose:
                print('\t\tAncilla error in CNOT')
                print(f'\t\t\t{syndrome=}')
            syndrome = qubit_phase_flip(
                index=i, values=syndrome, mode_size=self.N_ancilla
            )
            if self.verbose:
                print(f'\t\t\t{syndrome=}')
        # single photon loss on data
        if rand() < self.physical_gates['CNOT'].pIZ:
            self.error_count[0] += 1
            self.verb('\t\tQubit error on data, CNOT Z2')
            self.verb(f'\t\t\t{self.data_qubit_state=}')
            self.data_qubit_phase_flip(index=i + _cnot_round)
            self.verb(f'\t\t\t{self.data_qubit_state=}')
        # error propagation
        # after the cnot but before the ZZ error from the cnot
        if self.get_data_qubit_state(index=i + _cnot_round) == 1:
            syndrome = qubit_phase_flip(
                index=i, values=syndrome, mode_size=self.N_ancilla
            )

        # Z1Z2 error
        if rand() < self.physical_gates['CNOT'].pZZ:
            self.error_count[0] += 1
            self.verb('\t\tAncilla error in CNOT')
            self.verb(f'\t\t\t{syndrome=}')
            syndrome = qubit_phase_flip(
                index=i, values=syndrome, mode_size=self.N_ancilla
            )
            for txt in [
                f'\t\t\t{syndrome=}',
                '\t\tQubit error on data, CNOT Z1Z2',
                f'\t\t\t{self.data_qubit_state=}',
            ]:
                self.verb(txt)
            self.data_qubit_phase_flip(index=i + _cnot_round)
            self.verb(f'\t\t\t{self.data_qubit_state=}')

    def perfect_parity_measurement(self):
        # final destructive measurement of the data
        self.partial_reconvergence(p=self.p_ppm)
        final_state = [data // self.N_data for data in self.data_qubit_state]
        self.data_qubit_state = final_state

        # extract perfect final syndrome from
        # the data destructive measurement outcomes
        return [
            final_state[i] ^ final_state[i + 1]
            for i in range(len(final_state) - 1)
        ]




class LIdleGateReducedModelSymmetrical(LIdleGateReducedModel):
    def _repeated_imperfect_parity_measurements(self, index: int):
        pass


def reconvergence(transition_matrix, data_qubit_state):
    N = transition_matrix[:, 0].shape[0]
    return [
        choice(
            range(N),
            p=transition_matrix[:, data_qubit_state[state]]
            / np.sum(transition_matrix[:, data_qubit_state[state]]),
        )
        for state in range(len(data_qubit_state))
    ]


def single_qubit_reconvergence(transition_matrix, data_qubit_state, state):
    N = transition_matrix[:, 0].shape[0]
    data_qubit_state[state] = choice(
        range(N),
        p=transition_matrix[:, state],
    )


def qubit_phase_flip(index: int, values: list, mode_size: int) -> list:
    if values[index] >= mode_size and values[index] < 2 * mode_size:
        values[index] -= mode_size
    elif values[index] >= 0 and values[index] < mode_size:
        values[index] += mode_size
    else:
        raise ValueError(
            f'qubit of index {index} is'
            f' {values[index]} but should be between 0 and'
            f' {2*mode_size-1}'
        )
    return values


def perfect_transition_matrix(N: int) -> np.ndarray:
    a = np.zeros((2 * N, 2 * N))
    a[0, :N] = 1
    a[N, N:] = 1
    return a


def pheno_transition_matrix(N: int, p: float) -> np.ndarray:
    a = np.zeros((2 * N, 2 * N))
    a[0, :N] = 1 - p
    a[N, N:] = 1 - p
    a[N, :N] = p
    a[0, N:] = p
    return a


def tpl_cnot_tm(N_data: int, N_ancilla: int) -> Tuple[np.ndarray, np.ndarray]:
    half_dim = N_data * 2 * N_ancilla
    np_shape = ((2 * half_dim), (2 * half_dim))
    a = np.reshape(
        np.zeros(shape=np_shape),
        (2, N_ancilla, 2, N_data, 2, N_ancilla, 2, N_data),
    )
    to_switch = np.zeros(half_dim)
    to_switch = np.reshape(to_switch, (N_ancilla, 2, N_data))
    to_switch[:, 0, :] = 1
    return a, to_switch


def perfect_CNOT_transition_matrix(N_data: int, N_ancilla: int) -> np.ndarray:
    a, to_switch = tpl_cnot_tm(N_data=N_data, N_ancilla=N_ancilla)
    to_switch.astype('bool')
    a[0, 0, 0, 0, 0, :, :, :] = to_switch
    a[0, 0, 1, 0, 1, :, :, :] = to_switch != np.ones_like(to_switch)
    a[1, 0, 0, 0, 1, :, :, :] = to_switch
    a[1, 0, 1, 0, 0, :, :, :] = to_switch != np.ones_like(to_switch)
    return a.reshape(
        (2 * N_data * 2 * N_ancilla), (2 * N_data * 2 * N_ancilla)
    ).astype('int8')


def pra_CNOT_transition_matrix(
    N_data: int,
    N_ancilla: int,
    pZ1: float = 0.0,
    pZ2: float = 0.0,
    pZ1Z2: float = 0.0,
) -> np.ndarray:
    a, to_switch = tpl_cnot_tm(N_data=N_data, N_ancilla=N_ancilla)
    to_switch_dual = to_switch != np.ones_like(to_switch)
    to_switch_dual = to_switch_dual.astype('float64')
    fid = 1 - (pZ1 + pZ2 + pZ1Z2)

    a[0, 0, 0, 0, 0, :, :, :] += to_switch * fid
    a[1, 0, 0, 0, 0, :, :, :] += to_switch * pZ1
    a[0, 0, 1, 0, 0, :, :, :] += to_switch * pZ2
    a[1, 0, 1, 0, 0, :, :, :] += to_switch * pZ1Z2

    a[0, 0, 1, 0, 1, :, :, :] += to_switch_dual * fid
    a[1, 0, 1, 0, 1, :, :, :] += to_switch_dual * pZ1
    a[0, 0, 0, 0, 1, :, :, :] += to_switch_dual * pZ2
    a[1, 0, 0, 0, 1, :, :, :] += to_switch_dual * pZ1Z2

    a[1, 0, 0, 0, 1, :, :, :] += to_switch * fid
    a[0, 0, 0, 0, 1, :, :, :] += to_switch * pZ1
    a[1, 0, 1, 0, 1, :, :, :] += to_switch * pZ2
    a[0, 0, 1, 0, 1, :, :, :] += to_switch * pZ1Z2

    a[1, 0, 1, 0, 0, :, :, :] += to_switch_dual * fid
    a[0, 0, 1, 0, 0, :, :, :] += to_switch_dual * pZ1
    a[1, 0, 0, 0, 0, :, :, :] += to_switch_dual * pZ2
    a[0, 0, 0, 0, 0, :, :, :] += to_switch_dual * pZ1Z2
    return a.reshape((2 * N_data * 2 * N_ancilla), (2 * N_data * 2 * N_ancilla))


def plot_transition_matrix(tm):
    _, ax = plt.subplots(1, 1, facecolor='white')
    norm = colors.LogNorm(1e-4, tm.max(), clip='True')
    im = plt.imshow(tm, norm=norm, cmap='Blues', origin='upper')
    plt.colorbar(im, norm=norm)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout(pad=0.0)
    plt.show()


def second_max(lst):
    return max(lst, key=lambda x: min(lst) - 1 if (x == max(lst)) else x)


if __name__ == '__main__':
    params = {
        'nbar': 8,
        'distance': 11,
        'k1d': 1e-3,
        # 'k1': 1e-3,
        # 'k1a': 1e-3,
        'k2d': 1,
        'k2a': 1,
        # 'k2': 1,
        'gate_time': 1,
        'logic_fail_max': 100,
        'init_transition_matrix': True,
        'N_data': 2,
        'N_ancilla': 1,
    }
    params.update({'k1a': params['k2a'] * params['k1d']})
    params.update({'gate_time': 1 / params['k2a']})
    params.update({'num_rounds': params['distance'] * params['k2a']})
    lidle = LIdleGateCompleteModelAsym(**params)
    print(lidle)
    lidle.generate_error_correction_graph()
    # pdb.set_trace()
    # lidle.sample()
    lidle.simulate()
