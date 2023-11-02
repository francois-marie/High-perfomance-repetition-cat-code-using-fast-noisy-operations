from abc import abstractmethod
from time import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from numpy.random import rand
from pymatching import Matching

from qsim.utils.error_correction_graph import ErrorCorrectionGraph
from qsim.utils.utils import combine, error_bar

from .lidle import LIdleGate


class Phenomenological(LIdleGate):
    def __init__(
        self,
        distance: int,
        p: float,
        q: Optional[float] = None,
        num_rounds: Optional[int] = None,
        verbose: bool = False,
        traj_max: int = 500_000,
        logic_fail_max: int = 10000,
    ):
        if q is None:
            q = p
        if num_rounds is None:
            num_rounds = distance + 1
            # num_rounds = distance
        super().__init__(
            distance=distance,
            physical_gates={},
            num_rounds=num_rounds,
            verbose=verbose,
            traj_max=traj_max,
            logic_fail_max=logic_fail_max,
        )
        self.p = p
        self.q = q

    def _save_name(self):
        return f'{self.distance}_{self.num_rounds}_{self.p}_{self.q}'

    def _get_results(self) -> Dict[str, Any]:
        return {
            **super()._get_results(),
            'p': self.p,
            'q': self.q,
            'distance': self.distance,
            'num_rounds': self.num_rounds,
        }

    def __str__(self):
        return (
            f'{super().__str__()}'
            f'\t Repetition : {self.num_rounds} \n\t Max traj :'
            f' {self.traj_max} \n\t Max Logic Fails :'
            f' {self.logic_fail_max} \n'
        )

    def _simulate_Z(self):
        self.monte_carlo()

    @abstractmethod
    def monte_carlo(self):
        pass

    def _generate_error_correction_graph(self):
        pass

    def parity_measurement(self):
        pass

    def perfect_parity_measurement(self):
        pass


class CheckMatrixPhenomenological(Phenomenological):
    """
    Each column of Hx corresponds to a qubit, and each
    row corresponds to a X stabiliser.

    Hx[i,j]==1 if X stabiliser i acts non-trivially
    on qubit j, and is 0 otherwise.
    """

    def __init__(
        self,
        distance: int,
        p: float,
        q: Optional[float] = None,
        num_rounds: Optional[int] = None,
        verbose: bool = False,
        traj_max: int = 10_000_000,
        logic_fail_max: int = 1_000,
    ):
        super().__init__(
            distance=distance,
            p=p,
            q=q,
            num_rounds=num_rounds,
            verbose=verbose,
            traj_max=traj_max,
            logic_fail_max=logic_fail_max,
        )

        self.check_matrix = (
            np.identity(self.distance) + np.diag([1] * (self.distance - 1), 1)
        )[:-1]
        self.check_matrix = sp.csr_matrix(self.check_matrix)

        self.m2d = Matching(
            H=self.check_matrix,
            spacelike_weights=np.log((1 - self.p) / self.p) if p != 0 else None,
            repetitions=self.num_rounds,
            timelike_weights=np.log((1 - self.q) / self.q) if q != 0 else None,
            error_probabilities=self.p,
            measurement_error_probability=self.q,
        )
        g = self.m2d.to_networkx()
        boundary = (self.distance + 1) * (self.distance - 1)
        g.remove_edge(self.distance * (self.distance - 1), boundary)
        for i in range(self.distance * (self.distance - 1), boundary):
            g.remove_edge(i, i + 1)
        self.m2d = Matching(g)
        self.p_prepw = p
        self.pvin = p
        self.p0 = combine([self.p_prepw, 0])
        self.pvout = 0
        self.p3 = 0
        self.pbord = 0
        self.phor = q
        self.pdiag = 0

    def _save_name(self) -> str:
        return f'{self.distance}_{self.num_rounds}_{self.p}_{self.q}'

    def _get_results(self) -> Dict[str, Any]:
        return {
            **super()._get_results(),
            'p': self.p,
            'q': self.q,
            'distance': self.distance,
            'num_rounds': self.num_rounds,
            # 'log_errors': self.log_errors,
        }

    def sample(self) -> int:
        # simulate noisy syndromes
        num_stabilisers, _ = self.check_matrix.shape
        # np.random.seed(1) # Keep RNG deterministic
        np.random.seed()
        noise = (
            np.random.rand(self.distance, self.num_rounds) < self.p
        ).astype(np.uint8)
        self.p_errors = noise.sum()
        # remove last round of errors
        # no data qubit error during perfect_parity_measurement
        noise[:, -1] = [0] * self.distance
        self.noise = noise
        if self.verbose:
            print('noise')
            print(noise)

        noise_cumulative = (np.cumsum(noise, 1) % 2).astype(np.uint8)
        noise_total = noise_cumulative[
            :, -1
        ]  # total cumulative noise at the last round
        self.noise_total = noise_total
        if self.verbose:
            print('noise total')
            print(self.noise_total)

        noiseless_syndrome = (self.check_matrix @ noise_cumulative % 2).astype(
            np.uint8
        )

        # we assume each syndrome measurement is incorrect with probability q,
        # but that the last round of measurements
        # is perfect to ensure an even number of defects
        syndrome_error = (
            np.random.rand(num_stabilisers, self.num_rounds) < self.q
        ).astype(np.uint8)
        syndrome_error[:, -1] = 0
        self.q_errors = syndrome_error.sum()
        self.syndrome_error = syndrome_error
        if self.verbose:
            print('syndrome error')
            print(self.syndrome_error)

        noisy_syndrome = ((noiseless_syndrome + syndrome_error) % 2).astype(
            np.uint8
        )
        detection_events = np.copy(noisy_syndrome)
        detection_events[:, 1:] = (
            detection_events[:, 1:] - detection_events[:, 0:-1]
        ) % 2
        self.noisy_syndrome = noisy_syndrome
        self.detection_events = detection_events

        # decode
        self.correction = self.m2d.decode(
            detection_events,
            num_neighbours=self.num_neighbours,
            # return_weight=True,
        )
        if self.verbose:
            print('correction')
            print(self.correction)
        self.error_count = [np.sum(self.noise), np.sum(self.syndrome_error)]
        self.error_counts.append(self.error_count.copy())
        self.res = int(all(self.noise_total == self.correction))

        return self.res

    def multi_sample(self, multiplicity: int = 1) -> int:
        # simulate noisy syndromes
        num_stabilisers, _ = self.check_matrix.shape
        # np.random.seed(1) # Keep RNG deterministic
        np.random.seed()
        noise_m = (
            np.random.rand(multiplicity, self.distance, self.num_rounds)
            < self.p
        ).astype(np.uint8)
        print(noise_m)
        self.p_errors = noise_m.sum(axis=(1, 2))
        # remove last round of errors
        # no data qubit error during perfect_parity_measurement
        # noise[:, -1] = [0] * self.distance
        self.noise_m = noise_m
        if self.verbose:
            print('noise')
            print(noise_m)

        noise_cumulative = (np.cumsum(noise_m, axis=2) % 2).astype(np.uint8)
        noise_total = noise_cumulative[
            :, :, -1
        ]  # total cumulative noise at the last round
        self.noise_total = noise_total
        if self.verbose:
            print('noise total')
            print(self.noise_total)

        noiseless_syndrome = (
            self.check_matrix @ noise_cumulative[:] % 2
        ).astype(np.uint8)

        # we assume each syndrome measurement is incorrect with probability q,
        # but that the last round of measurements
        # is perfect to ensure an even number of defects
        syndrome_error_m = (
            np.random.rand(multiplicity, num_stabilisers, self.num_rounds)
            < self.q
        ).astype(np.uint8)
        syndrome_error_m[:, :, -1] = 0
        self.q_errors = syndrome_error_m.sum(axis=(1, 2))
        self.syndrome_error_m = syndrome_error_m
        if self.verbose:
            print('syndrome error')
            print(self.syndrome_error_m)

        noisy_syndrome_m = ((noiseless_syndrome + syndrome_error_m) % 2).astype(
            np.uint8
        )
        detection_events = np.copy(noisy_syndrome_m)
        detection_events[:, :, 1:] = (
            detection_events[:, :, 1:] - detection_events[:, :, 0:-1]
        ) % 2
        self.noisy_syndrome = noisy_syndrome_m
        self.detection_events = detection_events

        # decode
        self.correction = self.m2d.decode(
            detection_events[:],
            num_neighbours=self.num_neighbours,
            # return_weight=True,
        )
        if self.verbose:
            print('correction')
            print(self.correction)
        self.error_count = [np.sum(self.noise), np.sum(self.syndrome_error)]
        self.error_counts.append(self.error_count.copy())
        self.res = int(all(self.noise_total[:] == self.correction[:]))

        return self.res

    def monte_carlo_from_graph(self) -> None:
        print(
            f'(d, p, q, r): {(self.distance, self.p, self.q, self.num_rounds)}'
        )
        num_trajectories = 0
        logical_failure = 0
        while (
            num_trajectories < self.traj_max
            and logical_failure < self.logic_fail_max
        ):
            noise, syndrome = self.m2d.add_noise()
            correction = self.m.decode(syndrome)
            if not np.allclose(correction, noise):
                logical_failure += 1
            num_trajectories += 1
        self.num_trajectories['Z'] = num_trajectories
        self.fails = logical_failure
        self.outcome['Z'] = 1 - self.fails / num_trajectories
        err = error_bar(N=self.num_trajectories['Z'], p=1 - self.outcome['Z'])
        print(f'p={1 - self.outcome["Z"]} + /- {err}')

    def monte_carlo(self) -> None:
        print(
            f'(d, p, q, r): {(self.distance, self.p, self.q, self.num_rounds)}'
        )
        self.log_errors = []
        verbose = self.verbose
        self.verbose = False
        fails = 0
        num_trajectories = 0
        while fails < self.logic_fail_max and num_trajectories < self.traj_max:
            num_trajectories += 1
            # if not self.sample():
            #     fails += 1 only if self.sample() == 0
            fails += self.sample() ^ 1
            # fails += self.multi_sample() ^ 1
            self.log_errors.append((self.res, self.p_errors, self.q_errors))
        self.num_trajectories['Z'] = num_trajectories
        self.fails = fails
        self.verbose = verbose
        self.outcome['Z'] = 1 - fails / num_trajectories
        err = error_bar(N=self.num_trajectories['Z'], p=1 - self.outcome['Z'])
        print(f'p={1 - self.outcome["Z"]} + /- {err}')

    def build_ecg(self):
        e = ErrorCorrectionGraph(
            num_rows=self.distance + 1, num_columns=self.distance - 1
        )
        e.g = self.m2d.to_networkx()
        nodes = sorted(list(e.g.nodes()))
        for node in nodes:
            e.g.nodes[node]['pos'] = (
                node // (self.distance - 1),
                node % (self.distance - 1),
            )
        e.g.nodes[nodes[-1]]['pos'] = (0, -1)
        for u, v in e.g.edges():
            e.g[u][v]['color'] = 'black'
        self.e = e


class CheckMatrixPhenomenological5(CheckMatrixPhenomenological):
    pass


class CheckMatrixPhenomenological7(CheckMatrixPhenomenological):
    pass


class CheckMatrixPhenomenological29(CheckMatrixPhenomenological):
    pass


class CheckMatrixPhenomenological33(CheckMatrixPhenomenological):
    pass


class CheckMatrixPhenomenologicalold(CheckMatrixPhenomenological):
    pass


def find_threshold_dichotomy(
    q: float,
    relative_tolerance=1e-1,
    d_inf: int = 5,
    d_sup: int = 9,
    p_min=1e-3,
    p_max=0.5,
) -> Tuple[float, float]:
    while (p_max - p_min) / p_max > relative_tolerance:
        p_current = np.mean([p_min, p_max])
        pheno_inf = CheckMatrixPhenomenological(
            distance=d_inf, p=p_current, q=q
        )
        pheno_inf.monte_carlo()
        pl_inf = pheno_inf.fails / pheno_inf.num_trajectories

        pheno_sup = CheckMatrixPhenomenological(
            distance=d_sup, p=p_current, q=q
        )
        pheno_sup.monte_carlo()
        pl_sup = pheno_sup.fails / pheno_sup.num_trajectories

        if pl_inf > pl_sup:
            # in the exponentially suppressed regime
            # threshold on the right
            p_min = p_current
        else:
            p_max = p_current
    return p_min, p_max


class FilterPhenomenological(Phenomenological):
    def __init__(
        self,
        distance: int,
        p: float,
        q: float,
        num_rounds: Optional[int] = None,
        verbose: bool = False,
        traj_max: int = 500_000,
    ):
        if num_rounds is None:
            num_rounds = distance
        super().__init__(
            distance=distance,
            p=p,
            q=q,
            num_rounds=num_rounds,
            verbose=verbose,
            traj_max=traj_max,
            logic_fail_max=1,
        )

        I = np.array([[1, 0], [0, 1]])
        Z = np.array([[0, 1], [1, 0]])
        plus = np.array([[1, 0], [0, 0]])
        minus = np.array([[0, 0], [0, 1]])

        ket_plusa = np.kron(np.array([[1], [0]]), np.kron(I, I))
        ket_minusa = np.kron(np.array([[0], [1]]), np.kron(I, I))

        Id = np.kron(I, np.kron(I, I))
        Za = np.kron(Z, np.kron(I, I))
        # Z1 = np.kron(I, np.kron(Z, I))
        Z2 = np.kron(I, np.kron(I, Z))

        plus1 = np.kron(I, np.kron(plus, I))
        minus1 = np.kron(I, np.kron(minus, I))
        plus2 = np.kron(I, np.kron(I, plus))
        minus2 = np.kron(I, np.kron(I, minus))

        cnot_a1 = plus1 @ Id + minus1 @ Za
        cnot_a2 = plus2 @ Id + minus2 @ Za

        # faulty ancilla preparation
        K_data_error = (1 - p) * Id + p * Z2

        K_ancilla_error = (1 - self.q) * Id + self.q * Za

        K = K_ancilla_error @ cnot_a1 @ cnot_a2 @ K_data_error

        self.k_plus = ket_plusa.T @ K @ ket_plusa
        self.k_minus = ket_minusa.T @ K @ ket_plusa

        # build POVM basis on d-qubits
        self.k_plus_list = [
            sp.kron(
                sp.eye(2**i),
                sp.kron(self.k_plus, sp.eye(2 ** (self.distance - i - 2))),
            )
            for i in range(self.distance - 1)
        ]
        self.k_minus_list = [
            sp.kron(
                sp.eye(2**i),
                sp.kron(self.k_minus, sp.eye(2 ** (self.distance - i - 2))),
            )
            for i in range(self.distance - 1)
        ]

        # build Kraus basis for (physical) boundary qubits
        self.k_first = (1 - p) * I + p * Z
        self.k_first = sp.kron(self.k_first, sp.eye(2 ** (self.distance - 1)))

    def stabilizer_measurement(self, psi) -> Tuple[np.ndarray, List[int]]:
        """Perform one stabilizer measurement

        Args:
            psi (np.ndarray()): wavefunction of the data qubits

        Returns:
            tuple(psi, syndrome): (wavefunction of the data qubits after the
                execution of one round of QEC,
                result of the measurements of the stabilizers
                - list of length self.distance-1)
        """
        self.distance = len(psi)

        # data errors
        for i in range(self.distance):
            psi[i] ^= rand() < self.p

        # perfect syndrom readout
        syndrom = [psi[i] ^ psi[i + 1] for i in range(self.distance - 1)]

        # measurement errors
        for i in range(self.distance - 1):
            syndrom[i] ^= rand() < self.q

        return psi, syndrom

    def update_filter(self, syndrom) -> None:
        """Update the filter

        Args:
            syndrom (List[int]): result of the measurements of the stabilizers
                - list of length self.distance-1)
        """
        # Load
        self.rho_filter = self.k_first.dot(self.rho_filter)

        for idx, outcome in enumerate(syndrom):
            if outcome:
                self.rho_filter = self.k_minus_list[idx].dot(self.rho_filter)
            else:
                self.rho_filter = self.k_plus_list[idx].dot(self.rho_filter)

        self.rho_filter = self.rho_filter / sum(self.rho_filter)  # normalize

    def index(self, psi) -> int:
        index = 0
        self.distance = len(psi)
        for i in range(self.distance):
            index += psi[i] * 2 ** (self.distance - 1 - i)
        return index

    def sample(self) -> None:
        # initial state of the real system & initial quantum filter state
        psi = np.zeros(self.distance, dtype=int)
        self.rho_filter = np.zeros(2**self.distance)
        self.rho_filter[0] = 1
        for stabilizer_round in range(self.num_rounds):
            # run QEC circuit
            psi, syndrom = self.stabilizer_measurement(psi)
            # Update quantum filter
            self.update_filter(syndrom)
            # max eigenvalue of the filter after a fictive
            # perfect parity measurement
            # compute the output of a perfect parity measurement
            system_state = self.index(psi)
            # compute largest eigenvalue of the filter
            # and the square value to compute the numerical variance
            max_eig = max(
                self.rho_filter[system_state],
                self.rho_filter[system_state ^ (2**self.distance - 1)],
            ) / (
                self.rho_filter[system_state]
                + self.rho_filter[system_state ^ (2**self.distance - 1)]
            )
            self.max_eig_array[stabilizer_round] += max_eig
            self.max_eig_array2[stabilizer_round] += max_eig**2

    def monte_carlo(self) -> None:
        # run code
        t_start = time()

        self.max_eig_array = np.zeros(self.num_rounds)
        self.max_eig_array2 = np.zeros(self.num_rounds)

        for _ in trange(self.traj_max):
            self.sample()

        self.max_eig_array /= self.traj_max
        self.max_eig_array2 /= self.traj_max
        self.err_bar = (
            1.96
            * np.sqrt(self.max_eig_array2 - self.max_eig_array**2)
            / np.sqrt(self.traj_max)
        )
        self.outcome['Z'] = self.max_eig_array[-1]
        self.t_stop = time() - t_start

        # print
        print(f'Time elapsed : {self.t_stop}\n')

    def _save(self) -> None:
        # save results
        try:
            self.max_eig_dict = np.load(
                'max_eig_dict.npy', allow_pickle=True
            ).item()
        except FileNotFoundError:
            self.max_eig_dict = {}
        self.max_eig_dict[self.distance, self.p] = self.max_eig_array
        np.save('max_eig_dict.npy', self.max_eig_dict)

        try:
            self.err_bar_dict = np.load(
                'err_bar_dict.npy', allow_pickle=True
            ).item()
        except FileNotFoundError:
            self.err_bar_dict = {}
        self.err_bar_dict[self.distance, self.p] = self.err_bar
        np.save('err_bar_dict.npy', self.err_bar_dict)

        try:
            timing_dict = np.load('timing.npy', allow_pickle=True).item()
        except FileNotFoundError:
            timing_dict = {}
        timing_dict[self.distance, self.p] = self.t_stop
        np.save('timing.npy', timing_dict)
