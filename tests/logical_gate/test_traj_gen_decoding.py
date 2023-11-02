import numpy as np
import pytest

from qsim.logical_gate.lidle import LIdleGatePhenomenological
from qsim.logical_gate.phenomenological import CheckMatrixPhenomenological


@pytest.mark.parametrize('p', [1e-1])
@pytest.mark.parametrize('q', [2e-1])
@pytest.mark.parametrize('distance', [3])
@pytest.mark.parametrize('logic_fail_max', [5000])
@pytest.mark.parametrize('num_traj', [10000])
@pytest.mark.parametrize('num_neighbours', [None])
class TestTrajGenDecoding:
    def test_traj_gen_decoding(
        self, p, q, distance, logic_fail_max, num_traj, num_neighbours
    ):
        # initialize gates
        pheno = CheckMatrixPhenomenological(
            distance=distance,
            p=p,
            q=q,
        )

        lidle = LIdleGatePhenomenological(
            distance=distance,
            p=p,
            q=q,
            logic_fail_max=logic_fail_max,
            num_rounds=distance,
        )
        lidle.generate_error_correction_graph()
        # traj gen
        samples_check = []
        samples_lidle = []
        for i in range(num_traj):
            res = pheno.sample()
            samples_check.append(
                (pheno.detection_events, pheno.noise_total, res)
            )
            res = lidle.sample()
            samples_lidle.append((lidle.syndrome, lidle.data_qubit_state, res))

        # decodings
        # from check generated trajectories
        for i in range(num_traj):
            noisy_syndrome, noise_total, res = samples_check[i]
            correction = pheno.m2d.decode(
                noisy_syndrome, num_neighbours=num_neighbours
            )
            assert int(all(noise_total == correction)) == res

            detection_event_list = []
            for x in noisy_syndrome.transpose():
                detection_event_list += list(x)
            detection_event_list.append(sum(detection_event_list) % 2)

            correction = lidle.m.decode(
                detection_event_list, num_neighbours=num_neighbours
            )
            assert int(all(noise_total == correction)) == res

        # from lidle generated trajectories
        for i in range(num_traj):
            syndrome, logical_control, res = samples_lidle[i]
            lidle.syndrome_to_detection_event(syndrome=syndrome)
            correction = lidle.m.decode(
                lidle.detection_event_list, num_neighbours=num_neighbours
            )
            assert int(all(logical_control == correction)) == res

            noisy_syndrome = np.copy(syndrome)
            syndrome[1:, :] = noisy_syndrome[1:, :] ^ noisy_syndrome[0:-1, :]
            correction = pheno.m2d.decode(
                syndrome.transpose(), num_neighbours=num_neighbours
            )
            assert all(logical_control == correction) == res
