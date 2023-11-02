from typing import Dict, List
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qutip import sigmax, sigmay, sigmaz

def gen_params(cardinal_states: dict, N: int, physical_params:Optional[dict]=None):
    if physical_params is None:
        physical_params = {
            'nbar': 4,
            'k1': 1e-3,
            'k2': 1,
            'gate_time': 1,
            'truncature': N,
            'num_tslots_pertime': 1e3,
        }

    if 'truncature' not in physical_params.keys():
        physical_params['truncature'] = N
    return [
        {**physical_params, **{
            'initial_state': initial_state,
            'initial_state_name': initial_state_name,
        }}
        for (initial_state_name, initial_state) in cardinal_states.items()
    ]


def build_one_qubit_output(E0, E1, EP, EM):
    rho0 =E0
    rho1 = EP + 1j * EM - (1 + 1j) * E0 / 2 - (1 + 1j) * E1 / 2
    rho2 = EP - 1j * EM - (1 - 1j) * E0 / 2 - (1 - 1j) * E1 / 2
    rho3 = E1
    r0 = np.concatenate((rho0, rho1), axis=1)
    r1 = np.concatenate((rho2, rho3), axis=1)
    return np.concatenate((r0, r1))


def one_qubit_process_matrix(rhop):
    # Compute process matrix
    X = np.array([[0, 1], [1, 0]])
    # Z = np.array([[1, 0], [0, -1]])
    Lambda = (
        1
        / 2
        * np.concatenate(
            (
                np.concatenate((np.eye(2), X), axis=1),
                np.concatenate((X, -np.eye(2)), axis=1),
            )
        )
    )
    chi_2 = (Lambda.T).dot(rhop).dot(Lambda)
    # ChiError

    # rhopp = (np.kron(np.eye(2), Z)).dot(rhop).dot(np.kron(np.eye(2), Z))
    # chi_2_error = (Lambda.T).dot(rhopp).dot(Lambda)
    # return chi_2_error
    return chi_2


def factorized_Z_process_tomography(rhop, theta):
    # Compute process matrix
    X = np.array([[0, 1], [1, 0]])
    Z_theta = np.array([[1, 0], [0, np.exp(1.0j * theta)]])
    Lambda = (
        1
        / 2
        * np.concatenate(
            (
                np.concatenate((np.eye(2), X), axis=1),
                np.concatenate((X, -np.eye(2)), axis=1),
            )
        )
    )
    # ChiError
    # Y = np.array([[0, -1j], [1j, 0]])
    # OneQbasis = (np.eye(2), X, Y, Z)

    rhopp = (
        (np.kron(np.eye(2), Z_theta).T.conj())
        .dot(rhop)
        .dot(np.kron(np.eye(2), Z_theta))
    )
    chi_2_error = (Lambda.T).dot(rhopp).dot(Lambda)
    return chi_2_error


def build_two_qubit_output(
    E0,
    E1,
    E2,
    E3,
    EP01,
    EP02,
    EP03,
    EP12,
    EP13,
    EP23,
    EM01,
    EM02,
    EM03,
    EM12,
    EM13,
    EM23,
):
    rho00 = E0
    rho01 = EP01 + 1j * EM01 - (1 + 1j) * E0 / 2 - (1 + 1j) * E1 / 2
    rho02 = EP02 + 1j * EM02 - (1 + 1j) * E0 / 2 - (1 + 1j) * E2 / 2
    rho03 = EP03 + 1j * EM03 - (1 + 1j) * E0 / 2 - (1 + 1j) * E3 / 2

    rho10 = EP01 - 1j * EM01 - (1 - 1j) * E1 / 2 - (1 - 1j) * E0 / 2
    rho11 = E1
    rho12 = EP12 + 1j * EM12 - (1 + 1j) * E1 / 2 - (1 + 1j) * E2 / 2
    rho13 = EP13 + 1j * EM13 - (1 + 1j) * E1 / 2 - (1 + 1j) * E3 / 2

    rho20 = EP02 - 1j * EM02 - (1 - 1j) * E2 / 2 - (1 - 1j) * E0 / 2
    rho21 = EP12 - 1j * EM12 - (1 - 1j) * E2 / 2 - (1 - 1j) * E1 / 2
    rho22 = E2
    rho23 = EP23 + 1j * EM23 - (1 + 1j) * E2 / 2 - (1 + 1j) * E3 / 2

    rho30 = EP03 - 1j * EM03 - (1 - 1j) * E3 / 2 - (1 - 1j) * E0 / 2
    rho31 = EP13 - 1j * EM13 - (1 - 1j) * E3 / 2 - (1 - 1j) * E1 / 2
    rho32 = EP23 - 1j * EM23 - (1 - 1j) * E3 / 2 - (1 - 1j) * E2 / 2
    rho33 = E3

    r0 = np.concatenate((rho00, rho01, rho02, rho03), axis=1)
    r1 = np.concatenate((rho10, rho11, rho12, rho13), axis=1)
    r2 = np.concatenate((rho20, rho21, rho22, rho23), axis=1)
    r3 = np.concatenate((rho30, rho31, rho32, rho33), axis=1)

    return np.concatenate((r0, r1, r2, r3))


def lambda2():
    x = np.array([[0, 1], [1, 0]])
    Lambda = (
        1
        / 2
        * np.concatenate(
            (
                np.concatenate((np.eye(2), x), axis=1),
                np.concatenate((x, -np.eye(2)), axis=1),
            )
        )
    )
    Lambda2 = np.kron(Lambda, Lambda)
    return Lambda2


def perm():
    zero, one = np.array([[1], [0]]), np.array([[0], [1]])
    Perm = (
        np.kron(zero, zero).dot(np.kron(zero, zero).T)
        + np.kron(zero, one).dot(np.kron(one, zero).T)
        + np.kron(one, zero).dot(np.kron(zero, one).T)
        + np.kron(one, one).dot(np.kron(one, one).T)
    )
    Perm = np.kron(np.eye(2), np.kron(Perm, np.eye(2)))
    return Perm


def CNOT_process_tomography(rhop):
    return two_qubit_process_matrix(rhop=rhop)

def two_qubit_process_matrix(rhop):
    # chi_2: process tomography of the real process
    # Compute process matrix
    Perm = perm()

    Lambda2 = lambda2()
    chi_2 = (Perm.dot(Lambda2)).T.dot(rhop).dot(Perm.dot(Lambda2))
    return chi_2


def factorized_CNOT_process_tomography(rhop):
    # only for fock, no need for SFB since we do not do the cnot
    # chi_2_error: process matrix once the perfect CNOT is fctorized
    # (to see only the errors)
    # In the perfect CNOT case, it gives a matrix with a 1 on
    # IIxII and 0 every where else

    # ChiError
    Y = sigmay().data.A
    Z = sigmaz().data.A
    X = sigmax().data.A
    U_CNOT = 1 / 2 * np.kron(np.eye(2) + Z, np.eye(2)) + 1 / 2 * np.kron(
        np.eye(2) - Z, X
    )
    OneQbasis = (np.eye(2), X, Y, Z)
    TwoQbasis = []
    for i in np.arange(0, 4):
        for j in np.arange(0, 4):
            TwoQbasis.append(np.kron(OneQbasis[i], OneQbasis[j]))

    rhopp = (
        (np.kron(np.eye(4), U_CNOT)).dot(rhop).dot(np.kron(np.eye(4), U_CNOT))
    )
    rhopp = (
        (np.kron(U_CNOT, np.eye(4))).dot(rhop).dot(np.kron(U_CNOT, np.eye(4)))
    )
    Lambda2 = lambda2()
    Perm = perm()
    chi_2_error = (Perm.dot(Lambda2)).T.dot(rhopp).dot(Perm.dot(Lambda2))
    # chi_2_error = (Perm.dot(Lambda2)).T.dot(rhop).dot(Perm.dot(Lambda2))
    return chi_2_error


def chi_2_CNOT():
    # chi_2_CNOT: chi_2 of the perfect CNOT unitary,
    # Ideal Chi
    chi_2_CNOT = np.zeros((16, 16))
    chi_2_CNOT[0, 0] = 1 / 4
    chi_2_CNOT[0, 1] = 1 / 4
    chi_2_CNOT[0, 12] = 1 / 4
    chi_2_CNOT[0, 13] = -1 / 4
    chi_2_CNOT[1, 0] = 1 / 4
    chi_2_CNOT[1, 1] = 1 / 4
    chi_2_CNOT[1, 12] = 1 / 4
    chi_2_CNOT[1, 13] = -1 / 4
    chi_2_CNOT[12, 0] = 1 / 4
    chi_2_CNOT[12, 1] = 1 / 4
    chi_2_CNOT[12, 12] = 1 / 4
    chi_2_CNOT[12, 13] = -1 / 4
    chi_2_CNOT[13, 0] = -1 / 4
    chi_2_CNOT[13, 1] = -1 / 4
    chi_2_CNOT[13, 12] = -1 / 4
    chi_2_CNOT[13, 13] = 1 / 4
    return chi_2_CNOT


def diagonal_real_part(chi_2):
    return np.real(np.diag(chi_2))


def build_error_model(chi_2, error_names: List[str]) -> Dict[str, float]:
    return dict(zip(error_names, diagonal_real_part(chi_2)))


def build_one_qubit_error_model(chi_2) -> Dict[str, float]:
    error_names = [
        'fid',
        'pX',
        'pY',
        'pZ',
    ]
    return build_error_model(chi_2, error_names)


def build_two_qubits_error_model(chi_2) -> Dict[str, float]:
    error_names = [
        'fid',
        'pX2',
        'pY2',
        'pZ2',
        'pX1',
        'pX1X2',
        'pX1Y2',
        'pX1Z2',
        'pY1',
        'pY1X2',
        'pY1Y2',
        'pY1Z2',
        'pZ1',
        'pZ1X2',
        'pZ1Y2',
        'pZ1Z2',
    ]
    return build_error_model(chi_2, error_names)
