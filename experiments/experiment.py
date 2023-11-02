from abc import ABC, abstractmethod

from numpy import sqrt

from qsim import VERSION


class Experiment(ABC):
    def __init__(self, verbose: bool = False):
        self.VERSION = VERSION
        # self.circuit_path = f'{qsim_repo_path}/experiments/circuit.png'
        self.verbose = verbose

    @abstractmethod
    def single_simulation(self):
        pass

    # @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def plot(self, **kwargs):
        pass


class ExperimentPhysicalParameters(Experiment):
    def __init__(
        self,
        nbar: int,
        k2a: float,
        k2: float,
        k1: float,
        k1a: float,
        gate_time: float,
        N: int,
        verbose: bool = False,
    ):
        super().__init__()
        self.nbar = nbar
        self.alpha = sqrt(nbar)
        self.k1 = k1
        self.k2 = k2
        self.gate_time = gate_time
        self.k2a = k2a
        self.k1a = k1a
        self.N = N
        self.verbose = verbose
