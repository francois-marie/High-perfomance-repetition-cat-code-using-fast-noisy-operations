from abc import ABC, abstractmethod
from time import time
from typing import Any, Dict, Iterable, Optional

from qsim import VERSION


class Gate(ABC):
    """Abstract base class for all gates.

    General lifecycle of a gate:
        1. Initialization. The initialization should be lazy, meaning that no
           heavy computation is done inside the `__init__()` method (leave that
           work to `_pre_simulate()` or `_simulate()`).
        2. Simulation. The process for a gate simulation is implemented in
           `simulate()`:
            a. The `_pre_simulate()` method is called. This method should
               initialize attributes used in `_simulate()` but not provided to
               `__init__()`.
            b. The `_simulate()` method is called. This method implements the
               simulation logic of the gate.
            c. The `_get_results()` method is called, and the dict it returns is
               put in `self.results`.
            d. `simulate()` returns `self.results`.
        3. The user may then change some attributes of the gate directly and
           call `simulate()` again.
    """

    def __init__(self, verbose: bool = False):
        self.results: Optional[Dict[str, Any]] = None
        """Contains the results of the gate simulation, as returned by
        `_get_results`. `None` if `simulate` has not been called yet."""

        self.elapsed_time: Optional[float] = None
        """Global time taken by the simulation, in seconds. `None` if `simulate`
        has not been called yet."""

        self.version = VERSION  # may be used by `_get_results`
        self.class_name = self.__class__.__name__  # same
        self.verbose = verbose

    def verb(self, text: str):
        if self.verbose:
            print(text)

    def simulate(self) -> Dict[str, Any]:
        """Simulate the gate."""
        self._pre_simulate()

        start = time()
        self._simulate()
        self.elapsed_time = time() - start
        self.verb(f'Elapsed time: {self.elapsed_time:.2f}s')

        self.results = self._get_results()
        return self.results

    def _pre_simulate(self):
        """This method is called at the beginning of `simulate()`. It may be
        overridden to set attributes used by the simulation which are not
        directly provided to `__init__()`.

        You usually want to call `super()._pre_simulate()` when overriding this
        method.
        """

    @abstractmethod
    def _simulate(self) -> Dict[str, Any]:
        """Implement the simulation logic of the gate."""

    @abstractmethod
    def _get_results(self) -> Dict[str, Any]:
        """Return the result of the simulation as a dict."""

    @staticmethod
    @abstractmethod
    def get_non_key_attributes() -> Iterable[str]:
        """Return the keys of `_get_results` which do not qualify as "key
        attributes". Key attributes are the attributes used to detect
        duplicates in the database.

        In other words, attributes listed here are not checked when detecting
        duplicates."""
        return ()
