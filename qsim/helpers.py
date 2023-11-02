import multiprocessing
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
from qutip import Qobj

from qsim.database import Database
from qsim.gate import Gate
from qsim.logical_gate.lgate import LGate
from qsim.physical_gate.pgate import PGate
from qsim.serializer import serialize


def base_data_path() -> Path:
    return Path(__file__).parent.parent.absolute() / 'data'


data_path = base_data_path()


def _default_db_path(gate_class: Type[Gate]) -> Path:
    single_db_path = base_data_path()

    if issubclass(gate_class, PGate):
        single_db_path /= 'physical_gate'
    elif issubclass(gate_class, LGate):
        single_db_path /= 'logical_gate'

    return single_db_path / f'{gate_class.__name__}.json'


def all_simulations(
    gate_class_or_db_path: Union[str, Type[Gate]]
) -> List[Dict[str, Any]]:
    """Return all the simulations in a database.

    Args:
        gate_class_or_db_path: if a gate class is provided, the database path
            is automatically determined. Otherwise, that's the path to the
            database.

    Example:
        >>> all_simulations(Gate)
        >>> all_simulations('my_database.json')
    """
    db_path = (
        gate_class_or_db_path
        if isinstance(gate_class_or_db_path, str)
        else _default_db_path(gate_class_or_db_path)
    )
    with Database(db_path) as db:
        return db.all()


def first_simulation(
    gate_class_or_db_path: Union[str, Type[Gate]]
) -> Dict[str, Any]:
    """Iterates over the simulations in a database.

    Args:
        gate_class_or_db_path: if a gate class is provided, the database path
            is automatically determined. Otherwise, that's the path to the
            database.

    Example:
        >>> first_simulation(Gate)
        >>> first_simulation('my_database.json')
    """
    db_path = (
        gate_class_or_db_path
        if isinstance(gate_class_or_db_path, str)
        else _default_db_path(gate_class_or_db_path)
    )
    with Database(db_path) as db:
        return next(db.iter())


def possible_keys(gate_class_or_db_path: Union[str, Type[Gate]]) -> List[str]:
    """Returns the keys of the simulations in a database.

    Args:
        gate_class_or_db_path: if a gate class is provided, the database path
            is automatically determined. Otherwise, that's the path to the
            database.

    Example:
        >>> iter_simulations(Gate)
        >>> iter_simulations('my_database.json')
    """
    it = first_simulation(gate_class_or_db_path)
    return list(it.keys())


def search_simulation(
    gate_class_or_db_path: Union[str, Type[Gate]], **kwargs
) -> List[Dict[str, Any]]:
    """Read a database and return the simulations matching the provided
    keyword arguments.

    Args:
        gate_class_or_db_path: if a gate class is provided, the database path
            is automatically determined. Otherwise, that's the path to the
            database.
        kwargs: the search conditions.

    Example:
        >>> search_simulation(Gate, k1=3, k2=4)
    """
    if len(kwargs) == 0:
        raise ValueError('no arguments specified')

    db_path = (
        gate_class_or_db_path
        if isinstance(gate_class_or_db_path, str)
        else _default_db_path(gate_class_or_db_path)
    )
    with Database(db_path) as db:
        return db.match(**kwargs)


class _SimulateGateInner:
    """This routine returns the result of `Gate.simulate` called with specific
    arguments."""

    def __init__(self, gate_class: Type[Gate]):
        self._gate_class = gate_class

    def __call__(self, params):
        return self._gate_class(**params).simulate()


class _SimulatePGateOuter:
    """This routine calls `_SimulateGateInner` with every argument in `params`
    and records the results in the database at `db_path`.

    If an entry is a duplicate of an already existing entry, the old entry is
    replaced with the new data.
    """

    def __init__(
        self,
        gate_class: Type[PGate],
        params: Iterable[Dict[str, Any]],
        db_path: str,
        n_proc: int = 2,
    ):
        self._gate_class = gate_class
        self._params = params
        self._db_path = db_path
        self._n_proc = n_proc

    def __call__(self):
        with Database(self._db_path) as db:
            with multiprocessing.Pool(self._n_proc) as pool:
                results = pool.map(
                    _SimulateGateInner(self._gate_class), self._params
                )

            non_key_attributes = self._gate_class.get_non_key_attributes()
            for result in results:
                conditions = {
                    k: v
                    for k, v in result.items()
                    if k not in non_key_attributes
                }
                db.upsert_matching(serialize(result), **conditions)


class _SimulateLGateOuter:
    """This routine calls `_SimulateGateInner` with every argument in `params`
    and records the results in the database at `db_path`.

    If an entry is a duplicate of an already existing entry and
    `overwrite` is `False`, the entries are
    merged according to this strategy:
      * `LGate.outcome` are rebalanced depending on the number of trajectories.
      * `LGate.num_trajectories` are added.

    If `overwrite` is `True`, the old entry is overwritten.
    """

    def __init__(
        self,
        gate_class: Type[LGate],
        params: Iterable[Dict[str, Any]],
        db_path: str,
        n_proc: int = 2,
        overwrite: bool = False,
    ):
        self._gate_class = gate_class
        self._params = params
        self._db_path = db_path
        self._n_proc = n_proc
        self._overwrite = overwrite

    @staticmethod
    def _merge_states(
        existing: Dict[str, Any], update: Dict[str, Any]
    ) -> Tuple[Dict[str, int], Dict[str, float]]:
        """Merge two `state` dictionaries from `LGate._get_results`."""
        new_num_trajectories = {
            k: existing['num_trajectories'][k] + update['num_trajectories'][k]
            for k in ('X', 'Y', 'Z')
        }

        # compute the weighted average between the two outcomes
        new_outcome = {'X': None, 'Y': None, 'Z': None}
        for k in ('X', 'Y', 'Z'):
            if new_num_trajectories[k] != 0:
                e = (
                    0
                    if existing['outcome'][k] is None
                    else existing['num_trajectories'][k]
                    * existing['outcome'][k]
                )
                u = (
                    0
                    if update['outcome'][k] is None
                    else update['num_trajectories'][k] * update['outcome'][k]
                )
                new_outcome[k] = (e + u) / new_num_trajectories[k]

        return new_num_trajectories, new_outcome

    def __call__(self):
        with Database(self._db_path) as db:
            with multiprocessing.Pool(self._n_proc) as pool:
                results = pool.map(
                    _SimulateGateInner(self._gate_class), self._params
                )

            non_key_attributes = self._gate_class.get_non_key_attributes()
            for result in results:
                result = serialize(result)
                conditions = {
                    k: v
                    for k, v in result.items()
                    if k not in non_key_attributes
                }

                duplicates = db.match(**conditions)

                if len(duplicates) == 0:
                    db.insert(result)
                    continue

                if self._overwrite:
                    new_outcome = result['state']['outcome']
                    new_num_trajectories = result['state']['num_trajectories']
                else:
                    duplicate = max(duplicates, key=lambda x: x['datetime'])
                    new_num_trajectories, new_outcome = self._merge_states(
                        duplicate['state'], result['state']
                    )

                db.update_matching(
                    {
                        'state': {
                            'outcome': new_outcome,
                            'num_trajectories': new_num_trajectories,
                        }
                    },
                    **conditions,
                )


def simulate(
    gate_class: Type[Gate],
    params: Iterable[Dict[str, Any]],
    db_path: Optional[str] = None,
    n_proc: int = 2,
    overwrite_logical: bool = False,
) -> multiprocessing.Process:
    """Simulate a gate with the provided parameters using a pool of workers
    and write the results to the database.

    Args:
        gate_class: the class of the gate to simulate.
        params: an iterable of dictionaries. Each dictionary is unpacked as
            keyword parameters to the `__init__` method of `gate_class`.
        db_path: the path to the database. If `None`, the database path is
            automatically determined.
        n_proc: number of processors used by the multiprocessing pool.
        overwrite_logical: if `True` and the gate is a logical gate, the
            simulation will overwrite an older simulation. This parameter has no
            effect if the gate is a physical gate.

    Note:
        For a physical gate, in case of conflict, the old entry is replaced with
        the new entry.

        For a logical gate, in case of conflict, if `overwrite_logical` is
        `False`, the new entry is merged with the new results according to this
        strategy:
            * `outcome` is rebalanced depending on the number of trajectories.
            * `num_trajectories` are added.
        If `overwrite_logical` is `True`, the new entry replaces the old entry.
    """

    print(f'{n_proc=}')
    db_path = _default_db_path(gate_class) if db_path is None else db_path

    if issubclass(gate_class, PGate):
        target = _SimulatePGateOuter(gate_class, params, str(db_path), n_proc)
    elif issubclass(gate_class, LGate):
        target = _SimulateLGateOuter(
            gate_class, params, str(db_path), n_proc, overwrite_logical
        )
    else:
        raise ValueError('unsupported gate class')

    p = multiprocessing.Process(target=target)
    p.start()
    return p


def gen_params_simulate(**kwargs) -> List[Dict[str, Any]]:
    """Generates an iterable (list) of parameters for simulate().
    Each list item is a dict with keys corresponding to the gate parameters.
    The kwargs of this methods are thus the parameters of the gate
    followed by '_l' with 'l' for 'list'.

    Args:
        kwargs: the parameters to simulate.

    Example:
        >>>gen_params_simulate(
        p_l = np.logspace(-2, -1, 11),
        distance_l=range(3, 8, 2),
        num_rounds_l = range(4, 20, 2)
        )
    """
    keys = [key[:-2] for (key, _) in kwargs.items()]
    return [dict(zip(keys, param)) for param in product(*kwargs.values())]


def format_logical_search(
    gate_class_or_db_path: Union[str, Type[Gate]],
    keys: Optional[List[str]] = None,
    **kwargs,
):
    if keys is None:
        keys = ['p', 'q', 'num_rounds']
    res = search_simulation(gate_class_or_db_path, **kwargs)
    res_dict = {}
    for single_res in res:
        f = single_res['state']['outcome']['Z']
        N = single_res['state']['num_trajectories']['Z']
        res_dict.update({tuple(single_res[key] for key in keys): (f, N)})
    return res_dict


def build_rho_from_db_res(
    single_res: Dict[str, Any],
    N_sfb: Optional[int] = None,
    N_fock: Optional[int] = None,
) -> Qobj:
    return build_rho_from_list(
        arr_list=single_res['state'], N_sfb=N_sfb, N_fock=N_fock
    )


def build_rho_from_list(
    arr_list: list,
    N_sfb: Optional[int] = None,
    N_fock: Optional[int] = None,
) -> Qobj:
    return build_rho_from_arr(
        rho=np.array(arr_list, dtype=complex), N_sfb=N_sfb, N_fock=N_fock
    )


def build_rho_from_arr(
    rho: np.ndarray,
    N_sfb: Optional[int] = None,
    N_fock: Optional[int] = None,
) -> Qobj:
    if N_sfb is not None:
        dims = [[2, N_sfb, 2, N_sfb], [2, N_sfb, 2, N_sfb]]
    elif N_fock is not None:
        dims = [[N_fock, N_fock], [N_fock, N_fock]]
    else:
        return Qobj(dims=[])

    rho = rho[0] + 1.0j * rho[1]
    rho = Qobj(rho, dims=dims)
    return rho
