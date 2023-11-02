import datetime
from typing import Any, List, cast

import numpy as np
from qutip import Qobj


def serialize_complex(obj: complex) -> List[float]:
    return [obj.real, obj.imag]


def serialize_date(obj: datetime.datetime) -> str:
    return obj.isoformat()


def serialize_ndarray(obj: np.ndarray) -> Any:
    if obj.ndim > 0 and obj.dtype in (np.complex64, np.complex128):
        return [obj.real.tolist(), obj.imag.tolist()]

    tolist = obj.tolist()
    if obj.ndim == 0:  # tolist return a scalar value
        if isinstance(tolist, complex):
            return serialize_complex(cast(complex, tolist))
        return tolist
    return tolist


def serialize_qobj(obj: Qobj) -> Any:
    return serialize_ndarray(obj.data.A)


def serialize(obj: Any) -> Any:  # pylint: disable=too-many-return-statements
    """Serialize `obj`, that is transform it into a primitive type.

    Primitive types are `bool`, `int`, `float`, `str` and `dict` and `list`
    of these types.

    Raises:
        ValueError: if the provided object cannot be serialized.
    """

    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [serialize(x) for x in obj]

    if isinstance(obj, complex):
        return serialize_complex(obj)

    if isinstance(obj, datetime.datetime):
        return serialize_date(obj)

    if isinstance(obj, np.ndarray):
        return serialize_ndarray(obj)

    if isinstance(obj, Qobj):
        return serialize_qobj(obj)

    if isinstance(obj, (bool, int, float, str, np.integer)) or obj is None:
        return obj

    raise ValueError('`obj` cannot be serialized')
