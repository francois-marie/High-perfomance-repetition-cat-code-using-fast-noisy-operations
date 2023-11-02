import datetime

import numpy as np
from qutip import Qobj

from qsim.serializer import (
    serialize,
    serialize_complex,
    serialize_date,
    serialize_ndarray,
    serialize_qobj,
)


class TestSerializer:
    def test_serialize_date(self):
        now = datetime.datetime.now()
        assert serialize_date(now) == now.isoformat()
        assert (
            serialize_date(datetime.datetime(2020, 5, 3, 11, 3, 20))
            == '2020-05-03T11:03:20'
        )

    def test_serialize_complex(self):
        assert serialize_complex(1.2 + 3j) == [1.2, 3.0]
        assert serialize_complex(-4.4j) == [0.0, -4.4]

    def test_serialize_ndarray(self):
        assert serialize_ndarray(np.array([[4.3, 5], [-1.8, 2]])) == [
            [4.3, 5],
            [-1.8, 2],
        ]

        assert serialize_ndarray(
            np.array(
                [
                    [1.3 + 5j, 2 + 1.4j],
                    [-1, 1j],
                ]
            )
        ) == [[[1.3, 2.0], [-1.0, 0.0]], [[5.0, 1.4], [0.0, 1.0]]]

    def test_serialize_qobj(self):
        array = np.array([[1.2 + 3j, 2.4 - 0.1j], [3.8 + 3j, 5.1]])
        obj = Qobj(inpt=array)
        assert serialize_qobj(obj) == [
            [[1.2, 2.4], [3.8, 5.1]],
            [[3, -0.1], [3, 0.0]],
        ]

    def test_serialize(self):
        now = datetime.datetime.now()

        assert serialize(
            {
                'date': now,
                'nested': [1, 2.3, True, None],
                'complex': 1.2 + 3j,
                'array': np.array([[1, 4], [3, 2]]),
                'complex_array': np.array([[1 + 2j, 9 - 3j]]),
            }
        ) == {
            'date': now.isoformat(),
            'nested': [1, 2.3, True, None],
            'complex': [1.2, 3.0],
            'array': [[1, 4], [3, 2]],
            'complex_array': [[[1, 9]], [[2, -3]]],
        }
