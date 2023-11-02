from typing import Any, Dict, Iterable

from qsim import VERSION
from qsim.gate import Gate


class FakeGate(Gate):
    def _simulate(self) -> Dict[str, Any]:
        pass

    def _get_results(self) -> Dict[str, Any]:
        pass

    @staticmethod
    def get_non_key_attributes() -> Iterable[str]:
        pass


class TestGate:
    def test_version(self):
        assert FakeGate().version == VERSION

    def test_class_name(self):
        assert FakeGate().class_name == 'FakeGate'
