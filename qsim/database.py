from __future__ import annotations

from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Union

from tinydb import Query, TinyDB, where


def _make_and_query(conditions: Dict[str, Any]) -> Query:
    """Transform a `dict` into a `Query` where the equality conditions
    `key == value` are ANDed."""
    return reduce(
        lambda a, b: a & b,
        [where(field) == value for field, value in conditions.items()],
    )


class Database:
    """Thin wrapper around a `tinydb.TinyDB` object which implements a lock
    mechanism to avoid concurrent writing.

    Example:
        >>> with Database(...) as db:
        ...     db.match(k1=3, k2=4)
    """

    def __init__(self, db_path: Union[str, Path]):
        self._db_path = Path(db_path)
        self._lock_path = self._db_path.with_suffix('.lock')

        if self._lock_path.exists():
            raise RuntimeError(
                f'The database {self._db_path} is in use. If you think that is '
                f'an error, remove {self._lock_path}.'
            )
        self._lock_path.touch()

        self._tiny_db = TinyDB(db_path)

    def close(self):
        try:
            self._lock_path.unlink()
        except FileNotFoundError:
            pass
        self._tiny_db.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __len__(self):
        return len(self._tiny_db)

    def all(self):
        """Return every document in the database."""
        return self._tiny_db.all()

    def iter(self):
        """Iterates over the documents in the database."""
        return iter(self._tiny_db)

    def insert(self, data: Dict[str, Any]):
        """Insert `data` into the database.

        Note:
            This method does not check for duplicates.
        """
        self._tiny_db.insert(data)

    def insert_multiple(self, data: List[Dict[str, Any]]):
        """Insert several entries into the database.

        This method is faster than repeatedly calling `insert`.

        Note:
            This method does not check for duplicates.
        """
        self._tiny_db.insert_multiple(data)

    def search(self, query: Query) -> List[Dict[str, Any]]:
        return self._tiny_db.search(query)

    def match(self, **kwargs) -> List[Dict[str, Any]]:
        """Return the entries where each specified argument is equal to the
        specified value.

        Example:
            >>> with Database(...) as db:
            >>>     db.match(nbar=3, gate_time=1.3)
            ...     # return all the entries with nbar = 3 and gate_time = 1.3
        """
        return self.search(_make_and_query(kwargs))

    def update(self, data: Dict[str, Any], query: Query):
        """Update the entries selected by `query` with `data`."""
        return self._tiny_db.update(data, query)

    def update_matching(self, data: Dict[str, Any], **kwargs):
        """Update the entries where each specified argument is equal to the
        specified value with `data`.

        Example:
            >>> with Database(...) as db:
            >>>     db.update({'state': ...}, nbar=1, k1=3)
            ...     # change the state of the entries where nbar=1, k1=3
        """
        return self.update(data, _make_and_query(kwargs))

    def upsert(self, data: Dict[str, Any], query: Query):
        """Update the entries selected by `query` with `data`. If no entry is
        found, insert `data` into the database."""
        return self._tiny_db.upsert(data, query)

    def upsert_matching(self, data: Dict[str, Any], **kwargs):
        """Update the entries where each specified argument is equal to the
        specified value with `data`. If no entry is found, insert `data` into
        the database.

        Example:
            >>> with Database(...) as db:
            >>>     db.upsert({'state': ...}, nbar=1, k1=3)
            ...     # update the entries where nbar=1, k1=3, or insert a new one
        """
        return self.upsert(data, _make_and_query(kwargs))
