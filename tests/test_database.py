import datetime
import shutil
from pathlib import Path

import pytest
from tinydb import Query

from qsim.database import Database
from qsim.serializer import serialize_date


@pytest.fixture
def db_path(tmpdir) -> Path:
    database_path = Path('tests') / 'data' / 'test_database.json'
    shutil.copy(database_path, tmpdir)
    return Path(tmpdir) / database_path.name


@pytest.fixture
def db(db_path: Path) -> Database:
    return Database(db_path)


class TestDatabase:
    def test_lock(self, tmpdir):
        db = Database(tmpdir / 'db.json')
        with pytest.raises(RuntimeError):
            Database(tmpdir / 'db.json')

        db.close()
        Database(tmpdir / 'db.json')

    def test_context_manager(self, tmpdir):
        with Database(tmpdir / 'db.json'):
            with pytest.raises(RuntimeError):
                Database(tmpdir / 'db.json')

        Database(tmpdir / 'db.json')

    def test_len(self, db: Database):
        assert len(db) == 3

    def test_search(self, db: Database):
        Q = Query()
        assert len(db.search(Q.nbar == 4)) == 2
        assert len(db.search(Q.nbar == 42)) == 0
        assert (
            len(
                db.search(
                    (Q.nbar == 4) & (Q.datetime == "2021-10-29T15:44:30.827183")
                )
            )
            == 1
        )
        assert (
            len(
                db.search(
                    (Q.datetime >= "2021-10-01") & (Q.datetime <= "2021-10-30")
                )
            )
            == 1
        )
        assert len(db.search((Q.datetime > "2021-11-01") & (Q.nbar <= 3))) == 1

    def test_match(self, db: Database):
        assert len(db.match(nbar=4)) == 2
        assert len(db.match(nbar=42)) == 0
        assert len(db.match(nbar=4, datetime="2021-10-29T15:44:30.827183")) == 1

    def test_update(self, db: Database):
        Q = Query()

        d = datetime.datetime(1990, 1, 1)
        assert len(db.search(Q.datetime == serialize_date(d))) == 0
        db.update({'datetime': serialize_date(d)}, Q.nbar == 42)
        assert len(db.search(Q.datetime == serialize_date(d))) == 0

        assert len(db.search(Q.nbar == 100)) == 0
        db.update(
            {'nbar': 100},
            (Q.datetime >= "2021-10-01") & (Q.datetime <= "2021-10-30"),
        )
        assert len(db.search(Q.nbar == 100)) == 1

    def test_update_matching(self, db: Database):
        Q = Query()

        d = datetime.datetime(1990, 1, 1)
        assert len(db.search(Q.datetime == serialize_date(d))) == 0
        db.update_matching({'datetime': serialize_date(d)}, nbar=42)
        assert len(db.search(Q.datetime == serialize_date(d))) == 0

        d = datetime.datetime(2000, 1, 1)
        assert len(db.search(Q.datetime == serialize_date(d))) == 0
        db.update_matching({'datetime': serialize_date(d)}, nbar=4)
        assert len(db.search(Q.datetime == serialize_date(d))) == 2

    def test_upsert(self, db: Database):
        Q = Query()

        d = datetime.datetime(1990, 1, 1)
        assert len(db.search(Q.datetime == serialize_date(d))) == 0
        db.upsert({'datetime': serialize_date(d)}, Q.nbar == 4)
        assert len(db.search(Q.datetime == serialize_date(d))) == 2

        assert len(db) == 3
        db.upsert(
            {
                'datetime': serialize_date(datetime.datetime.now()),
                'nbar': 30,
                'state': [],
            },
            Q.nbar == 23,
        )
        assert len(db) == 4

    def test_upsert_matching(self, db: Database):
        d = datetime.datetime(1990, 1, 1)
        assert len(db.match(datetime=serialize_date(d))) == 0
        db.upsert_matching({'datetime': serialize_date(d)}, nbar=4)
        assert len(db.match(datetime=serialize_date(d))) == 2

        assert len(db) == 3
        db.upsert_matching(
            {
                'datetime': serialize_date(datetime.datetime.now()),
                'nbar': 30,
                'state': [],
            },
            nbar=23,
        )
        assert len(db) == 4
