import sqlite3
from pathlib import Path
from tempfile import mkdtemp

import pytest

from sca import SCA


@pytest.fixture(scope="module")
def tsv_file():
    here = Path(__file__).parent
    tsv_path = here / "uk_hansard_100rows.tsv"
    return tsv_path


@pytest.fixture(scope="module")
def temp_dir():
    return Path(mkdtemp())


@pytest.fixture(scope="module")
def sca_filled(temp_dir, tsv_file):
    db_path = temp_dir / "sca.sqlite3"
    sca = SCA(db_path=db_path, tsv_path=tsv_file)
    return sca


def test_count_entries(sca_filled):
    count, *_ = sca_filled.conn.execute("select count(*) from raw").fetchone()
    assert count == 100


def test_columns(sca_filled):
    cursor = sca_filled.conn.execute("select * from raw")
    columns = [_[0] for _ in cursor.description]
    assert columns == [
        "speech_id",
        "year",
        "date",
        "parliament",
        "topic",
        "function",
        "speaker_id",
        "party",
        "party_in_power",
        "cabinet",
        "district",
        "district_class",
        "times_in_house",
        "seniority",
        "speech_text",
    ]
