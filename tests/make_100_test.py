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


@pytest.fixture(scope="module")
def speeches_with_collocates(sca_filled):
    sca_filled.add_collocates((("govern*", "minister*"),))
    return sca_filled.conn.execute("select * from collocate_window").fetchall()


def test_collocate_len(speeches_with_collocates):
    assert len(speeches_with_collocates) == 20


def test_collocate_min(speeches_with_collocates):
    assert min(w for *_, w in speeches_with_collocates) == 1


def test_collocate_max(speeches_with_collocates):
    assert max(w for *_, w in speeches_with_collocates) == 82


def test_collocate_sum(speeches_with_collocates):
    assert sum(w for *_, w in speeches_with_collocates) == 384


def test_collocate_lenw10(speeches_with_collocates):
    assert len([w for *_, w in speeches_with_collocates if w <= 10]) == 9


@pytest.mark.xfail(strict=True, reason="Red Phase")
def test_name_id_col():
    sca = SCA(db_path=db_path, tsv_path=tsv_file, id_col="id_col_name")
    assert sca.id_col == "id_col_name"


@pytest.mark.xfail(strict=True, reason="Red Phase")
def test_name_id_col(sca_filled):
    assert sca_filled.id_col == "speech_id"
