import sqlite3
from pathlib import Path
from tempfile import mkdtemp

import pytest
from yaml import safe_load

import sca


@pytest.fixture(scope="module")
def tsv_file():
    here = Path(__file__).parent
    tsv_path = here / "uk_hansard_100rows.csv"
    return tsv_path


@pytest.fixture(scope="module")
def temp_dir():
    return Path(mkdtemp())


@pytest.fixture(scope="module")
def sca_filled(temp_dir, tsv_file):
    db_path = temp_dir / "sca.sqlite3"
    corpus = sca.from_tsv(
        db_path=db_path,
        tsv_path=tsv_file,
        id_col="speech_id",
        text_column="speech_text",
    )
    corpus.add_collocates((("govern*", "minister*"),))
    corpus.save()
    return corpus


@pytest.fixture(scope="module")
def speeches_with_collocates(sca_filled):
    return sca_filled.conn.execute("select * from collocate_window").fetchall()


@pytest.fixture(scope="module")
def settings(sca_filled):
    return sca_filled.settings_dict()


@pytest.fixture(scope="module")
def yml_settings(sca_filled):
    with open(sca_filled.yaml_path, "r", encoding="utf8") as f:
        return safe_load(f)


@pytest.fixture(scope="module")
def yml_loaded(sca_filled):
    return sca.from_yml(sca_filled.yaml_path)


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


def test_name_id_col():
    corpus = SCA(db_path=db_path, tsv_path=tsv_file, id_col="id_col_name")
    assert corpus.id_col == "id_col_name"


def test_name_id_col(sca_filled):
    assert sca_filled.id_col == "speech_id"


class TestSavedSettings:
    def test_collocates(self, settings):
        assert settings["collocates"] == {
            ("govern*", "minister*"),
        }

    def test_stored(self, settings, sca_filled):
        assert set(settings["collocates"]) == sca_filled.collocates

    def test_yml_collocates(self, yml_settings, settings):
        assert (
            set(tuple(collocate) for collocate in yml_settings["collocates"])
            == settings["collocates"]
        )

    def test_yaml_read_collocates(self, yml_loaded, sca_filled):
        assert (
            yml_loaded.settings_dict()["collocates"]
            == sca_filled.settings_dict()["collocates"]
        )

    def test_yaml_read_dbpath(self, yml_loaded, sca_filled):
        assert (
            yml_loaded.settings_dict()["db_path"]
            == sca_filled.settings_dict()["db_path"]
        )

    def test_keys(self, yml_loaded, sca_filled):
        assert (
            yml_loaded.settings_dict().keys()
            == sca_filled.settings_dict().keys()
        )
