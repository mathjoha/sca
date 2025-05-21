import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from sca import SCA


@pytest.fixture
def sca_instance(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    tsv_content = (
        "id\ttext\tparliament\tparty\tparty_in_power\tdistrict_class\tseniority\n"
        "1\tHello world, this is a test.\t1\tA\tGov\tUrban\t1\n"
        "2\tAnother sentence with world hello.\t1\tB\tOpp\tRural\t2\n"
        "3\tHello again dear world, how are you?\t2\tA\tGov\tUrban\t3\n"
        "4\tThis is a new world for us.\t2\tC\tOpp\tRural\t1\n"
        "5\tNo target words here.\t1\tB\tOpp\tUrban\t2"
    )
    tsv_path = tmp_path / "test.tsv"
    with open(tsv_path, "w") as f:
        f.write(tsv_content)

    sca = SCA()
    sca.read_file(
        tsv_path=tsv_path, id_col="id", text_column="text", db_path=db_path
    )
    sca.columns = {
        "parliament",
        "party",
        "party_in_power",
        "district_class",
        "seniority",
    }
    sca.set_data_cols()
    return sca


def test_add_and_mark_collocates(sca_instance):
    sca_instance.add_collocates([("hello", "world")])

    conn = sqlite3.connect(sca_instance.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM collocate_window WHERE window IS NOT NULL")
    rows = cursor.fetchall()
    conn.close()

    assert (
        len(rows) > 0
    ), "collocate_window table should contain actual collocations"
    assert len(rows) == 3, "Expected 3 collocations for 'hello' and 'world'"

    found_collocation_for_speech1 = any(
        row[0] == "1" and row[1] == "hello" and row[2] == "world"
        for row in rows
    )
    assert (
        found_collocation_for_speech1
    ), "Expected collocation for speech_id '1'"

    assert "hello" in sca_instance.terms
    assert "world" in sca_instance.terms
    assert ("hello", "world") in sca_instance.collocates
