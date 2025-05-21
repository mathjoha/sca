from pathlib import Path

import pytest

from sca.corpus import SCA

TEST_DIR = Path(__file__).parent


@pytest.mark.xfail(
    strict=True, reason="Red Phase: Test for dynamic header processing"
)
def test_dynamic_headers():
    test_file = TEST_DIR / "dynamic_headers_test.csv"
    db_file = TEST_DIR / "test_dynamic_headers.sqlite3"
    if db_file.exists():
        db_file.unlink()

    corpus = SCA()
    corpus.read_file(
        tsv_path=test_file,
        id_col="Unique ID",
        text_column="Main Text Content",
        db_path=db_file,
    )

    expected_data_cols = {
        "meta_data_one",
        "another_meta_column",
        "a_final_header",
    }
    assert set(corpus.columns) == expected_data_cols

    settings = corpus.settings_dict()
    assert "columns" in settings
    assert set(settings["columns"]) == expected_data_cols

    if db_file.exists():
        db_file.unlink()
