from pathlib import Path

import pytest

from sca.corpus import SCA  # Assuming SCA is in sca.corpus

TEST_DIR = Path(__file__).parent


@pytest.mark.xfail(
    strict=True, reason="Red Phase: Test for dynamic header processing"
)
def test_dynamic_headers():
    test_file = TEST_DIR / "dynamic_headers_test.csv"
    db_file = TEST_DIR / "test_dynamic_headers.sqlite3"
    if db_file.exists():
        db_file.unlink()  # Ensure a clean database for each test run

    corpus = SCA()
    # Use header names as they appear in the CSV
    corpus.read_file(
        tsv_path=test_file,
        id_col="Unique ID",
        text_column="Main Text Content",
        db_path=db_file,
    )

    # Check if the columns were correctly identified and stored,
    # (case-insensitively and with spaces replaced by underscores)
    expected_data_cols = {
        "meta_data_one",
        "another_meta_column",
        "a_final_header",
    }
    assert set(corpus.columns) == expected_data_cols

    # Verify that data can be queried using the dynamic columns
    # This is a basic check, more comprehensive checks can be added
    # depending on how these columns are used.
    # For instance, checking if 'counts_by_subgroups' works with these new columns.

    # For now, we'll just check if the columns are present in the settings_dict
    # as this is a good proxy for them being correctly processed.
    settings = corpus.settings_dict()
    assert "columns" in settings
    assert set(settings["columns"]) == expected_data_cols

    # Clean up the created database file
    if db_file.exists():
        db_file.unlink()
