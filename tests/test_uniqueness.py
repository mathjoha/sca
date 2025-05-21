import sqlite3
from pathlib import Path

import pytest

from sca.corpus import (
    SCA,  # Assuming src is in PYTHONPATH or project is installed editable
)


# Helper to create a minimal TSV for testing
def create_minimal_tsv(
    tmp_path: Path,
    filename: str = "test.tsv",
    content: str = "id\ttext\n1\tpatterna then patternb\n",
) -> Path:
    """Creates a minimal TSV file in tmp_path with given content."""
    tsv_path = tmp_path / filename
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write(content)
    return tsv_path


@pytest.mark.xfail(
    strict=True,
    reason="Red Phase: DB uniqueness for collocate_window not yet implemented",
)
def test_collocate_window_prevents_duplicates_at_db_level(tmp_path: Path):
    """
    Tests that an attempt to insert a duplicate row into the 'collocate_window'
    table would raise an IntegrityError if uniqueness constraints were active.
    This test is expected to fail (xfail) in the Red Phase as constraints
    are not yet implemented.
    """
    db_path = tmp_path / "test.sqlite3"
    # Using a specific text content to make window calculation predictable for the test.
    # Text: "text1	patterna then patternb"
    # id_col='id', text_column='text'
    # sca_instance.text_column in collocate_window will store 'text1' (the id_col value).
    tsv_path = create_minimal_tsv(
        tmp_path, content="id\ttext\ntext1\tpatterna then patternb\n"
    )

    sca_instance = SCA()
    sca_instance.read_file(
        tsv_path=tsv_path, id_col="id", text_column="text", db_path=db_path
    )

    pattern1 = "patterna"
    pattern2 = "patternb"

    # Call mark_windows once to populate the table.
    # pattern1 and pattern2 are sorted ('patterna', 'patternb') and lowercased by mark_windows.
    sca_instance.mark_windows(pattern1, pattern2)

    # Connect to the database to verify and attempt duplicate insertion
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Verify that one row was inserted as expected.
    # The value for the foreign key (sca_instance.text_column) in collocate_window
    # is 'text1' from the id_col of the TSV.
    # Patterns are 'patterna', 'patternb'.
    cursor.execute(
        f"SELECT COUNT(*) FROM collocate_window WHERE {sca_instance.text_column}=? AND pattern1=? AND pattern2=?",
        ("text1", "patterna", "patternb"),
    )
    count = cursor.fetchone()[0]
    assert count == 1, "Initial data not inserted by mark_windows as expected"

    # Retrieve the exact row that was inserted to attempt a duplicate insert.
    # Order of columns in SELECT should match VALUES (?, ?, ?, ?) later.
    # Table columns are defined as: {self.text_column: str, "pattern1": str, "pattern2": str, "window": int}
    cursor.execute(
        f"SELECT {sca_instance.text_column}, pattern1, pattern2, window FROM collocate_window WHERE {sca_instance.text_column}=? AND pattern1=? AND pattern2=?",
        ("text1", "patterna", "patternb"),
    )
    inserted_row_data = cursor.fetchone()
    assert (
        inserted_row_data is not None
    ), "Failed to retrieve the initially inserted row"
    # Expected inserted_row_data: ('text1', 'patterna', 'patternb', 2)
    # 'text1' is the speech_id (value of id_col)
    # 'patterna', 'patternb' are the sorted, lowercased patterns
    # 2 is the window for "patterna then patternb" (pos 0, pos 2, diff 2)

    # Attempt to insert the exact same row data again.
    # This should raise sqlite3.IntegrityError if a UNIQUE constraint was on
    # (e.g. on sca_instance.text_column, pattern1, pattern2).
    with pytest.raises(sqlite3.IntegrityError):
        cursor.execute(
            f"INSERT INTO collocate_window ({sca_instance.text_column}, pattern1, pattern2, window) VALUES (?, ?, ?, ?)",
            inserted_row_data,
        )
        conn.commit()

    # If IntegrityError is raised, test xpasses (unexpectedly passes for red phase but pytest handles xpass).
    # If IntegrityError is NOT raised (current state), code in `with` block finishes,
    # and pytest.raises itself raises an AssertionError, making the test fail.
    # This failure is expected by @pytest.mark.xfail.

    conn.close()


@pytest.mark.xfail(
    strict=True,
    reason="Red Phase: DB uniqueness for term tables not yet implemented",
)
def test_term_tables_prevent_duplicate_text_fk(tmp_path: Path):
    """
    Tests that an attempt to insert a duplicate text_fk into a term table
    would raise an IntegrityError if uniqueness constraints were active.
    This test is expected to fail (xfail) in the Red Phase.
    """
    db_path = tmp_path / "test_terms.sqlite3"
    tsv_content = "id\ttext\ntext1\tsome patterna here\ntext2\tanother patterna example\ntext3\tno target term"
    tsv_path = create_minimal_tsv(
        tmp_path, filename="terms_test.tsv", content=tsv_content
    )

    sca_instance = SCA()
    sca_instance.read_file(
        tsv_path=tsv_path, id_col="id", text_column="text", db_path=db_path
    )

    term_to_test = "patterna"
    cleaned_term = "patterna"  # cleaner("patterna") is "patterna"

    # Call tabulate_term to create the table and populate it.
    # 'text1' and 'text2' should be in the 'patterna' table.
    sca_instance.tabulate_term(cleaned_term)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Verify initial state: text1 and text2 should be in the table
    cursor.execute(
        f"SELECT text_fk FROM {cleaned_term} WHERE text_fk = ?", ("text1",)
    )
    assert (
        cursor.fetchone() is not None
    ), f"text1 not found in {cleaned_term} table after tabulation"
    cursor.execute(
        f"SELECT text_fk FROM {cleaned_term} WHERE text_fk = ?", ("text2",)
    )
    assert (
        cursor.fetchone() is not None
    ), f"text2 not found in {cleaned_term} table after tabulation"

    # Attempt to insert a duplicate text_fk ('text1') into the term table.
    # This should raise sqlite3.IntegrityError if a UNIQUE constraint was on text_fk.
    with pytest.raises(sqlite3.IntegrityError):
        cursor.execute(
            f"INSERT INTO {cleaned_term} (text_fk) VALUES (?)", ("text1",)
        )
        conn.commit()

    conn.close()
