import sqlite3
from pathlib import Path

import pytest

from sca import SCA


def create_minimal_tsv(
    tmp_path: Path,
    filename: str = "test.tsv",
    content: str = "id\ttext\n1\tpatterna then patternb\n",
) -> Path:
    tsv_path = tmp_path / filename
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write(content)
    return tsv_path


def test_collocate_window_prevents_duplicates_at_db_level(tmp_path: Path):
    db_path = tmp_path / "test.sqlite3"
    tsv_path = create_minimal_tsv(
        tmp_path, content="id\ttext\ntext1\tpatterna then patternb\n"
    )

    sca_instance = SCA()
    sca_instance.read_file(
        tsv_path=tsv_path, id_col="id", text_column="text", db_path=db_path
    )

    pattern1 = "patterna"
    pattern2 = "patternb"

    sca_instance.mark_windows(pattern1, pattern2)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        f"SELECT COUNT(*) FROM collocate_window WHERE {sca_instance.id_col}=? AND pattern1=? AND pattern2=?",
        ("text1", "patterna", "patternb"),
    )
    count = cursor.fetchone()[0]
    assert count == 1, "Initial data not inserted by mark_windows as expected"

    cursor.execute(
        f"SELECT {sca_instance.id_col}, pattern1, pattern2, window FROM collocate_window WHERE {sca_instance.id_col}=? AND pattern1=? AND pattern2=?",
        ("text1", "patterna", "patternb"),
    )
    inserted_row_data = cursor.fetchone()
    assert (
        inserted_row_data is not None
    ), "Failed to retrieve the initially inserted row"

    with pytest.raises(sqlite3.IntegrityError):
        cursor.execute(
            f"INSERT INTO collocate_window ({sca_instance.id_col}, pattern1, pattern2, window) VALUES (?, ?, ?, ?)",
            inserted_row_data,
        )
        conn.commit()

    conn.close()


def test_term_tables_prevent_duplicate_text_fk(tmp_path: Path):
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
    cleaned_term = "patterna"

    sca_instance.tabulate_term(cleaned_term)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

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

    with pytest.raises(sqlite3.IntegrityError):
        cursor.execute(
            f"INSERT INTO {cleaned_term} (text_fk) VALUES (?)", ("text1",)
        )
        conn.commit()

    conn.close()


def test_named_collocate_prevents_duplicates(tmp_path: Path):
    db_path = tmp_path / "test_named_collocate.sqlite3"
    tsv_content = "id\ttext\ntext1\tAlice met Bob.\ntext2\tCharlie met David."
    tsv_path = create_minimal_tsv(
        tmp_path, filename="named_collocate_test.tsv", content=tsv_content
    )

    sca_instance = SCA()
    sca_instance.read_file(
        tsv_path=tsv_path, id_col="id", text_column="text", db_path=db_path
    )

    group_name = "test_group"
    collocates = [("alice*", "bob*", 5), ("charlie*", "david*", 5)]

    sca_instance.add_collocates([c[:2] for c in collocates])

    sca_instance.create_collocate_group(group_name, collocates)

    cursor = sca_instance.conn.cursor()

    cursor.execute("SELECT * FROM named_collocate")
    rows = cursor.fetchall()
    assert len(rows) == 2, "Expected two entries in named_collocate table"

    # Attempt to create the same collocate group again, which should populate named_collocate
    # and trigger an IntegrityError if uniqueness is enforced.
    with pytest.raises(sqlite3.IntegrityError):
        cursor.execute(
            "INSERT INTO named_collocate (name, table_name, term1, term2, window) VALUES (?, ?, ?, ?, ?)",
            (group_name, f"group_{group_name}", "alice*", "bob*", 5),
        )

    with pytest.raises(sqlite3.IntegrityError):
        sca_instance.create_collocate_group(group_name, collocates)

    cursor.execute("SELECT * FROM named_collocate")
    rows = cursor.fetchall()
    assert len(rows) == 2, "Expected two entries in named_collocate table"
