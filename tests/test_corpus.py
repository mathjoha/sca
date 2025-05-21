import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from sca.corpus import SCA  # Assuming SCA is in sca.corpus


def create_dummy_csv(file_path: Path, num_headers: int, num_rows: int):
    headers = [f"header{i}" for i in range(num_headers)]
    data = [
        [f"data_{r}_{h}" for h in range(num_headers)] for r in range(num_rows)
    ]
    df = pd.DataFrame(data, columns=headers)
    df["id"] = [f"id_{i}" for i in range(num_rows)]
    df["text"] = [f"text_{i}" for i in range(num_rows)]
    df.to_csv(file_path, index=False)


def create_dummy_tsv(file_path: Path, num_headers: int, num_rows: int):
    headers = [f"header_tsv_{i}" for i in range(num_headers)]
    data = [
        [f"data_tsv_{r}_{h}" for h in range(num_headers)]
        for r in range(num_rows)
    ]
    df = pd.DataFrame(data, columns=headers)
    df["id_tsv"] = [f"id_tsv_{i}" for i in range(num_rows)]
    df["text_tsv"] = [f"text_tsv_{i}" for i in range(num_rows)]
    df.to_csv(file_path, index=False, sep="\t")


@pytest.mark.xfail(
    strict=True,
    reason="Red Phase: Test for headers with spaces and special characters",
)
def test_header_cleaning(tmp_path):
    csv_path = tmp_path / "special_headers.csv"
    db_path = tmp_path / "test_special.sqlite3"
    yml_path = tmp_path / "test_special.yml"

    headers_original = [
        "First Header",
        "Second.Header",
        "Header With-Hyphen",
        "header_ok",
    ]
    headers_cleaned = [
        "first_header",
        "secondheader",
        "header_withhyphen",
        "header_ok",
    ]

    data = [
        [f"data_{r}_{h}" for h in range(len(headers_original))]
        for r in range(2)
    ]
    df = pd.DataFrame(data, columns=headers_original)
    df["id_col"] = ["id_1", "id_2"]
    df["text_c"] = ["text_1", "text_2"]
    df.to_csv(csv_path, index=False)

    corpus_write = SCA()
    corpus_write.read_file(
        tsv_path=csv_path,
        id_col="id_col",
        text_column="text_c",
        db_path=db_path,
    )
    corpus_write.save()

    corpus_load = SCA()
    corpus_load.load(yml_path)

    assert sorted(list(corpus_load.columns)) == sorted(headers_cleaned)
    for header in corpus_load.columns:
        assert (
            header.isidentifier()
        ), f"Header '{header}' is not a valid SQLite table name."


@pytest.mark.xfail(
    strict=True, reason="Red Phase: Test for duplicate headers after cleaning"
)
def test_duplicate_headers_after_cleaning(tmp_path):
    csv_path = tmp_path / "duplicate_headers.csv"
    db_path = tmp_path / "test_duplicate.sqlite3"

    headers_original = ["Header One", "Header.One", "UniqueHeader"]
    data = [
        [f"data_{r}_{h}" for h in range(len(headers_original))]
        for r in range(2)
    ]
    df = pd.DataFrame(data, columns=headers_original)
    df["id_col"] = ["id_1", "id_2"]
    df["text_c"] = ["text_1", "text_2"]
    df.to_csv(csv_path, index=False)

    corpus = SCA()
    with pytest.raises(
        ValueError, match="Duplicate column names found after cleaning:"
    ):
        corpus.read_file(
            tsv_path=csv_path,
            id_col="id_col",
            text_column="text_c",
            db_path=db_path,
        )


@pytest.mark.xfail(strict=True, reason="Red Phase: Test empty file handling")
def test_empty_file(tmp_path):
    csv_path = tmp_path / "empty.csv"
    db_path = tmp_path / "test_empty.sqlite3"
    with open(csv_path, "w") as f:
        f.write("id,text,header1\n")

    corpus = SCA()
    # Expecting an sqlite3.OperationalErro  r if table 'raw' isn't created due to empty insert
    with pytest.raises(sqlite3.OperationalError, match="no such table: raw"):
        corpus.read_file(
            tsv_path=csv_path,
            id_col="id",
            text_column="text",
            db_path=db_path,
        )


def test_dynamic_csv_headers(tmp_path):
    csv_path = tmp_path / "dynamic_headers.csv"
    db_path = tmp_path / "test_dynamic.sqlite3"
    yml_path = tmp_path / "test_dynamic.yml"

    create_dummy_csv(csv_path, 5, 10)

    corpus_write = SCA()
    corpus_write.read_file(
        tsv_path=csv_path,
        id_col="id",
        text_column="text",
        db_path=db_path,
    )
    corpus_write.save()

    assert yml_path.exists()

    corpus_load = SCA()
    corpus_load.load(yml_path)

    expected_headers = sorted([f"header{i}" for i in range(5)])
    assert (
        hasattr(corpus_load, "columns")
        and sorted(list(corpus_load.columns)) == expected_headers
    ), f"Expected {expected_headers}, got {getattr(corpus_load, 'columns', 'attribute missing')}"

    for header in corpus_load.columns:
        assert (
            header.isidentifier()
        ), f"Header '{header}' is not a valid SQLite table name."


def test_dynamic_tsv_headers(tmp_path):
    tsv_path = tmp_path / "dynamic_headers.tsv"
    db_path = tmp_path / "test_dynamic_tsv.sqlite3"
    yml_path = tmp_path / "test_dynamic_tsv.yml"

    create_dummy_tsv(tsv_path, 3, 5)

    corpus_write = SCA()
    corpus_write.read_file(
        tsv_path=tsv_path,
        id_col="id_tsv",
        text_column="text_tsv",
        db_path=db_path,
    )
    corpus_write.save()

    assert yml_path.exists()

    corpus_load = SCA()
    corpus_load.load(yml_path)

    expected_headers = sorted([f"header_tsv_{i}" for i in range(3)])
    assert (
        hasattr(corpus_load, "columns")
        and sorted(list(corpus_load.columns)) == expected_headers
    ), f"Expected {expected_headers}, got {getattr(corpus_load, 'columns', 'attribute missing')}"
    for header in corpus_load.columns:
        assert (
            header.isidentifier()
        ), f"Header '{header}' is not a valid SQLite table name."


def test_file_only_id_text(tmp_path):
    csv_path = tmp_path / "only_id_text.csv"
    db_path = tmp_path / "test_only_id_text.sqlite3"
    yml_path = tmp_path / "test_only_id_text.yml"

    df = pd.DataFrame({"id": ["id1"], "text": ["text1"]})
    df.to_csv(csv_path, index=False)

    corpus_write = SCA()
    corpus_write.read_file(
        tsv_path=csv_path,
        id_col="id",
        text_column="text",
        db_path=db_path,
    )
    corpus_write.save()

    assert yml_path.exists()
    corpus_load = SCA()
    corpus_load.load(yml_path)

    assert (
        hasattr(corpus_load, "columns") and list(corpus_load.columns) == []
    ), f"Expected empty list for columns, got {getattr(corpus_load, 'columns', 'attribute missing')}"
