import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from sca import SCA


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


def test_header_sanitation_check(tmp_path: Path):
    csv_path = tmp_path / "special_headers.csv"
    db_path = tmp_path / "test_special.sqlite3"
    yml_path = tmp_path / "test_special.yml"

    headers_original = [
        "First Header",
        "Second.Header",
        "Header-With-Hyphen",
        "Semi;Colon",
        "Full:Colon",
        "co,ma",
        "id_1",
        "text_1",
    ]

    data = [
        [f"data_{r}_{h}" for h in range(len(headers_original))]
        for r in range(2)
    ]
    df = pd.DataFrame(data, columns=headers_original)
    df.to_csv(csv_path, index=False)

    corpus_write = SCA()
    with pytest.raises(
        ValueError, match="Column name .+ is not SQLite-friendly."
    ):
        corpus_write.read_file(
            tsv_path=csv_path,
            id_col="id_1",
            text_column="text_1",
            db_path=db_path,
        )


def test_duplicate_headers_detection(tmp_path: Path):
    csv_path = tmp_path / "duplicate_headers.csv"
    db_path = tmp_path / "test_duplicate.sqlite3"

    headers_original = [
        "headerone",
        "headeronE",
        "uniqueheader",
        "id_col",
        "text_c",
    ]
    data = [
        [f"data_{r}_{h}" for h in range(len(headers_original))]
        for r in range(2)
    ]
    df = pd.DataFrame(data)
    df.columns = headers_original
    df.to_csv(csv_path, index=False)

    corpus = SCA()
    with pytest.raises(ValueError, match="Duplicate column names found."):
        corpus.read_file(
            tsv_path=csv_path,
            id_col="id_col",
            text_column="text_c",
            db_path=db_path,
        )


def test_duplicate_keys(tmp_path: Path):
    csv_path = tmp_path / "duplicate_headers.csv"
    db_path = tmp_path / "test_duplicate.sqlite3"

    headers_original = [
        "id_col",
        "id_col",
    ]
    data = [
        [f"data_{r}_{h}" for h in range(len(headers_original))]
        for r in range(2)
    ]
    df = pd.DataFrame(data)
    df.columns = headers_original
    df.to_csv(csv_path, index=False)

    corpus = SCA()
    with pytest.raises(
        ValueError, match="text_column and id_col cannot be the same"
    ):
        corpus.read_file(
            tsv_path=csv_path,
            id_col="id_col",
            text_column="id_col",
            db_path=db_path,
        )


def test_dynamic_csv_headers(tmp_path: Path):
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


def test_dynamic_tsv_headers(tmp_path: Path):
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


def test_file_only_id_text(tmp_path: Path):
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


def test_loading(tmp_path: Path):
    csv_path = tmp_path / "small_csv.csv"
    db_path = tmp_path / "small_csv.sqlite3"
    yml_path = tmp_path / "small_csv.yml"

    create_dummy_csv(csv_path, 5, 5)

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

    assert corpus_write != None

    assert corpus_write == corpus_load
