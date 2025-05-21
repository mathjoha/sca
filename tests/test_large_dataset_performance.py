import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from sca import SCA


@pytest.mark.slow
def test_large_dataset_performance(tmp_path: Path):
    csv_path = tmp_path / "large_dataset.csv"
    db_path = tmp_path / "large_dataset.sqlite3"

    # Create test data with realistic looking content
    num_rows = 100_000
    data = {
        "id": [f"speech_{i}" for i in range(num_rows)],
        "text": [
            f"This is speech {i} with some meaningful content about policy and debate number {i % 100}"
            for i in range(num_rows)
        ],
        "year": [2020 + (i % 5) for i in range(num_rows)],
        "parliament": [50 + (i % 3) for i in range(num_rows)],
        "party": ["Party" + str(i % 5) for i in range(num_rows)],
        "district": ["District" + str(i % 20) for i in range(num_rows)],
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    # Test reading and processing the large dataset
    corpus = SCA()
    corpus.read_file(
        tsv_path=csv_path, id_col="id", text_column="text", db_path=db_path
    )

    # Set and verify columns
    expected_columns = {"year", "parliament", "party", "district"}
    corpus.columns = expected_columns
    corpus.set_data_cols()

    # Verify data integrity
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check row count
    cursor.execute("SELECT COUNT(*) FROM raw")
    assert cursor.fetchone()[0] == num_rows, "Not all rows were imported"

    # Check column presence
    cursor.execute("PRAGMA table_info(raw)")
    columns = {row[1] for row in cursor.fetchall()}
    for col in expected_columns:
        assert col in columns, f"Column {col} missing from database"

    # Test collocate functionality with large dataset
    corpus.add_collocates([("policy", "debate")])
    cursor.execute("SELECT COUNT(*) FROM collocate_window")
    collocate_count = cursor.fetchone()[0]
    assert collocate_count > 0, "No collocates found in large dataset"

    conn.close()
