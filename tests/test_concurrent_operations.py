import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from sca import SCA


def create_test_data(
    tmp_path: Path, prefix: str, num_rows: int = 5
) -> tuple[Path, Path]:
    """Creates test data files for concurrent operation testing."""
    csv_path = tmp_path / f"{prefix}_data.csv"
    db_path = tmp_path / f"{prefix}_data.sqlite3"

    content = "id,text\n"
    content += "\n".join(
        f"{prefix}_{i},This is text {i} with hello and world"
        for i in range(num_rows)
    )

    csv_path.write_text(content)
    return csv_path, db_path


def test_concurrent_database_access(tmp_path: Path):
    """Tests concurrent access to different SCA instances with different databases."""
    # Create two separate test datasets
    csv_path1, db_path1 = create_test_data(tmp_path, "test1")
    csv_path2, db_path2 = create_test_data(tmp_path, "test2")

    def process_dataset(csv_path: Path, db_path: Path):
        sca = SCA()
        sca.read_file(
            tsv_path=csv_path, id_col="id", text_column="text", db_path=db_path
        )
        sca.add_collocates([("hello", "world")])
        return len(sca.collocates)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(process_dataset, csv_path1, db_path1)
        future2 = executor.submit(process_dataset, csv_path2, db_path2)

        result1 = future1.result()
        result2 = future2.result()

    assert result1 == 1, "First SCA instance should have one collocate pair"
    assert result2 == 1, "Second SCA instance should have one collocate pair"
