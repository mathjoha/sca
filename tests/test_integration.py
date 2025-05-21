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


def test_counts_by_subgroups(sca_instance, tmp_path):
    # Crucial: Add collocates to this specific sca_instance for this test
    # Use a valid call to add_collocates (pairs of terms)
    sca_instance.add_collocates([("hello", "world")])

    # Now, the collocate_window table should be populated for ("hello", "world")
    # The `collocates` argument for counts_by_subgroups specifies terms and a window for querying
    output_file = tmp_path / "subgroup_counts.tsv"
    sca_instance.counts_by_subgroups([("hello", "world", 5)], output_file)

    assert (
        output_file.exists()
    ), "Output file for subgroup counts was not created."

    df_counts = pd.read_csv(output_file, sep="\t")
    assert (
        not df_counts.empty
    ), "Subgroup counts DataFrame should not be empty."

    expected_cols = sorted(
        list(sca_instance.columns) + ["total", "collocate_count"]
    )
    assert (
        sorted(list(df_counts.columns)) == expected_cols
    ), f"Output CSV missing/has extra columns. Got {sorted(list(df_counts.columns))}, expected {expected_cols}"

    # Check a specific row, e.g., for parliament 1, party A
    row_gov_party_A = df_counts[
        (df_counts["parliament"] == 1)
        & (df_counts["party"] == "A")
        & (df_counts["party_in_power"] == "Gov")
        & (df_counts["district_class"] == "Urban")
        & (df_counts["seniority"] == 1)
    ]
    assert (
        not row_gov_party_A.empty
    ), "Expected data for P1, Party A, Gov, Urban, Sen 1"
    assert row_gov_party_A.iloc[0]["total"] == 1
    assert row_gov_party_A.iloc[0]["collocate_count"] == 1

    row_opp_party_C = df_counts[
        (df_counts["parliament"] == 2)
        & (df_counts["party"] == "C")
        & (df_counts["party_in_power"] == "Opp")
        & (df_counts["district_class"] == "Rural")
        & (df_counts["seniority"] == 1)
    ]
    assert (
        not row_opp_party_C.empty
    ), "Expected data for P2, Party C, Opp, Rural, Sen 1"
    assert row_opp_party_C.iloc[0]["total"] == 1
    assert (
        row_opp_party_C.iloc[0]["collocate_count"] == 0
    )  # This speech has "world" but not "hello"

    assert (
        len(df_counts) == 5
    ), f"Expected 5 rows in output, got {len(df_counts)}"


def test_count_with_collocates(sca_instance):
    sca_instance.add_collocates([("hello", "world")])

    # Query for the collocate ("hello", "world") with a window of 5
    # Based on test data, 3 speeches contain this collocate within this window.
    # Speech 1: id=1, parliament=1, party=A, ...
    # Speech 2: id=2, parliament=1, party=B, ...
    # Speech 3: id=3, parliament=2, party=A, ...
    cursor = sca_instance.count_with_collocates([("hello", "world", 5)])
    results = cursor.fetchall()

    # Expected columns in results: parliament, party, party_in_power, district_class, seniority, count
    # These are from sca_instance.data_cols + "count(rowid)"
    # sca_instance.data_cols is "parliament, party, party_in_power, district_class, seniority"

    # Results are grouped by data_cols. Speeches 1 and 3 have party A, speech 2 has party B.
    # Speech 1: (1, 'A', 'Gov', 'Urban', 1, 1)
    # Speech 2: (1, 'B', 'Opp', 'Rural', 2, 1)
    # Speech 3: (2, 'A', 'Gov', 'Urban', 3, 1)
    assert len(results) == 3, "Expected 3 groups with the collocate"

    # Convert results to a list of dicts for easier checking
    # Description: (name, type_code, display_size, internal_size, precision, scale, null_ok)
    column_names = [desc[0] for desc in cursor.description]
    results_dicts = [dict(zip(column_names, row)) for row in results]

    # Check for speech 1 data (parliament=1, party='A')
    speech1_data = next(
        (
            r
            for r in results_dicts
            if r["parliament"] == 1 and r["party"] == "A"
        ),
        None,
    )
    assert speech1_data is not None, "Data for P1, Party A not found"
    assert speech1_data["count(rowid)"] == 1
    assert speech1_data["party_in_power"] == "Gov"
    assert speech1_data["district_class"] == "Urban"
    assert speech1_data["seniority"] == 1

    # Check for speech 2 data (parliament=1, party='B')
    speech2_data = next(
        (
            r
            for r in results_dicts
            if r["parliament"] == 1 and r["party"] == "B"
        ),
        None,
    )
    assert speech2_data is not None, "Data for P1, Party B not found"
    assert speech2_data["count(rowid)"] == 1
    assert speech2_data["party_in_power"] == "Opp"
    assert speech2_data["district_class"] == "Rural"
    assert speech2_data["seniority"] == 2

    # Check for speech 3 data (parliament=2, party='A')
    speech3_data = next(
        (
            r
            for r in results_dicts
            if r["parliament"] == 2 and r["party"] == "A"
        ),
        None,
    )
    assert speech3_data is not None, "Data for P2, Party A not found"
    assert speech3_data["count(rowid)"] == 1
    assert speech3_data["party_in_power"] == "Gov"
    assert speech3_data["district_class"] == "Urban"
    assert speech3_data["seniority"] == 3
