import sqlite3
from pathlib import Path

import pandas as pd
import pytest
import yaml

from sca import SCA, from_file


# Fixture for a basic SCA instance with some data
@pytest.fixture
def sca_initial_data(tmp_path):
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

    sca = from_file(
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


# Alias original fixture name for any potential external uses (though all tests here are refactored)
@pytest.fixture
def sca_instance(sca_initial_data):
    return sca_initial_data


@pytest.fixture
def sca_with_hello_world_collocates(sca_initial_data):
    """SCA instance after ('hello', 'world') has been added as a collocate."""
    sca = sca_initial_data
    collocate_pair = ("hello", "world")
    sca.add_collocates([collocate_pair])
    return sca


@pytest.fixture
def sca_with_test_collocate_group(sca_initial_data):
    """SCA instance after creating a collocate group named 'test_hw_group'."""
    sca = sca_initial_data
    if ("hello", "world") not in sca.collocates:
        sca.add_collocates([("hello", "world")])
    group_name = "test_hw_group"
    collocates_for_group = [("hello", "world", 5)]
    table_name_expected = "group_" + group_name
    sca.create_collocate_group(group_name, collocates_for_group)
    return sca, table_name_expected


@pytest.fixture
def sca_after_empty_pattern_collocate_error(sca_initial_data):
    """SCA instance after attempting to add a collocate that cleans to an empty string."""
    sca = sca_initial_data
    try:
        sca.add_collocates([("!@#", "world")])
    except sqlite3.OperationalError:
        pass  # Expected error
    return sca


class TestFileAndConfigLoading:
    """Tests for file reading, database seeding, and configuration loading."""

    def test_read_file_non_existent_tsv_raises_file_not_found(self, tmp_path):
        # Arrange
        db_path = tmp_path / "test_no_tsv.sqlite3"
        non_existent_tsv_path = tmp_path / "does_not_exist.tsv"
        sca = SCA()
        if db_path.exists():  # Ensure seed_db is called by read_file
            db_path.unlink()

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            sca.read_file(
                tsv_path=non_existent_tsv_path,
                id_col="id",
                text_column="text",
                db_path=db_path,
            )

    def test_read_file_missing_id_col_raises_attribute_error(self, tmp_path):
        # Arrange
        db_path = tmp_path / "test_missing_id.sqlite3"
        tsv_content = "some_other_id_col\ttext\n1\tHello world.\t1\n"
        tsv_path = tmp_path / "missing_id.tsv"
        with open(tsv_path, "w") as f:
            f.write(tsv_content)

        sca = SCA()
        if db_path.exists():  # Ensure seed_db is called
            db_path.unlink()

        expected_error_msg = r"The specified 'id_col' \('id_col_not_present'\) was not found in the columns of the input file '.*missing_id\.tsv'\. Available columns are: \['some_other_id_col', 'text'\]\. Please ensure the column name is correct and present in the file\."

        # Act & Assert
        with pytest.raises(AttributeError, match=expected_error_msg):
            sca.read_file(
                tsv_path=tsv_path,
                id_col="id_col_not_present",
                text_column="text",
                db_path=db_path,
            )

    def test_read_file_missing_text_col_raises_attribute_error(self, tmp_path):
        # Arrange
        db_path = tmp_path / "test_missing_text.sqlite3"
        tsv_content = "id\tsome_other_text_col\n1\tHello world.\n"
        tsv_path = tmp_path / "missing_text.tsv"
        with open(tsv_path, "w") as f:
            f.write(tsv_content)

        sca = SCA()
        if db_path.exists():  # Ensure seed_db is called
            db_path.unlink()

        expected_error_msg = r"The specified 'text_column' \('text_col_not_present'\) was not found in the columns of the input file '.*missing_text\.tsv'\. Available columns are: \['id', 'some_other_text_col'\]\. Please ensure the column name is correct and present in the file\."

        # Act & Assert
        with pytest.raises(AttributeError, match=expected_error_msg):
            sca.read_file(
                tsv_path=tsv_path,
                id_col="id",
                text_column="text_col_not_present",
                db_path=db_path,
            )

    def test_seed_db_with_empty_tsv_file_raises_empty_data_error(
        self, tmp_path
    ):
        # Arrange
        db_path = tmp_path / "empty_db.sqlite3"
        empty_tsv_path = tmp_path / "empty.tsv"
        with open(empty_tsv_path, "w") as f:
            pass  # Create an empty file

        sca = SCA()
        if db_path.exists():  # Ensure seed_db is called
            db_path.unlink()

        # Act & Assert
        # pd.read_csv raises EmptyDataError for a completely empty file.
        with pytest.raises(pd.errors.EmptyDataError):
            sca.read_file(
                tsv_path=empty_tsv_path,
                id_col="id",
                text_column="text",
                db_path=db_path,
            )

    def test_seed_db_with_tsv_headers_only_raises_db_error(self, tmp_path):
        # Arrange
        db_path = tmp_path / "headers_only_db.sqlite3"
        headers_only_tsv_path = tmp_path / "headers_only.tsv"
        with open(headers_only_tsv_path, "w") as f:
            f.write("id\ttext\tmeta1\n")  # Headers but no data lines

        sca = SCA()
        if db_path.exists():  # Ensure seed_db is called
            db_path.unlink()

        # Act & Assert
        # sqlite_utils doesn't create 'raw' table if insert_all gets empty data.
        # Subsequent create_index call fails.
        # Updated: Now expecting ValueError due to changes in seed_db for empty files.
        with pytest.raises(
            ValueError,
            match=rf"The input file '{headers_only_tsv_path}' is empty and does not contain any data\. Please provide a file with content\.",
        ):
            sca.read_file(
                tsv_path=headers_only_tsv_path,
                id_col="id",
                text_column="text",
                db_path=db_path,
            )

    def test_load_non_existent_yml_raises_file_not_found(self, tmp_path):
        # Arrange
        sca = SCA()
        non_existent_yml_path = tmp_path / "does_not_exist.yml"

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            sca.load(non_existent_yml_path)

    def test_load_yml_missing_key_raises_key_error(self, tmp_path):
        # Arrange
        yml_content = (
            "db_path: test.sqlite3\n"
            "collocates: []\n"
            # id_col is missing
            "text_column: text\n"
            "columns: [col1, col2]\n"
            "language: english\n"
            "custom_stopwords: [custom1, custom2]\n"
        )
        yml_path = tmp_path / "missing_key.yml"
        with open(yml_path, "w") as f:
            f.write(yml_content)

        sca = SCA()

        # Act & Assert
        with pytest.raises(KeyError, match="'id_col'"):
            sca.load(yml_path)

    def test_load_malformed_yml_raises_yaml_error(self, tmp_path):
        # Arrange
        malformed_yml_path = tmp_path / "malformed.yml"
        # Invalid YAML due to unindented colon
        with open(malformed_yml_path, "w") as f:
            f.write("db_path: test.sqlite3\n: unindented_colon")

        sca = SCA()

        # Act & Assert
        with pytest.raises(yaml.YAMLError):
            sca.load(malformed_yml_path)


class TestSCAOperations:
    """Tests for SCA analytical methods using a pre-populated instance."""

    def test_add_collocates_db_row_count_correct(
        self, sca_with_hello_world_collocates
    ):
        # Arrange: Done by sca_with_hello_world_collocates fixture
        sca = sca_with_hello_world_collocates

        # Act: Query the database
        conn = sqlite3.connect(sca.db_path)
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT {sca.id_col} FROM collocate_window WHERE window IS NOT NULL"
        )
        rows = cursor.fetchall()
        conn.close()

        # Assert
        assert (
            len(rows) == 3
        ), "Expected 3 total collocations in DB for ('hello', 'world')"

    def test_add_collocates_db_has_speech1_collocation(
        self, sca_with_hello_world_collocates
    ):
        # Arrange: Done by sca_with_hello_world_collocates fixture
        sca = sca_with_hello_world_collocates

        # Act: Query the database for the specific collocation
        conn = sqlite3.connect(sca.db_path)
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT COUNT(*) FROM collocate_window "
            f"WHERE {sca.id_col} = '1' AND pattern1 = 'hello' AND pattern2 = 'world' AND window IS NOT NULL"
        )
        count = cursor.fetchone()[0]
        conn.close()

        # Assert
        assert (
            count == 1
        ), "Expected specific collocation for speech_id '1' ('hello', 'world') in DB"

    def test_add_collocates_updates_internal_terms_with_hello(
        self, sca_with_hello_world_collocates
    ):
        # Arrange: Done by sca_with_hello_world_collocates fixture
        sca = sca_with_hello_world_collocates
        # Act: (Implicitly done by fixture)
        # Assert
        assert "hello" in sca.terms, "'hello' should be in internal terms list"

    def test_add_collocates_updates_internal_terms_with_world(
        self, sca_with_hello_world_collocates
    ):
        # Arrange: Done by sca_with_hello_world_collocates fixture
        sca = sca_with_hello_world_collocates
        # Act: (Implicitly done by fixture)
        # Assert
        assert "world" in sca.terms, "'world' should be in internal terms list"

    def test_add_collocates_updates_internal_collocates_set(
        self, sca_with_hello_world_collocates
    ):
        # Arrange: Done by sca_with_hello_world_collocates fixture
        sca = sca_with_hello_world_collocates
        collocate_pair = ("hello", "world")
        # Act: (Implicitly done by fixture)
        # Assert
        assert (
            collocate_pair in sca.collocates
        ), "('hello', 'world') should be in internal collocates set"

    def test_counts_by_subgroups_generates_correct_output_file(
        self, sca_initial_data, tmp_path
    ):
        # Arrange
        sca = sca_initial_data
        sca.add_collocates([("hello", "world")])  # Prerequisite for counts

        output_file = tmp_path / "subgroup_counts.tsv"
        collocates_to_query = [("hello", "world", 5)]

        # Act
        sca.counts_by_subgroups(collocates_to_query, output_file)

        # Assert: File properties
        assert (
            output_file.exists()
        ), "Output file for subgroup counts was not created."

        df_counts = pd.read_csv(output_file, sep="\t")
        assert (
            not df_counts.empty
        ), "Subgroup counts DataFrame should not be empty."

        expected_cols = sorted(
            list(sca.columns) + ["total", "collocate_count"]
        )
        assert (
            sorted(list(df_counts.columns)) == expected_cols
        ), f"Output CSV columns mismatch. Got {sorted(list(df_counts.columns))}, expected {expected_cols}"
        assert (
            len(df_counts) == 5
        ), f"Expected 5 rows in output, got {len(df_counts)}"

    def test_counts_by_subgroups_correct_counts_for_group1(
        self, sca_initial_data, tmp_path
    ):
        # Arrange
        sca = sca_initial_data
        sca.add_collocates([("hello", "world")])
        output_file = tmp_path / "subgroup_counts_g1.tsv"
        collocates_to_query = [("hello", "world", 5)]

        # Act
        sca.counts_by_subgroups(collocates_to_query, output_file)
        df_counts = pd.read_csv(output_file, sep="\t")

        # Assert: Specific group data (Gov, Party A)
        row_gov_party_a = df_counts[
            (df_counts["parliament"] == 1)
            & (df_counts["party"] == "A")
            & (df_counts["party_in_power"] == "Gov")
            & (df_counts["district_class"] == "Urban")
            & (df_counts["seniority"] == 1)
        ]
        assert (
            not row_gov_party_a.empty
        ), "Expected data for P1, Party A, Gov, Urban, Sen 1"
        assert row_gov_party_a.iloc[0]["total"] == 1
        assert row_gov_party_a.iloc[0]["collocate_count"] == 1

    def test_counts_by_subgroups_correct_counts_for_group2(
        self, sca_initial_data, tmp_path
    ):
        # Arrange
        sca = sca_initial_data
        sca.add_collocates([("hello", "world")])
        output_file = tmp_path / "subgroup_counts_g2.tsv"
        collocates_to_query = [("hello", "world", 5)]

        # Act
        sca.counts_by_subgroups(collocates_to_query, output_file)
        df_counts = pd.read_csv(output_file, sep="\t")

        # Assert: Specific group data (Opp, Party C) - no "hello"
        row_opp_party_c = df_counts[
            (df_counts["parliament"] == 2)
            & (df_counts["party"] == "C")
            & (df_counts["party_in_power"] == "Opp")
            & (df_counts["district_class"] == "Rural")
            & (df_counts["seniority"] == 1)
        ]
        assert (
            not row_opp_party_c.empty
        ), "Expected data for P2, Party C, Opp, Rural, Sen 1"
        assert row_opp_party_c.iloc[0]["total"] == 1
        assert row_opp_party_c.iloc[0]["collocate_count"] == 0

    def test_count_with_collocates_returns_correct_groups(
        self, sca_initial_data
    ):
        # Arrange
        sca = sca_initial_data
        sca.add_collocates([("hello", "world")])
        collocates_to_query = [("hello", "world", 5)]

        # Act
        cursor = sca.count_with_collocates(collocates_to_query)
        results = cursor.fetchall()

        # Assert
        assert (
            len(results) == 3
        ), "Expected 3 groups with the collocate ('hello', 'world')"

    def test_count_with_collocates_data_for_speech1_group(
        self, sca_initial_data
    ):
        # Arrange
        sca = sca_initial_data
        sca.add_collocates([("hello", "world")])
        collocates_to_query = [("hello", "world", 5)]

        # Act
        cursor = sca.count_with_collocates(collocates_to_query)
        column_names = [desc[0] for desc in cursor.description]
        results_dicts = [
            dict(zip(column_names, row)) for row in cursor.fetchall()
        ]

        # Assert
        speech1_data = next(
            (
                r
                for r in results_dicts
                if r["parliament"] == 1 and r["party"] == "A"
            ),
            None,
        )
        assert (
            speech1_data is not None
        ), "Data for P1, Party A (speech 1) not found"
        assert speech1_data["count(rowid)"] == 1
        assert speech1_data["party_in_power"] == "Gov"
        assert speech1_data["district_class"] == "Urban"
        assert speech1_data["seniority"] == 1

    def test_count_with_collocates_data_for_speech2_group(
        self, sca_initial_data
    ):
        # Arrange
        sca = sca_initial_data
        sca.add_collocates([("hello", "world")])
        collocates_to_query = [("hello", "world", 5)]

        # Act
        cursor = sca.count_with_collocates(collocates_to_query)
        column_names = [desc[0] for desc in cursor.description]
        results_dicts = [
            dict(zip(column_names, row)) for row in cursor.fetchall()
        ]

        # Assert
        speech2_data = next(
            (
                r
                for r in results_dicts
                if r["parliament"] == 1 and r["party"] == "B"
            ),
            None,
        )
        assert (
            speech2_data is not None
        ), "Data for P1, Party B (speech 2) not found"
        assert speech2_data["count(rowid)"] == 1
        assert speech2_data["party_in_power"] == "Opp"
        assert speech2_data["district_class"] == "Rural"
        assert speech2_data["seniority"] == 2

    def test_count_with_collocates_data_for_speech3_group(
        self, sca_initial_data
    ):
        # Arrange
        sca = sca_initial_data
        sca.add_collocates([("hello", "world")])
        collocates_to_query = [("hello", "world", 5)]

        # Act
        cursor = sca.count_with_collocates(collocates_to_query)
        column_names = [desc[0] for desc in cursor.description]
        results_dicts = [
            dict(zip(column_names, row)) for row in cursor.fetchall()
        ]

        # Assert
        speech3_data = next(
            (
                r
                for r in results_dicts
                if r["parliament"] == 2 and r["party"] == "A"
            ),
            None,
        )
        assert (
            speech3_data is not None
        ), "Data for P2, Party A (speech 3) not found"
        assert speech3_data["count(rowid)"] == 1
        assert speech3_data["party_in_power"] == "Gov"
        assert speech3_data["district_class"] == "Urban"
        assert speech3_data["seniority"] == 3

    # --- Start of refactored edge case tests ---
    def test_add_collocates_skips_pair_with_numeric_term(
        self, sca_initial_data
    ):
        # Arrange
        sca = sca_initial_data
        initial_collocates_count = len(sca.collocates)
        # This pair should be skipped because "123" is a digit-only string
        # and cleaner("123") results in "", leading to len(clean_pair) != 2
        # if not str(pattern).isdigit() is the primary filter here.

        # Act
        with pytest.raises(ValueError, match="No clean collocates to add."):
            sca.add_collocates([("numericterm", "123")])

        # Assert
        assert (
            len(sca.collocates) == initial_collocates_count
        ), "Collocate pair with numeric string should be skipped"
        assert ("numericterm", "123") not in sca.collocates

    def test_add_collocates_handles_duplicate_collocate_gracefully(
        self, sca_initial_data
    ):
        # Arrange
        sca = sca_initial_data
        collocate_pair = ("firstcall", "term")
        sca.add_collocates([collocate_pair])  # Add it once
        assert collocate_pair in sca.collocates
        count_after_first_add = len(sca.collocates)

        # Act: Add the same collocate again
        with pytest.raises(ValueError, match="No clean collocates to add."):
            sca.add_collocates([collocate_pair])

        # Assert
        assert (
            len(sca.collocates) == count_after_first_add
        ), "Adding a duplicate collocate should not change the count"

    def test_add_collocates_duplicate_collocate(self, sca_initial_data: SCA):
        # Arrange
        sca = sca_initial_data
        collocate_pair = ("firstcall", "term")
        with pytest.raises(
            ValueError, match="Aborting: Could not add ALL collocates."
        ):
            sca.add_collocates([collocate_pair] * 2)
        assert collocate_pair not in sca.collocates
        sca.add_collocates([collocate_pair] * 2, allow_duplicates=True)
        assert collocate_pair in sca.collocates

        with pytest.raises(ValueError, match="No clean collocates to add."):
            sca.add_collocates([collocate_pair] * 2)

    def test_mark_windows_handles_fnmatch_mismatch(self, tmp_path):
        # Arrange: text has "alpha" and "betaX", but we search for "beta"
        # This means get_positions for "beta" will be empty due to fnmatch.
        db_path = tmp_path / "edge_fnmatch.sqlite3"
        tsv_content = (
            "id\ttext\n"
            "40\talpha also has beta text\n"  # Exact match for ("alpha", "beta")
            "41\talpha only has betaX variant\n"  # "betaX" won't fnmatch "beta"
        )
        tsv_path = tmp_path / "edge_fnmatch.tsv"
        with open(tsv_path, "w") as f:
            f.write(tsv_content)

        sca = SCA()
        sca.read_file(
            tsv_path=tsv_path, id_col="id", text_column="text", db_path=db_path
        )

        # Act
        sca.add_collocates([("alpha", "beta")])  # This triggers mark_windows

        # Assert: Only the text with exact "beta" should have a collocate window
        conn = sqlite3.connect(sca.db_path)
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT {sca.id_col} FROM collocate_window WHERE pattern1='alpha' AND pattern2='beta' AND window IS NOT NULL"
        )
        rows_with_window = cursor.fetchall()
        conn.close()

        assert (
            len(rows_with_window) == 1
        ), "Expected one actual collocation for alpha-beta from text 40"
        assert (
            rows_with_window[0][0] == "40"
        ), "The actual collocation should be from speech 40 (exact match)"

    def test_mark_windows_handles_no_cooccurrence_of_terms(self, tmp_path):
        # Arrange: Terms "gamma" and "delta" appear in different texts.
        # The SQL join in mark_windows for the tqdm loop should be empty.
        db_path = tmp_path / "edge_no_cooccurrence.sqlite3"
        tsv_content = (
            "id\ttext\n50\tGamma word only here\n51\tDelta word only here\n"
        )
        tsv_path = tmp_path / "edge_no_cooccurrence.tsv"
        with open(tsv_path, "w") as f:
            f.write(tsv_content)

        sca = SCA()
        sca.read_file(
            tsv_path=tsv_path, id_col="id", text_column="text", db_path=db_path
        )

        # Act
        sca.add_collocates([("gamma", "delta")])

        # Assert: A placeholder should be inserted into collocate_window
        conn = sqlite3.connect(sca.db_path)
        cursor = conn.cursor()
        # The placeholder has pattern1='delta', pattern2='gamma' because they are sorted.
        cursor.execute(
            f"SELECT {sca.id_col}, window FROM collocate_window WHERE pattern1='delta' AND pattern2='gamma'"
        )
        rows = cursor.fetchall()
        conn.close()

        assert len(rows) == 1, "Expected one row for gamma-delta"
        assert rows[0][0] is None, "speech_id should be None for placeholder"
        assert rows[0][1] is None, "window should be None for placeholder"

    # --- End of refactored edge case tests ---

    def test_create_collocate_group_table_exists(
        self, sca_with_test_collocate_group
    ):
        # Arrange: Done by fixture
        sca, table_name_expected = sca_with_test_collocate_group

        # Act: Connect and check for table
        cursor = sca.conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name_expected,),
        )
        table_exists = cursor.fetchone()

        # Assert
        assert (
            table_exists is not None
        ), f"Table {table_name_expected} was not created."

    def test_counts_by_subgroups_with_empty_collocates_list_raises_db_error(
        self, sca_initial_data, tmp_path
    ):
        # Arrange
        sca = sca_initial_data
        output_file = tmp_path / "subgroup_counts_empty.tsv"

        # Act & Assert
        # pd.read_sql_query raises DatabaseError for malformed SQL from empty collocates.
        with pytest.raises(
            pd.errors.DatabaseError, match=r"Execution failed on sql"
        ):
            sca.counts_by_subgroups([], output_file)

    def test_create_collocate_group_with_empty_collocates_list_raises_db_error(
        self, sca_initial_data
    ):
        # Arrange
        sca = sca_initial_data
        group_name = "test_empty_group"

        # Act & Assert
        # Malformed SQL from empty collocates list.
        with pytest.raises(
            sqlite3.OperationalError, match=r"incomplete input"
        ):
            sca.create_collocate_group(group_name, [])

    def test_add_collocates_with_pattern_cleaning_to_empty_raises_db_error(
        self, sca_initial_data
    ):
        # Arrange
        sca = sca_initial_data
        initial_collocates_count = len(sca.collocates)
        initial_terms_count = len(sca.terms)  # Capture initial terms count

        # Act & Assert for the exception
        with pytest.raises(
            sqlite3.OperationalError, match=r'near "\(": syntax error'
        ):
            sca.add_collocates([("!@#", "world")])

        # Assert state immediately after expected error
        assert (
            len(sca.collocates) == initial_collocates_count
        ), "Collocates set should not change after the error."
        assert (
            "" not in sca.terms
        ), "Empty string should not be added as a term."
        # Check if 'world' was added before the error, if it wasn't there already
        # This depends on the internal logic of add_collocates if terms are added before DB op.
        # For a stricter check on what happened to terms:
        if (
            "world" not in sca_initial_data.terms
        ):  # If world wasn't an initial term
            assert (
                "world" not in sca.terms
            ), "'world' should not be added if operation failed mid-way for empty pattern."
        else:
            assert (
                len(sca.terms) == initial_terms_count
            ), "Terms set should not change if 'world' was already a term."

    def test_sca_usable_after_empty_pattern_collocate_error(
        self, sca_after_empty_pattern_collocate_error
    ):
        # Arrange: Done by fixture
        sca = sca_after_empty_pattern_collocate_error

        # Act: Try a valid operation
        sca.add_collocates([("newvalid", "pairvalid")])

        # Assert: Valid operation succeeds
        assert ("newvalid", "pairvalid") in sca.collocates
        assert "newvalid" in sca.terms
        assert "pairvalid" in sca.terms
