import sqlite3
from pathlib import Path

import pandas as pd
import pytest
import yaml

from sca import SCA


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


@pytest.mark.xfail(strict=True, reason="Red Phase [Expand as necessary]")
class TestStopwordsIntegration:
    """Tests for stopwords functionality integration with other SCA methods."""

    @pytest.fixture
    def sca_with_custom_stopwords(self, tmp_path):
        """SCA instance with initial data and custom stopwords."""
        db_path = tmp_path / "test_stopwords.sqlite3"
        tsv_content = (
            "id\ttext\n"
            "1\tThe quick brown fox jumps over the lazy dog.\n"  # NLTK: the, over, custom: quick, lazy
            "2\tA fast hare and a sleepy tortoise.\n"  # NLTK: a, and, a
            "3\tThe quick and lazy fox is quick.\n"  # NLTK: the, and, is, custom: quick, lazy
            "4\tNo stopwords here just words.\n"
            "5\tquick lazy quick lazy the the a a.\n"  # all stopwords
        )
        tsv_path = tmp_path / "test_stopwords.tsv"
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write(tsv_content)

        sca = SCA(language="english", db_path=db_path)
        sca.read_file(tsv_path=tsv_path, id_col="id", text_column="text")
        sca.add_stopwords(["quick", "lazy"])  # Custom stopwords
        return sca

    def test_get_positions_with_combined_stopwords(
        self, sca_with_custom_stopwords
    ):
        sca = sca_with_custom_stopwords

        # Test with count_stopwords=False
        positions_no_stopwords = sca.get_positions(
            ["fox", "dog", "hare", "tortoise", "words"],
            count_stopwords=False,
            search_term="fox",
        )
        expected_positions_no_stopwords = {
            "fox": [
                2,
                7,
            ],  # "brown" (1), "jumps" (3), "dog" (5) | "hare" (1), "sleepy" (3), "tortoise" (4)
            # Original: 1: The quick brown fox jumps over the lazy dog. (the, over, quick, lazy are stopwords)
            # Cleaned: brown fox jumps dog -> fox is at index 2
            # Original: 3: The quick and lazy fox is quick. (the, and, is, quick, lazy are stopwords)
            # Cleaned: fox -> fox is at index 1. Wait, this is wrong.
            # "The quick brown fox jumps over the lazy dog." -> "brown fox jumps dog"
            #   - "The" (stop)
            #   - "quick" (custom stop)
            #   - "brown" (0)
            #   - "fox" (1)
            #   - "jumps" (2)
            #   - "over" (stop)
            #   - "the" (stop)
            #   - "lazy" (custom stop)
            #   - "dog" (3)
            # So, for text 1, with search_term="fox", positions are: "fox" at 1.
            # "The quick and lazy fox is quick." -> "fox"
            #   - "The" (stop)
            #   - "quick" (custom stop)
            #   - "and" (stop)
            #   - "lazy" (custom stop)
            #   - "fox" (0)
            #   - "is" (stop)
            #   - "quick" (custom stop)
            # So, for text 3, with search_term="fox", positions are: "fox" at 0.
            # Expected for "fox": [1, 0] not [2,7]
            # Let's re-evaluate the expected values for count_stopwords=False
        }
        # Text 1: "The quick brown fox jumps over the lazy dog." -> "brown fox jumps dog"
        # Positions (0-indexed) for search_term="fox":
        #   - "brown" is 0, "fox" is 1, "jumps" is 2, "dog" is 3.
        # Text 2: "A fast hare and a sleepy tortoise." -> "fast hare sleepy tortoise"
        #   - "fast" 0, "hare" 1, "sleepy" 2, "tortoise" 3
        # Text 3: "The quick and lazy fox is quick." -> "fox"
        #   - "fox" 0
        # Text 4: "No stopwords here just words." -> "no stopwords here just words"
        #   - "no" 0, "stopwords" 1, "here" 2, "just" 3, "words" 4
        # Text 5: "quick lazy quick lazy the the a a." -> "" (empty after stopword removal)

        expected_positions_no_stopwords_fox = {"fox": [1, 0]}  # Doc IDs 1, 3
        positions_fox_no_sw = sca.get_positions(
            ["fox"], count_stopwords=False, search_term="fox"
        )
        assert positions_fox_no_sw == expected_positions_no_stopwords_fox

        expected_positions_no_stopwords_dog = {"dog": [3]}  # Doc ID 1
        positions_dog_no_sw = sca.get_positions(
            ["dog"], count_stopwords=False, search_term="dog"
        )
        assert positions_dog_no_sw == expected_positions_no_stopwords_dog

        expected_positions_no_stopwords_hare = {"hare": [1]}  # Doc ID 2
        positions_hare_no_sw = sca.get_positions(
            ["hare"], count_stopwords=False, search_term="hare"
        )
        assert positions_hare_no_sw == expected_positions_no_stopwords_hare

        expected_positions_no_stopwords_words = {"words": [4]}  # Doc ID 4
        positions_words_no_sw = sca.get_positions(
            ["words"], count_stopwords=False, search_term="words"
        )
        assert positions_words_no_sw == expected_positions_no_stopwords_words

        # Test with count_stopwords=True
        # Text 1: "The quick brown fox jumps over the lazy dog." (9 words)
        # "fox" is at index 3 (0-indexed)
        # Text 3: "The quick and lazy fox is quick." (7 words)
        # "fox" is at index 4
        positions_stopwords_fox = sca.get_positions(
            ["fox"],
            count_stopwords=True,
            search_term="fox",
        )
        expected_positions_stopwords_fox = {"fox": [3, 4]}  # Doc IDs 1, 3
        assert positions_stopwords_fox == expected_positions_stopwords_fox

        positions_stopwords_dog = sca.get_positions(
            ["dog"],
            count_stopwords=True,
            search_term="dog",
        )
        expected_positions_stopwords_dog = {"dog": [8]}  # Doc ID 1
        assert positions_stopwords_dog == expected_positions_stopwords_dog

        # Test case where target word itself is a stopword (custom)
        positions_quick_no_sw = sca.get_positions(
            ["quick"], count_stopwords=False, search_term="quick"
        )
        assert positions_quick_no_sw == {"quick": []}  # "quick" is a stopword

        positions_quick_sw = sca.get_positions(
            ["quick"], count_stopwords=True, search_term="quick"
        )
        # Text 1: "The quick brown fox jumps over the lazy dog." -> quick at 1
        # Text 3: "The quick and lazy fox is quick." -> quick at 1, quick at 6
        # Text 5: "quick lazy quick lazy the the a a." -> quick at 0, quick at 2
        expected_positions_quick_sw = {"quick": [1, 1, 6, 0, 2]}
        assert positions_quick_sw == expected_positions_quick_sw

        # Test case where target word is NLTK stopword
        positions_the_no_sw = sca.get_positions(
            ["the"], count_stopwords=False, search_term="the"
        )
        assert positions_the_no_sw == {"the": []}

        positions_the_sw = sca.get_positions(
            ["the"], count_stopwords=True, search_term="the"
        )
        # Text 1: "The quick brown fox jumps over the lazy dog." -> The at 0, the at 6
        # Text 3: "The quick and lazy fox is quick." -> The at 0
        # Text 5: "quick lazy quick lazy the the a a." -> the at 4, the at 5
        expected_positions_the_sw = {"the": [0, 6, 0, 4, 5]}
        assert positions_the_sw == expected_positions_the_sw

    def test_mark_windows_with_combined_stopwords(
        self, sca_with_custom_stopwords
    ):
        sca = sca_with_custom_stopwords

        # Test with count_stopwords=False
        # Text 1: "brown fox jumps dog", target "fox" (pos 1), context "brown" (pos 0)
        # Window: [-1, 1] around "fox" (1) -> [0, 2] -> "brown fox jumps"
        sca.mark_windows(
            search_pair=("fox", "brown"),
            left_span=1,
            right_span=1,
            count_stopwords=False,
            table_name="fox_brown_false",
        )
        conn = sqlite3.connect(sca.db_path)
        df_false = pd.read_sql_query("SELECT * FROM fox_brown_false", conn)
        conn.close()
        assert len(df_false) == 1
        assert df_false.iloc[0]["id"] == "1"
        assert df_false.iloc[0]["window"] == "brown fox jumps"

        # Test with count_stopwords=True
        # Text 1: "The quick brown fox jumps over the lazy dog.", target "fox" (pos 3), context "brown" (pos 2)
        # Window: [-1, 1] around "fox" (3) -> [2, 4] -> "brown fox jumps"
        sca.mark_windows(
            search_pair=("fox", "brown"),
            left_span=1,
            right_span=1,
            count_stopwords=True,
            table_name="fox_brown_true",
        )
        conn = sqlite3.connect(sca.db_path)
        df_true = pd.read_sql_query("SELECT * FROM fox_brown_true", conn)
        conn.close()
        assert len(df_true) == 1
        assert df_true.iloc[0]["id"] == "1"
        assert (
            df_true.iloc[0]["window"] == "brown fox jumps"
        )  # "The quick brown fox jumps over the lazy dog" -> window is "brown fox jumps"

        # Test where one of the search pair is a stopword (e.g. "quick", "fox")
        # "quick" is a custom stopword. mark_windows should skip it if count_stopwords=False
        sca.mark_windows(
            search_pair=("quick", "fox"),
            left_span=1,
            right_span=1,
            count_stopwords=False,
            table_name="quick_fox_false",
        )
        conn = sqlite3.connect(sca.db_path)
        with pytest.raises(
            pd.io.sql.DatabaseError, match="no such table: quick_fox_false"
        ):  # Or check if table is empty
            _ = pd.read_sql_query("SELECT * FROM quick_fox_false", conn)
        conn.close()

        # "quick" is a custom stopword. mark_windows should include it if count_stopwords=True
        # Text 1: "The quick brown fox jumps over the lazy dog."
        # "quick" (1), "fox" (3). Window [-1,1] around "quick" -> [0,2] -> "The quick brown"
        # Window [-1,1] around "fox" -> [2,4] -> "brown fox jumps"
        # For ("quick", "fox"), "quick" is target, "fox" is context or vice versa.
        # If "quick" is target (pos 1), "fox" (pos 3) is at offset +2.
        # If "fox" is target (pos 3), "quick" (pos 1) is at offset -2.
        # Let's assume ("target", "context")
        # Target "quick" (pos 1 in text 1), context "fox" (pos 3 in text 1). span_tuple should contain 3-1=2.
        # Window for ("quick", "fox") where "quick" is target. Text 1: "The quick brown fox" -> "quick" (1), "fox" (3)
        # Left 1, Right 1. Window around "quick" (1) -> [0,2] -> "The quick brown"
        sca.mark_windows(
            search_pair=("quick", "fox"),  # target, context
            left_span=1,  # from target
            right_span=1,  # from target
            count_stopwords=True,
            table_name="quick_fox_true",
        )
        conn = sqlite3.connect(sca.db_path)
        df_qf_true = pd.read_sql_query("SELECT * FROM quick_fox_true", conn)
        conn.close()
        # Expected for Text 1 ("The quick brown fox..."): target "quick" (idx 1), context "fox" (idx 3) -> distance +2
        # Window for "quick" (idx 1) with L1,R1 is "The quick brown"
        # Expected for Text 3 ("The quick and lazy fox is quick."):
        #   "quick" (idx 1), "fox" (idx 4) -> distance +3. Window for "quick" (idx 1) -> "The quick and"
        #   "quick" (idx 6), "fox" (idx 4) -> distance -2. Window for "quick" (idx 6) -> "is quick ." (if period included) -> "is quick" (if not)
        # The current implementation of mark_windows finds co-occurrences within the *document*, not necessarily within a span of each other.
        # It then stores the window around the *target* word.
        # So for ("quick", "fox"), "quick" is target.
        # Text 1: "quick" at 1. Window "The quick brown"
        # Text 3: "quick" at 1. Window "The quick and"
        # Text 3: "quick" at 6. Window "is quick" (assuming punctuation is tokenized and then removed or handled)
        # Let's check the actual output.
        # Assuming text is tokenized then punctuation is removed, then stopwords are processed
        # "The quick and lazy fox is quick." -> "the quick and lazy fox is quick"
        # if count_stopwords=True: tokens are ['the', 'quick', 'and', 'lazy', 'fox', 'is', 'quick']
        # "quick" at index 1. Window L1,R1: "the quick and"
        # "quick" at index 6. Window L1,R1: "is quick" (no following token)

        # Let's simplify: check if the table has rows for texts 1 and 3.
        assert (
            len(df_qf_true) >= 2
        )  # Should have entries from text 1 and text 3
        # Check windows for text 1
        row_text1_qf = df_qf_true[df_qf_true["id"] == "1"]
        assert len(row_text1_qf) == 1
        assert row_text1_qf.iloc[0]["window"] == "The quick brown"

        # Check windows for text 3. "quick" appears twice.
        # "The quick and lazy fox is quick."
        # Target "quick" (idx 1), context "fox" (idx 4) -> window "The quick and"
        # Target "quick" (idx 6), context "fox" (idx 4) -> window "is quick" (no next token if split by space and remove punctuation)
        # If 'quick.' becomes 'quick'. Then last token is 'quick'. Text: "The quick and lazy fox is quick"
        # "quick" at 1, window "The quick and"
        # "quick" at 6, window "is quick"
        rows_text3_qf = df_qf_true[df_qf_true["id"] == "3"]
        assert len(rows_text3_qf) == 2
        windows_text3 = sorted(
            list(rows_text3_qf["window"])
        )  # sort because order is not guaranteed
        assert windows_text3[0] == "The quick and"  # Around first "quick"
        assert windows_text3[1] == "is quick"  # Around second "quick"

    def test_create_collocate_group_with_combined_stopwords(
        self, sca_with_custom_stopwords
    ):
        sca = sca_with_custom_stopwords

        # Add a collocate pair that involves a custom stopword
        # ("brown", "fox") - neither are stopwords.
        # ("quick", "fox") - "quick" is a custom stopword.
        # ("the", "fox") - "the" is an NLTK stopword.

        sca.add_collocates(
            [("brown", "fox"), ("quick", "fox"), ("the", "fox")]
        )

        # When creating a group, the behavior depends on how add_collocates handles stopwords.
        # `add_collocates` filters out pairs where *either* term becomes empty after cleaning (which includes stopword removal).
        # So ("quick", "fox") and ("the", "fox") should not have been added effectively.
        # `self.terms` will contain 'brown', 'fox'.
        # `self.collocates` will contain ('brown', 'fox').
        assert ("brown", "fox") in sca.collocates
        assert (
            "quick",
            "fox",
        ) not in sca.collocates  # because "quick" is a stopword
        assert (
            "the",
            "fox",
        ) not in sca.collocates  # because "the" is a stopword

        # Now, create a group.
        # If we provide a collocate for the group that involves a stopword, it should be ignored.
        collocates_for_group = [
            ("brown", "fox", 5),  # Valid
            ("quick", "fox", 5),  # Invalid because "quick" is a stopword
            ("the", "fox", 5),  # Invalid because "the" is a stopword
        ]
        group_name = "test_group_stopwords"
        sca.create_collocate_group(group_name, collocates_for_group)

        conn = sqlite3.connect(sca.db_path)
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='group_{group_name}'"
        )
        assert (
            cursor.fetchone() is not None
        ), f"Table group_{group_name} was not created."

        df_group = pd.read_sql_query(f"SELECT * FROM group_{group_name}", conn)
        conn.close()

        # The group table should only contain the valid collocate ("brown", "fox")
        assert len(df_group) == 1
        assert (
            df_group.iloc[0]["collocate1"] == "brown"
        )  # add_collocates stores them sorted
        assert df_group.iloc[0]["collocate2"] == "fox"
        assert df_group.iloc[0]["max_window"] == 5

        # Test with count_stopwords=False when group was created (default in create_collocate_group's call to mark_windows)
        # For ("brown", "fox") in Text 1: "The quick brown fox jumps over the lazy dog."
        # Cleaned (no stopwords): "brown fox jumps dog"
        # "brown" is at 0, "fox" is at 1.
        # mark_windows called by create_collocate_group uses count_stopwords=False by default.
        # mark_windows(search_pair=("brown", "fox"), ..., count_stopwords=False)
        # Target "brown", context "fox". Window around "brown" (0) with L5,R5: "brown fox jumps dog"
        assert df_group.iloc[0]["id"] == "1"  # From Text 1
        assert df_group.iloc[0]["window_text"] == "brown fox jumps dog"


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
        sca.add_collocates([collocate_pair])

        # Assert
        assert (
            len(sca.collocates) == count_after_first_add
        ), "Adding a duplicate collocate should not change the count"

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
