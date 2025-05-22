from pathlib import Path

import pandas as pd
import pytest

from sca import SCA, from_file


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


@pytest.fixture
def minimal_corpus_for_collocation(tmp_path: Path) -> SCA:
    """Creates a minimal SCA instance with a few texts for testing collocations."""
    csv_path = tmp_path / "minimal_colloc.csv"
    db_path = tmp_path / "minimal_colloc.sqlite3"

    data = {
        "doc_id": ["text1", "text2", "text3", "text4"],
        "content": [
            "alpha bravo charlie delta",  # alpha, bravo together
            "alpha foxtrot charlie golf",  # alpha, charlie, not close
            "hotel india bravo xray",  # bravo, no alpha
            "alpha bravo alpha bravo echo",  # multiple alpha, bravo
        ],
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    corpus = SCA(language="english")
    # Clear default stopwords for predictable testing of window sizes
    corpus.stopwords = set()
    corpus.custom_stopwords = set()

    corpus.read_file(
        tsv_path=csv_path,
        id_col="doc_id",
        text_column="content",
        db_path=db_path,
    )
    return corpus


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
        ValueError,
        match=r"Column name .+ is not SQLite-friendly\.",
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
    with pytest.raises(
        ValueError,
        match=r"Duplicate column names found\.",
    ):
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
        ValueError,
        match=r"The 'id_col' .+ and 'text_column' .+ parameters cannot specify the same column name\. Please provide distinct column names for identifiers and text content\.",
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

    expected = [f"header{i}" for i in range(5)]
    assert corpus_load.columns == [
        f"header{i}" for i in range(5)
    ], f"Expected {expected}, got {getattr(corpus_load, 'columns', 'attribute missing')}"

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

    expected_headers = [f"header_tsv_{i}" for i in range(3)]
    assert (
        expected_headers == corpus_load.columns
    ), f"Expected {expected_headers}, got {corpus_load.columns}"
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
        corpus_load.columns == []
    ), f"Expected {sorted(['id', 'text'])}, got {getattr(corpus_load, 'columns', 'attribute missing')}"


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

    db_path.touch()

    with pytest.raises(
        FileExistsError,
        match=rf"Database file '{db_path}' already exists\. Seeding is only allowed to a non-existent database\. If you intend to re-seed, please provide a new database path or delete the existing file '{db_path}'\.",
    ):
        from_file(
            tsv_path=db_path,
            id_col="id",
            text_column="text",
            db_path=db_path,
        )


def test_compare_same(tmp_path: Path):
    csv_path = tmp_path / "small_csv.csv"
    db_path = tmp_path / "small_csv.sqlite3"
    db_path2 = tmp_path / "small_csv2.sqlite3"
    yml_path = tmp_path / "small_csv.yml"

    create_dummy_csv(csv_path, 5, 5)

    corpus_1 = SCA()
    corpus_1.read_file(
        tsv_path=csv_path,
        id_col="id",
        text_column="text",
        db_path=db_path,
    )
    corpus_2 = from_file(
        tsv_path=csv_path,
        id_col="id",
        text_column="text",
        db_path=db_path2,
    )

    assert corpus_1 == corpus_2


def test_compare_different(tmp_path: Path):
    csv_path = tmp_path / "small_csv.csv"
    csv_path2 = tmp_path / "small_csv2.csv"
    db_path = tmp_path / "small_csv.sqlite3"
    db_path2 = tmp_path / "small_csv2.sqlite3"

    create_dummy_csv(csv_path, 5, 5)
    create_dummy_csv(csv_path2, 10, 10)

    corpus_1 = SCA()
    corpus_1.read_file(
        tsv_path=csv_path,
        id_col="id",
        text_column="text",
        db_path=db_path,
    )
    corpus_2 = from_file(
        tsv_path=csv_path2,
        id_col="id",
        text_column="text",
        db_path=db_path2,
    )

    assert corpus_1 != corpus_2


def test_seed_existing_db(tmp_path: Path):
    db_path = tmp_path / "small_csv.sqlite3"

    db_path.touch()

    with pytest.raises(
        FileExistsError,
        match=r"Database file '.*small_csv\.sqlite3' already exists\. Seeding is only allowed to a non-existent database\. If you intend to re-seed, please provide a new database path or delete the existing file '.*small_csv\.sqlite3'\.",
    ):
        from_file(
            tsv_path=db_path,
            id_col="id",
            text_column="text",
            db_path=db_path,
        )


def test_seed_db_with_tsv_headers_only_raises_db_error(tmp_path: Path):
    headers_only_tsv_path = tmp_path / "headers_only.tsv"
    db_path = tmp_path / "test_headers_only.sqlite3"

    with open(headers_only_tsv_path, "w") as f:
        f.write("id\ttext\n")

    with pytest.raises(
        ValueError,
        match=rf"The input file '{headers_only_tsv_path}' is empty and does not contain any data\. Please provide a file with content\.",
    ):
        from_file(
            tsv_path=headers_only_tsv_path,
            id_col="id",
            text_column="text",
            db_path=db_path,
        )


def test_language_initialization():
    """Test initializing SCA with different languages."""
    english_corpus = SCA(language="english")
    french_corpus = SCA(language="french")
    german_corpus = SCA(language="german")

    assert "the" in english_corpus.stopwords
    assert "le" in french_corpus.stopwords
    assert "der" in german_corpus.stopwords

    with pytest.raises(
        ValueError, match="Invalid language code 'invalid_lang'"
    ):
        SCA(language="invalid_lang")


@pytest.mark.xfail(strict=True, reason="Red Phase")
def test_empty_stopwords():
    corpus = SCA()
    # Simulate clearing all stopwords, including base language and custom
    corpus.stopwords = set()
    corpus.custom_stopwords = set()
    # We need to manually clear corpus.language related stopwords too for this test case
    # or re-initialize with a language that has no stopwords if such a thing exists in NLTK (unlikely)
    # For now, explicitly setting corpus.stopwords = set() covers the intent.

    positions = corpus.get_positions(
        ["the", "word"],  # Tokens: "the" at 0, "word" at 1
        "word",
        count_stopwords=False,  # This means stopwords SHOULD be skipped
    )
    # If "the" were a stopword and count_stopwords=False, it would be skipped,
    # and "word" would be at effective position 0.
    # Since stopwords is empty, "the" is NOT a stopword.
    # So, "the" is token 0, "word" is token 1.
    assert positions["word"] == [1]

    # Test with count_stopwords=True (shouldn't change behavior if list is empty)
    positions_count_true = corpus.get_positions(
        ["the", "word"],
        "word",
        count_stopwords=True,  # Stopwords (if any) are counted
    )
    assert positions_count_true["word"] == [1]

    # Test with a different default stopword to ensure it's not skipped
    corpus_original_stopwords = SCA()  # Standard English stopwords
    assert "the" in corpus_original_stopwords.stopwords

    positions_with_stops = corpus_original_stopwords.get_positions(
        ["the", "word"],
        "word",
        count_stopwords=False,  # "the" should be skipped
    )
    # "the" is skipped, "word" is at effective position 0
    assert positions_with_stops["word"] == [0]

    # Now, let's test the instance where stopwords were cleared
    corpus_original_stopwords.stopwords.clear()
    corpus_original_stopwords.custom_stopwords.clear()
    positions_after_clear = corpus_original_stopwords.get_positions(
        ["the", "word"], "word", count_stopwords=False
    )
    assert positions_after_clear["word"] == [
        1
    ]  # "the" is no longer a stopword


@pytest.mark.xfail(strict=True, reason="Red Phase [Expand as necessary]")
def test_add_remove_stopwords_impact_on_get_positions():
    corpus = SCA(language="english")  # Starts with default English stopwords
    tokens = ["a", "custom", "word", "the", "another"]

    # Initially, "a" and "the" should be stopwords
    positions = corpus.get_positions(
        tokens, "custom", "word", "another", count_stopwords=False
    )
    assert positions["custom"] == [0]  # "a" is skipped
    assert positions["word"] == [1]
    assert positions["another"] == [2]  # "the" is skipped

    # Add "custom" as a stopword
    corpus.add_stopwords({"custom"})
    positions_after_add = corpus.get_positions(
        tokens, "word", "another", count_stopwords=False
    )
    assert (
        "custom" not in positions_after_add
    )  # "custom" should now be skipped
    assert positions_after_add["word"] == [0]  # "a", "custom" skipped
    assert positions_after_add["another"] == [1]  # "the" skipped

    # Remove "the" from stopwords
    corpus.remove_stopwords({"the"})
    positions_after_remove = corpus.get_positions(
        tokens, "custom", "word", "the", "another", count_stopwords=False
    )
    # "custom" is still a stopword (from previous add)
    # "a" is a default stopword
    assert positions_after_remove["word"] == [0]  # "a", "custom" skipped
    assert positions_after_remove["the"] == [
        1
    ]  # "the" is no longer a stopword
    assert positions_after_remove["another"] == [2]


@pytest.mark.xfail(strict=True, reason="Red Phase [Expand as necessary]")
def test_stopwords_impact_on_mark_windows(minimal_corpus_for_collocation: SCA):
    corpus = minimal_corpus_for_collocation
    # In the fixture, stopwords are initially empty.

    # Scenario 1: count_stopwords=False (default for mark_windows)
    corpus.mark_windows("alpha", "bravo")  # default count_stopwords=False
    windows_before_stopword = (
        pd.read_sql_query(
            "SELECT doc_id, window FROM collocate_window WHERE pattern1='alpha' AND pattern2='bravo'",
            corpus.conn,
        )
        .set_index("doc_id")["window"]
        .to_dict()
    )
    assert windows_before_stopword.get("text1") == 1  # alpha bravo (0, 1)
    assert (
        windows_before_stopword.get("text4") == 1
    )  # alpha bravo (0, 1) and (2,3) -> min is 1
    assert "text2" not in windows_before_stopword  # alpha foxtrot charlie
    assert "text3" not in windows_before_stopword  # hotel india bravo

    # Add "bravo" as a stopword. This will trigger _reset_stopwords_dependent_calculations,
    # clearing collocate_window.
    corpus.add_stopwords({"bravo"})

    # Rerun mark_windows with count_stopwords=False
    # "bravo" is now a stopword. get_positions(..., "bravo", count_stopwords=False) will be empty.
    # So, no entries for ("alpha", "bravo") should be made.
    corpus.mark_windows("alpha", "bravo", count_stopwords=False)
    windows_after_stopword_false = (
        pd.read_sql_query(
            "SELECT doc_id, window FROM collocate_window WHERE pattern1='alpha' AND pattern2='bravo'",
            corpus.conn,
        )
        .set_index("doc_id")["window"]
        .to_dict()
    )
    assert (
        len(windows_after_stopword_false) == 0
    )  # No windows because "bravo" is skipped

    # Scenario 2: count_stopwords=True
    # _reset_stopwords_dependent_calculations was called by add_stopwords, so collocate_window is empty again.
    corpus.mark_windows("alpha", "bravo", count_stopwords=True)
    # "bravo" is a stopword, but count_stopwords=True means get_positions will find it.
    windows_after_stopword_true = (
        pd.read_sql_query(
            "SELECT doc_id, window FROM collocate_window WHERE pattern1='alpha' AND pattern2='bravo'",
            corpus.conn,
        )
        .set_index("doc_id")["window"]
        .to_dict()
    )
    assert windows_after_stopword_true.get("text1") == 1
    assert windows_after_stopword_true.get("text4") == 1
    assert "text2" not in windows_after_stopword_true
    assert "text3" not in windows_after_stopword_true

    # Remove "bravo" from stopwords. collocate_window is cleared again.
    corpus.remove_stopwords({"bravo"})
    corpus.mark_windows("alpha", "bravo")  # count_stopwords=False (default)
    windows_after_remove = (
        pd.read_sql_query(
            "SELECT doc_id, window FROM collocate_window WHERE pattern1='alpha' AND pattern2='bravo'",
            corpus.conn,
        )
        .set_index("doc_id")["window"]
        .to_dict()
    )
    assert windows_after_remove.get("text1") == 1
    assert windows_after_remove.get("text4") == 1


@pytest.mark.xfail(strict=True, reason="Red Phase [Expand as necessary]")
def test_stopwords_impact_on_create_collocate_group(
    minimal_corpus_for_collocation: SCA,
):
    corpus = minimal_corpus_for_collocation
    # In the fixture, stopwords are initially empty. `alpha` and `bravo` are not stopwords.

    collocates_spec = [("alpha", "bravo", 5)]
    group_name_v1 = "test_group_v1"
    corpus.create_collocate_group(group_name_v1, collocates_spec)

    group_table_name_v1 = f"group_{group_name_v1}"
    df_v1 = pd.read_sql_query(
        f"SELECT token, sw FROM {group_table_name_v1}", corpus.conn
    )

    # Check specific tokens from text1: "alpha bravo charlie delta"
    # None of these are stopwords initially in the fixture
    assert not df_v1[df_v1["token"] == "alpha"]["sw"].iloc[0]
    assert not df_v1[df_v1["token"] == "bravo"]["sw"].iloc[0]
    assert not df_v1[df_v1["token"] == "charlie"]["sw"].iloc[0]

    # Add "alpha" as a stopword. This should trigger a reset.
    corpus.add_stopwords({"alpha"})

    # Recreate the group (or create a new one). The old group table was dropped by the reset.
    group_name_v2 = "test_group_v2"
    corpus.create_collocate_group(group_name_v2, collocates_spec)
    group_table_name_v2 = f"group_{group_name_v2}"
    df_v2 = pd.read_sql_query(
        f"SELECT token, sw FROM {group_table_name_v2}", corpus.conn
    )

    # Now "alpha" should be marked as a stopword
    assert df_v2[df_v2["token"] == "alpha"]["sw"].iloc[0]
    # "bravo" and "charlie" should still not be stopwords
    assert not df_v2[df_v2["token"] == "bravo"]["sw"].iloc[0]
    assert not df_v2[df_v2["token"] == "charlie"]["sw"].iloc[0]

    # Remove "alpha" from stopwords. Reset is triggered.
    corpus.remove_stopwords({"alpha"})
    group_name_v3 = "test_group_v3"
    corpus.create_collocate_group(group_name_v3, collocates_spec)
    group_table_name_v3 = f"group_{group_name_v3}"
    df_v3 = pd.read_sql_query(
        f"SELECT token, sw FROM {group_table_name_v3}", corpus.conn
    )

    # "alpha" should no longer be a stopword
    assert not df_v3[df_v3["token"] == "alpha"]["sw"].iloc[0]
