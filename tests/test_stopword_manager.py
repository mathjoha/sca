from pathlib import Path

import pytest
import yaml

from sca import SCA, from_yml


def test_sca_language_initialization():
    corpus = SCA(language="french")
    assert "le" in corpus.stopwords
    assert "the" not in corpus.stopwords


def test_french_stopwords():
    corpus = SCA(language="french")
    assert "le" in corpus.stopwords
    assert "the" not in corpus.stopwords


def test_invalid_language():
    with pytest.raises(
        ValueError, match="Invalid language code 'invalid_lang'"
    ):
        SCA(language="invalid_lang")


def test_load_invalid_language(tmpdir: Path):
    yml_path = tmpdir / "test_invalid_language.yml"

    with open(yml_path, "w", encoding="utf8") as f:
        yaml.safe_dump(data={"language": "invalid_lang"}, stream=f)

    with pytest.raises(
        ValueError, match="Invalid language code 'invalid_lang'"
    ):
        from_yml(yml_path)


def test_load_stopwords_from_file(tmp_path):
    sw_file = tmp_path / "custom_stopwords.txt"
    sw_file.write_text("custom1\ncustom2\ncustom3")

    corpus = SCA()
    corpus.load_stopwords_from_file(sw_file)
    assert "custom1" in corpus.stopwords
    assert "custom2" in corpus.stopwords


def test_invalid_stopwords_file():
    with pytest.raises(FileNotFoundError):
        corpus = SCA()
        corpus.load_stopwords_from_file("nonexistent.txt")


def test_modify_stopwords():
    corpus = SCA()
    corpus.add_stopwords({"new1", "new2"})
    assert "new1" in corpus.stopwords
    assert "new2" in corpus.stopwords

    corpus.remove_stopwords({"new1"})
    assert "new1" not in corpus.stopwords
    assert "new2" in corpus.stopwords


def test_invalid_stopwords_modification():
    corpus = SCA()
    with pytest.raises(TypeError, match="Stopwords must be provided as a set"):
        corpus.add_stopwords("not_a_set")

    with pytest.raises(TypeError, match="Stopwords must be provided as a set"):
        corpus.remove_stopwords(None)


def test_stopwords_persistence(tmp_path):
    # Create initial corpus with custom stopwords
    yml_path = tmp_path / "test_stopwords.yml"
    db_path = tmp_path / "test_stopwords.sqlite3"

    corpus = SCA(language="french")
    corpus.db_path = db_path
    corpus.yaml_path = yml_path
    corpus.id_col = "id"  # Initialize required attributes
    corpus.text_column = "text"
    corpus.columns = []
    corpus.add_stopwords({"custom1", "custom2"})
    corpus.save()

    # Load corpus from saved configuration
    loaded_corpus = from_yml(yml_path)
    assert loaded_corpus.language == "french"
    assert "custom1" in loaded_corpus.custom_stopwords
    assert "custom1" in loaded_corpus.stopwords
    assert "custom2" in loaded_corpus.stopwords
    assert "le" in loaded_corpus.stopwords  # French stopword
    assert "the" not in loaded_corpus.stopwords  # English stopword


def test_invalid_stopwords_config(tmp_path):
    yml_path = tmp_path / "test_invalid_stopwords.yml"
    db_path = tmp_path / "test_invalid_stopwords.sqlite3"

    corpus = SCA()
    corpus.language = "invalid_lang"  # Invalid language
    corpus.db_path = db_path
    corpus.yaml_path = yml_path
    corpus.id_col = "id"
    corpus.text_column = "text"
    corpus.columns = []

    with pytest.raises(ValueError, match="Invalid language configuration"):
        corpus.save()


def test_get_positions_with_custom_stopwords():
    corpus = SCA()
    corpus.add_stopwords({"custom_stop"})

    tokens = ["word1", "custom_stop", "word2"]
    positions = corpus.get_positions(tokens, False, "word*")

    assert positions["word*"] == [0, 1]

    positions_with_stopwords = corpus.get_positions(tokens, True, "word*")
    assert positions_with_stopwords["word*"] == [0, 2]


def test_get_cyrillic_positions_with_custom_stopwords():
    corpus = SCA()
    corpus.add_stopwords({"стоп"})

    tokens = ["слово1", "стоп", "слово2"]
    positions = corpus.get_positions(tokens, False, "слово*")

    assert positions["слово*"] == [0, 1]

    positions_with_stopwords = corpus.get_positions(tokens, True, "слово*")
    assert positions_with_stopwords["слово*"] == [0, 2]


def test_modify_stopwords():
    corpus = SCA()
    corpus.add_stopwords({"new1", "new2"})
    assert "new1" in corpus.stopwords

    corpus.remove_stopwords({"new1"})
    assert "new1" not in corpus.stopwords

    assert "new2" in corpus.stopwords
    assert "new1" not in corpus.stopwords
