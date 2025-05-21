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
        corpus.remove_stopwords("not_a_set")
