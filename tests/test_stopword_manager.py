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
