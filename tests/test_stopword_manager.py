import pytest

from sca import SCA


@pytest.mark.xfail(
    strict=True, reason="Red Phase - Language support not implemented yet"
)
def test_sca_language_initialization():
    corpus = SCA(language="french")
    assert "le" in corpus.stopwords  # French stopword
    assert "the" not in corpus.stopwords  # English stopword


@pytest.mark.xfail(
    strict=True, reason="Red Phase - Language support not implemented yet"
)
def test_french_stopwords():
    corpus = SCA(language="french")
    assert "le" in corpus.stopwords
    assert "the" not in corpus.stopwords


@pytest.mark.xfail(
    strict=True, reason="Red Phase - Language support not implemented yet"
)
def test_invalid_language():
    with pytest.raises(
        ValueError, match="Invalid language code 'invalid_lang'"
    ):
        SCA(language="invalid_lang")
