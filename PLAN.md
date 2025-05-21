# Plan for Dynamic Stopwords Support

## 1. Add Language Support for NLTK Stopwords
1. Add language parameter to SCA initialization
    - Happy path test: Test initializing SCA with different languages (e.g., 'english', 'french', 'german') ✅
    - Implement feature:
        - Move stopwords initialization to SCA class ✅
        - Add language parameter to __init__ ✅
        - Load stopwords from nltk based on language ✅
        - Store stopwords in instance variable ✅
    - Unhappy path test: Test invalid language codes ✅
    ```python
    def test_invalid_language():
        with pytest.raises(ValueError, match="Invalid language code 'invalid_lang'"):
            SCA(language='invalid_lang')
    ```

## 2. Add Custom Stopwords Support
1. Add method to load stopwords from file
    - Happy path test: Test loading stopwords from file
    ```python
    def test_load_stopwords_from_file(tmp_path):
        # Create test file
        sw_file = tmp_path / "custom_stopwords.txt"
        sw_file.write_text("custom1\ncustom2\ncustom3")

        corpus = SCA()
        corpus.load_stopwords_from_file(sw_file)
        assert "custom1" in corpus.stopwords
        assert "custom2" in corpus.stopwords
    ```
    - Implement feature:
        - Add load_stopwords_from_file method
        - Support txt files with one word per line
        - Add validation for file format
    - Unhappy path test: Test invalid file formats and missing files
    ```python
    def test_invalid_stopwords_file():
        with pytest.raises(FileNotFoundError):
            corpus = SCA()
            corpus.load_stopwords_from_file("nonexistent.txt")
    ```

2. Add method to add/remove stopwords programmatically
    - Happy path test: Test adding and removing stopwords
    ```python
    def test_modify_stopwords():
        corpus = SCA()
        corpus.add_stopwords({"new1", "new2"})
        assert "new1" in corpus.stopwords

        corpus.remove_stopwords({"new1"})
        assert "new1" not in corpus.stopwords
    ```
    - Implement feature:
        - Add add_stopwords and remove_stopwords methods
        - Add validation for input types
        - Update stopwords set in place
    - Unhappy path test: Test invalid inputs
    ```python
    def test_invalid_stopwords_modification():
        corpus = SCA()
        with pytest.raises(TypeError):
            corpus.add_stopwords("not_a_set")
    ```

## 3. Add Stopwords Persistence
1. Add stopwords to YAML configuration
    - Happy path test: Test saving and loading stopwords configuration
    ```python
    def test_stopwords_persistence():
        corpus = SCA(language='french')
        corpus.add_stopwords({"custom1", "custom2"})
        corpus.save()

        loaded_corpus = SCA.from_yml("sca.yml")
        assert loaded_corpus.language == 'french'
        assert "custom1" in loaded_corpus.stopwords
    ```
    - Implement feature:
        - Add language and custom stopwords to settings_dict
        - Update save and load methods
        - Add validation for loaded stopwords
    - Unhappy path test: Test loading invalid configurations
    ```python
    def test_invalid_stopwords_config():
        with pytest.raises(ValueError):
            corpus = SCA()
            corpus.language = None  # Invalid state
            corpus.save()
    ```

## 4. Update Existing Functionality
1. Update stopwords-dependent methods
    - Happy path test: Test existing functionality with new stopwords system
    ```python
    def test_get_positions_with_custom_stopwords():
        corpus = SCA()
        corpus.add_stopwords({"custom_stop"})
        positions = corpus.get_positions(
            ["word1", "custom_stop", "word2"],
            count_stopwords=False,
            "word*"
        )
        assert positions["word*"] == [0, 1]  # custom_stop is ignored
    ```
    - Implement feature:
        - Update get_positions to use instance stopwords
        - Update mark_windows to use instance stopwords
        - Update create_collocate_group to use instance stopwords
    - Unhappy path test: Test edge cases with modified stopwords
    ```python
    def test_empty_stopwords():
        corpus = SCA()
        corpus.stopwords.clear()
        positions = corpus.get_positions(
            ["the", "word"],
            count_stopwords=False,
            "word"
        )
        assert positions["word"] == [1]  # "the" not treated as stopword
    ```

## Implementation Order
1. Add language support (1)
2. Add custom stopwords support (2)
3. Add persistence (3)
4. Update existing functionality (4)

## Notes
- All new features should be documented in docstrings
- All changes should maintain backward compatibility
- Default language should remain 'english' for backward compatibility
- Custom stopwords should be additive to language-based stopwords by default
