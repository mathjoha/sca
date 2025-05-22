# Plan for Dynamic Stopwords Support

## 1. Add Language Support for NLTK Stopwords
1. Add language parameter to SCA initialization
    - Happy path test: Test initializing SCA with different languages (e.g., 'english', 'french', 'german') ✅
    - Implement feature: ✅
        - Move stopwords initialization to SCA class ✅
        - Add language parameter to __init__ ✅
        - Load stopwords from nltk based on language ✅
        - Store stopwords in instance variable ✅
    - Unhappy path test: Test invalid language codes ✅


## 2. Add Custom Stopwords Support
1. Add method to load stopwords from file
    - Happy path test: Test loading stopwords from file  ✅
    - Implement feature:  ✅
        - Add load_stopwords_from_file method  ✅
        - Support txt files with one word per line  ✅
    - Unhappy path test: Test invalid file formats and missing files  ✅

2. Add method to add/remove stopwords programmatically
    - Happy path test: Test adding and removing stopwords ✅
    - Implement feature:
        - Add add_stopwords and remove_stopwords methods ✅
        - Add validation for input types ✅
        - Update stopwords set in place ✅
    - Unhappy path test: Test invalid inputs ✅

## 3. Add Stopwords Persistence
1. Add stopwords to YAML configuration ✅
    - Happy path test: Test saving and loading stopwords configuration ✅

    - Implement feature: ✅
        - Add language and custom stopwords to settings_dict ✅
        - Update save and load methods ✅
        - Add validation for loaded stopwords ✅
    - Unhappy path test: Test loading invalid configurations ✅

## 4. Update Existing Functionality
1. Update stopwords-dependent methods
    - Happy path test: Test existing functionality with new stopwords system

    - Implement feature:
        - Update get_positions to use instance stopwords
        - Update mark_windows to use instance stopwords
        - Update create_collocate_group to use instance stopwords
        - Changing stopwords resets the stopwords-dependent calculations
        - Resetting calculations recalculates and vacuums.
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
    - SCA __eq__ needs to compare stopwords too. ✅

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
