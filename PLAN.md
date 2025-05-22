# Plan for Dynamic Stopwords Support

## 1. Add Language Support for NLTK Stopwords ✅

## 2. Add Custom Stopwords Support ✅

## 3. Add Stopwords Persistence ✅

## 4. Update Existing Functionality
1. Update stopwords-dependent methods
    - Happy path test: Test existing functionality with new stopwords system ✅

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
