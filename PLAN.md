# Plan: Add Uniqueness Checks to SCA Database

## Overview
Currently, the SCA database allows duplicate rows in tables other than the 'raw' table. We need to implement uniqueness checks to ensure data integrity.

## Implementation Steps

### 1. Red Phase: Add Tests for Uniqueness
1. Create test file `tests/test_uniqueness.py`
   - Test duplicate prevention in `collocate_window` table
   - Test duplicate prevention in term tables
   - Test duplicate prevention in `named_collocate` table
   - Test duplicate prevention in collocate group tables

Suggested prompt: "Let's start by implementing the uniqueness tests in test_uniqueness.py"

### 2. Green Phase: Implement Uniqueness Constraints
1. Modify `seed_db` method:
   - Add unique constraints to `collocate_window` table
   - Add error handling for duplicate entries

2. Modify `tabulate_term` method:
   - Add unique constraint to term tables
   - Add error handling for duplicate entries

3. Modify `create_collocate_group` method:
   - Add unique constraints to `named_collocate` table
   - Add unique constraints to group tables
   - Add error handling for duplicate entries

Suggested prompt: "Let's implement the uniqueness constraints in the seed_db method"

### 3. Blue Phase: Refactor and Optimize
1. Extract common uniqueness constraint logic:
   - Create helper method for adding unique constraints
   - Create helper method for handling duplicate entries
   - Update existing methods to use new helpers

2. Update documentation:
   - Add docstring information about uniqueness constraints
   - Update error messages to be more descriptive
   - Add logging for uniqueness violations

Suggested prompt: "Let's refactor the uniqueness constraint implementation to use helper methods"

### 4. Testing and Validation
1. Run all tests to ensure no regressions
2. Test edge cases:
   - Large datasets
   - Concurrent operations
   - Error recovery
   - Data migration for existing databases

Suggested prompt: "Let's run comprehensive tests on the uniqueness implementation"

## Commit Structure
1. "Red Phase: Add uniqueness tests"
2. "Green Phase: Implement uniqueness constraints"
3. "Blue Phase: Refactor uniqueness implementation"
4. "Update documentation for uniqueness constraints"

## Notes
- Each commit should be small and focused
- Run pre-commit hooks before each commit
- Follow pythonic code style
- Use descriptive variable names
- Add docstrings for new methods
- Maintain test coverage
