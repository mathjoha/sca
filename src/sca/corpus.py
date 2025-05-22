import atexit
import logging
import re
import sqlite3
from collections import defaultdict
from fnmatch import fnmatch
from pathlib import Path

import pandas as pd
import sqlite_utils
from nltk.corpus import stopwords
from tqdm.auto import tqdm
from yaml import safe_dump, safe_load

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="sca.log",
)
logger = logging.getLogger(__name__)

cleaner_pattern = re.compile(r"[^a-zа-я]+")
tokenizer_pattern = re.compile(r"\s+")


def tokenizer(text):
    """Tokenizes text by splitting on whitespace and converting to lowercase.

    Args:
        text: The input string to tokenize.

    Returns:
        A list of tokens.
    """
    return tokenizer_pattern.split(text.lower())


def cleaner(token):
    """Removes non-alphabetic characters from a token.

    Args:
        token: The input token string.

    Returns:
        The cleaned token string.
    """
    return cleaner_pattern.sub("", token)


def get_min_window(pos1, pos2):
    """Calculates the minimum distance between two lists of positions.

    Args:
        pos1: A list of integer positions.
        pos2: A list of integer positions.

    Returns:
        The minimum absolute difference between any pair of positions
        from pos1 and pos2.
    """
    return min(abs(p1 - p2) for p1 in pos1 for p2 in pos2)


def sqlite3_friendly(column_name):
    """Checks if a column name is SQLite-friendly.

    A column name is considered SQLite-friendly if it contains only
    alphanumeric characters and underscores.

    Args:
        column_name: The column name string to check.

    Returns:
        True if the column name is SQLite-friendly, False otherwise.
    """
    return not re.search(r"[^a-zA-Z0-9_А-Яа-я]", column_name)


def from_file(
    tsv_path: str | Path,
    db_path: str | Path,
    id_col: str,
    text_column: str,
    language: str = "english",
):
    """Creates an SCA object from a TSV/CSV file and a database path.

    Args:
        tsv_path: Path to the input TSV or CSV file.
        db_path: Path to the SQLite database file.
        id_col: Name of the column containing unique identifiers.
        text_column: Name of the column containing the text data.

    Returns:
        An SCA object.
    """
    corpus = SCA(language=language)
    corpus.read_file(
        db_path=db_path,
        tsv_path=tsv_path,
        id_col=id_col,
        text_column=text_column,
    )

    return corpus


def from_yml(yml_path):
    """Creates an SCA object from a YAML configuration file.

    Args:
        yml_path: Path to the YAML configuration file.

    Returns:
        An SCA object.
    """
    corpus = SCA()
    corpus.load(yml_path)
    return corpus


class SCA:
    db_path = Path("sca.sqlite3")

    def __init__(self, language="english"):
        self.set_language(language)
        self.collocates = set()
        logger.info(
            f"Initialized SCA with language '{language}' and {len(self.stopwords)} stopwords"
        )

    def set_language(self, language):
        if language is None:
            self.language = None
            self.stopwords = set()
        elif not language in stopwords.fileids():
            raise ValueError(f"Invalid language code '{language}'")
        else:
            self.language = language
            self.stopwords = set(stopwords.words(language))
        self.custom_stopwords = set()

    def load_stopwords_from_file(self, file_path: str | Path):
        """Load custom stopwords from a text file.

        Args:
            file_path: Path to a text file containing stopwords, one per line.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Stopwords file not found: {file_path}")

        with open(file_path, "r", encoding="utf8") as f:
            custom_stopwords_from_file = {
                word.lower() for word in f.read().split()
            }

        # Update overall stopwords and only add to custom_stopwords those not in language set
        newly_added_custom = custom_stopwords_from_file - set(
            stopwords.words(self.language)
        )
        self.custom_stopwords.update(newly_added_custom)
        self.stopwords.update(
            custom_stopwords_from_file
        )  # ensure all from file are added to main set

        logger.info(
            f"Loaded {len(custom_stopwords_from_file)} stopwords from {file_path}. "
            f"{len(newly_added_custom)} were added to custom stopwords."
        )
        self._reset_stopwords_dependent_calculations()

    def read_file(
        self,
        tsv_path: Path | str,
        id_col: str,
        text_column: str,
        db_path="sca.sqlite3",
    ):
        """Reads data from a TSV/CSV file and initializes the SCA object.

        If the database file specified by db_path does not exist, it seeds the
        database from the tsv_path.

        Args:
            tsv_path: Path to the input TSV or CSV file.
            id_col: Name of the column containing unique identifiers.
            text_column: Name of the column containing the text data.
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        logger.info(f"Set db_path to {self.db_path}")

        self.yaml_path = self.db_path.with_suffix(".yml")
        logger.info(f"Set yaml_path to {self.yaml_path}")

        self.id_col = id_col
        self.text_column = text_column
        logger.info(
            f"Set id_col to '{self.id_col}' and text_column to '{self.text_column}'"
        )

        if not self.db_path.exists():
            logger.info(
                f"Database file {self.db_path} does not exist. Seeding database."
            )
            self.seed_db(tsv_path)
        else:
            logger.warning(
                f"Database file {self.db_path} already exists. Seeding will be aborted as it's only allowed for a non-existent database."
            )
            raise FileExistsError(
                f"Database file '{self.db_path}' already exists. Seeding is only allowed to a non-existent database. If you intend to re-seed, please provide a new database path or delete the existing file '{self.db_path}'."
            )

        self.conn = sqlite3.connect(db_path)
        logger.info(f"Connected to database: {db_path}")
        self.terms = set(
            _[0]
            for _ in self.conn.execute(
                """
                select name from sqlite_master
                where type == "table"
                and instr(name, "_") == 0
                """
            ).fetchall()
        )
        logger.info(f"Loaded {len(self.terms)} terms from the database.")
        self.collocates = set(
            self.conn.execute(
                "select distinct pattern1, pattern2 from collocate_window"
            ).fetchall()
        )
        logger.info(
            f"Loaded {len(self.collocates)} collocate pairs from collocate_window table."
        )
        self.conn.execute(
            """create table if not exists named_collocate (
            name text,
            table_name text,
            term1 text,
            term2 text,
            window integer,
            UNIQUE(name, term1, term2, window))"""
        )
        logger.info("Ensured 'named_collocate' table exists.")
        atexit.register(self.save)

    def settings_dict(self):
        """Returns a dictionary of the current SCA settings.

        Returns:
            A dictionary containing settings like database path, collocates, and stopwords configuration.
        """
        settings = {
            "db_path": str(
                self.db_path.resolve().relative_to(
                    self.yaml_path.resolve().parent,
                )
            ),
            "collocates": self.collocates,
            "language": self.language,
            "custom_stopwords": list(
                self.stopwords - set(stopwords.words(self.language))
            ),
            "id_col": self.id_col,
            "text_column": self.text_column,
            "columns": sorted(self.columns),
        }
        return settings

    def save(self):
        """Saves the current SCA settings to a YAML file.

        The YAML file is saved with the same name as the database file but
        with a .yml extension.

        Raises:
            ValueError: If the language configuration is invalid
        """
        if (
            self.language is not None
            and self.language not in stopwords.fileids()
        ):
            raise ValueError("Invalid language configuration")

        logger.info(f"Saving SCA settings to {self.yaml_path}")
        settings = self.settings_dict()
        settings["collocates"] = sorted(settings["collocates"])
        settings["id_col"] = self.id_col
        settings["text_column"] = self.text_column
        settings["columns"] = sorted(list(self.columns))
        settings["custom_stopwords"] = sorted(self.custom_stopwords)
        with open(self.yaml_path, "w", encoding="utf8") as f:
            safe_dump(data=settings, stream=f)
        logger.info("Successfully saved SCA settings")

    def load(self, settings_path: str | Path):
        """Loads SCA settings from a YAML file.

        Args:
            settings_path: Path to the YAML settings file.

        Raises:
            ValueError: If the language configuration is invalid.
            KeyError: If required fields are missing.
        """
        self.yaml_path = Path(settings_path)
        logger.info(f"Loading SCA settings from {settings_path}")
        with open(settings_path, "r", encoding="utf8") as f:
            settings = safe_load(f)
        logger.info(f"Successfully loaded settings from {self.yaml_path}")

        self.set_language(settings["language"])

        logger.info(
            f"Loaded language '{self.language}' with {len(self.stopwords)} stopwords"
        )
        self.custom_stopwords = set(settings["custom_stopwords"])
        self.stopwords.update(self.custom_stopwords)

        self.db_path = Path(settings_path).parent / Path(settings["db_path"])
        logger.info(f"Set db_path to {self.db_path} from settings file.")
        self.collocates = set(
            tuple(collocate) for collocate in settings["collocates"]
        )
        logger.info(
            f"Loaded {len(self.collocates)} collocate pairs from settings."
        )
        self.id_col = settings["id_col"]
        self.text_column = settings["text_column"]
        logger.info(
            f"Set id_col to '{self.id_col}' and text_column to '{self.text_column}' from settings."
        )
        self.columns = sorted(settings["columns"])
        logger.info(
            f"Loaded {len(self.columns)} data columns from settings: {self.columns}"
        )
        self.set_data_cols()

        self.conn = sqlite3.connect(self.db_path)
        logger.info(f"Connected to database: {self.db_path}")
        self.terms = set(
            _[0]
            for _ in self.conn.execute(
                """
                select name from sqlite_master
                where type == "table"
                and instr(name, "_") == 0
                """
            ).fetchall()
        )
        logger.info(f"Loaded {len(self.terms)} terms from the database.")
        self.conn.execute(
            """create table if not exists named_collocate (
            name text,
            table_name text,
            term1 text,
            term2 text,
            window integer,
            UNIQUE(name, term1, term2, window))"""
        )
        logger.info("Ensured 'named_collocate' table exists.")

    def set_data_cols(self):
        """Sets the data columns for the SCA object.

        This is used for constructing SQL queries.
        """
        self.data_cols = ", ".join(self.columns)

    def __hash__(self):
        with open(self.db_path, "rb") as f:
            return hash(f.read())

    def __eq__(self, other):
        """Checks if two SCA objects are equal.

        Two SCA objects are considered equal if they have the same:
        - language
        - stopwords
        - collocates
        - id_col
        - text_column
        - columns
        - terms
        - database hash

        Args:
            other: Another SCA object to compare with.

        Returns:
            True if the objects are equal, False otherwise.
        """
        if not isinstance(other, SCA):
            return False

        return (
            self.language == other.language
            and self.stopwords == other.stopwords
            and self.collocates == other.collocates
            and self.id_col == other.id_col
            and self.text_column == other.text_column
            and self.columns == other.columns
            and self.terms == other.terms
            and hash(self) == hash(other)
        )

    def _add_term(self, term):
        """Adds a term to the database and updates the internal terms set.

        This involves tabulating the term occurrences in the raw text data.

        Args:
            term: The term string to add.
        """
        self.tabulate_term(term)
        self.terms |= {
            term,
        }
        logger.info(f"Added term: {term}. Total terms: {len(self.terms)}")

    def seed_db(self, source_path):
        """Seeds the SQLite database from a source CSV or TSV file.

        This method reads the source file, validates column names, creates
        necessary tables and indexes in the database.

        Args:
            source_path: Path to the source CSV or TSV file.

        Raises:
            ValueError: If text_column and id_col are the same, if the input
                file is empty, if column names are not SQLite-friendly, or if
                duplicate column names are found.
            AttributeError: If id_col or text_column are not found in the input file.
            TypeError: If the input file is not a valid TSV or CSV file.
        """

        logger.info(f"Starting to seed database from {source_path}")
        if self.text_column == self.id_col:
            logger.error("text_column and id_col cannot be the same.")
            raise ValueError(
                f"The 'id_col' ('{self.id_col}') and 'text_column' ('{self.text_column}') parameters cannot specify the same column name. Please provide distinct column names for identifiers and text content."
            )

        db = sqlite_utils.Database(self.db_path)
        logger.info(f"Initialized database object for {self.db_path}")

        if source_path.suffix.lower() == ".tsv":
            sep = "\t"
            logger.info(f"Detected TSV file: {source_path}")
        else:
            sep = ","
            logger.info(f"Assuming CSV file: {source_path}")

        data = pd.read_csv(source_path, sep=sep)
        logger.info(f"Read {len(data)} rows from {source_path}")

        if data.empty:
            logger.error(f"Input file {source_path} is empty.")
            raise ValueError(
                f"The input file '{source_path}' is empty and does not contain any data. Please provide a file with content."
            )

        if self.id_col not in data.columns:
            logger.error(f"Column {self.id_col} not found in {source_path}")
            raise AttributeError(
                f"The specified 'id_col' ('{self.id_col}') was not found in the columns of the input file '{source_path}'. Available columns are: {list(data.columns)}. Please ensure the column name is correct and present in the file."
            )
        if self.text_column not in data.columns:
            logger.error(
                f"Column {self.text_column} not found in {source_path}"
            )
            raise AttributeError(
                f"The specified 'text_column' ('{self.text_column}') was not found in the columns of the input file '{source_path}'. Available columns are: {list(data.columns)}. Please ensure the column name is correct and present in the file."
            )

        for column_name in data.columns:
            if not sqlite3_friendly(column_name):
                logger.error(
                    f"Column name {column_name} is not SQLite-friendly."
                )
                raise ValueError(
                    f"Column name {column_name} is not SQLite-friendly."
                )

        self.columns = sorted(
            set(
                map(
                    str.lower,
                    set(data.columns)
                    - {
                        self.id_col,
                        self.text_column,
                    },
                )
            )
        )
        logger.info(f"Set data columns: {self.columns}")

        if len(self.columns) != (len(data.columns) - 2):
            logger.error(f"Duplicate column names found: {self.columns}")
            raise ValueError("Duplicate column names found.")

        self.set_data_cols()

        db["raw"].insert_all(data.to_dict(orient="records"))
        logger.info(f"Inserted {len(data)} records into 'raw' table.")

        db["raw"].create_index([self.id_col], unique=True)
        logger.info(f"Created unique index on '{self.id_col}' in 'raw' table.")

        db["collocate_window"].create(
            {
                self.id_col: str,
                "pattern1": str,
                "pattern2": str,
                "window": int,
            },
            pk=[self.id_col, "pattern1", "pattern2"],
        )
        logger.info("Created 'collocate_window' table.")
        logger.info(f"Finished seeding database from {source_path}")

    def get_positions(self, tokens, count_stopwords=False, *patterns):
        """Retrieves the positions of specified patterns within a list of tokens.

        Args:
            tokens: A list of token strings.
            count_stopwords: If True, stopwords are included in position counts.
                             Defaults to False.
            *patterns: One or more patterns to search for. Patterns can include
                       wildcards ('*') for partial matching.

        Returns:
            A dictionary where keys are patterns and values are lists of
            integer positions where each pattern occurs.
        """
        pos_dict = defaultdict(list)
        stops = 0
        for i, token in enumerate(tokens):
            if token.lower() in self.stopwords and not count_stopwords:
                stops += 1
            else:
                for pattern in patterns:
                    if fnmatch(token, pattern):
                        pos_dict[pattern].append(i - stops)
                        break

        return pos_dict

    def tabulate_term(self, cleaned_pattern):
        """Creates a table in the database for a given term (cleaned_pattern).

        The table stores the foreign keys (text_fk) of texts from the 'raw' table
        that contain the term.
        If the table already exists, this method does nothing.

        Args:
            cleaned_pattern: The cleaned term string (no special characters) for
                             which to create a table.
        """
        if (cleaned_pattern,) not in self.conn.execute(
            "select tbl_name from sqlite_master"
        ).fetchall():
            logger.info(
                f"Table for term '{cleaned_pattern}' does not exist. Creating and populating."
            )
            self.conn.execute(
                f"create table {cleaned_pattern} (text_fk text unique)",
            )
            sqlite_utils.Database(self.conn).table(cleaned_pattern).upsert_all(
                [
                    {"text_fk": row[0]}
                    for row in self.conn.execute(
                        f"select {self.id_col} from raw where {self.text_column} like ?",
                        [f"%{cleaned_pattern}%"],
                    )
                ],
                pk="text_fk",
            )
            self.conn.commit()
            logger.info(
                f"Successfully created and populated table for term '{cleaned_pattern}'."
            )
        else:
            logger.info(f"Table for term '{cleaned_pattern}' already exists.")
            logger.debug(
                f"Skipping calculation for already tabulated term '{cleaned_pattern}'."
            )

    def mark_windows(self, pattern1, pattern2, count_stopwords=False):
        """Calculates and stores the minimum window between two patterns in texts.

        This method identifies texts containing both pattern1 and pattern2,
        calculates the minimum distance (window) between their occurrences in each
        such text, and stores this information in the 'collocate_window' table.

        Args:
            pattern1: The first pattern string.
            pattern2: The second pattern string.
            count_stopwords: If True, stopwords are included in position counts
                             when calculating windows. If False (default), stopwords
                             are ignored.
        """
        pattern1, pattern2 = sorted((pattern1, pattern2))
        logger.info(
            f"Marking windows for patterns: '{pattern1}' and '{pattern2}'. count_stopwords={count_stopwords}"
        )

        clean1 = cleaner(pattern1)
        clean2 = cleaner(pattern2)
        logger.info(f"Cleaned patterns: '{clean1}' and '{clean2}'.")

        self.tabulate_term(clean1)
        self.tabulate_term(clean2)

        data = []
        logger.info(
            f"Querying texts containing both '{clean1}' and '{clean2}'."
        )
        for speech_id, text in tqdm(
            self.conn.execute(
                f"""
                select {self.id_col}, {self.text_column} from raw
                join {clean1}
                on {clean1}.text_fk == {self.id_col}
                join {clean2}
                on {clean2}.text_fk == {self.id_col}
                """,
                {"term1": clean2, "term2": clean2},
            ),
            desc=f"Calculating windows for {pattern1} - {pattern2}",
            total=self.conn.execute(
                f"""
                select count(*) from {clean1}
                join {clean2}
                on {clean1}.text_fk == {clean2}.text_fk
                """
            ).fetchone()[0],
        ):
            pos_dict = self.get_positions(
                [cleaner(token) for token in tokenizer(text)],
                count_stopwords,
                pattern1,
                pattern2,
            )

            pos1 = pos_dict[pattern1]
            pos2 = pos_dict[pattern2]

            if len(pos1) == 0 or len(pos2) == 0:
                continue
            else:
                data.append(
                    (
                        speech_id,
                        pattern1,
                        pattern2,
                        get_min_window(pos1, pos2),
                    )
                )
        if len(data) == 0:
            # For tracking that no collocates were found
            data.append((None, pattern1, pattern2, None))
            logger.info(
                f"No occurrences found for '{pattern1}' - '{pattern2}'. "
                "Storing placeholder."
            )
        else:
            logger.info(
                f"Found {len(data)} instances for '{pattern1}' - '{pattern2}'."
            )

        db = sqlite_utils.Database(self.db_path)
        db["collocate_window"].upsert_all(
            [
                {
                    self.id_col: speech_id,
                    "pattern1": pattern1,
                    "pattern2": pattern2,
                    "window": window,
                }
                for speech_id, pattern1, pattern2, window in data
            ],
            pk=[self.id_col, "pattern1", "pattern2"],
        )
        logger.info(
            f"Stored window information for '{pattern1}' - '{pattern2}' in 'collocate_window' table."
        )

    def collocate_to_condition(self, pattern1, pattern2, window):
        """Generates an SQL condition string for a collocate pair and window size.

        Args:
            pattern1: The first pattern string of the collocate pair.
            pattern2: The second pattern string of the collocate pair.
            window: The maximum window size (distance) between the patterns.

        Returns:
            An SQL WHERE clause condition string.
        """
        return (
            f"(pattern1 == '{pattern1}'"
            f"and pattern2 == '{pattern2}'"
            f"and window <= {window})"
        )

    def add_collocates(self, collocates, allow_duplicates=False):
        """Adds new collocate pairs to the SCA object.

        This involves cleaning the input patterns, adding any new terms to the
        database, marking windows for new collocate pairs, and updating the
        internal set of collocates.

        Args:
            collocates: An iterable of collocate pairs (tuples of two pattern
                        strings). e.g. [("patternA", "patternB"), ...]
        """
        logger.info(f"Adding {len(collocates)} collocate pairs.")
        prepared_collocates = set()
        clean_terms = set()
        for collocate in collocates:
            clean_pair = {
                cleaner(pattern)
                for pattern in collocate
                if not str(pattern).isdigit()
            }

            if len(clean_pair) != 2:
                continue

            collocate = tuple(p.lower() for p in sorted(collocate))
            if collocate in self.collocates:
                continue

            clean_terms |= clean_pair
            prepared_collocates |= {
                tuple(sorted(collocate)),
            }
        if not prepared_collocates:
            logger.warning(
                f"No valid collocates to add from input: {collocates=}"
            )
            raise ValueError("No clean collocates to add.")
        elif (
            len(prepared_collocates) != len(collocates)
            and not allow_duplicates
        ):
            logger.warning(
                f"Could not add all provided collocates due to duplicates or invalid pairs (and allow_duplicates=False). Original: {collocates=}. "
                f"Only {prepared_collocates=} could have been added."
            )
            raise ValueError(f"Aborting: Could not add ALL collocates.")

        logger.info(
            f"Prepared {len(prepared_collocates)} new collocate pairs for processing."
        )
        logger.info(
            f"Identified {len(clean_terms)} unique clean terms from new collocates."
        )

        new_terms_to_add = clean_terms - self.terms
        if new_terms_to_add:
            logger.info(
                f"Adding {len(new_terms_to_add)} new terms to the database: {new_terms_to_add}"
            )
            for term in new_terms_to_add:
                self._add_term(term)
        else:
            logger.info("No new terms to add from the provided collocates.")

        logger.info(
            f"Marking windows for {len(prepared_collocates)} new collocate pairs."
        )
        for collocate in prepared_collocates:
            self.mark_windows(*collocate)
        self.collocates |= prepared_collocates
        logger.info(
            f"Successfully added {len(prepared_collocates)} new collocate pairs. Total collocates: {len(self.collocates)}."
        )

    def collocate_to_textID_query(self, collocates):
        """Generates an SQL subquery to select distinct text IDs based on collocates.

        Args:
            collocates: An iterable of collocate specifications, where each
                        specification is a tuple (pattern1, pattern2, window).

        Returns:
            An SQL subquery string that selects distinct text IDs (speech_ids)
            matching the given collocate conditions.
        """
        conditions = " or ".join(
            self.collocate_to_condition(p1, p2, w) for p1, p2, w in collocates
        )

        id_query = (
            f" (select {self.id_col} as window from "
            f"collocate_window where {conditions}) "
        )

        return id_query

    def count_with_collocates(self, collocates):
        """Counts occurrences in the raw data grouped by data columns, filtered by collocates.

        Args:
            collocates: An iterable of collocate specifications (pattern1, pattern2, window)
                        used to filter the texts before counting.

        Returns:
            A database cursor pointing to the results of the count query.
            The query groups by all columns specified in `self.data_cols`.
        """
        id_query = self.collocate_to_textID_query(collocates)

        c = self.conn.execute(
            f"""
            select {self.data_cols}, count(rowid) from raw
            where {self.id_col} in {id_query}
            group by {self.data_cols}
            """
        )

        return c

    def counts_by_subgroups(self, collocates, out_file):
        """Calculates and saves counts by subgroups, comparing baseline and collocate-filtered data.

        This method first calculates baseline counts from the 'raw' table, grouped
        by specified data columns. It then calculates counts for texts filtered by
        the given collocates, grouped similarly. Finally, it merges these counts
        and saves the result to a TSV file.

        Args:
            collocates: An iterable of collocate specifications (pattern1, pattern2, window)
                        used for filtering.
            out_file: Path to the output TSV file where results will be saved.
        """
        logger.info(
            f"Calculating counts by subgroups. Output file: {out_file}"
        )
        logger.info(
            f"Using {len(collocates)} collocate specifications for filtering."
        )
        # todo: test pre-calculating the baseline
        logger.info(
            f"Calculating baseline counts from 'raw' table, grouping by {self.data_cols}."
        )
        df_baseline = pd.read_sql_query(
            f"""
            select {self.data_cols}, count(rowid) as total
            from raw
            group by {self.data_cols}
            """,
            self.conn,
        ).fillna("N/A")
        logger.info(
            f"Baseline calculation complete. Found {len(df_baseline)} baseline groups."
        )

        id_query = self.collocate_to_textID_query(collocates)
        logger.info("Generated ID query for collocates.")

        logger.info("Calculating collocate-filtered counts.")
        df_collocates = pd.read_sql_query(
            f"""
            select parliament, party, party_in_power, district_class,
            seniority, count(rowid) as collocate_count
            from raw
            where {self.id_col} in {id_query}
            group by parliament, party, party_in_power,
            district_class, seniority
            """,
            self.conn,
        )
        logger.info(
            f"Collocate-filtered count calculation complete. Found {len(df_collocates)} groups."
        )

        logger.info("Merging baseline and collocate-filtered counts.")
        df_all = df_baseline.merge(
            df_collocates,
            on=[
                "parliament",
                "party",
                "party_in_power",
                "district_class",
                "seniority",
            ],
            how="outer",
        ).fillna(0)
        logger.info("Merge complete.")

        df_all["collocate_count"] = df_all["collocate_count"].apply(int)

        df_all.to_csv(out_file, sep="\t", encoding="utf8", index=False)
        logger.info(f"Successfully saved counts by subgroups to {out_file}")

    # add function for tabulation of the results ...
    ## headers = [d[0] for d in cursor.description]

    def create_collocate_group(self, collocate_name, collocates):
        """Creates a named group of collocates and stores matched text-ids"""

        table_name = "group_" + collocate_name.strip().replace(" ", "_")
        logger.info(
            f"Creating collocate group: '{collocate_name}'. Table name: '{table_name}'. "
            f"Using {len(collocates)} collocate specifications for this group."
        )

        self.conn.execute(
            f"""
            create table if not exists named_collocate (
            name, table_name, term1, term2, window,
            UNIQUE(term1, term2, window))
            """
        )

        named_collocate_records = [
            {
                "name": collocate_name,
                "table_name": table_name,
                "term1": pattern1,
                "term2": pattern2,
                "window": window,
            }
            for pattern1, pattern2, window in collocates
        ]
        self.conn.executemany(
            f"""
            insert into named_collocate (name, table_name, term1, term2, window)
            values (:name, :table_name, :term1, :term2, :window)
            """,
            named_collocate_records,
        )

        logger.info(
            f"Inserted {len(collocates)} specifications into 'named_collocate' for group '{collocate_name}'."
        )

        self.conn.execute(
            f"""
            create table {table_name} (text_fk, collocate_name fk)
            """
        )

        id_query = self.collocate_to_textID_query(collocates)
        logger.info("Generated ID query for collocates in this group.")

        ids = self.conn.execute(id_query[2:-2]).fetchall()
        self.conn.executemany(
            f"""
            insert into {table_name} (text_fk, collocate_name)
            values (?, "{collocate_name}")
            """,
            ids,
        )

        logger.info(f"Logged {len(ids)} used texts for {collocate_name}")

    def _reset_stopwords_dependent_calculations(self):
        """Resets database tables and internal states affected by stopword changes."""
        logger.info("Resetting stopwords-dependent calculations.")
        if hasattr(self, "conn") and self.conn is not None:
            cursor = self.conn.cursor()

            # Clear collocate_window table
            cursor.execute("DELETE FROM collocate_window")
            logger.info("Cleared 'collocate_window' table.")

            # Drop group tables and clear named_collocate entries
            group_tables_to_drop = cursor.execute(
                "SELECT table_name FROM named_collocate"
            ).fetchall()
            for (table_name,) in group_tables_to_drop:
                if (
                    table_name
                    and isinstance(table_name, str)
                    and sqlite3_friendly(table_name)
                ):  # Ensure table_name is valid and safe
                    cursor.execute(
                        f"DROP TABLE IF EXISTS [{table_name}]"
                    )  # Quote table name
                    logger.info(f"Dropped table '{table_name}'.")

            cursor.execute("DELETE FROM named_collocate")
            logger.info("Cleared 'named_collocate' table.")

            self.conn.commit()
            # Vacuum to reclaim space after deletions and drops
            self.conn.execute("VACUUM")
            logger.info("Vacuumed database.")
            logger.info("Committed changes for resetting calculations.")
        else:
            logger.info(
                "No active database connection, skipping database resets."
            )

        # Reset internal collocate set if it's derived from db that's now cleared
        self.collocates = set()
        logger.info("Reset internal collocates set.")

    def add_stopwords(self, new_stopwords: set):
        """Add custom stopwords programmatically.

        Args:
            stopwords: A set of strings to add as stopwords.

        Raises:
            TypeError: If stopwords is not a set.
        """
        if not isinstance(new_stopwords, set):
            raise TypeError("Stopwords must be provided as a set")

        self.custom_stopwords |= new_stopwords - self.stopwords
        self.stopwords |= new_stopwords
        logger.info(
            f"Added {len(new_stopwords - (self.stopwords - self.custom_stopwords))} custom stopwords"
        )
        self._reset_stopwords_dependent_calculations()

    def remove_stopwords(self, stopwords_to_remove: set):
        """Remove stopwords programmatically.

        Args:
            stopwords: A set of strings to remove from stopwords.

        Raises:
            TypeError: If stopwords is not a set.
        """
        if not isinstance(stopwords_to_remove, set):
            raise TypeError("Stopwords must be provided as a set")

        removed_custom = self.custom_stopwords.intersection(
            stopwords_to_remove
        )
        self.custom_stopwords.difference_update(stopwords_to_remove)

        removed_lang_sw = (
            set(stopwords.words(self.language))
            - (self.stopwords - stopwords_to_remove)
        ).intersection(stopwords_to_remove)

        self.stopwords.difference_update(stopwords_to_remove)
        logger.info(
            f"Removed {len(stopwords_to_remove)} stopwords. {len(removed_custom)} were custom, {len(removed_lang_sw)} were language-specific."
        )
        self._reset_stopwords_dependent_calculations()
