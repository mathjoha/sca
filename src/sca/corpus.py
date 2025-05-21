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

sw = set(stopwords.words("english"))
sw |= {
    "hon",
    "house",
    "member",
    "common",
    "speaker",
    "mr",
    "friend",
    "gentleman",
}
sw |= {"one", "would"}


# todo: move to separate file.
cleaner_pattern = re.compile(r"[^a-z]+")
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
    return not re.search(r"[^a-zA-Z0-9_]", column_name)


def from_file(
    tsv_path: str | Path, db_path: str | Path, id_col: str, text_column: str
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
    corpus = SCA()
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
            logger.info(
                f"Database file {self.db_path} already exists. Skipping seed."
            )
            raise FileExistsError(
                f"Trying to seed to an existing database file: {self.db_path}"
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
        atexit.register(self.save)

    def settings_dict(self):
        """Returns a dictionary of the current SCA settings.

        Returns:
            A dictionary containing settings like database path and collocates.
        """
        settings = {
            "db_path": str(
                self.db_path.resolve().relative_to(
                    self.yaml_path.resolve().parent,
                )
            ),
            # Source file w/ hash?
            "collocates": self.collocates,
        }
        return settings

    def save(self):
        """Saves the current SCA settings to a YAML file.

        The YAML file is saved with the same name as the database file but
        with a .yml extension.
        """
        logger.info(f"Saving SCA settings to {self.yaml_path}")
        settings = self.settings_dict()
        settings["collocates"] = list(settings["collocates"])
        settings["id_col"] = self.id_col
        settings["text_column"] = self.text_column
        settings["columns"] = sorted(list(self.columns))
        with open(self.yaml_path, "w", encoding="utf8") as f:
            safe_dump(data=settings, stream=f)
        logger.info(f"Successfully saved SCA settings to {self.yaml_path}")

    def load(self, settings_path: str | Path):
        """Loads SCA settings from a YAML configuration file.

        Args:
            settings_path: Path to the YAML configuration file.
        """
        self.yaml_path = Path(settings_path)
        logger.info(f"Loading SCA settings from {self.yaml_path}")
        with open(settings_path, "r", encoding="utf8") as f:
            settings = safe_load(f)
        logger.info(f"Successfully loaded settings from {self.yaml_path}")

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

    def set_data_cols(self):
        """Sets the data_cols attribute as a comma-separated string of columns.

        This is used for constructing SQL queries.
        """
        self.data_cols = ", ".join(self.columns)

    def __hash__(self):
        with open(self.db_path, "rb") as f:
            return hash(f.read())

    def __eq__(self, other):
        if not isinstance(other, SCA):
            return False

        return (
            self.collocates == other.collocates
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
        """

        logger.info(f"Starting to seed database from {source_path}")
        if self.text_column == self.id_col:
            logger.error("text_column and id_col cannot be the same.")
            raise ValueError("text_column and id_col cannot be the same")

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
            raise ValueError(f"Input file {source_path} is empty.")

        if self.id_col not in data.columns:
            logger.error(f"Column {self.id_col} not found in {source_path}")
            raise AttributeError(
                f"Column {self.id_col} not found in {source_path}",
            )
        if self.text_column not in data.columns:
            logger.error(
                f"Column {self.text_column} not found in {source_path}"
            )
            raise AttributeError(
                f"Column {self.text_column} not found in {source_path}"
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
            set(map(str.lower, data.columns))
            - {
                self.id_col,
                self.text_column,
            }
        )
        logger.info(f"Set data columns: {self.columns}")

        if len(self.columns) != (len(data.columns) - 2):
            logger.error(f"Duplicate column names found: {self.columns}")
            raise ValueError(
                "Duplicate column names found." + ", ".join(self.columns)
            )

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
        """Finds all occurrences of patterns in a list of tokens.

        Args:
            tokens: A list of token strings.
            count_stopwords: If True, stopwords are included in position counts.
                             If False (default), stopwords are ignored and do not
                             affect position numbering.
            *patterns: Variable length argument list of pattern strings to search for.
                       Patterns can include wildcards (e.g., "*ing").

        Returns:
            A dictionary where keys are the input patterns and values are lists
            of integer positions where each pattern was found. Positions are
            0-indexed.
        """
        pos_dict = {_: [] for _ in patterns}
        stops = 0
        for i, token in enumerate(tokens):
            if token.lower() in sw:
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
        data = {"table": cleaned_pattern}
        if (cleaned_pattern,) not in self.conn.execute(
            "select tbl_name from sqlite_master"
        ).fetchall():
            logger.info(
                f"Table for term '{cleaned_pattern}' does not exist. Creating and populating."
            )
            self.conn.execute(
                f"create table {cleaned_pattern} (text_fk)",
                data,
            )
            self.conn.execute(
                f"""
                insert into {cleaned_pattern} select {self.id_col} from
                raw where {self.text_column} like "%" || :table || "%"
                """,
                data,
            )
            self.conn.commit()
            logger.info(
                f"Successfully created and populated table for term '{cleaned_pattern}'."
            )
        else:
            logger.info(
                f"Table for term '{cleaned_pattern}' already exists. Skipping calculation."
            )
            print(cleaned_pattern, "not calculated")

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
        return f"""
        (pattern1 == "{pattern1}"
        and pattern2 == "{pattern2}"
        and window <= {window})
        """

    def add_collocates(self, collocates):
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

        if prepared_collocates:
            logger.info(
                f"Marking windows for {len(prepared_collocates)} new collocate pairs."
            )
            for collocate in prepared_collocates:
                self.mark_windows(*collocate)
            self.collocates |= prepared_collocates
            logger.info(
                f"Successfully added {len(prepared_collocates)} new collocate pairs. Total collocates: {len(self.collocates)}."
            )
        else:
            logger.info(
                "No new collocate pairs to add (either duplicates or invalid). "
            )

    def collocate_to_speech_query(self, collocates):
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
            f" (select distinct {self.id_col} from "
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
        id_query = self.collocate_to_speech_query(collocates)

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

        id_query = self.collocate_to_speech_query(collocates)
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
        """Creates a named group of collocates and stores detailed token information.

        This method performs several actions:
        1. Creates (if not exists) a 'named_collocate' table to store metadata about
           the collocate group (name, table_name, term1, term2, window).
        2. Inserts the provided collocate specifications into 'named_collocate'.
        3. Creates a new table named 'group_<collocate_name>' (with spaces in
           collocate_name replaced by underscores).
        4. Populates this new table with token-level information for all texts that
           match any of the specified collocates. For each token, it stores:
           - text_fk: Foreign key to the original text in the 'raw' table.
           - raw_text: The original token string.
           - token: The cleaned token string.
           - sw: Boolean indicating if the token is a stopword.
           - conterm: (Currently seems to be set to None or False, purpose might be
                      related to "context term" - needs clarification).
           - collocate_begin: (Currently seems to be set to None, purpose unclear).
           - collocate_end: (Currently seems to be set to None, purpose unclear).

        Args:
            collocate_name: A string name for the collocate group.
            collocates: An iterable of collocate specifications, where each
                        specification is a tuple (pattern1, pattern2, window).
        """
        table_name = "group_" + collocate_name.strip().replace(" ", "_")
        logger.info(
            f"Creating collocate group: '{collocate_name}'. Table name: '{table_name}'."
        )
        logger.info(
            f"Using {len(collocates)} collocate specifications for this group."
        )

        self.conn.execute(
            """create table if not exists named_collocate (
            name text,
            table_name text,
            term1 text,
            term2 text,
            window integer)"""
        )
        logger.info("Ensured 'named_collocate' table exists.")

        self.conn.executemany(
            f"""
            insert into named_collocate (name, table_name, term1,
              term2, window)
            values ("{collocate_name}", "{table_name}", ?, ?, ?)
            """,
            collocates,
        )
        logger.info(
            f"Inserted {len(collocates)} specifications into 'named_collocate' for group '{collocate_name}'."
        )

        self.conn.execute(
            f"""
            create table {table_name} (text_fk, raw_text, token,
            sw, conterm, collocate_begin, collocate_end)
            """
        )
        logger.info(
            f"Created table '{table_name}' for collocate group details."
        )

        id_query = self.collocate_to_speech_query(collocates)
        logger.info("Generated ID query for collocates in this group.")

        collocate_patterns = {
            pattern for collocate in collocates for pattern in collocate[:2]
        }
        logger.info(
            f"Identified {len(collocate_patterns)} unique patterns for this group: {collocate_patterns}"
        )

        pattern_to_targets = defaultdict(set)
        for pattern1, pattern2, window in collocates:
            pattern_to_targets[pattern1] |= {
                (pattern2, window),
            }
            pattern_to_targets[pattern2] |= {
                (pattern1, window),
            }

        logger.info(
            f"Processing texts for collocate group '{collocate_name}'."
        )
        speech_data_to_insert = []
        for text_fk, text in self.conn.execute(
            f"""
            select {self.id_col}, {self.text_column} from raw
            where {self.id_col} in {id_query}
            """
        ):
            sw_pos_adjust = 0
            current_speech_tokens = []
            for pos, raw_token in enumerate(tokenizer(text)):
                token = cleaner(raw_token)

                is_sw = token in sw

                if is_sw:
                    sw_pos_adjust += 1
                    sw_pos = None
                    conterm = False
                    collocate = False

                else:
                    sw_pos = pos - sw_pos_adjust
                    conterm = None
                    collocates_match = [
                        pattern
                        for pattern in collocate_patterns
                        if fnmatch(token, pattern)
                    ]

                current_speech_tokens.append(
                    [
                        text_fk,
                        pos,
                        sw_pos,
                        raw_token,
                        token,
                        is_sw,
                        conterm,
                    ]
                )

            for i, token_data in enumerate(current_speech_tokens):
                if token_data[5]:
                    current_speech_tokens[i] = token_data[:6] + [
                        False,
                        None,
                        None,
                    ]
                else:
                    current_speech_tokens[i] = token_data[:6] + [
                        False,
                        None,
                        None,
                    ]
            speech_data_to_insert.extend(current_speech_tokens)

        logger.info(
            f"Inserting {len(speech_data_to_insert)} token entries into '{table_name}'."
        )
        self.conn.executemany(
            f"""
            insert into {table_name} (text_fk, raw_text, token,
            sw, conterm, collocate_begin, collocate_end)
            values (?, ?, ?, ?, ?, ?, ?)
            """,
            [item[:7] for item in speech_data_to_insert],
        )
        self.conn.commit()
        logger.info(f"Successfully inserted token data into '{table_name}'.")
