import atexit
import logging
import re
import sqlite3
from collections import defaultdict
from fnmatch import fnmatch
from pathlib import Path

import pandas as pd
from nltk.corpus import stopwords
from tqdm.auto import tqdm
from yaml import safe_dump, safe_load

# sw = set(stopwords.words("english"))
sw = {
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
cleaner_pattern = re.compile(r"\W+")
tokenizer_pattern = re.compile(r"\s+")


def tokenizer(text):
    return tokenizer_pattern.split(text.lower())


def cleaner(token):
    return cleaner_pattern.sub("", token)


def get_min_window(pos1, pos2):
    return min(abs(p1 - p2) for p1 in pos1 for p2 in pos2)


def from_tsv(tsv_path: str | Path, db_path: str | Path, id_col: str):
    corpus = SCA()
    corpus.read_tsv(db_path=db_path, tsv_path=tsv_path, id_col=id_col)

    return corpus


def from_yml(yml_path):
    corpus = SCA()
    corpus.load(yml_path)
    return corpus


class SCA:
    db_path = Path("sca.sqlite3")
    sep = "\t"

    def read_tsv(
        self,
        *,
        db_path="sca.sqlite3",
        tsv_path: Path | None = None,
        id_col=None,
    ):
        self.db_path = Path(db_path)
        self.yaml_path = self.db_path.with_suffix(".yml")
        if self.yaml_path.exists():
            self.load(self.yaml_path)
        if self.id_col is None:
            self.id_col = id_col

        if not self.db_path.exists():
            self.seed_db(tsv_path)

        self.conn = sqlite3.connect(db_path)
        self.terms = set(
            _[0]
            for _ in self.conn.execute(
                'select name from sqlite_master where type == "table" and instr(name, "_") == 0'
            ).fetchall()
        )
        self.collocates = set(
            self.conn.execute(
                "select distinct pattern1, pattern2 from collocate_window"
            ).fetchall()
        )
        atexit.register(self.save)

    def settings_dict(self):
        settings = {
            "db_path": str(
                self.db_path.resolve().relative_to(
                    self.yaml_path.resolve().parent
                )
            ),
            # Source file w/ hash?
            "collocates": self.collocates,
        }
        return settings

    def save(self):
        settings = self.settings_dict()
        settings["collocates"] = list(settings["collocates"])
        settings["id_col"] = self.id_col
        settings["columns"] = self.columns
        settings["text_column"] = self.text_column
        with open(self.yaml_path, "w", encoding="utf8") as f:
            safe_dump(data=settings, stream=f)

    def load(self, settings_path: str | Path):
        self.yaml_path = Path(settings_path)
        with open(settings_path, "r", encoding="utf8") as f:
            settings = safe_load(f)

        self.db_path = Path(settings_path).parent / Path(settings["db_path"])
        self.collocates = set(
            tuple(collocate) for collocate in settings["collocates"]
        )
        self.id_col = settings["id_col"]
        if self.id_col is None:
            self.id_col = "row"

        if settings["columns"] is None:
            raise ValueError("Cannot find columns!")

        self.set_columns(columns=settings["columns"])

        if type(settings["text_column"]) is not str:
            raise ValueError("Cannot find text_column name.")
        self.text_column = settings["text_column"]

    def set_columns(self, columns: list[str]):
        self.columns = columns
        self.columns_str = ", ".join(self.columns)

    def _add_term(self, term):
        self.tabulate_term(term)
        self.terms |= {
            term,
        }

    def seed_db(self, source_path):
        df = pd.read_csv(source_path, sep=self.sep)
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql("raw", conn, if_exists="replace", index=False)
            conn.execute(
                f"CREATE UNIQUE INDEX index_sentence ON raw ({self.id_col})"
            )

            conn.execute(
                "CREATE TABLE collocate_window (speech_fk, pattern1, pattern2, window)"
            )

            conn.commit()

        self.set_columns([_ for _ in df.columns if _ != self.text_column])

    def get_positions(self, tokens, count_stopwords=False, *patterns):
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
        data = {"table": cleaned_pattern}
        if (
            not (cleaned_pattern,)
            in self.conn.execute(
                "select tbl_name from sqlite_master"
            ).fetchall()
        ):
            self.conn.execute(
                f"create table {cleaned_pattern} (speech_fk)", data
            )
            self.conn.execute(
                f'insert into {cleaned_pattern} select {self.id_col} from raw where {self.text_column} like "%" || :table || "%"',
                data,
            )
            self.conn.commit()
        else:
            print(cleaned_pattern, "not calculated")

    def mark_windows(self, pattern1, pattern2, count_stopwords=False):
        pattern1, pattern2 = sorted((pattern1, pattern2))

        clean1 = cleaner(pattern1)
        clean2 = cleaner(pattern2)

        self.tabulate_term(clean1)
        self.tabulate_term(clean2)

        data = []
        for speech_id, text in tqdm(
            self.conn.execute(
                f"select {self.id_col}, {self.text_column} from raw join {clean1} on {clean1}.speech_fk == {self.id_col} join {clean2} on {clean2}.speech_fk == {self.id_col}",
                {"term1": clean2, "term2": clean2},
            ),
            desc=f"Calculating windows for {pattern1} - {pattern2}",
            total=self.conn.execute(
                f"select count(*) from {clean1} join {clean2} on {clean1}.speech_fk == {clean2}.speech_fk"
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

        self.conn.executemany(
            "insert into collocate_window (speech_fk, pattern1, pattern2, window) values (?, ?, ?, ?)",
            data,
        )
        self.conn.commit()

    def collocate_to_condition(self, pattern1, pattern2, window):
        return f' (pattern1 == "{pattern1}" and pattern2 == "{pattern2}" and window <= {window}) '

    def add_collocates(self, collocates):
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

        for term in clean_terms - self.terms:
            self._add_term(term)

        for collocate in prepared_collocates:
            self.mark_windows(*collocate)
        self.collocates |= prepared_collocates

    def collocate_to_speech_query(self, collocates):
        conditions = " or ".join(
            self.collocate_to_condition(p1, p2, w) for p1, p2, w in collocates
        )

        id_query = f" (select distinct speech_fk from collocate_window where {conditions}) "

        return id_query

    def count_with_collocates(self, collocates):
        id_query = self.collocate_to_speech_query(collocates)

        c = self.conn.execute(
            f"select {self.columns_str}, count(rowid) from raw where {self.id_col} in {id_query} group by {self.columns_str}"
        )

        return c

    def counts_by_subgroups(self, collocates, out_file):
        # todo: test pre-calculating the baseline
        df_baseline = pd.read_sql_query(
            f"select {self.columns_str}, count(rowid) as total from raw group by {self.columns_str}",
            self.conn,
        ).fillna("N/A")

        id_query = self.collocate_to_speech_query(collocates)

        df_collocates = pd.read_sql_query(
            f"select {self.columns_str}, count(rowid) as collocate_count from raw where {self.id_col} in {id_query} group by {self.columns_str}",
            self.conn,
        )

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

        df_all["collocate_count"] = df_all["collocate_count"].apply(int)

        df_all.to_csv(out_file, sep="\t", encoding="utf8", index=False)

    # add function for tabulation of the results ...
    ## headers = [d[0] for d in cursor.description]

    def create_collocate_group(self, name, collocates):
        """Collocate groups are used to avoid double counting when searching for multiple collocates at the same time."""
        table_name = "group_" + name.strip().replace(" ", "_")

        self.conn.execute(
            f"insert into named_collocate (name, table_name, term1, term2, window) values ({name}, {table_name}, ?, ?, ?)",
            collocates,
        )

        self.conn.execute(
            f"create table {table_name} (speech_fk, raw_text, token, sw, conterm, collocate_begin, collocate_end)"
        )

        id_query = self.collocate_to_speech_query(collocates)

        collocate_patterns = {
            pattern for collocate in collocates for pattern in collocate[:2]
        }

        pattern_to_targets = defaultdict(set)
        for pattern1, pattern2, window in collocates:
            pattern_to_targets[pattern1] |= {
                (pattern_to_targets, window),
            }
            pattern_to_targets[pattern2] |= {
                (pattern1, window),
            }

        data = []
        for speech_fk, text in self.conn.execute(
            f"select {self.id_col}, {self.text_column} from raw where speechid in {id_query}"
        ):
            sw_pos_adjust = 0
            speech_data = []
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
                    collocates = [
                        pattern
                        for pattern in collocate_patterns
                        if fnmatch(token, pattern)
                    ]

                speech_data.append(
                    [
                        speech_fk,
                        pos,
                        sw_pos,
                        raw_token,
                        token,
                        is_sw,
                        conterm,
                    ]
                )

            for i, (_, _, pos_sw, _, token, is_sw, _) in speech_data:
                if is_sw:
                    speech_data[i].append(False, None, None)

                else:
                    pass


class Speech:
    def __init__(self, speech_id, raw_text, collocates):
        self.speech_id = speech_id
        self.tokens = [
            Token(self, raw_token) for raw_token in tokenizer(raw_text)
        ]
        self.collocates = collocates

        self.collocate_tokens = []

        self.patterns = set(
            pattern
            for collocate in self.collocates
            for pattern in collocate[:2]
        )


class Token:
    def __init__(self, speech, raw_token, pos):
        self.speech = speech
        self.token = cleaner(raw_token)

        self.sw = self.Token in sw
        self.pos = pos
        self.sw_pos = pos

        self.coll_patterns = [
            pattern
            for pattern in self.speech.patterns
            if fnmatch(self.token, pattern)
        ]

        if len(self.coll_patterns) > 0:
            self.speech.collocate_tokens.append(self)

    def next_collocate(self):
        pass


if __name__ == "__main__":
    pass
