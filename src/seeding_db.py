import os
import sqlite3

import requests
from tqdm.auto import tqdm

raw_data = "uk_hansard_1935_2014_BvW_2022.tsv"
db_path = "sca.sqlite3"


def download_data():
    resp = requests.get(
        "https://zenodo.org/record/7348819/files/uk_hansard_1935_2014_BvW_2022.tsv?download=1",
        stream=True,
    )

    with open(raw_data, "wb") as f:
        for chunk in tqdm(resp.iter_content(chunk_size=1024)):
            f.write(chunk)


def tsv2db(source=raw_data, db=db_path):
    with sqlite3.connect(db) as conn:
        with open(source, "r", encoding="utf8") as f:
            data = []
            for i, line in tqdm(enumerate(f), total=3_390_083):
                if i == 0:
                    headers = ",".join(line.rstrip().split("\t"))
                    qmarks = ",".join("?" for _ in line.split("\t"))
                    conn.execute(f"CREATE TABLE raw ({headers})")
                    conn.execute(
                        "CREATE UNIQUE INDEX index_sentence on raw (speech_id)"
                    )
                else:
                    data.append(line.split("\t"))

                if len(data) == 500_000:
                    conn.executemany(
                        f"INSERT INTO raw ({headers}) values ({qmarks})", data
                    )
                    data = []

        conn.execute(
            "CREATE TABLE collocate_window (speech_fk, pattern1, pattern2, window)"
        )

        conn.commit()


if __name__ == "__main__":
    if not os.path.exists(raw_data):
        download_data()

    tsv2db()
