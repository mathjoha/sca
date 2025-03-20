from src.sca import SCA
from pathlib import Path

data_file = Path("sca.sqlite3")
# data_file.unlink()

sca = SCA()

sca.id_col = "row"
sca.text_column = "Content"

sca.read_tsv(tsv_path="spod100.tsv")

collocates = [("изве*", "съв*")]
sca.add_collocates(collocates)


collocates = [("изве*", "съв*", 10)]

print(sca.count_with_collocates(collocates).fetchall())
