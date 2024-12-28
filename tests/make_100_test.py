from pathlib import Path
from tempfile import mkdtemp

import pytest

from sca import SCA


@pytest.fixture(scope="module")
def tsv_file():
    here = Path(__file__).parent
    tsv_path = here / "uk_hansard_100rows.tsv"
    return tsv_path


@pytest.fixture(scope="module")
def temp_dir():
    return Path(mkdtemp())


def test_convert_tsv(temp_dir, tsv_file):
    db_path = temp_dir / "sca.sqlite3"
    sca = SCA(db_path=db_path, tsv_path=tsv_file)
