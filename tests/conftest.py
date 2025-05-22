from pathlib import Path
from tempfile import mkdtemp

import pytest


@pytest.fixture(scope="module")
def temp_dir():
    return Path(mkdtemp())


@pytest.fixture(scope="module")
def db_path(temp_dir):
    return temp_dir / "sca.sqlite3"
