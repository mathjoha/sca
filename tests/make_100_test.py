import pytest

from sca import SCA


@pytest.mark.xfail(strict=True, reason="Red Phase")
def test_null():
    assert False
