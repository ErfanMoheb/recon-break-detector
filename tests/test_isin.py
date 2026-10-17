import pytest

from recon.isin import check_digit, is_valid, make_isin


def test_known_isin():
    assert check_digit("US037833100") == 5
    assert make_isin("US", "037833100") == "US0378331005"
    assert is_valid("US0378331005")


def test_letters_in_body():
    assert is_valid("GB0002634946")
    assert is_valid("AU0000XVGZA3")


@pytest.mark.parametrize("bad", ["US0378331004", "US037833100", "12345678901X", ""])
def test_rejects_bad_isins(bad):
    assert not is_valid(bad)


def test_generated_securities_are_valid(small_ds):
    assert small_ds.securities["isin"].map(is_valid).all()
