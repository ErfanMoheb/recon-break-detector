"""ISIN construction and validation (ISO 6166)."""

from __future__ import annotations

import string

_ALNUM = string.digits + string.ascii_uppercase


def _expand(body: str) -> str:
    """Replace letters by two-digit numbers (A=10 ... Z=35)."""
    return "".join(str(_ALNUM.index(ch)) for ch in body)


def check_digit(body: str) -> int:
    """Luhn check digit for the first 11 characters of an ISIN."""
    if len(body) != 11 or any(ch not in _ALNUM for ch in body):
        raise ValueError(f"ISIN body must be 11 upper-case alphanumerics, got {body!r}")
    digits = [int(d) for d in _expand(body)]
    total = 0
    # The check digit will sit to the right, so the rightmost body digit is doubled.
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (10 - total % 10) % 10


def make_isin(country: str, nsin: str) -> str:
    """Build a full ISIN from a two-letter country code and a nine-character NSIN."""
    body = country.upper() + nsin.upper()
    return body + str(check_digit(body))


def is_valid(isin: str) -> bool:
    """True when ``isin`` has the right shape and check digit."""
    if len(isin) != 12 or not isin[:2].isalpha() or not isin[-1].isdigit():
        return False
    try:
        return check_digit(isin[:11]) == int(isin[-1])
    except ValueError:
        return False
