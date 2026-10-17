"""Matcher behaviour on small hand-built ledgers, one scenario per test."""

import pandas as pd
import pytest

from recon import schema as S
from recon.match import normalise_reference, reconcile

ACCOUNT_MAP = {"CUS111": "ACC01", "CUS222": "ACC02"}
FX = {"USD": 1.0, "JPY": 0.0067}
D1, D2, D3 = "2026-03-02", "2026-03-03", "2026-03-04"


def booking(n, day=D1, qty=100, amount=10_000.00, ccy="USD", isin="US0378331005", acct="ACC01"):
    return {
        "internal_id": f"INT{n:08d}",
        "trade_ref": f"T{n:08d}",
        "account": acct,
        "isin": isin,
        "currency": ccy,
        "side": "BUY",
        "quantity": qty,
        "price": amount / qty,
        "amount": amount,
        "counterparty": "CP01",
        "booking_date": day,
    }


def line(
    n, ref, day=D1, qty=100, amount=10_000.00, ccy="USD", isin="US0378331005", custody="CUS111"
):
    return {
        "external_id": f"EXT{n:08d}",
        "statement_ref": ref,
        "custody_account": custody,
        "isin": isin,
        "currency": ccy,
        "side": "BUY",
        "quantity": qty,
        "amount": amount,
        "counterparty": "CP01",
        "statement_date": day,
    }


def run(internal, external):
    res = reconcile(pd.DataFrame(internal), pd.DataFrame(external), ACCOUNT_MAP, FX)
    exc = res.exceptions
    if exc.empty:
        exc = pd.DataFrame(columns=["break_type", "resolved_date", "first_seen"])
    return res.pairs, exc


@pytest.mark.parametrize(
    "ref, expected",
    [
        ("T00001234", 1234),
        ("t00001234", 1234),
        ("REF-T00001234", 1234),
        ("1234", 1234),
        ("T00001234/1", 1234),
        ("CUST 00001234", 1234),
        ("", None),
        (None, None),
    ],
)
def test_normalise_reference(ref, expected):
    assert normalise_reference(ref) == expected


def test_clean_match_across_reference_styles():
    pairs, exc = run([booking(1)], [line(1, "REF-T00000001")])
    assert pairs["status"].tolist() == ["MATCHED"]
    assert exc.empty


def test_rounding_inside_tolerance_is_not_a_break():
    pairs, exc = run([booking(1)], [line(1, "T00000001", amount=10_000.40)])
    assert pairs["status"].tolist() == ["MATCHED"]
    assert exc.empty


def test_amount_outside_tolerance_is_a_break():
    pairs, exc = run([booking(1)], [line(1, "T00000001", amount=10_050.00)])
    assert pairs["status"].tolist() == [S.B_AMOUNT]
    assert exc["break_type"].tolist() == [S.B_AMOUNT]
    assert exc["exposure_usd"].iloc[0] == pytest.approx(50.0)


def test_tolerance_is_in_usd_not_local_currency():
    # 100 yen is about USD 0.67: inside tolerance. 1,000 yen is about USD 6.70: outside.
    inside = run(
        [booking(1, ccy="JPY", amount=1_500_000)],
        [line(1, "T00000001", ccy="JPY", amount=1_500_100)],
    )
    outside = run(
        [booking(1, ccy="JPY", amount=1_500_000)],
        [line(1, "T00000001", ccy="JPY", amount=1_501_000)],
    )
    assert inside[0]["status"].tolist() == ["MATCHED"]
    assert outside[0]["status"].tolist() == [S.B_AMOUNT]


def test_partial_delivery_is_a_quantity_break():
    pairs, exc = run([booking(1)], [line(1, "T00000001", qty=60, amount=6_000.00)])
    assert pairs["status"].tolist() == [S.B_QTY]
    assert exc["exposure_usd"].iloc[0] == pytest.approx(4_000.0)


def test_blank_reference_matched_on_attributes():
    pairs, exc = run(
        [booking(1), booking(2, qty=200, amount=20_000.00)],
        [line(1, "", qty=200, amount=20_000.10)],
    )
    assert pairs["method"].tolist() == ["attributes"]
    assert pairs["internal_id"].tolist() == ["INT00000002"]
    assert exc["break_type"].tolist() == [S.B_MISSING_EXTERNAL]  # trade 1 is still unpaired


def test_blank_reference_needs_same_quantity():
    pairs, exc = run([booking(1)], [line(1, "", qty=50, amount=5_000.00)])
    assert pairs.empty
    assert sorted(exc["break_type"]) == [S.B_MISSING_EXTERNAL, S.B_MISSING_INTERNAL]


def test_late_posting_raises_then_resolves():
    pairs, exc = run([booking(1, day=D1)], [line(1, "T00000001", day=D2)])
    assert pairs["match_date"].tolist() == [D2]
    assert exc["break_type"].tolist() == [S.B_MISSING_EXTERNAL]
    assert exc["first_seen"].tolist() == [D1]
    assert exc["resolved_date"].tolist() == [D2]


def test_late_booking_raises_then_resolves():
    pairs, exc = run([booking(1, day=D2)], [line(1, "T00000001", day=D1)])
    assert exc["break_type"].tolist() == [S.B_MISSING_INTERNAL]
    assert exc["resolved_date"].tolist() == [D2]


def test_never_booked_stays_open():
    pairs, exc = run([booking(1, day=D3)], [line(1, "T00000099", day=D1)])
    missing = exc[exc["break_type"] == S.B_MISSING_INTERNAL]
    assert missing["resolved_date"].tolist() == [""]


def test_duplicate_same_day():
    pairs, exc = run([booking(1)], [line(1, "T00000001"), line(2, "t00000001")])
    assert pairs["external_id"].tolist() == ["EXT00000001"]
    assert exc["break_type"].tolist() == [S.B_DUPLICATE]
    assert exc["external_id"].tolist() == ["EXT00000002"]


def test_duplicate_next_day_without_reference():
    pairs, exc = run([booking(1)], [line(1, "T00000001", day=D1), line(2, "", day=D2)])
    assert exc["break_type"].tolist() == [S.B_DUPLICATE]


def test_wrong_account_is_not_matched():
    pairs, exc = run([booking(1)], [line(1, "T00000001", custody="CUS222")])
    assert pairs.empty
    assert sorted(exc["break_type"]) == [S.B_MISSING_EXTERNAL, S.B_MISSING_INTERNAL]


def test_truth_columns_do_not_change_the_result(small_ds):
    with_truth = reconcile(small_ds.internal, small_ds.external, small_ds.account_map, small_ds.fx)
    scrambled = small_ds.external.copy()
    scrambled["truth_trade_ref"] = (
        scrambled["truth_trade_ref"].sample(frac=1, random_state=0).values
    )
    scrambled["truth_event"] = "none"
    again = reconcile(small_ds.internal, scrambled, small_ds.account_map, small_ds.fx)
    pd.testing.assert_frame_equal(with_truth.pairs, again.pairs)
    pd.testing.assert_frame_equal(with_truth.exceptions, again.exceptions)


def test_opposite_side_with_same_reference_is_not_matched():
    sell = line(1, "T00000001") | {"side": "SELL"}
    pairs, exc = run([booking(1)], [sell])
    assert pairs.empty
    assert sorted(exc["break_type"]) == [S.B_MISSING_EXTERNAL, S.B_MISSING_INTERNAL]
