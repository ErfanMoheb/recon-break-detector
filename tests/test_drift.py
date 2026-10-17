from dataclasses import replace

import pandas as pd
import pytest

from recon import drift as D
from recon import generate as G
from recon import schema as S
from recon import triage as T
from recon.match import reconcile
from tests.conftest import SMALL

DRIFT_DAY = 20


def _late_share(internal: pd.DataFrame, cp: str, after: bool, drift_date: str) -> float:
    rows = internal[internal["counterparty"] == cp]
    rows = rows[(rows["booking_date"] >= drift_date) == after]
    return float((rows["truth_event"] == S.TIMING_EXTERNAL).mean())


def _same_data(a: dict, b: dict) -> None:
    pd.testing.assert_frame_equal(a["internal_book"], b["internal_book"])
    pd.testing.assert_frame_equal(a["custodian_statement"], b["custodian_statement"])
    assert a["meta"] == b["meta"]


def test_empty_drift_leaves_data_unchanged(small_data):
    _same_data(small_data, G.generate(replace(SMALL, drift=())))
    assert "drift" not in small_data["meta"]


def test_drift_to_the_same_rate_leaves_trades_unchanged(small_data):
    """The drift plumbing itself must not move any random draw."""
    prof = G.counterparty_profile(SMALL)
    same = tuple(G.LateRateChange(0, r.counterparty, r.late_rate) for r in prof.itertuples())
    other = G.generate(replace(SMALL, drift=same))
    pd.testing.assert_frame_equal(small_data["internal_book"], other["internal_book"])
    pd.testing.assert_frame_equal(small_data["custodian_statement"], other["custodian_statement"])


def test_drifted_counterparty_posts_late_more_often_after_the_drift_day(small_data):
    prof = G.counterparty_profile(SMALL)
    cp = prof.loc[prof["weight"].idxmax(), "counterparty"]
    drifted = G.generate(replace(SMALL, drift=(G.LateRateChange(DRIFT_DAY, cp, 0.25),)))
    drift_date = drifted["meta"]["trade_days"][DRIFT_DAY]
    assert drifted["meta"]["drift"][0]["date"] == drift_date

    base, new = small_data["internal_book"], drifted["internal_book"]
    before = _late_share(new, cp, after=False, drift_date=drift_date)
    after = _late_share(new, cp, after=True, drift_date=drift_date)
    assert after > 3 * before
    assert after > 2 * _late_share(base, cp, after=True, drift_date=drift_date)
    # Days before the drift are untouched.
    refs = base.loc[base["booking_date"] < drift_date, "trade_ref"]
    pd.testing.assert_frame_equal(
        base.set_index("trade_ref").loc[refs], new.set_index("trade_ref").loc[refs]
    )


def test_unknown_counterparty_is_rejected():
    with pytest.raises(ValueError, match="unknown counterparty"):
        G.generate(replace(SMALL, n_days=3, drift=(G.LateRateChange(1, "CP99", 0.1),)))


def test_drift_schedule_picks_one_busy_fast_and_one_slow_counterparty():
    cfg = G.GeneratorConfig()
    prof = G.counterparty_profile(cfg).set_index("counterparty")
    to_slow, to_fast = D.drift_schedule(cfg)
    slow = prof.nlargest(D.N_SLOW, "late_rate").index
    assert to_slow.counterparty not in slow
    assert prof.loc[to_slow.counterparty, "weight"] == prof.drop(slow)["weight"].max()
    assert to_fast.counterparty in slow
    assert to_slow.late_rate > prof.loc[to_slow.counterparty, "late_rate"]
    assert to_fast.late_rate < prof.loc[to_fast.counterparty, "late_rate"]
    assert to_slow.day_index == to_fast.day_index == D.DRIFT_DAY


def test_history_window_only_drops_old_outcomes(small_ds):
    exc = reconcile(
        small_ds.internal, small_ds.external, small_ds.account_map, small_ds.fx
    ).exceptions
    cal = small_ds.meta["calendar"]
    col = "counterparty_self_resolve_rate"
    full = T.build_features(exc, cal)[col]
    pd.testing.assert_series_equal(full, T.build_features(exc, cal, history_window=len(cal))[col])
    short = T.build_features(exc, cal, history_window=5)[col]
    early = exc["first_seen"] < cal[T.HISTORY_DELAY_BDAYS + 5]  # nothing has expired yet
    pd.testing.assert_series_equal(full[early], short[early])
    assert not full[~early].equals(short[~early])
