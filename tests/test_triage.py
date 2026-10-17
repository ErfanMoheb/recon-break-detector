import numpy as np
import pandas as pd
import pytest

from recon import schema as S
from recon import triage as T
from recon.evaluate import _queue_metrics
from recon.match import reconcile


@pytest.fixture(scope="module")
def exceptions(small_ds):
    return reconcile(
        small_ds.internal, small_ds.external, small_ds.account_map, small_ds.fx
    ).exceptions


def test_history_features_do_not_see_recent_outcomes(small_ds, exceptions):
    """Rewriting the outcomes of the last two days must not move today's features."""
    cal = small_ds.meta["calendar"]
    X = T.build_features(exceptions, cal)
    day = small_ds.meta["trade_days"][30]
    k = cal.index(day)
    recent = exceptions["first_seen"].isin(cal[k - T.SETTLE_WINDOW_BDAYS : k + 1])
    altered = exceptions.copy()
    altered.loc[recent, "resolved_date"] = np.where(
        altered.loc[recent, "resolved_date"] == "", altered.loc[recent, "first_seen"], ""
    )
    X2 = T.build_features(altered, cal)
    today = exceptions["first_seen"] == day
    hist_cols = [c for c in X.columns if c.endswith("_self_resolve_rate")]
    pd.testing.assert_frame_equal(X.loc[today, hist_cols], X2.loc[today, hist_cols])


def test_history_features_do_use_older_outcomes(small_ds, exceptions):
    cal = small_ds.meta["calendar"]
    X = T.build_features(exceptions, cal)
    late = exceptions["first_seen"] >= small_ds.meta["trade_days"][20]
    assert X.loc[late, "counterparty_self_resolve_rate"].nunique() > 3


def test_labels_follow_truth(small_ds, exceptions):
    y = T.label_needs_action(exceptions, small_ds.internal, small_ds.external)
    certain = exceptions["break_type"].isin([S.B_QTY, S.B_AMOUNT, S.B_DUPLICATE])
    assert y[certain].all()
    resolved = exceptions["resolved_date"] != ""
    assert not y[resolved].any()  # anything that cleared on its own was benign


def test_queue_metrics_on_a_hand_case():
    day = pd.DataFrame(
        {"exposure_usd": [100.0, 50.0, 10.0, 1.0], "needs_action": [False, True, True, False]}
    )
    m = _queue_metrics(day, np.array([4.0, 3.0, 2.0, 1.0]), capacity=2)
    assert m["precision_at_k"] == pytest.approx(0.5)
    assert m["value_recall_at_k"] == pytest.approx(50 / 60)
    assert m["items_to_90pct_value"] == 3


def test_every_strategy_returns_one_score_per_item(exceptions):
    day = exceptions[exceptions["first_seen"] == exceptions["first_seen"].iloc[-1]].assign(
        counterparty_self_resolve_rate=0.5
    )
    proba = np.full(len(day), 0.5)
    for fn in T.STRATEGIES.values():
        assert len(fn(day, proba=proba)) == len(day)
