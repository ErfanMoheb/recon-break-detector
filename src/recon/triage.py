"""Triage: rank each day's new exceptions so analysts work the real breaks first.

An exception "needs action" when the underlying trade has a real break. Missing-movement
exceptions are the ambiguous ones: most are late postings that clear within two business
days. The model estimates P(needs action) from the break itself and from how similar
exceptions behaved in the past, and the queue is ordered by expected exposure,
P(needs action) x exposure in USD.

History features use only outcomes an operations team could have observed: an exception
counts towards history three business days after it was raised, by which time a late
posting would have cleared. Nothing from the scored day's outcomes is used.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import schema as S

MISSING_TYPES = (S.B_MISSING_EXTERNAL, S.B_MISSING_INTERNAL)
SETTLE_WINDOW_BDAYS = 2
HISTORY_DELAY_BDAYS = SETTLE_WINDOW_BDAYS + 1
SMOOTHING = 10.0
CURRENCIES = tuple(S.MARKETS)
GROUP_KEYS = ("counterparty", "account", "currency")


def observed_self_resolved(exc: pd.DataFrame) -> pd.Series:
    """True when an exception cleared on its own within the settlement window."""
    resolved = exc["resolved_date"].fillna("").astype(str)
    gap = [np.busday_count(a, b) if b else 10**6 for a, b in zip(exc["first_seen"], resolved)]
    return pd.Series(np.asarray(gap) <= SETTLE_WINDOW_BDAYS, index=exc.index)


def build_features(exc: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    """Feature matrix, one row per exception, indexed like ``exc``."""
    day_idx = {d: k for k, d in enumerate(calendar)}
    t = exc["first_seen"].map(day_idx).to_numpy()
    cal = pd.to_datetime(pd.Series(calendar))
    month_end = (
        cal.groupby(cal.dt.to_period("M")).rank(ascending=False, method="first") <= 2
    ).to_numpy()

    out = pd.DataFrame(index=exc.index)
    for b in S.BREAK_TYPES:
        out[f"type_{b}"] = (exc["break_type"] == b).astype(int)
    for c in CURRENCIES:
        out[f"ccy_{c}"] = (exc["currency"] == c).astype(int)
    out["log_exposure"] = np.log10(1.0 + exc["exposure_usd"].astype(float))
    out["ref_present"] = exc["ref_present"].astype(int)
    out["weekday"] = cal.dt.weekday.to_numpy()[t]
    out["month_end"] = month_end[t].astype(int)
    out["side_buy"] = (exc["side"] == "BUY").astype(int)

    q_i = exc["quantity_internal"].astype(float)
    q_e = exc["quantity_external"].astype(float)
    a_i = exc["amount_internal"].astype(float)
    a_e = exc["amount_external"].astype(float)
    out["qty_ratio"] = (q_e / q_i).where(exc["break_type"] == S.B_QTY, 1.0).fillna(1.0)
    out["amount_diff_pct"] = (
        ((a_e - a_i).abs() / a_i).where(exc["break_type"] == S.B_AMOUNT, 0.0).fillna(0.0)
    )

    # Same-day burst: many missing items from one source at once.
    missing = exc["break_type"].isin(MISSING_TYPES)
    for key in ("counterparty", "account"):
        today = exc[missing].groupby(["first_seen", key]).size()
        idx = pd.MultiIndex.from_arrays([exc["first_seen"], exc[key]])
        out[f"{key}_missing_today"] = np.log1p(today.reindex(idx).fillna(0).to_numpy())

    # Historical self-resolution rates, revealed with a delay.
    outcome = observed_self_resolved(exc).to_numpy()
    order = np.argsort(t, kind="stable")
    counts = {k: defaultdict(lambda: [0, 0]) for k in GROUP_KEYS}
    total = [0, 0]
    rates = {k: np.zeros(len(exc)) for k in GROUP_KEYS}
    n_hist = np.zeros(len(exc))
    revealed = 0
    miss = missing.to_numpy()
    keys = {k: exc[k].to_numpy() for k in GROUP_KEYS}
    for pos in order:
        while revealed < len(order) and t[order[revealed]] <= t[pos] - HISTORY_DELAY_BDAYS:
            r = order[revealed]
            if miss[r]:
                total[0] += outcome[r]
                total[1] += 1
                for k in GROUP_KEYS:
                    c = counts[k][keys[k][r]]
                    c[0] += outcome[r]
                    c[1] += 1
            revealed += 1
        prior = (total[0] + 1.0) / (total[1] + 2.0)
        for k in GROUP_KEYS:
            hits, n = counts[k][keys[k][pos]]
            rates[k][pos] = (hits + SMOOTHING * prior) / (n + SMOOTHING)
        n_hist[pos] = np.log1p(counts["counterparty"][keys["counterparty"][pos]][1])
    for k in GROUP_KEYS:
        out[f"{k}_self_resolve_rate"] = rates[k]
    out["counterparty_history_n"] = n_hist
    return out


def label_needs_action(
    exc: pd.DataFrame, internal: pd.DataFrame, external: pd.DataFrame
) -> pd.Series:
    """Ground truth: the exception's trade carries a real (non-benign) event."""
    ev_int = internal.set_index("internal_id")["truth_event"]
    ev_ext = external.set_index("external_id")["truth_event"]
    event = np.where(
        exc["internal_id"] != "",
        exc["internal_id"].map(ev_int),
        exc["external_id"].map(ev_ext),
    )
    return pd.Series(~pd.Series(event).isin(S.BENIGN_EVENTS).to_numpy(), index=exc.index)


def fit_model(X: pd.DataFrame, y: pd.Series, seed: int = 0) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=seed,
    )
    return model.fit(X, y.astype(int))


# --- Ranking strategies: higher score = earlier in the analyst queue ---


def score_random(exc: pd.DataFrame, **_) -> np.ndarray:
    """No prioritisation: a fixed pseudo-random order per day."""
    day = str(exc["first_seen"].iloc[0]) if len(exc) else ""
    rng = np.random.default_rng(int(day.replace("-", "") or 0))
    return rng.random(len(exc))


def score_exposure(exc: pd.DataFrame, **_) -> np.ndarray:
    """Largest USD exposure first."""
    return exc["exposure_usd"].to_numpy(dtype=float)


def score_rules(exc: pd.DataFrame, **_) -> np.ndarray:
    """A typical desk rule: certain breaks (mismatch, duplicate) first, then by exposure."""
    certain = (~exc["break_type"].isin(MISSING_TYPES)).to_numpy()
    return certain * 1e12 + exc["exposure_usd"].to_numpy(dtype=float)


def score_counterparty_rule(exc: pd.DataFrame, **_) -> np.ndarray:
    """One-line rule: exposure x chance the item is real, read off counterparty history.

    Mismatches and duplicates count as certain; a missing item is discounted by how often the
    same counterparty's missing items have cleared on their own.
    """
    missing = exc["break_type"].isin(MISSING_TYPES).to_numpy()
    p_real = np.where(missing, 1.0 - exc["counterparty_self_resolve_rate"].to_numpy(), 1.0)
    return p_real * exc["exposure_usd"].to_numpy(dtype=float)


def score_model_probability(exc: pd.DataFrame, proba: np.ndarray, **_) -> np.ndarray:
    """Most-likely-real first, ignoring size."""
    return proba


def score_model(exc: pd.DataFrame, proba: np.ndarray, **_) -> np.ndarray:
    """Expected exposure: P(needs action) x exposure."""
    return proba * exc["exposure_usd"].to_numpy(dtype=float)


STRATEGIES = {
    "random": score_random,
    "largest_exposure": score_exposure,
    "desk_rules": score_rules,
    "counterparty_rule": score_counterparty_rule,
    "model_probability": score_model_probability,
    "model": score_model,
}
