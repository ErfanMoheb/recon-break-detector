"""End-to-end evaluation against the generator's ground truth.

Three questions, each answered on its own:

1. Matching: are the pairs right, and does each real break end up with the right type?
2. Triage: on held-out days, how much real exposure does an analyst clear in the first K
   items of the queue, for each ranking strategy?
3. Surveillance: are the injected incidents flagged on the day they happen, and how many
   alerts fire on ordinary days?

The time split is strict: the model is trained on exceptions raised before the split day
and scored on exceptions raised on or after it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import schema as S
from . import triage as T
from .explain import facts_for, template_explanation
from .io import Dataset
from .match import MatchResult, reconcile
from .surveil import WINDOW_DAYS, daily_alerts

TRAIN_DAYS = 80
CAPACITY = 10
COVERAGE_TARGET = 0.90
MAX_K = 40
# Labels of the last few training days would not be known yet on the first test day.
LABEL_EMBARGO_BDAYS = T.HISTORY_DELAY_BDAYS


@dataclass
class EvaluationResult:
    metrics: dict
    match: MatchResult
    exceptions: pd.DataFrame  # with features, labels and scores
    daily: pd.DataFrame  # one row per test day per strategy
    curves: pd.DataFrame  # mean value recall at k, per strategy
    alerts: pd.DataFrame


# --- 1. Matching ---


def _exception_trade(exc: pd.DataFrame, ds: Dataset) -> pd.Series:
    """Trade reference behind each exception (uses ground truth: evaluation only)."""
    int_ref = ds.internal.set_index("internal_id")["trade_ref"]
    ext_ref = ds.external.set_index("external_id")["truth_trade_ref"]
    return pd.Series(
        np.where(
            exc["internal_id"] != "",
            exc["internal_id"].map(int_ref),
            exc["external_id"].map(ext_ref),
        ),
        index=exc.index,
    )


def matching_metrics(ds: Dataset, res: MatchResult) -> dict:
    int_ref = ds.internal.set_index("internal_id")["trade_ref"]
    ext_ref = ds.external.set_index("external_id")["truth_trade_ref"]
    p = res.pairs
    correct = p["internal_id"].map(int_ref).to_numpy() == p["external_id"].map(ext_ref).to_numpy()

    two_sided = ds.external.loc[~ds.external["truth_is_duplicate"], "truth_trade_ref"]
    two_sided = set(two_sided) & set(ds.internal["trade_ref"])
    paired_trades = set(p.loc[correct, "internal_id"].map(int_ref))

    # Final state: mismatch and duplicate exceptions are final; missing ones count only
    # if they never cleared.
    exc = res.exceptions
    final = exc[(exc["resolved_date"] == "") | ~exc["break_type"].isin(T.MISSING_TYPES)]
    final_types = final.groupby(_exception_trade(final, ds))["break_type"].agg(set)

    expected = {}
    for r in ds.internal.itertuples():
        if r.truth_break_type:
            expected[r.trade_ref] = r.truth_break_type
    for r in ds.external.itertuples():
        if r.truth_break_type:
            expected[r.truth_trade_ref] = r.truth_break_type

    by_type = {}
    for b in S.BREAK_TYPES:
        refs = [t for t, e in expected.items() if e == b]
        detected = sum(t in final_types.index for t in refs)
        typed = sum(t in final_types.index and b in final_types[t] for t in refs)
        by_type[b] = {"n": len(refs), "detected": detected, "correct_type": typed}

    benign_with_open = [t for t in final_types.index if t not in expected]
    n_real = len(expected)
    return {
        "pairs": int(len(p)),
        "pair_precision": float(correct.mean()),
        "pair_recall": len(paired_trades & two_sided) / len(two_sided),
        "pairs_by_method": p["method"].value_counts().to_dict(),
        "real_breaks": n_real,
        "break_recall": sum(v["detected"] for v in by_type.values()) / n_real,
        "break_type_accuracy": sum(v["correct_type"] for v in by_type.values()) / n_real,
        "false_breaks_open_at_end": len(benign_with_open),
        "by_type": by_type,
    }


# --- 2. Triage ---


def _queue_metrics(day: pd.DataFrame, score: np.ndarray, capacity: int) -> dict:
    order = np.argsort(-score, kind="stable")
    real = day["needs_action"].to_numpy()[order]
    value = (day["exposure_usd"].to_numpy() * day["needs_action"].to_numpy())[order]
    total = value.sum()
    cum = np.cumsum(value) / total if total > 0 else np.ones(len(value))
    k = min(capacity, len(day))
    reach = int(np.searchsorted(cum, COVERAGE_TARGET - 1e-12) + 1)
    curve = np.array([cum[min(j, len(cum)) - 1] for j in range(1, MAX_K + 1)])
    return {
        "precision_at_k": real[:k].mean(),
        "value_recall_at_k": cum[k - 1],
        "items_to_90pct_value": min(reach, len(day)),
        "curve": curve,
    }


def _bootstrap_ci(x: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate(
    ds: Dataset,
    train_days: int = TRAIN_DAYS,
    capacity: int = CAPACITY,
    seed: int = 0,
    n_boot: int = 2000,
) -> EvaluationResult:
    res = reconcile(ds.internal, ds.external, ds.account_map, ds.fx)
    metrics: dict = {"matching": matching_metrics(ds, res)}

    trade_days = ds.meta["trade_days"]
    split = trade_days[train_days]
    exc = res.exceptions.copy()
    X = T.build_features(exc, ds.meta["calendar"])
    exc["needs_action"] = T.label_needs_action(exc, ds.internal, ds.external)
    train = (exc["first_seen"] < trade_days[train_days - LABEL_EMBARGO_BDAYS]).to_numpy()
    test = ((exc["first_seen"] >= split) & (exc["first_seen"] <= trade_days[-1])).to_numpy()

    model = T.fit_model(X[train], exc.loc[train, "needs_action"], seed=seed)
    exc["p_needs_action"] = model.predict_proba(X)[:, 1]
    exc["counterparty_self_resolve_rate"] = X["counterparty_self_resolve_rate"]
    exc["split"] = np.where(train, "train", np.where(test, "test", "embargo or lag"))

    y_te = exc.loc[test, "needs_action"].astype(int)
    p_te = exc.loc[test, "p_needs_action"]
    miss = test & exc["break_type"].isin(T.MISSING_TYPES).to_numpy()
    metrics["model"] = {
        "train_exceptions": int(train.sum()),
        "test_exceptions": int(test.sum()),
        "test_needs_action_rate": float(y_te.mean()),
        "roc_auc": float(roc_auc_score(y_te, p_te)),
        "average_precision": float(average_precision_score(y_te, p_te)),
        "missing_only_roc_auc": float(
            roc_auc_score(exc.loc[miss, "needs_action"], exc.loc[miss, "p_needs_action"])
        ),
        "missing_only_needs_action_rate": float(exc.loc[miss, "needs_action"].mean()),
    }

    rows, curves = [], {}
    test_exc = exc[test]
    for name, fn in T.STRATEGIES.items():
        curve_sum = np.zeros(MAX_K)
        n_days = 0
        for day, grp in test_exc.groupby("first_seen", sort=True):
            if grp["needs_action"].sum() == 0:
                continue
            score = fn(grp, proba=grp["p_needs_action"].to_numpy())
            m = _queue_metrics(grp, score, capacity)
            curve_sum += m.pop("curve")
            n_days += 1
            rows.append({"date": day, "strategy": name, "items": len(grp), **m})
        curves[name] = curve_sum / n_days
    daily = pd.DataFrame(rows)

    summary = {}
    for k, (name, grp) in enumerate(daily.groupby("strategy", sort=False)):
        entry = {"days": int(len(grp))}
        for col in ("precision_at_k", "value_recall_at_k", "items_to_90pct_value"):
            x = grp[col].to_numpy(dtype=float)
            lo, hi = _bootstrap_ci(x, n_boot, seed + k)
            entry[col] = {"mean": float(x.mean()), "ci95": [lo, hi]}
        summary[name] = entry
    # Paired comparisons on the same days: model minus each other ordering.
    piv = daily.pivot(index="date", columns="strategy", values="value_recall_at_k")
    paired = {}
    for j, other in enumerate(n for n in T.STRATEGIES if n != "model"):
        delta = (piv["model"] - piv[other]).to_numpy()
        lo, hi = _bootstrap_ci(delta, n_boot, seed + 100 + j)
        paired[other] = {
            "mean": float(delta.mean()),
            "ci95": [lo, hi],
            "days_model_better": int((delta > 0).sum()),
            "days_equal": int((delta == 0).sum()),
            "days_other_better": int((delta < 0).sum()),
        }
    metrics["triage"] = {
        "capacity": capacity,
        "split_date": split,
        "train_exceptions_end": trade_days[train_days - LABEL_EMBARGO_BDAYS],
        "strategies": summary,
        "model_minus_other_value_recall": paired,
    }

    # --- 3. Surveillance ---
    alerts = daily_alerts(exc, trade_days)
    incidents = ds.meta["incidents"]
    found = []
    for inc in incidents:
        hit = alerts[(alerts["date"] == inc["date"]) & (alerts["key"] == inc["target"])]
        found.append({**inc, "detected": bool(len(hit)), "robust_z": hit["robust_z"].max()})
    incident_keys = {(i["date"], i["target"]) for i in incidents}
    false_alerts = alerts[[(d, k) not in incident_keys for d, k in zip(alerts.date, alerts.key)]]
    metrics["surveillance"] = {
        "incidents": [
            {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in f.items()}
            for f in found
        ],
        "monitored_days": len(trade_days) - WINDOW_DAYS,
        "false_alerts": int(len(false_alerts)),
    }

    curves_df = pd.DataFrame(curves, index=pd.RangeIndex(1, MAX_K + 1, name="k"))
    return EvaluationResult(metrics, res, exc, daily, curves_df, alerts)


def typical_day(daily: pd.DataFrame, strategy: str = "model") -> str:
    """The held-out day whose value cleared is closest to the median for ``strategy``."""
    d = daily[daily["strategy"] == strategy].sort_values("date")
    gap = (d["value_recall_at_k"] - d["value_recall_at_k"].median()).abs()
    return str(d.loc[gap.idxmin(), "date"])


def analyst_queue(exc: pd.DataFrame, day: str, capacity: int = CAPACITY) -> pd.DataFrame:
    """The ranked work queue for one day, with a plain-English reason per item."""
    grp = exc[exc["first_seen"] == day].copy()
    grp["priority_score"] = T.score_model(grp, grp["p_needs_action"].to_numpy())
    grp = grp.sort_values("priority_score", ascending=False).head(capacity)
    grp.insert(0, "rank", range(1, len(grp) + 1))
    grp["reason"] = [template_explanation(facts_for(r)) for r in grp.to_dict("records")]
    cols = [
        "rank",
        "exception_id",
        "break_type",
        "trade_ref",
        "account",
        "counterparty",
        "currency",
        "exposure_usd",
        "p_needs_action",
        "needs_action",
        "reason",
    ]
    return grp[cols].reset_index(drop=True)
