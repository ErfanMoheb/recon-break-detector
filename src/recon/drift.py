"""Drift experiment: does the model beat the counterparty rule once behaviour shifts?

Both the model and the counterparty-history rule lean on how often each counterparty's
missing items have cleared on their own. Here two counterparties change behaviour half-way
through the held-out days (day 100 of 80-119):

* the busiest counterparty outside the three slow posters starts posting late (rate 0.18);
* the busiest of the three slow posters gets fast (rate 0.02).

Both are picked by rule from the generated universe, so every seed drifts its own pair.
The model is trained once on pre-drift days, as in the main evaluation. Four orderings are
compared on days 80-99 (before) and 100-119 (after): the model, the counterparty rule on
all history, the same rule on the last 20 business days of history, and desk rules. A
control run with no drift on the same seeds separates the drift from ordinary
day-to-day variation.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from . import generate as G
from . import triage as T
from .evaluate import CAPACITY, TRAIN_DAYS, _bootstrap_ci, _queue_metrics, evaluate
from .io import from_generated

DRIFT_DAY = 100
SLOW_LATE_RATE = 0.18
FAST_LATE_RATE = 0.02
N_SLOW = 3  # the generator makes three counterparties slow posters
RECENT_WINDOW_BDAYS = 20
SCENARIOS = ("no_drift", "drift")
PERIODS = ("before", "after")
STRATEGIES = ("model", "counterparty_rule", "counterparty_rule_recent", "desk_rules")
RULES = ("counterparty_rule", "counterparty_rule_recent", "desk_rules")


def drift_schedule(
    cfg: G.GeneratorConfig, day_index: int = DRIFT_DAY
) -> tuple[G.LateRateChange, ...]:
    """One busy fast poster turns slow and one slow poster turns fast, chosen by rule."""
    prof = G.counterparty_profile(cfg)
    slow = prof.nlargest(N_SLOW, "late_rate")
    fast = prof.drop(slow.index)
    to_slow = fast.loc[fast["weight"].idxmax(), "counterparty"]
    to_fast = slow.loc[slow["weight"].idxmax(), "counterparty"]
    return (
        G.LateRateChange(day_index, to_slow, SLOW_LATE_RATE),
        G.LateRateChange(day_index, to_fast, FAST_LATE_RATE),
    )


def _scorers() -> dict:
    def recent_rule(g: pd.DataFrame) -> np.ndarray:
        return T.score_counterparty_rule(
            g.assign(counterparty_self_resolve_rate=g["recent_self_resolve_rate"])
        )

    return {
        "model": lambda g: T.score_model(g, g["p_needs_action"].to_numpy()),
        "counterparty_rule": T.score_counterparty_rule,
        "counterparty_rule_recent": recent_rule,
        "desk_rules": T.score_rules,
    }


def run_seed(
    seed: int, drifted: bool, train_days: int = TRAIN_DAYS, capacity: int = CAPACITY
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Daily queue results on held-out days, plus per-counterparty rate tracking."""
    cfg = replace(G.GeneratorConfig(), seed=seed)
    schedule = drift_schedule(cfg) if drifted else ()
    ds = from_generated(G.generate(replace(cfg, drift=schedule)))
    exc = evaluate(ds, train_days=train_days, capacity=capacity, seed=0, n_boot=10).exceptions
    recent = T.build_features(exc, ds.meta["calendar"], history_window=RECENT_WINDOW_BDAYS)
    exc["recent_self_resolve_rate"] = recent["counterparty_self_resolve_rate"]
    drift_date = ds.meta["trade_days"][DRIFT_DAY]

    rows = []
    test = exc[exc["split"] == "test"]
    for day, grp in test.groupby("first_seen", sort=True):
        if grp["needs_action"].sum() == 0:
            continue
        for name, fn in _scorers().items():
            m = _queue_metrics(grp, fn(grp), capacity)
            rows.append(
                {
                    "scenario": "drift" if drifted else "no_drift",
                    "date": day,
                    "period": "after" if day >= drift_date else "before",
                    "strategy": name,
                    "items": len(grp),
                    "precision_at_k": m["precision_at_k"],
                    "value_recall_at_k": m["value_recall_at_k"],
                }
            )

    # How each rule's view of the drifted counterparties moves, against what really happened.
    track = []
    miss = exc[exc["break_type"].isin(T.MISSING_TYPES) & (exc["split"] == "test")]
    for ch in schedule:
        cp = miss[miss["counterparty"] == ch.counterparty]
        for period, part in cp.groupby(np.where(cp["first_seen"] >= drift_date, "after", "before")):
            last = part[part["first_seen"] == part["first_seen"].max()]
            track.append(
                {
                    "counterparty": ch.counterparty,
                    "new_late_rate": ch.late_rate,
                    "period": period,
                    "missing_items": len(part),
                    "observed_self_clear_share": float(T.observed_self_resolved(part).mean()),
                    "all_history_rate_last_day": float(
                        last["counterparty_self_resolve_rate"].mean()
                    ),
                    "recent_rate_last_day": float(last["recent_self_resolve_rate"].mean()),
                }
            )
    info = {"seed": seed, "drift": [vars(ch) for ch in schedule], "drift_date": drift_date}
    return pd.DataFrame(rows), pd.DataFrame(track), info


def _mean_ci(x: np.ndarray, n_boot: int, seed: int) -> dict:
    lo, hi = _bootstrap_ci(x, n_boot, seed)
    return {"mean": float(x.mean()), "ci95": [lo, hi], "days": int(len(x))}


def summarise(daily: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """Value cleared in the first K per scenario, period and strategy, with paired gaps.

    ``daily`` may hold several seeds; days are then pooled across seeds (one row per seed-day).
    """
    out: dict = {}
    for scenario in SCENARIOS:
        sc = daily[daily["scenario"] == scenario]
        if sc.empty:
            continue
        key = ["seed", "date"] if "seed" in sc else ["date"]
        piv = sc.pivot_table(index=["period", *key], columns="strategy", values="value_recall_at_k")
        entry: dict = {}
        for j, period in enumerate(PERIODS):
            p = piv.loc[period]
            s = seed + 100 * j  # same draws for both scenarios: identical days, identical CIs
            entry[period] = {
                "value_recall_at_k": {
                    name: _mean_ci(p[name].to_numpy(), n_boot, s + k)
                    for k, name in enumerate(STRATEGIES)
                },
                "model_minus": {
                    rule: {
                        **_mean_ci((p["model"] - p[rule]).to_numpy(), n_boot, s + 10 + k),
                        "days_model_better": int((p["model"] > p[rule]).sum()),
                        "days_rule_better": int((p["model"] < p[rule]).sum()),
                    }
                    for k, rule in enumerate(RULES)
                },
            }
        # After minus before, per strategy: unpaired, the periods are different days.
        rng = np.random.default_rng(seed + 500)
        change = {}
        for name in STRATEGIES:
            b, a = piv.loc["before", name].to_numpy(), piv.loc["after", name].to_numpy()
            boot_a = rng.choice(a, size=(n_boot, len(a))).mean(axis=1)
            boot_b = rng.choice(b, size=(n_boot, len(b))).mean(axis=1)
            lo, hi = np.percentile(boot_a - boot_b, [2.5, 97.5])
            change[name] = {"mean": float(a.mean() - b.mean()), "ci95": [float(lo), float(hi)]}
        entry["after_minus_before"] = change
        out[scenario] = entry
    return out


def run(
    seeds, train_days: int = TRAIN_DAYS, capacity: int = CAPACITY, n_boot: int = 2000
) -> tuple[dict, dict[int, pd.DataFrame]]:
    """Run drift and control on every seed. Returns metrics and one daily table per seed."""
    per_seed, tables, tracking, chosen = {}, {}, [], []
    for seed in seeds:
        parts = []
        for drifted in (False, True):
            d, track, info = run_seed(seed, drifted, train_days, capacity)
            parts.append(d)
            if drifted:
                tracking.append(track.assign(seed=seed))
                chosen.append(info)
        tables[seed] = pd.concat(parts, ignore_index=True)
        per_seed[str(seed)] = summarise(tables[seed], n_boot=n_boot, seed=seed)
    pooled = pd.concat([t.assign(seed=s) for s, t in tables.items()], ignore_index=True)
    metrics = {
        "setup": {
            "drift_day_index": DRIFT_DAY,
            "train_days": train_days,
            "capacity": capacity,
            "slow_late_rate": SLOW_LATE_RATE,
            "fast_late_rate": FAST_LATE_RATE,
            "recent_window_bdays": RECENT_WINDOW_BDAYS,
            "seeds": list(seeds),
            "drift_by_seed": chosen,
        },
        "pooled": summarise(pooled, n_boot=n_boot),
        "by_seed": per_seed,
        "drifted_counterparties": pd.concat(tracking, ignore_index=True)
        .round(4)
        .to_dict("records"),
    }
    return metrics, tables
