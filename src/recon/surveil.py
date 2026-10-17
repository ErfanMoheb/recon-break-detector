"""Day-level surveillance: spot bursts that point at an operational incident.

A single late feed or a stale price file produces dozens of exceptions sharing one
counterparty or one market on one day. Ranking them one by one hides the cause, so the
daily count per source is compared with its own recent history using a robust z-score
(median and MAD of the trailing window, today excluded).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema as S

WINDOW_DAYS = 20
Z_THRESHOLD = 4.0
MIN_COUNT = 6

# (dimension, break types counted) pairs that are monitored.
MONITORS = (
    ("counterparty", (S.B_MISSING_EXTERNAL, S.B_MISSING_INTERNAL)),
    ("currency", (S.B_AMOUNT, S.B_QTY)),
)


def robust_z(x: float, history: np.ndarray) -> float:
    med = float(np.median(history))
    mad = float(np.median(np.abs(history - med)))
    return (x - med) / (1.4826 * mad + 1.0)


def daily_alerts(exc: pd.DataFrame, days: list[str]) -> pd.DataFrame:
    """One row per (day, source) whose exception count is anomalously high."""
    alerts = []
    for dim, types in MONITORS:
        sub = exc[exc["break_type"].isin(types)]
        counts = (
            sub.groupby(["first_seen", dim])
            .size()
            .unstack(fill_value=0)
            .reindex(days, fill_value=0)
        )
        values = counts.to_numpy()
        for t in range(WINDOW_DAYS, len(days)):
            hist = values[t - WINDOW_DAYS : t]
            for j, key in enumerate(counts.columns):
                x = values[t, j]
                if x < MIN_COUNT:
                    continue
                z = robust_z(x, hist[:, j])
                if z >= Z_THRESHOLD:
                    alerts.append(
                        {
                            "date": days[t],
                            "dimension": dim,
                            "key": key,
                            "count": int(x),
                            "baseline_median": float(np.median(hist[:, j])),
                            "robust_z": round(z, 1),
                        }
                    )
    return pd.DataFrame(
        alerts, columns=["date", "dimension", "key", "count", "baseline_median", "robust_z"]
    )
