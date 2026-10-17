"""Figures for the README. Static PNGs, Times-family font throughout."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import schema as S  # noqa: E402

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

STRATEGY_LABELS = {
    "random": "Unranked (random order)",
    "largest_exposure": "Largest exposure first",
    "desk_rules": "Desk rules",
    "counterparty_rule": "Counterparty-history rule",
    "model_probability": "Model: most likely real first",
    "model": "Model: expected exposure",
}
STRATEGY_COLORS = dict(zip(STRATEGY_LABELS, SERIES))
# Label placement for the trade-off scatter, where two points sit close together.
LABEL_OFFSETS = {
    "counterparty_rule": (-10, 12, "right"),
    "model": (10, -16, "left"),
}
TYPE_COLORS = dict(zip(S.BREAK_TYPES, SERIES))


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "TeX Gyre Termes", "Times"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.edgecolor": INK_2,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "legend.frameon": False,
        }
    )


def value_recall_curve(curves: pd.DataFrame, capacity: int, path: Path) -> None:
    """Share of real-break exposure cleared after reviewing the first k items."""
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for name in STRATEGY_LABELS:
        if name not in curves:
            continue
        y = curves[name].to_numpy() * 100
        lw = 2.6 if name == "model" else 1.8
        ax.plot(curves.index, y, color=STRATEGY_COLORS[name], lw=lw, label=STRATEGY_LABELS[name])
    ax.axvline(capacity, color=INK_2, lw=1, ls=(0, (3, 3)))
    ax.text(capacity + 0.4, 4, f"daily capacity\n(K = {capacity})", color=INK_2, fontsize=9)
    ax.set_xlim(1, curves.index.max())
    ax.set_ylim(0, 101)
    ax.set_xlabel("Items reviewed (k), in queue order")
    ax.set_ylabel("Real-break exposure cleared (%)")
    ax.set_title("How much at-risk value an analyst clears in the first k items")
    ax.legend(loc="lower right", fontsize=9)
    fig.text(
        0.01, 0.01, "Mean over 40 held-out business days; synthetic data.", color=INK_2, fontsize=8
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def precision_vs_value(summary: dict, path: Path) -> None:
    """Trade-off at the daily capacity: hit rate against value cleared, with 95% CIs."""
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for name, label in STRATEGY_LABELS.items():
        s = summary.get(name)
        if s is None:
            continue
        x, y = s["precision_at_k"], s["value_recall_at_k"]
        xm, ym = x["mean"] * 100, y["mean"] * 100
        ax.errorbar(
            xm,
            ym,
            xerr=[[xm - x["ci95"][0] * 100], [x["ci95"][1] * 100 - xm]],
            yerr=[[ym - y["ci95"][0] * 100], [y["ci95"][1] * 100 - ym]],
            fmt="o",
            ms=8,
            color=STRATEGY_COLORS[name],
            ecolor=STRATEGY_COLORS[name],
            elinewidth=1.4,
            capsize=3,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
        )
        dx, dy, ha = LABEL_OFFSETS.get(name, (8, 6, "left"))
        ax.annotate(
            label,
            (xm, ym),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            color=INK,
            fontsize=9,
        )
    ax.set_xlim(20, 105)
    ax.set_ylim(20, 85)
    ax.set_xlabel("Share of the first 10 items that are real breaks (%)")
    ax.set_ylabel("Real-break exposure cleared in the first 10 (%)")
    ax.set_title("Hit rate versus value cleared at a capacity of 10 items a day")
    fig.text(
        0.01,
        0.01,
        "Bars: 95% bootstrap intervals over 40 held-out days; synthetic data.",
        color=INK_2,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def daily_exceptions(
    exc: pd.DataFrame, trade_days: list[str], split_date: str, alerts: pd.DataFrame, path: Path
) -> None:
    """New exceptions per day by type, with the train/test split and surveillance alerts."""
    _style()
    counts = (
        exc.groupby(["first_seen", "break_type"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=trade_days, columns=list(S.BREAK_TYPES), fill_value=0)
    )
    x = np.arange(len(trade_days))
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    bottom = np.zeros(len(x))
    for b in S.BREAK_TYPES:
        vals = counts[b].to_numpy()
        ax.bar(
            x,
            vals,
            bottom=bottom,
            width=0.82,
            color=TYPE_COLORS[b],
            edgecolor=SURFACE,
            linewidth=0.6,
            label=b.replace("_", " ").title(),
        )
        bottom += vals
    split = trade_days.index(split_date)
    ax.axvline(split - 0.5, color=INK_2, lw=1, ls=(0, (3, 3)))
    ax.text(split + 0.5, bottom.max() * 0.97, "held-out days →", color=INK_2, fontsize=9)
    ax.text(
        split - 1.5, bottom.max() * 0.97, "← training days", color=INK_2, fontsize=9, ha="right"
    )
    for row in alerts.itertuples():
        if row.date in trade_days:
            k = trade_days.index(row.date)
            ax.annotate(
                f"alert: {row.key}",
                (k, bottom[k]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color=INK,
                arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.8},
            )
    ticks = x[::10]
    ax.set_xticks(ticks, [trade_days[i][5:] for i in ticks])
    ax.set_xlim(-1, len(x))
    ax.set_ylim(0, bottom.max() * 1.18)
    ax.set_xlabel("Business day (2026, month-day)")
    ax.set_ylabel("New exceptions")
    ax.set_title("New exceptions per day, by break type")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
