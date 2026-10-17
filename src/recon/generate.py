"""Synthetic custody settlement data with labelled reconciliation breaks.

Two ledgers are produced for the same trades:

* the internal book: what the firm booked, one row per trade, dated by booking day;
* the custodian statement: what the custodian reports, dated by statement day.

Real breaks (missing movements, partial deliveries, wrong amounts, duplicates) are
injected next to benign noise that a naive matcher would also flag: reference
formatting differences, blank references, rounding inside tolerance, and late postings
that resolve on their own a day or two later. Late-posting propensity depends on the
counterparty, the market and month end, so a triage model has something real to learn.
An optional drift schedule changes a counterparty's late-posting rate part-way through.

Ground truth sits in ``truth_*`` columns and in ``meta.json``. The matcher drops those
columns before it reads anything (see ``recon.io.observable``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import schema as S
from .isin import make_isin


@dataclass(frozen=True)
class Incident:
    """A one-day operational incident affecting one counterparty or one market."""

    day_index: int
    kind: str  # S.FEED_DELAY (target = counterparty) or S.PRICE_FEED (target = currency)
    target: str
    share: float = 1.0


@dataclass(frozen=True)
class LateRateChange:
    """From ``day_index`` on, ``counterparty`` posts late at ``late_rate`` instead."""

    day_index: int
    counterparty: str
    late_rate: float


DEFAULT_INCIDENTS = (
    Incident(40, S.FEED_DELAY, "CP03"),
    Incident(95, S.PRICE_FEED, "GBP", 0.6),
    Incident(108, S.FEED_DELAY, "CP08"),
)


@dataclass(frozen=True)
class GeneratorConfig:
    n_days: int = 120
    start: str = "2026-01-05"
    trades_per_day: int = 500
    n_accounts: int = 8
    n_securities: int = 300
    n_counterparties: int = 15
    seed: int = 7
    p_missing_external: float = 0.005
    p_missing_internal: float = 0.004
    p_qty_mismatch: float = 0.004
    p_amount_mismatch: float = 0.004
    p_duplicate: float = 0.003
    p_rounding_noise: float = 0.06
    p_blank_reference: float = 0.04
    incidents: tuple[Incident, ...] = field(default=DEFAULT_INCIDENTS)
    drift: tuple[LateRateChange, ...] = ()


_CCY_WEIGHTS = {"USD": 0.45, "EUR": 0.20, "GBP": 0.12, "JPY": 0.13, "HKD": 0.10}
# Asian markets settle overnight relative to a New York operations team.
_CCY_LATE_MULT = {"USD": 1.0, "EUR": 1.0, "GBP": 1.0, "JPY": 1.8, "HKD": 1.8}
_LAG_DAYS = 3  # statement days generated past the last trade day so late postings land


def format_reference(ref: str, rng: np.random.Generator, p_blank: float) -> str:
    """Render a trade reference the way a custodian feed might."""
    u = rng.random()
    if u < p_blank:
        return ""
    digits = ref[1:]
    style = rng.choice(6, p=[0.55, 0.10, 0.10, 0.10, 0.10, 0.05])
    return [
        ref,
        ref.lower(),
        f"REF-{ref}",
        digits.lstrip("0"),
        f"{ref}/1",
        f"CUST {digits}",
    ][style]


@dataclass
class _Universe:
    accounts: pd.DataFrame
    securities: pd.DataFrame
    counterparties: pd.DataFrame


def _build_universe(cfg: GeneratorConfig, rng: np.random.Generator) -> _Universe:
    n_acc = cfg.n_accounts
    custody = rng.choice(np.arange(100000, 999999), size=n_acc, replace=False)
    manual = np.full(n_acc, 0.004)
    manual[rng.choice(n_acc, size=2, replace=False)] = 0.03  # desks that book by hand
    acct_err = np.ones(n_acc)
    acct_err[rng.integers(n_acc)] = 2.0
    accounts = pd.DataFrame(
        {
            "account": [f"ACC{i + 1:02d}" for i in range(n_acc)],
            "custody_account": [f"CUS{c}" for c in custody],
            "weight": rng.dirichlet(np.full(n_acc, 3.0)),
            "manual_rate": manual,
            "error_mult": acct_err,
        }
    )

    ccys = list(_CCY_WEIGHTS)
    rows, seen = [], set()
    while len(rows) < cfg.n_securities:
        ccy = rng.choice(ccys, p=list(_CCY_WEIGHTS.values()))
        mkt = S.MARKETS[ccy]
        country = rng.choice(mkt.countries)
        isin = make_isin(country, "".join(rng.choice(list("0123456789"), size=9)))
        if isin in seen:
            continue
        seen.add(isin)
        rows.append(
            {
                "isin": isin,
                "currency": ccy,
                "base_price": mkt.typical_price * float(rng.lognormal(0.0, 0.6)),
                "illiquid": bool(rng.random() < 0.15),
            }
        )
    securities = pd.DataFrame(rows)
    securities["weight"] = rng.dirichlet(np.full(cfg.n_securities, 0.7))

    n_cp = cfg.n_counterparties
    late = rng.uniform(0.01, 0.04, size=n_cp)
    late[rng.choice(n_cp, size=3, replace=False)] = rng.uniform(0.12, 0.20, size=3)
    err = rng.lognormal(0.0, 0.3, size=n_cp)
    err[rng.choice(n_cp, size=2, replace=False)] *= 2.5
    counterparties = pd.DataFrame(
        {
            "counterparty": [f"CP{i + 1:02d}" for i in range(n_cp)],
            "weight": rng.dirichlet(np.full(n_cp, 2.0)),
            "late_rate": late,
            "error_mult": err,
        }
    )
    return _Universe(accounts, securities, counterparties)


def counterparty_profile(cfg: GeneratorConfig | None = None) -> pd.DataFrame:
    """Each counterparty's volume weight and base late-posting rate for this config.

    The universe is drawn first from the seeded generator, so this rebuilds exactly the
    counterparties ``generate(cfg)`` uses without generating any trades.
    """
    cfg = cfg or GeneratorConfig()
    cps = _build_universe(cfg, np.random.default_rng(cfg.seed)).counterparties
    return cps[["counterparty", "weight", "late_rate"]].copy()


def _late_rates_by_day(cps: pd.DataFrame, cfg: GeneratorConfig) -> np.ndarray:
    """Counterparty late-posting rates per trade day, shape (n_days, n_counterparties)."""
    rates = np.tile(cps["late_rate"].to_numpy(), (cfg.n_days, 1))
    pos = {cp: k for k, cp in enumerate(cps["counterparty"])}
    for ch in sorted(cfg.drift, key=lambda c: c.day_index):
        if ch.counterparty not in pos:
            raise ValueError(f"unknown counterparty in drift: {ch.counterparty}")
        rates[max(ch.day_index, 0) :, pos[ch.counterparty]] = ch.late_rate
    return rates


def _month_end_flags(days: pd.DatetimeIndex) -> np.ndarray:
    """True on the last two business days of each month."""
    month = days.to_period("M")
    pos_from_end = pd.Series(range(len(days)))[::-1].groupby(month[::-1]).cumcount()[::-1]
    return pos_from_end.to_numpy() < 2


def _round_amount(x: np.ndarray, decimals: np.ndarray) -> np.ndarray:
    return np.array([round(float(v), int(d)) for v, d in zip(x, decimals)])


def generate(cfg: GeneratorConfig | None = None) -> dict[str, pd.DataFrame | dict]:
    """Return the internal book, custodian statement, reference data and metadata."""
    cfg = cfg or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    uni = _build_universe(cfg, rng)
    acc, sec, cps = uni.accounts, uni.securities, uni.counterparties

    all_days = pd.bdate_range(cfg.start, periods=cfg.n_days + _LAG_DAYS)
    month_end = _month_end_flags(all_days)
    incidents = [inc for inc in cfg.incidents if 0 <= inc.day_index < cfg.n_days]
    incidents_by_day: dict[int, list[Incident]] = {}
    for inc in incidents:
        incidents_by_day.setdefault(inc.day_index, []).append(inc)

    sec_ccy = sec["currency"].to_numpy()
    sec_lot = np.array([S.MARKETS[c].lot for c in sec_ccy])
    sec_dec = np.array([S.MARKETS[c].decimals for c in sec_ccy])
    late_mult = np.array([_CCY_LATE_MULT[c] for c in sec_ccy])
    cp_late_by_day = _late_rates_by_day(cps, cfg)

    internal, external = [], []
    trade_no, ext_no = 0, 0

    def next_ext_id() -> str:
        nonlocal ext_no
        ext_no += 1
        return f"EXT{ext_no:08d}"

    for d in range(cfg.n_days):
        day = all_days[d]
        n = int(rng.poisson(cfg.trades_per_day))
        a = rng.choice(len(acc), size=n, p=acc["weight"])
        s = rng.choice(len(sec), size=n, p=sec["weight"])
        c = rng.choice(len(cps), size=n, p=cps["weight"])
        side = rng.choice(["BUY", "SELL"], size=n)
        qty = rng.integers(1, 50, size=n) * sec_lot[s]
        price = sec["base_price"].to_numpy()[s] * (1.0 + rng.normal(0.0, 0.01, size=n))
        price = np.where(sec_dec[s] == 0, np.round(price), np.round(price, 4))
        amount = _round_amount(qty * price, sec_dec[s])

        cp_err = cps["error_mult"].to_numpy()[c]
        p = np.column_stack(
            [
                cp_late_by_day[d, c] * late_mult[s] * (1.5 if month_end[d] else 1.0),
                acc["manual_rate"].to_numpy()[a],
                cfg.p_missing_external * cp_err * acc["error_mult"].to_numpy()[a],
                cfg.p_qty_mismatch * cp_err,
                cfg.p_amount_mismatch * np.where(sec["illiquid"].to_numpy()[s], 3.0, 1.0),
                cfg.p_duplicate * cp_err,
            ]
        )
        names = np.array(
            [
                S.TIMING_EXTERNAL,
                S.TIMING_INTERNAL,
                S.MISSING_EXTERNAL,
                S.QTY_MISMATCH,
                S.AMOUNT_MISMATCH,
                S.DUPLICATE_EXTERNAL,
                S.NONE,
            ]
        )
        cum = np.cumsum(np.column_stack([p, 1.0 - p.sum(axis=1)]), axis=1)
        event = names[(rng.random(n)[:, None] > cum).sum(axis=1)]

        for inc in incidents_by_day.get(d, []):
            if inc.kind == S.FEED_DELAY:
                hit = cps["counterparty"].to_numpy()[c] == inc.target
            else:
                hit = sec_ccy[s] == inc.target
            hit &= (event == S.NONE) & (rng.random(n) < inc.share)
            event[hit] = inc.kind

        for i in range(n):
            trade_no += 1
            ref = f"T{trade_no:08d}"
            ev = str(event[i])
            si, ccy, dec, lot = s[i], sec_ccy[s[i]], int(sec_dec[s[i]]), int(sec_lot[s[i]])
            booked = all_days[d + 1] if ev == S.TIMING_INTERNAL else day
            row = {
                "internal_id": f"INT{trade_no:08d}",
                "trade_ref": ref,
                "account": acc["account"].iat[a[i]],
                "isin": sec["isin"].iat[si],
                "currency": ccy,
                "side": side[i],
                "quantity": int(qty[i]),
                "price": float(price[i]),
                "amount": float(amount[i]),
                "counterparty": cps["counterparty"].iat[c[i]],
                "booking_date": booked.date().isoformat(),
                "truth_event": ev,
                "truth_break_type": S.EXPECTED_BREAK.get(ev, "")
                if ev != S.MISSING_INTERNAL
                else "",
                "truth_needs_action": ev
                in {S.MISSING_EXTERNAL, S.QTY_MISMATCH, S.AMOUNT_MISMATCH, S.PRICE_FEED},
            }
            internal.append(row)
            if ev == S.MISSING_EXTERNAL:
                continue

            stmt_day = day
            if ev == S.TIMING_EXTERNAL:
                stmt_day = all_days[d + (1 if rng.random() < 0.8 else 2)]
            elif ev == S.FEED_DELAY:
                stmt_day = all_days[d + 1]

            ext_qty, ext_amt = int(qty[i]), float(amount[i])
            fx = S.FX_TO_USD[ccy]
            if ev == S.QTY_MISMATCH:
                if ext_qty >= 2 * lot:
                    partial = int(round(ext_qty * rng.uniform(0.4, 0.9) / lot)) * lot
                    partial = min(max(partial, lot), ext_qty - lot)
                else:
                    partial = ext_qty + lot  # over-delivery on a single lot
                ext_amt = round(ext_amt * partial / ext_qty, dec)
                ext_qty = partial
            elif ev in (S.AMOUNT_MISMATCH, S.PRICE_FEED):
                lo, hi = (0.004, 0.04) if ev == S.AMOUNT_MISMATCH else (0.01, 0.03)
                diff = ext_amt * rng.uniform(lo, hi) * rng.choice([-1.0, 1.0])
                if abs(diff) * fx < 5.0:
                    diff = np.sign(diff) * 5.0 / fx
                ext_amt = round(ext_amt + diff, dec)
            elif rng.random() < cfg.p_rounding_noise:
                bump = rng.integers(1, 31) if dec == 0 else rng.uniform(0.01, 0.40)
                ext_amt = round(ext_amt + float(bump) * rng.choice([-1.0, 1.0]), dec)

            ext_row = {
                "external_id": next_ext_id(),
                "statement_ref": format_reference(ref, rng, cfg.p_blank_reference),
                "custody_account": acc["custody_account"].iat[a[i]],
                "isin": sec["isin"].iat[si],
                "currency": ccy,
                "side": side[i],
                "quantity": ext_qty,
                "amount": ext_amt,
                "counterparty": cps["counterparty"].iat[c[i]],
                "statement_date": stmt_day.date().isoformat(),
                "truth_trade_ref": ref,
                "truth_event": ev,
                "truth_is_duplicate": False,
                "truth_break_type": "",
                "truth_needs_action": False,
            }
            external.append(ext_row)
            if ev == S.DUPLICATE_EXTERNAL:
                dup_day = day if rng.random() < 0.7 else all_days[d + 1]
                external.append(
                    {
                        **ext_row,
                        "external_id": next_ext_id(),
                        "statement_ref": format_reference(ref, rng, cfg.p_blank_reference),
                        "statement_date": dup_day.date().isoformat(),
                        "truth_is_duplicate": True,
                        "truth_break_type": S.B_DUPLICATE,
                        "truth_needs_action": True,
                    }
                )

        # Custodian movements the firm never booked.
        for _ in range(int(rng.poisson(n * cfg.p_missing_internal))):
            trade_no += 1
            ref = f"T{trade_no:08d}"
            ai = int(rng.choice(len(acc), p=acc["weight"]))
            si = int(rng.choice(len(sec), p=sec["weight"]))
            ccy, dec, lot = sec_ccy[si], int(sec_dec[si]), int(sec_lot[si])
            q = int(rng.integers(1, 50)) * lot
            px = sec["base_price"].iat[si] * (1.0 + rng.normal(0.0, 0.01))
            px = round(px) if dec == 0 else round(px, 4)
            external.append(
                {
                    "external_id": next_ext_id(),
                    "statement_ref": format_reference(ref, rng, cfg.p_blank_reference),
                    "custody_account": acc["custody_account"].iat[ai],
                    "isin": sec["isin"].iat[si],
                    "currency": ccy,
                    "side": str(rng.choice(["BUY", "SELL"])),
                    "quantity": q,
                    "amount": round(q * px, dec),
                    "counterparty": cps["counterparty"].iat[int(rng.choice(len(cps)))],
                    "statement_date": day.date().isoformat(),
                    "truth_trade_ref": ref,
                    "truth_event": S.MISSING_INTERNAL,
                    "truth_is_duplicate": False,
                    "truth_break_type": S.B_MISSING_INTERNAL,
                    "truth_needs_action": True,
                }
            )

    meta = {
        "config": {k: v for k, v in asdict(cfg).items() if k not in ("incidents", "drift")},
        "incidents": [
            {**asdict(inc), "date": all_days[inc.day_index].date().isoformat()} for inc in incidents
        ],
        "trade_days": [d.date().isoformat() for d in all_days[: cfg.n_days]],
        "calendar": [d.date().isoformat() for d in all_days],
    }
    if cfg.drift:
        meta["drift"] = [
            {**asdict(ch), "date": all_days[min(ch.day_index, cfg.n_days - 1)].date().isoformat()}
            for ch in cfg.drift
        ]
    return {
        "internal_book": pd.DataFrame(internal),
        "custodian_statement": pd.DataFrame(external),
        "accounts": acc[["account", "custody_account"]].copy(),
        "securities": sec[["isin", "currency", "illiquid"]].copy(),
        "fx_rates": pd.DataFrame(
            {"currency": list(S.FX_TO_USD), "usd_per_unit": list(S.FX_TO_USD.values())}
        ),
        "meta": meta,
    }


def write(data: dict[str, pd.DataFrame | dict], out_dir: str | Path) -> Path:
    """Write generated tables as CSV plus ``meta.json``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, obj in data.items():
        if name == "meta":
            (out / "meta.json").write_text(json.dumps(obj, indent=2))
        else:
            obj.to_csv(out / f"{name}.csv", index=False)
    return out
