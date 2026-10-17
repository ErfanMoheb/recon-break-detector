"""Daily two-ledger matching with break classification.

The engine replays business days in order. Each day it pools the new internal bookings
and custodian statement lines with every item still open from earlier days, then:

1. Reference pass: a statement line whose normalised reference equals a booked trade's
   reference (same account, ISIN, side and currency) is paired with it. The pair is classified as
   ``MATCHED``, ``QTY_MISMATCH`` or ``AMOUNT_MISMATCH``. A reference that was already
   paired earlier, with the same account, ISIN, quantity and amount, is a
   ``DUPLICATE_EXTERNAL``.
2. Attribute pass: statement lines with no reference are paired on account, ISIN, side,
   quantity, amount within tolerance and a booking date at most two business days away.
3. Whatever is still unpaired and new today becomes an exception: ``MISSING_EXTERNAL`` for
   a booked trade, ``MISSING_INTERNAL`` for a statement line. It stays open, ageing, and is
   marked resolved if a later day pairs it.

Ground-truth columns are dropped on entry, so the engine cannot see them.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import schema as S
from .io import observable

_DIGITS = re.compile(r"\d+")
MAX_DATE_GAP_BDAYS = 2
DUPLICATE_WINDOW_BDAYS = 2


def normalise_reference(ref: object) -> int | None:
    """Reduce a reference like ``REF-T00001234`` or ``1234/1`` to the trade number."""
    if ref is None or (isinstance(ref, float) and np.isnan(ref)):
        return None
    m = _DIGITS.search(str(ref))
    return int(m.group()) if m else None


@dataclass
class MatchResult:
    pairs: pd.DataFrame
    exceptions: pd.DataFrame


def _bday_gap(a: str, b: str) -> int:
    lo, hi = sorted((a, b))
    return int(np.busday_count(lo, hi))


def reconcile(
    internal: pd.DataFrame,
    external: pd.DataFrame,
    account_map: dict[str, str],
    fx: dict[str, float],
) -> MatchResult:
    """Run the daily matching loop over every booking and statement date."""
    ints = observable(internal).to_dict("records")
    exts = observable(external).to_dict("records")
    for r in ints:
        r["_ref"] = normalise_reference(r["trade_ref"])
    for r in exts:
        r["_ref"] = normalise_reference(r["statement_ref"])
        r["account"] = account_map.get(r["custody_account"], "")

    ints_by_day: dict[str, list[dict]] = defaultdict(list)
    exts_by_day: dict[str, list[dict]] = defaultdict(list)
    for r in ints:
        ints_by_day[r["booking_date"]].append(r)
    for r in exts:
        exts_by_day[r["statement_date"]].append(r)

    open_int: dict[str, dict] = {}
    open_ext: dict[str, dict] = {}
    matched_by_ref: dict[int, list[tuple[dict, str]]] = defaultdict(list)
    matched_by_attr: dict[tuple, list[tuple[dict, str]]] = defaultdict(list)
    pairs: list[dict] = []
    exceptions: list[dict] = []
    open_exc: dict[str, dict] = {}  # record id -> its open exception

    def usd(x: float, ccy: str) -> float:
        return float(x) * fx[ccy]

    def same_amount(a: float, b: float, ccy: str) -> bool:
        return abs(usd(a - b, ccy)) <= S.tolerance_usd(usd(a, ccy))

    def resolve(record_id: str, day: str) -> None:
        exc = open_exc.pop(record_id, None)
        if exc is not None:
            exc["resolved_date"] = day

    def raise_exception(day: str, kind: str, i: dict | None, e: dict | None) -> dict:
        base = i if i is not None else e
        ccy = base["currency"]
        q_i = i["quantity"] if i is not None else None
        q_e = e["quantity"] if e is not None else None
        a_i = i["amount"] if i is not None else None
        a_e = e["amount"] if e is not None else None
        if kind == S.B_QTY:
            exposure = abs(q_i - q_e) * i["price"] * fx[ccy]
        elif kind == S.B_AMOUNT:
            exposure = abs(usd(a_i - a_e, ccy))
        else:
            exposure = usd(base["amount"], ccy)
        exc = {
            "exception_id": f"X{len(exceptions) + 1:06d}",
            "first_seen": day,
            "break_type": kind,
            "internal_id": i["internal_id"] if i is not None else "",
            "external_id": e["external_id"] if e is not None else "",
            "trade_ref": i["trade_ref"] if i is not None else str(e["statement_ref"]),
            "account": base["account"],
            "isin": base["isin"],
            "currency": ccy,
            "side": base["side"],
            "counterparty": base["counterparty"],
            "quantity_internal": q_i,
            "quantity_external": q_e,
            "amount_internal": a_i,
            "amount_external": a_e,
            "exposure_usd": round(exposure, 2),
            "ref_present": e is None or e["_ref"] is not None,
            "resolved_date": "",
        }
        exceptions.append(exc)
        return exc

    def pair(i: dict, e: dict, day: str, method: str) -> None:
        if i["quantity"] != e["quantity"]:
            status = S.B_QTY
        elif not same_amount(i["amount"], e["amount"], i["currency"]):
            status = S.B_AMOUNT
        else:
            status = "MATCHED"
        pairs.append(
            {
                "internal_id": i["internal_id"],
                "external_id": e["external_id"],
                "match_date": day,
                "method": method,
                "status": status,
                "amount_diff_usd": round(usd(e["amount"] - i["amount"], i["currency"]), 2),
            }
        )
        resolve(i["internal_id"], day)
        resolve(e["external_id"], day)
        open_int.pop(i["internal_id"], None)
        open_ext.pop(e["external_id"], None)
        if status != "MATCHED":
            raise_exception(day, status, i, e)
        if e["_ref"] is not None:
            matched_by_ref[e["_ref"]].append((e, day))
        matched_by_attr[(e["account"], e["isin"], e["side"], e["quantity"])].append((e, day))

    def is_copy(e: dict, earlier: dict, earlier_day: str, day: str) -> bool:
        both_refs = e["_ref"] is not None and earlier["_ref"] is not None
        return (
            (not both_refs or e["_ref"] == earlier["_ref"])
            and earlier["account"] == e["account"]
            and earlier["isin"] == e["isin"]
            and earlier["quantity"] == e["quantity"]
            and same_amount(earlier["amount"], e["amount"], e["currency"])
            and _bday_gap(earlier_day, day) <= DUPLICATE_WINDOW_BDAYS
        )

    def flag_duplicate(e: dict, day: str) -> None:
        open_ext.pop(e["external_id"], None)
        resolve(e["external_id"], day)  # a line first left open, then recognised as a copy
        raise_exception(day, S.B_DUPLICATE, None, e)

    for day in sorted(set(ints_by_day) | set(exts_by_day)):
        for r in ints_by_day.get(day, []):
            open_int[r["internal_id"]] = r
            r["_first_seen"] = day
        for r in exts_by_day.get(day, []):
            open_ext[r["external_id"]] = r
            r["_first_seen"] = day

        int_by_ref: dict[int, list[dict]] = defaultdict(list)
        for r in open_int.values():
            int_by_ref[r["_ref"]].append(r)

        # 1. Reference pass.
        for e in sorted(open_ext.values(), key=lambda r: r["external_id"]):
            if e["_ref"] is None:
                continue
            cands = [
                i
                for i in int_by_ref.get(e["_ref"], [])
                if i["internal_id"] in open_int
                and i["account"] == e["account"]
                and i["isin"] == e["isin"]
                and i["side"] == e["side"]
                and i["currency"] == e["currency"]
            ]
            key = (e["account"], e["isin"], e["side"], e["quantity"])
            earlier = matched_by_ref.get(e["_ref"], []) + matched_by_attr.get(key, [])
            if cands:
                pair(min(cands, key=lambda r: r["internal_id"]), e, day, "reference")
            elif any(is_copy(e, x, xd, day) for x, xd in earlier):
                flag_duplicate(e, day)

        # 2. Attribute pass for lines that carry no reference.
        by_attr: dict[tuple, list[dict]] = defaultdict(list)
        for i in open_int.values():
            by_attr[(i["account"], i["isin"], i["side"], i["quantity"])].append(i)
        for e in sorted(open_ext.values(), key=lambda r: r["external_id"]):
            if e["_ref"] is not None:
                continue
            key = (e["account"], e["isin"], e["side"], e["quantity"])
            cands = [
                i
                for i in by_attr.get(key, [])
                if i["internal_id"] in open_int
                and same_amount(i["amount"], e["amount"], e["currency"])
                and _bday_gap(i["booking_date"], day) <= MAX_DATE_GAP_BDAYS
            ]
            if cands:
                best = min(
                    cands,
                    key=lambda r: (
                        abs(r["amount"] - e["amount"]),
                        _bday_gap(r["booking_date"], day),
                        r["internal_id"],
                    ),
                )
                pair(best, e, day, "attributes")
            elif any(is_copy(e, x, xd, day) for x, xd in matched_by_attr.get(key, [])):
                flag_duplicate(e, day)

        # 3. New, still-unpaired items become exceptions.
        for i in open_int.values():
            if i["_first_seen"] == day and i["internal_id"] not in open_exc:
                open_exc[i["internal_id"]] = raise_exception(day, S.B_MISSING_EXTERNAL, i, None)
        for e in open_ext.values():
            if e["_first_seen"] == day and e["external_id"] not in open_exc:
                open_exc[e["external_id"]] = raise_exception(day, S.B_MISSING_INTERNAL, None, e)

    return MatchResult(pairs=pd.DataFrame(pairs), exceptions=pd.DataFrame(exceptions))
