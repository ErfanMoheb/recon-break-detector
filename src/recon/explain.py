"""Plain-English reasons for each break, with a guard on anything a language model writes.

``template_explanation`` is deterministic and is what the pipeline uses by default.
``LLMExplainer`` lets a language model rephrase the same facts for an analyst and runs
``check_faithfulness`` on the draft before it is shown. A draft that fails is discarded and
the template is used instead.

The check is lexical. It verifies that every identifier, date and figure was copied from the
facts, that side, currency and break-type words agree with the facts, and that the text does
not advise dismissing the item. It cannot verify which role a copied figure is given (for
example, booked and custodian amounts written the wrong way round).
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass

from . import schema as S

_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_IDENT = re.compile(r"\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{4,}\b", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?%?")
_CURRENCY = re.compile(r"\b[A-Z]{3}\b")
_SIDE_WORDS = {
    "BUY": re.compile(r"\b(buy|buys|bought|purchase[sd]?)\b", re.IGNORECASE),
    "SELL": re.compile(r"\b(sell|sells|sold|sale)\b", re.IGNORECASE),
}
# Words that only fit some break types; using them for another type misstates the break.
_TYPE_WORDS = {
    S.B_DUPLICATE: re.compile(r"duplicat|second time|reported twice", re.IGNORECASE),
    S.B_QTY: re.compile(r"partial|quantit|short deliver|over-deliver", re.IGNORECASE),
    S.B_AMOUNT: re.compile(r"\b(price|fee|commission)s?\b", re.IGNORECASE),
    S.B_MISSING_EXTERNAL: re.compile(r"no custodian movement|not settled", re.IGNORECASE),
    S.B_MISSING_INTERNAL: re.compile(r"no matching booking|not booked|never booked", re.IGNORECASE),
}
_DISMISSIVE = re.compile(
    r"no action|nothing to do|ignore|safe to (close|clear)|can be (closed|ignored|cleared)"
    r"|will (clear|resolve|settle)|self-resolv|clear(s)? on its own",
    re.IGNORECASE,
)


def _missing(x: object) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _first(*values: object) -> object:
    return next((v for v in values if not _missing(v)), None)


def _fmt_qty(x: float | None) -> str:
    return "n/a" if _missing(x) else f"{int(x):,}"


def _fmt_amt(x: float | None, ccy: str) -> str:
    if _missing(x):
        return "n/a"
    return f"{ccy} {x:,.{S.MARKETS[ccy].decimals}f}"


def facts_for(row: dict) -> dict:
    """The fields an explanation may draw on, with derived values made explicit."""
    keys = (
        "exception_id",
        "break_type",
        "trade_ref",
        "account",
        "isin",
        "currency",
        "side",
        "counterparty",
        "first_seen",
        "quantity_internal",
        "quantity_external",
        "amount_internal",
        "amount_external",
        "exposure_usd",
        "p_needs_action",
        "counterparty_self_resolve_rate",
    )
    facts = {k: row.get(k) for k in keys if k in row}
    a_i, a_e = row.get("amount_internal"), row.get("amount_external")
    if row.get("break_type") == S.B_AMOUNT and a_i and a_e:
        facts["amount_difference"] = round(a_e - a_i, 2)
    q_i, q_e = row.get("quantity_internal"), row.get("quantity_external")
    if row.get("break_type") == S.B_QTY and q_i and q_e:
        facts["quantity_difference"] = q_e - q_i
    return facts


def template_explanation(f: dict) -> str:
    ccy = f["currency"]
    trade = (
        f"{f['side']} {_fmt_qty(_first(f.get('quantity_internal'), f.get('quantity_external')))}"
    )
    trade += f" {f['isin']} in {f['account']}"
    p = f.get("p_needs_action")
    rate = f.get("counterparty_self_resolve_rate")
    tail = ""
    if p is not None:
        tail = f" Estimated chance this needs action: {p:.0%}."
    kind = f["break_type"]
    if kind == S.B_MISSING_EXTERNAL:
        hist = (
            f" {f['counterparty']} has cleared {rate:.0%} of similar items on its own."
            if rate is not None
            else ""
        )
        return (
            f"Booked {trade} ({_fmt_amt(f.get('amount_internal'), ccy)}, ref {f['trade_ref']}) "
            f"has no custodian movement as of {f['first_seen']}.{hist}{tail}"
        )
    if kind == S.B_MISSING_INTERNAL:
        return (
            f"Custodian reports {trade} ({_fmt_amt(f.get('amount_external'), ccy)}, "
            f"counterparty {f['counterparty']}) on {f['first_seen']} with no matching "
            f"booking.{tail}"
        )
    if kind == S.B_QTY:
        return (
            f"Custodian settled {_fmt_qty(f.get('quantity_external'))} against "
            f"{_fmt_qty(f.get('quantity_internal'))} booked for ref {f['trade_ref']} "
            f"({f['isin']}, {f['account']}); exposure USD {f['exposure_usd']:,.2f}.{tail}"
        )
    if kind == S.B_AMOUNT:
        return (
            f"Amounts disagree for ref {f['trade_ref']}: booked "
            f"{_fmt_amt(f.get('amount_internal'), ccy)}, custodian "
            f"{_fmt_amt(f.get('amount_external'), ccy)} "
            f"(difference USD {f['exposure_usd']:,.2f}).{tail}"
        )
    if kind == S.B_DUPLICATE:
        return (
            f"Custodian reported {trade} ({_fmt_amt(f.get('amount_external'), ccy)}) a second "
            f"time on {f['first_seen']}; the first report is already matched.{tail}"
        )
    raise ValueError(f"unknown break type {kind!r}")


# --- Faithfulness check for model-written text ---


def _allowed_numbers(facts: dict) -> list[float]:
    nums: list[float] = []
    for v in facts.values():
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)) and not math.isnan(float(v)):
            x = float(v)
            nums += [x, abs(x)]
            if 0.0 <= x <= 1.0:  # probabilities and rates may be written as percentages
                nums += [x * 100.0, abs(x) * 100.0]
    return nums


def check_faithfulness(text: str, facts: dict) -> list[str]:
    """Return the problems found; an empty list means the text passed."""
    problems = []
    allowed_strings = {str(v).upper() for v in facts.values() if isinstance(v, str)}
    remaining = text
    for d in _DATE.findall(text):
        if d not in allowed_strings:
            problems.append(f"date {d} is not in the facts")
        remaining = remaining.replace(d, " ")
    for ident in _IDENT.findall(remaining):
        if ident.upper() not in allowed_strings:
            problems.append(f"identifier {ident} is not in the facts")
        remaining = remaining.replace(ident, " ")

    for code in _CURRENCY.findall(text):
        if code in S.MARKETS and code not in {facts.get("currency"), "USD"}:
            problems.append(f"currency {code} is not the trade's currency")
    side = facts.get("side")
    for other, pattern in _SIDE_WORDS.items():
        if side and other != side and pattern.search(text):
            problems.append(f"text describes a {other.lower()}, the trade is a {side.lower()}")
    for kind, pattern in _TYPE_WORDS.items():
        if kind != facts.get("break_type") and pattern.search(text):
            problems.append(f"wording fits a {kind} break, not {facts.get('break_type')}")
    if _DISMISSIVE.search(text):
        problems.append("text advises dismissing the item; that is the analyst's call")

    allowed = _allowed_numbers(facts)
    for tok in _NUMBER.findall(remaining):
        x = float(tok.rstrip("%").replace(",", ""))
        # Allow rounding to the precision the writer used.
        decimals = len(tok.rstrip("%").split(".")[1]) if "." in tok else 0
        half_unit = 0.5 * 10.0 ** (-decimals)
        if not any(abs(x - a) <= half_unit + 1e-9 for a in allowed):
            problems.append(f"number {tok} does not match any fact")
    return problems


PROMPT = """You explain reconciliation breaks to a custody operations analyst.
Write two sentences: what does not agree, and what the analyst should check first.
Use only the facts below. Copy identifiers, dates and figures exactly; do not compute
new figures and do not guess causes that the facts do not support.

Facts (JSON):
{facts}"""


Provider = Callable[[str], str]


@dataclass
class Explanation:
    text: str
    source: str  # "template" or "llm"
    problems: tuple[str, ...] = ()


class LLMExplainer:
    """Model-written explanations, accepted only when they pass ``check_faithfulness``."""

    def __init__(self, provider: Provider):
        self.provider = provider

    def explain(self, row: dict) -> Explanation:
        facts = facts_for(row)
        try:
            draft = self.provider(PROMPT.format(facts=json.dumps(facts, default=str)))
        except Exception as err:  # provider outage must not block the queue
            return Explanation(template_explanation(facts), "template", (f"provider error: {err}",))
        problems = check_faithfulness(draft, facts)
        if problems:
            return Explanation(template_explanation(facts), "template", tuple(problems))
        return Explanation(draft.strip(), "llm")


def anthropic_provider(model: str | None = None) -> Provider:
    """Provider backed by the Anthropic API; needs ``ANTHROPIC_API_KEY``."""
    import anthropic  # optional dependency: pip install .[llm]

    client = anthropic.Anthropic()
    model = model or os.environ.get("RECON_LLM_MODEL", "claude-sonnet-5-5")

    def call(prompt: str) -> str:
        msg = client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in msg.content if block.type == "text")

    return call
