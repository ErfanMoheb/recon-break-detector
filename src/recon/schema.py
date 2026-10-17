"""Shared vocabulary: markets, generator events, break types and tolerances."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Market:
    countries: tuple[str, ...]
    usd_per_unit: float
    typical_price: float
    lot: int
    decimals: int


# FX is held constant over the simulated horizon; a stated simplification.
MARKETS: dict[str, Market] = {
    "USD": Market(("US",), 1.00, 80.0, 100, 2),
    "EUR": Market(("DE", "FR", "NL"), 1.08, 60.0, 50, 2),
    "GBP": Market(("GB",), 1.27, 12.0, 100, 2),
    "JPY": Market(("JP",), 0.0067, 3200.0, 100, 0),
    "HKD": Market(("HK",), 0.128, 45.0, 500, 2),
}

FX_TO_USD: dict[str, float] = {ccy: m.usd_per_unit for ccy, m in MARKETS.items()}

# --- Generator events (ground truth, never visible to the matcher) ---
NONE = "none"
TIMING_EXTERNAL = "timing_external"  # custodian posts one or two days late; self-resolves
TIMING_INTERNAL = "timing_internal"  # desk books one day late; self-resolves
FEED_DELAY = "feed_delay"  # incident: a counterparty's whole feed arrives a day late
MISSING_EXTERNAL = "missing_external"  # settlement never reaches the custodian
MISSING_INTERNAL = "missing_internal"  # custodian movement the firm never booked
QTY_MISMATCH = "qty_mismatch"  # partial or over-delivery
AMOUNT_MISMATCH = "amount_mismatch"  # wrong price, fee or commission
PRICE_FEED = "price_feed"  # incident: stale prices on one market, amounts wrong
DUPLICATE_EXTERNAL = "duplicate_external"  # custodian reports the movement twice

BENIGN_EVENTS = frozenset({NONE, TIMING_EXTERNAL, TIMING_INTERNAL, FEED_DELAY})

# --- Break types reported by the matcher ---
B_MISSING_EXTERNAL = "MISSING_EXTERNAL"
B_MISSING_INTERNAL = "MISSING_INTERNAL"
B_QTY = "QTY_MISMATCH"
B_AMOUNT = "AMOUNT_MISMATCH"
B_DUPLICATE = "DUPLICATE_EXTERNAL"
BREAK_TYPES = (B_MISSING_EXTERNAL, B_MISSING_INTERNAL, B_QTY, B_AMOUNT, B_DUPLICATE)

# The break each real event should end up as once late postings have had time to land.
EXPECTED_BREAK: dict[str, str] = {
    MISSING_EXTERNAL: B_MISSING_EXTERNAL,
    MISSING_INTERNAL: B_MISSING_INTERNAL,
    QTY_MISMATCH: B_QTY,
    AMOUNT_MISMATCH: B_AMOUNT,
    PRICE_FEED: B_AMOUNT,
}

TRUTH_PREFIX = "truth_"

# A pair agrees on amount when the USD difference is within this band.
ABS_TOLERANCE_USD = 1.00
REL_TOLERANCE = 1e-5


def tolerance_usd(amount_usd: float) -> float:
    """Largest USD difference treated as rounding for a movement of this size."""
    return max(ABS_TOLERANCE_USD, REL_TOLERANCE * abs(amount_usd))
