"""Loading the generated tables and separating ground truth from observable data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .schema import TRUTH_PREFIX


def observable(df: pd.DataFrame) -> pd.DataFrame:
    """Drop every ground-truth column, so downstream code cannot read it by accident."""
    return df.loc[:, [c for c in df.columns if not c.startswith(TRUTH_PREFIX)]].copy()


@dataclass
class Dataset:
    internal: pd.DataFrame  # full table, truth columns included
    external: pd.DataFrame
    accounts: pd.DataFrame
    securities: pd.DataFrame
    fx: dict[str, float]
    meta: dict

    @property
    def account_map(self) -> dict[str, str]:
        """Custodian account code -> internal account."""
        return dict(zip(self.accounts["custody_account"], self.accounts["account"]))


def from_generated(data: dict) -> Dataset:
    fx = data["fx_rates"]
    return Dataset(
        internal=data["internal_book"],
        external=data["custodian_statement"],
        accounts=data["accounts"],
        securities=data["securities"],
        fx=dict(zip(fx["currency"], fx["usd_per_unit"])),
        meta=data["meta"],
    )


def load(data_dir: str | Path) -> Dataset:
    d = Path(data_dir)
    read = {"dtype": {"statement_ref": str}, "keep_default_na": False}
    fx = pd.read_csv(d / "fx_rates.csv")
    return Dataset(
        internal=pd.read_csv(d / "internal_book.csv"),
        external=pd.read_csv(d / "custodian_statement.csv", **read),
        accounts=pd.read_csv(d / "accounts.csv"),
        securities=pd.read_csv(d / "securities.csv"),
        fx=dict(zip(fx["currency"], fx["usd_per_unit"])),
        meta=json.loads((d / "meta.json").read_text()),
    )
