from dataclasses import replace

import pandas as pd

from recon import generate as G
from recon import schema as S
from recon.io import observable
from recon.match import normalise_reference
from tests.conftest import SMALL


def test_same_seed_same_data(small_data):
    again = G.generate(SMALL)
    pd.testing.assert_frame_equal(small_data["internal_book"], again["internal_book"])
    pd.testing.assert_frame_equal(small_data["custodian_statement"], again["custodian_statement"])


def test_different_seed_differs(small_data):
    other = G.generate(replace(SMALL, seed=SMALL.seed + 1))
    assert not small_data["internal_book"]["amount"].equals(other["internal_book"]["amount"])


def test_every_reference_style_normalises_to_the_trade(small_ds):
    ext = small_ds.external
    has_ref = ext["statement_ref"] != ""
    trade_no = ext.loc[has_ref, "truth_trade_ref"].str[1:].astype(int)
    assert (ext.loc[has_ref, "statement_ref"].map(normalise_reference) == trade_no).all()


def test_truth_is_separable(small_ds):
    obs = observable(small_ds.external)
    assert not any(c.startswith(S.TRUTH_PREFIX) for c in obs.columns)
    assert "statement_ref" in obs.columns


def test_event_mix_is_plausible(small_ds):
    share = small_ds.internal["truth_event"].value_counts(normalize=True)
    assert share[S.NONE] > 0.85
    for ev in (S.TIMING_EXTERNAL, S.MISSING_EXTERNAL, S.QTY_MISMATCH, S.DUPLICATE_EXTERNAL):
        assert share.get(ev, 0) > 0


def test_incident_outside_horizon_is_ignored():
    cfg = replace(SMALL, n_days=10, incidents=(G.Incident(50, S.FEED_DELAY, "CP01"),))
    assert G.generate(cfg)["meta"]["incidents"] == []


def test_incident_is_injected(small_ds):
    inc = small_ds.meta["incidents"][0]
    hit = small_ds.internal[small_ds.internal["truth_event"] == S.FEED_DELAY]
    assert len(hit) > 0
    assert set(hit["counterparty"]) == {inc["target"]}
