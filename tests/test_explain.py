import pytest

from recon import schema as S
from recon.explain import LLMExplainer, check_faithfulness, facts_for, template_explanation
from recon.match import reconcile

ROW = {
    "exception_id": "X000042",
    "break_type": S.B_AMOUNT,
    "trade_ref": "T00012345",
    "account": "ACC03",
    "isin": "US0378331005",
    "currency": "USD",
    "side": "BUY",
    "counterparty": "CP07",
    "first_seen": "2026-05-04",
    "quantity_internal": 1200,
    "quantity_external": 1200,
    "amount_internal": 96_400.26,
    "amount_external": 97_150.26,
    "exposure_usd": 750.00,
    "p_needs_action": 0.97,
}


def test_template_states_the_facts():
    text = template_explanation(facts_for(ROW))
    assert "T00012345" in text
    assert "USD 96,400.26" in text and "USD 97,150.26" in text
    assert check_faithfulness(text, facts_for(ROW)) == []


def test_every_generated_break_gets_a_faithful_template(small_ds):
    exc = reconcile(small_ds.internal, small_ds.external, small_ds.account_map, small_ds.fx)
    for row in exc.exceptions.to_dict("records"):
        facts = facts_for(row)
        assert check_faithfulness(template_explanation(facts), facts) == []


@pytest.mark.parametrize(
    "bad",
    [
        "Booked USD 96,400.26 but custodian shows USD 97,900.26.",  # altered amount
        "Trade T00099999 is off by USD 750.00.",  # invented reference
        "Raised on 2026-05-05 for ACC03.",  # wrong date
        "Probably a fee of USD 12.50 charged by CP07.",  # invented figure
        "Trade t00099999 in acc05 is off by USD 750.00.",  # invented ids, lower case
        "Sold 1,200 US0378331005; custodian shows USD 97,150.26.",  # wrong side
        "Booked EUR 96,400.26 against USD 97,150.26 at the custodian.",  # wrong currency
        "Partial delivery on T00012345: USD 750.00 short.",  # wrong break type
        "USD 750.00 difference on T00012345; it will clear on its own, no action.",  # dismissive
        "Posted 3 days late: T00012345, USD 750.00.",  # small number not in the facts
    ],
)
def test_checker_rejects_unsupported_claims(bad):
    assert check_faithfulness(bad, facts_for(ROW))


def test_checker_accepts_rounded_and_percent_forms():
    ok = (
        "T00012345 in ACC03: the custodian amount is about USD 750 higher than booked "
        "(USD 97,150.26 vs USD 96,400.26); 97% likely to need action, so check the "
        "price on 2026-05-04 first."
    )
    assert check_faithfulness(ok, facts_for(ROW)) == []


def test_llm_text_used_only_when_faithful():
    good = "T00012345: custodian shows USD 97,150.26 against USD 96,400.26 booked."
    ex = LLMExplainer(lambda prompt: good).explain(ROW)
    assert ex.source == "llm" and ex.text == good

    bad = LLMExplainer(lambda prompt: "Fee of USD 12.50 on T00012345.").explain(ROW)
    assert bad.source == "template" and bad.problems


def test_provider_failure_falls_back_to_template():
    def broken(prompt):
        raise TimeoutError("upstream timeout")

    ex = LLMExplainer(broken).explain(ROW)
    assert ex.source == "template"
    assert "provider error" in ex.problems[0]


@pytest.mark.xfail(strict=True, reason="lexical check: cannot tell which role a copied figure has")
def test_known_gap_swapped_roles():
    swapped = "T00012345: booked USD 97,150.26, custodian USD 96,400.26."
    assert check_faithfulness(swapped, facts_for(ROW))
