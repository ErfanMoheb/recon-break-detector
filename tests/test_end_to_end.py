from recon.cli import main
from recon.evaluate import analyst_queue, evaluate
from tests.conftest import SMALL_TRAIN_DAYS


def test_evaluate_small_dataset(small_ds):
    r = evaluate(small_ds, train_days=SMALL_TRAIN_DAYS, n_boot=100)
    m = r.metrics
    assert m["matching"]["pair_precision"] > 0.999
    assert m["matching"]["break_recall"] > 0.99
    assert m["matching"]["false_breaks_open_at_end"] == 0
    assert set(m["triage"]["strategies"]) >= {"random", "desk_rules", "model"}
    assert all(i["detected"] for i in m["surveillance"]["incidents"])

    day = small_ds.meta["trade_days"][SMALL_TRAIN_DAYS + 2]
    q = analyst_queue(r.exceptions, day, capacity=5)
    assert len(q) == 5
    assert q["rank"].tolist() == [1, 2, 3, 4, 5]
    assert q["reason"].str.len().min() > 40


def test_cli_generate_writes_files(tmp_path):
    main(["generate", "--out", str(tmp_path), "--days", "5", "--seed", "1"])
    names = {p.name for p in tmp_path.iterdir()}
    assert {"internal_book.csv", "custodian_statement.csv", "meta.json"} <= names
