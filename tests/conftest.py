import pytest

from recon import generate as G
from recon import schema as S
from recon.io import from_generated

SMALL = G.GeneratorConfig(
    n_days=40,
    trades_per_day=200,
    seed=3,
    incidents=(G.Incident(30, S.FEED_DELAY, "CP03"),),
)
SMALL_TRAIN_DAYS = 25


@pytest.fixture(scope="session")
def small_data():
    return G.generate(SMALL)


@pytest.fixture(scope="session")
def small_ds(small_data):
    return from_generated(small_data)
