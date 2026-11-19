"""EMA и пересечения."""

import math

import numpy as np
import pandas as pd
import pytest

from solution import add_signals, crossover, crossunder, default_warmup, ema


def test_ema_sma_seed_by_hand():
    # length 3 -> alpha = 0.5; первые 2 значения na, третье = SMA(1, 2, 3) = 2
    out = ema([1, 2, 3, 4, 5], 3)
    assert math.isnan(out[0]) and math.isnan(out[1])
    assert out[2:].tolist() == [2.0, 3.0, 4.0]


def test_ema_first_seed_equals_pandas_ewm():
    x = np.random.default_rng(1).normal(100, 5, 300)
    expected = pd.Series(x).ewm(span=9, adjust=False).mean().to_numpy()
    np.testing.assert_allclose(ema(x, 9, seed="first"), expected)


def test_ema_shorter_than_length_is_all_na():
    assert np.isnan(ema([1.0, 2.0], 3)).all()


def test_ema_rejects_bad_args():
    with pytest.raises(ValueError, match="length"):
        ema([1.0], 0)
    with pytest.raises(ValueError, match="seed"):
        ema([1.0], 1, seed="zero")


def test_crossover_and_crossunder():
    a = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0])
    b = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0])
    # бар 2: 3 > 2, а на баре 1 было 2 <= 2 -> crossover (касание тоже считается, как в Pine)
    assert crossover(a, b).tolist() == [False, False, True, False, False]
    # бар 4: 1 < 2, а на баре 3 было 2 >= 2 -> crossunder
    assert crossunder(a, b).tolist() == [False, False, False, False, True]


def test_cross_with_na_is_false():
    a = pd.Series([np.nan, 3.0, 1.0])
    b = pd.Series([np.nan, 2.0, 2.0])
    assert crossover(a, b).tolist() == [False, False, False]
    assert crossunder(a, b).tolist() == [False, False, True]


def random_walk(n: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 60_000 + np.cumsum(rng.normal(0, 25, n))


def test_warmup_hides_seed_difference():
    """У TradingView история длиннее наших 1000 баров. Считаем EMA по 3000 барам
    («как у TV») и только по последним 1000 («как у нас»): после прогрева 10 * slow
    расхождение меньше 0.01 (шаг цены BTCUSDT на споте Bybit 0.1), а сигналы совпадают."""
    full = random_walk(3000)
    ours = full[-1000:]
    warmup = 10 * 21
    for length in (9, 21):
        long_hist = ema(full, length)[-1000:]
        short_hist = ema(ours, length)
        assert np.max(np.abs(long_hist[warmup:] - short_hist[warmup:])) < 0.01

    tv = add_signals(pd.DataFrame({"close": full}), 9, 21).iloc[-1000:].reset_index(drop=True)
    us = add_signals(pd.DataFrame({"close": ours}), 9, 21)
    assert (tv["buy"].iloc[warmup:] == us["buy"].iloc[warmup:]).all()
    assert (tv["sell"].iloc[warmup:] == us["sell"].iloc[warmup:]).all()
    assert tv["buy"].iloc[warmup:].sum() > 0  # сигналы в окне вообще есть


def test_default_warmup_follows_the_longer_ema():
    assert default_warmup(9, 21) == 210
    assert default_warmup(50, 21) == 500


@pytest.mark.parametrize(("fast", "slow"), [(9, 21), (21, 9), (50, 21)])
def test_default_warmup_hides_seed_for_any_order(fast, slow):
    """Если fast длиннее slow, засев дольше помнит fast: прогрев 10 * slow тут не хватало."""
    warmup = default_warmup(fast, slow)
    for seed in range(5):
        full = random_walk(4000, seed)
        tv = add_signals(pd.DataFrame({"close": full}), fast, slow).iloc[-1000:].reset_index(drop=True)
        us = add_signals(pd.DataFrame({"close": full[-1000:]}), fast, slow)
        assert (tv["buy"].iloc[warmup:] == us["buy"].iloc[warmup:]).all()
        assert (tv["sell"].iloc[warmup:] == us["sell"].iloc[warmup:]).all()
