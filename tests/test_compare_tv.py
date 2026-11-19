"""Сверка с выгрузкой TradingView: синтетический «TV» с длинной историей."""

import numpy as np
import pandas as pd

from compare_tv import compare, load_tv_csv
from solution import BAR_MS, add_signals

T0 = 1_699_999_980_000
WARMUP = 210


def bars(n: int, seed: int = 3) -> pd.DataFrame:
    close = 60_000 + np.cumsum(np.random.default_rng(seed).normal(0, 25, n)).round(2)
    ts = T0 + np.arange(n) * BAR_MS
    return pd.DataFrame(
        {
            "ts": ts,
            "time": pd.to_datetime(ts, unit="ms", utc=True),
            "open": close,
            "high": close + 5,
            "low": close - 5,
            "close": close,
        }
    )


def tv_export(full: pd.DataFrame, path, iso_time: bool = False, with_shapes: bool = True) -> None:
    """Как Export chart data: EMA посчитаны по всей истории, plotshape = 1 или пусто."""
    s = add_signals(full, 9, 21)
    out = pd.DataFrame(
        {
            "time": s["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ") if iso_time else s["ts"] // 1000,
            "open": s["open"],
            "high": s["high"],
            "low": s["low"],
            "close": s["close"],
            "EMA Fast": s["ema_fast"],
            "EMA Slow": s["ema_slow"],
        }
    )
    if with_shapes:
        out["BUY"] = np.where(s["buy"], 1, np.nan)
        out["SELL"] = np.where(s["sell"], 1, np.nan)
    out.to_csv(path, index=False)


def ours_from(full: pd.DataFrame, n: int = 1000) -> pd.DataFrame:
    return add_signals(full.iloc[-n:].reset_index(drop=True), 9, 21)


def test_matches_after_warmup(tmp_path):
    full = bars(3000)
    tv_export(full, tmp_path / "tv.csv")
    tv, source = load_tv_csv(tmp_path / "tv.csv")
    res = compare(ours_from(full), tv, WARMUP)

    assert source.startswith("колонки BUY/SELL")
    assert res.common == res.ours_bars == res.tv_bars == 1000
    assert res.candle_diffs == []
    assert res.ema_diff["slow"][0] > 1  # в прогреве засев заметен
    assert res.ema_diff["slow"][1] < 0.01  # после прогрева меньше шага цены
    assert res.signals_after > 0
    assert res.mismatches_after == []


def test_reports_signal_and_candle_mismatch(tmp_path):
    full = bars(3000)
    tv_export(full, tmp_path / "tv.csv")
    tv, _ = load_tv_csv(tmp_path / "tv.csv")
    ours = ours_from(full)
    k = WARMUP + int(np.flatnonzero(ours["buy"].iloc[WARMUP:].to_numpy())[0])
    ours.loc[k, "buy"] = False  # «потеряли» сигнал после прогрева
    ours.loc[k, "close"] += 1

    res = compare(ours, tv, WARMUP)
    assert len(res.mismatches_after) == 1
    assert "у нас нет, у TV BUY" in res.mismatches_after[0]
    assert len(res.candle_diffs) == 1


def test_iso_time_and_signals_from_ema(tmp_path):
    full = bars(3000)
    tv_export(full, tmp_path / "tv.csv", iso_time=True, with_shapes=False)
    tv, source = load_tv_csv(tmp_path / "tv.csv")
    res = compare(ours_from(full), tv, WARMUP)

    assert source.startswith("пересчитаны")
    assert res.common == 1000
    assert res.mismatches_after == []


def test_partial_tv_coverage(tmp_path):
    full = bars(1500)
    tv_export(full.iloc[:1200], tmp_path / "tv.csv")  # у TV нет последних 300 баров
    tv, _ = load_tv_csv(tmp_path / "tv.csv")
    res = compare(ours_from(full), tv, WARMUP)
    assert res.common == 700
    assert len(res.only_ours) == 300
