"""Исполнение T+1, комиссии, PnL, проскальзывание и сквозной прогон без сети."""

import asyncio
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from solution import BAR_MS, backtest, main, trade_pnl

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "bybit_spot_BTCUSDT_1m.json"

T0 = 1_699_999_980_000


def frame(opens, closes, buys=(), sells=(), ts=None) -> pd.DataFrame:
    n = len(opens)
    ts = ts or [T0 + i * BAR_MS for i in range(n)]
    return pd.DataFrame(
        {
            "ts": ts,
            "time": pd.to_datetime(ts, unit="ms", utc=True),
            "open": opens,
            "close": closes,
            "buy": [i in buys for i in range(n)],
            "sell": [i in sells for i in range(n)],
        }
    )


def test_pnl_with_fees():
    qty, fee_in, fee_out, pnl, pct = trade_pnl(100.0, 110.0)
    assert fee_in == pytest.approx(1.0)  # 0.1% от 1000
    assert qty == pytest.approx(9.99)  # на 999 USDT
    assert fee_out == pytest.approx(9.99 * 110 * 0.0018)  # 0.18% от выручки 1098.9
    assert pnl == pytest.approx(1098.9 - 1.97802 - 1000)
    assert pct == pytest.approx(pnl / 10)


def test_flat_trade_loses_only_fees():
    *_, pnl, pct = trade_pnl(100.0, 100.0)
    assert pnl == pytest.approx(1000 * (0.999 * 0.9982 - 1))  # -2.7982 USDT
    assert pct == pytest.approx(-0.27982)


def test_execution_on_next_open_and_forced_close():
    df = frame(
        opens=[100, 101, 102, 103, 104, 105],
        closes=[100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
        buys={1, 2, 4},  # BUY на баре 2 уже в позиции
        sells={3},
    )
    res = backtest(df, warmup=0, warn=lambda _: None)

    first, second = res.trades
    # сигнал на Close бара 1 -> вход по Open бара 2; сигнал на баре 3 -> выход по Open бара 4
    assert (first.entry_time, first.entry_price) == (df["time"][2], 102)
    assert (first.exit_time, first.exit_price) == (df["time"][4], 104)
    assert first.entry_signal_price == 101.5 and not first.forced_exit
    assert first.pnl_usdt == pytest.approx(trade_pnl(102, 104)[3])
    # позиция открыта на конец данных -> закрываем по Close последнего бара
    assert second.entry_price == 105
    assert second.forced_exit and second.exit_price == 105.5
    assert res.ignored_signals == 1


def test_sell_without_position_is_ignored():
    df = frame(opens=[1, 1, 1], closes=[1, 1, 1], sells={0})
    res = backtest(df, warmup=0)
    assert res.trades == [] and res.ignored_signals == 1


def test_signals_in_warmup_are_not_traded():
    df = frame(opens=[1, 1, 1, 1], closes=[1, 1, 1, 1], buys={1}, sells={2})
    res = backtest(df, warmup=3)
    assert res.trades == [] and res.warmup_signals == 2


def test_buy_on_last_bar_is_not_executed():
    df = frame(opens=[1, 1, 1], closes=[1, 1, 1], buys={2})
    res = backtest(df, warmup=0)
    assert res.trades == []
    assert "BUY на последнем баре" in res.last_bar_note


def test_sell_on_last_bar_closes_at_last_close():
    df = frame(opens=[10, 10, 11, 12], closes=[10, 10, 11, 12.5], buys={0}, sells={3})
    res = backtest(df, warmup=0)
    (t,) = res.trades
    assert t.forced_exit and t.exit_price == 12.5
    assert "SELL на последнем баре" in res.last_bar_note


def test_slippage_warning():
    warnings = []
    # Close бара 1 = 100, Open бара 2 = 101 -> +1% > 0.5%
    df = frame(opens=[100, 100, 101, 101, 100.4], closes=[100, 100, 101, 100, 100], buys={1}, sells={3})
    res = backtest(df, warmup=0, warn=warnings.append)
    (t,) = res.trades
    assert t.entry_slippage == pytest.approx(0.01)
    assert t.exit_slippage == pytest.approx(0.004)  # 0.4% < 0.5%: без предупреждения
    assert len(warnings) == 1 and "входа" in warnings[0]
    assert res.slippage_warnings == 1
    assert t.notes == ["проскальзывание входа +1.00%"]


def test_execution_after_gap_is_marked():
    ts = [T0, T0 + BAR_MS, T0 + 5 * BAR_MS, T0 + 6 * BAR_MS]
    df = frame(opens=[1, 1, 1, 1], closes=[1, 1, 1, 1], buys={1}, sells={2}, ts=ts)
    (t,) = backtest(df, warmup=0).trades
    assert "вход после пропуска минут" in t.notes


def synthetic_bybit_response(n: int = 600) -> dict:
    """Синусоида цен: EMA 9/21 пересекаются несколько раз."""
    rows = []
    for i in range(n):
        c = 60_000 + 300 * math.sin(i / 15)
        o = 60_000 + 300 * math.sin((i - 1) / 15)
        rows.append(
            [
                str(T0 + i * BAR_MS),
                f"{o:.2f}",
                f"{max(o, c) + 5:.2f}",
                f"{min(o, c) - 5:.2f}",
                f"{c:.2f}",
                "1",
                "1",
            ]
        )
    rows.reverse()  # как у Bybit: от новых к старым
    return {"retCode": 0, "retMsg": "OK", "result": {"list": rows}, "time": T0 + n * BAR_MS - 1}


def test_main_end_to_end_from_file(tmp_path, capsys):
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(synthetic_bybit_response()), encoding="utf-8")
    assert asyncio.run(main(["--from-file", str(raw)])) == 0
    out = capsys.readouterr().out
    assert "незакрытая текущая свеча:    1" in out
    assert "Закрытые сделки" in out
    assert "Winrate:" in out
    assert "суммарный чистый PnL" in out


def test_exit_slippage_warning_and_gap_note():
    warnings = []
    ts = [T0 + i * BAR_MS for i in (0, 1, 2, 3, 9)]  # перед последним баром пропуск 5 минут
    df = frame(opens=[100, 100, 100, 100, 99], closes=[100, 100, 100, 100, 99], buys={1}, sells={3}, ts=ts)
    (t,) = backtest(df, warmup=0, warn=warnings.append).trades
    assert t.exit_slippage == pytest.approx(-0.01)
    assert len(warnings) == 1 and "выхода" in warnings[0]
    assert "проскальзывание выхода -1.00%" in t.notes
    assert "выход после пропуска минут" in t.notes


def test_example_file_matches_readme(capsys):
    """Сохранённый ответ Bybit из examples/: те же числа, что в README."""
    assert asyncio.run(main(["--from-file", str(EXAMPLE)])) == 0
    out = capsys.readouterr().out
    assert "осталось баров:              999" in out
    assert "сделок: 15 (из них закрыта принудительно в конце данных: 1)" in out
    assert "Winrate: 13.3% (2 из 15 с чистым PnL > 0)" in out
    assert "суммарный чистый PnL: -29.43 USDT" in out
