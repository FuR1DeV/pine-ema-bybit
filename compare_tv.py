#!/usr/bin/env python3
"""Сверка EMA и сигналов solution.py с выгрузкой графика TradingView бар за баром.

1. На TradingView открыть BYBIT:BTCUSDT (спот), 1m, добавить на график
   tradingview/ema_crossover_export.pine и выгрузить Export chart data.
2. python compare_tv.py --tv export.csv [--raw examples/bybit_spot_BTCUSDT_1m.json]

Сравниваются бары, которые есть и у нас, и в выгрузке TV (по времени открытия бара):
сами свечи, значения EMA и сигналы. Код возврата 1, если после прогрева есть
расхождения в сигналах.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from solution import add_signals, check_payload, clean_klines, crossover, crossunder, fmt_time

PRICE_COLS = ("open", "high", "low", "close")
PRICE_TOLERANCE = 1e-6  # цены у Bybit с шагом 0.01, всё, что больше, это другая свеча


def find_column(df: pd.DataFrame, name: str) -> str | None:
    """Колонка без учёта регистра и пробелов по краям: у TV это "EMA Fast", "BUY" и т.п."""
    for col in df.columns:
        if str(col).strip().lower() == name.lower():
            return col
    return None


def truthy(s: pd.Series) -> pd.Series:
    """Сигнал plotshape в выгрузке: 1 там, где он есть, пусто или 0 там, где нет."""
    num = pd.to_numeric(s, errors="coerce")
    return (num.fillna(0) != 0) | s.astype(str).str.strip().str.lower().eq("true")


def load_tv_csv(path: Path) -> tuple[pd.DataFrame, str]:
    """Читает Export chart data. Время бывает UNIX (секунды) или ISO с часовым поясом.

    Возвращает таблицу time, open..close, ema_fast, ema_slow, buy, sell и откуда взяты
    сигналы: из колонок BUY/SELL, если plotshape попал в выгрузку, иначе пересчитываем
    пересечения по значениям EMA самого TradingView.
    """
    raw = pd.read_csv(path)
    time_col = find_column(raw, "time")
    if time_col is None:
        raise ValueError(f"в выгрузке нет колонки time, есть: {list(raw.columns)}")
    t = raw[time_col]
    if pd.api.types.is_numeric_dtype(t):
        time = pd.to_datetime(t, unit="ms" if t.max() > 1e11 else "s", utc=True)
    else:
        time = pd.to_datetime(t, utc=True)

    out = pd.DataFrame({"time": time})
    for c in PRICE_COLS:
        col = find_column(raw, c)
        if col is not None:
            out[c] = pd.to_numeric(raw[col], errors="coerce")
    for key, name in (("ema_fast", "EMA Fast"), ("ema_slow", "EMA Slow")):
        col = find_column(raw, name)
        if col is None:
            raise ValueError(f"в выгрузке нет колонки {name!r}, есть: {list(raw.columns)}")
        out[key] = pd.to_numeric(raw[col], errors="coerce")
    buy_col, sell_col = find_column(raw, "BUY"), find_column(raw, "SELL")
    if buy_col is not None and sell_col is not None:
        out["buy"], out["sell"] = truthy(raw[buy_col]), truthy(raw[sell_col])

    out = out.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    if "buy" in out:
        return out, "колонки BUY/SELL из выгрузки TV"
    out["buy"] = crossover(out["ema_fast"], out["ema_slow"])
    out["sell"] = crossunder(out["ema_fast"], out["ema_slow"])
    return out, "пересчитаны по EMA из выгрузки TV"


@dataclass
class Comparison:
    ours_bars: int = 0
    tv_bars: int = 0  # баров TV в интервале наших данных
    common: int = 0
    only_ours: list[pd.Timestamp] = field(default_factory=list)
    only_tv: list[pd.Timestamp] = field(default_factory=list)
    candle_diffs: list[str] = field(default_factory=list)
    ema_diff: dict[str, tuple[float, float]] = field(default_factory=dict)  # (в прогреве, после)
    signals_after: int = 0  # сигналов у нас после прогрева на общих барах
    mismatches_warmup: int = 0
    mismatches_after: list[str] = field(default_factory=list)


def label(buy: bool, sell: bool) -> str:
    return "BUY" if buy else "SELL" if sell else "нет"


def compare(ours: pd.DataFrame, tv: pd.DataFrame, warmup: int) -> Comparison:
    """ours: результат solution.add_signals; номер бара в наших данных решает, прогрев это или нет."""
    res = Comparison(ours_bars=len(ours))
    ours = ours.assign(i=np.arange(len(ours)))
    lo, hi = ours["time"].iloc[0], ours["time"].iloc[-1]
    tv = tv[(tv["time"] >= lo) & (tv["time"] <= hi)]
    res.tv_bars = len(tv)
    res.only_ours = sorted(set(ours["time"]) - set(tv["time"]))
    res.only_tv = sorted(set(tv["time"]) - set(ours["time"]))

    m = ours.merge(tv, on="time", suffixes=("", "_tv"))
    res.common = len(m)
    after = m["i"] >= warmup

    for c in PRICE_COLS:
        if f"{c}_tv" not in m:
            continue
        bad = m[(m[c] - m[f"{c}_tv"]).abs() > PRICE_TOLERANCE]
        res.candle_diffs += [
            f"{fmt_time(r.time)} {c}: у нас {getattr(r, c)}, у TV {getattr(r, c + '_tv')}"
            for r in bad.itertuples()
        ]

    for k in ("fast", "slow"):
        d = (m[f"ema_{k}"] - m[f"ema_{k}_tv"]).abs()
        res.ema_diff[k] = (float(d[~after].max()), float(d[after].max()))

    mismatch = (m["buy"] != m["buy_tv"]) | (m["sell"] != m["sell_tv"])
    res.mismatches_warmup = int((mismatch & ~after).sum())
    res.signals_after = int((m["buy"] | m["sell"])[after].sum())
    res.mismatches_after = [
        f"{fmt_time(r.time)}: у нас {label(r.buy, r.sell)}, у TV {label(r.buy_tv, r.sell_tv)}"
        for r in m[mismatch & after].itertuples()
    ]
    return res


def print_comparison(res: Comparison, warmup: int, signals_source: str, args: argparse.Namespace) -> None:
    print(f"Сверка с TradingView: EMA {args.fast}/{args.slow}, засев {args.ema_seed}, прогрев {warmup} баров")
    print(f"  сигналы TV: {signals_source}")
    print(f"  баров у нас: {res.ours_bars}, у TV в том же интервале: {res.tv_bars}, общих: {res.common}")
    if res.only_ours or res.only_tv:
        print(f"  есть только у нас: {len(res.only_ours)}, только у TV: {len(res.only_tv)}")
        for t in (res.only_ours[:5] + res.only_tv[:5])[:10]:
            print(f"    {fmt_time(t)}")
    if res.common < res.ours_bars:
        print("  ВНИМАНИЕ: выгрузка TV покрывает не весь интервал наших данных")
    print(f"  свечи с другими ценами: {len(res.candle_diffs)}")
    for line in res.candle_diffs[:10]:
        print(f"    {line}")
    for k, length in (("fast", args.fast), ("slow", args.slow)):
        w, a = res.ema_diff[k]
        print(f"  EMA {length}: макс. разница в прогреве {w:.6f}, после прогрева {a:.6f}")
    print(f"  сигналы в прогреве: расходятся на {res.mismatches_warmup} барах (ожидаемо, засев)")
    print(
        f"  сигналы после прогрева: у нас {res.signals_after}, "
        f"расходятся на {len(res.mismatches_after)} барах"
    )
    for line in res.mismatches_after:
        print(f"    {line}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Сверка solution.py с выгрузкой графика TradingView")
    p.add_argument("--tv", type=Path, required=True, help="CSV из Export chart data")
    p.add_argument("--raw", type=Path, default=Path("examples/bybit_spot_BTCUSDT_1m.json"))
    p.add_argument("--fast", type=int, default=9)
    p.add_argument("--slow", type=int, default=21)
    p.add_argument("--warmup", type=int, default=None, help="по умолчанию 10 * slow, как в solution.py")
    p.add_argument("--ema-seed", choices=["sma", "first"], default="sma")
    args = p.parse_args(argv)
    warmup = 10 * args.slow if args.warmup is None else args.warmup

    payload = json.loads(args.raw.read_text(encoding="utf-8"))
    check_payload(payload)
    ours, _ = clean_klines(payload)
    ours = add_signals(ours, args.fast, args.slow, args.ema_seed)
    tv, source = load_tv_csv(args.tv)

    res = compare(ours, tv, warmup)
    print_comparison(res, warmup, source, args)
    return 1 if res.mismatches_after or res.common == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
