#!/usr/bin/env python3
"""Сверка EMA и сигналов solution.py с эталоном, у которого длинная история, бар за баром.

Эталон 1, выгрузка графика TradingView:
  на TradingView открыть BYBIT:BTCUSDT (спот), 1m, добавить на график
  tradingview/ema_crossover_export.pine и выгрузить Export chart data, затем
  python compare_tv.py --tv export.csv [--raw examples/bybit_spot_BTCUSDT_1m.json]

Эталон 2, без TradingView: та же стратегия на 3000 барах Bybit, которые кончаются
на последнем нашем баре (несколько запросов с параметром end):
  python compare_tv.py --bybit-history 3000

Сравниваются бары, которые есть и у нас, и у эталона (по времени открытия бара):
сами свечи, значения EMA и сигналы. Код возврата 1, если после прогрева есть
расхождения в сигналах.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from solution import (
    BAR_MS,
    BYBIT_BASE_URL,
    add_signals,
    check_payload,
    clean_klines,
    crossover,
    crossunder,
    fetch_klines,
    fmt_time,
)

PRICE_COLS = ("open", "high", "low", "close")
PRICE_TOLERANCE = 1e-6  # шаг цены BTCUSDT на споте Bybit 0.1, всё, что больше, это другая свеча


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


async def load_bybit_history(last_ts: int, total: int, args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    """Эталон без TradingView: total минутных баров Bybit, последний из которых равен
    нашему последнему бару. EMA по ним идёт с длинной историей, как на графике TV."""
    rows: list = []
    end = last_ts
    async with httpx.AsyncClient(base_url=args.base_url, timeout=10.0) as client:
        while len(rows) < total:
            chunk = (await fetch_klines(client, args.symbol, 1000, end_ms=end))["result"]["list"]
            if not chunk:
                break
            rows += chunk
            end = min(int(r[0]) for r in chunk) - 1
    ref, _ = clean_klines({"result": {"list": rows}}, now_ms=last_ts + BAR_MS)
    ref = add_signals(ref.iloc[-total:].reset_index(drop=True), args.fast, args.slow, args.ema_seed)
    return ref, f"посчитаны нами по {len(ref)} барам Bybit начиная с {fmt_time(ref['time'].iloc[0])} UTC"


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
            f"{fmt_time(r.time)} {c}: у нас {getattr(r, c)}, у эталона {getattr(r, c + '_tv')}"
            for r in bad.itertuples()
        ]

    for k in ("fast", "slow"):
        d = (m[f"ema_{k}"] - m[f"ema_{k}_tv"]).abs()
        res.ema_diff[k] = (float(d[~after].max()), float(d[after].max()))

    mismatch = (m["buy"] != m["buy_tv"]) | (m["sell"] != m["sell_tv"])
    res.mismatches_warmup = int((mismatch & ~after).sum())
    res.signals_after = int((m["buy"] | m["sell"])[after].sum())
    res.mismatches_after = [
        f"{fmt_time(r.time)}: у нас {label(r.buy, r.sell)}, у эталона {label(r.buy_tv, r.sell_tv)}"
        for r in m[mismatch & after].itertuples()
    ]
    return res


def print_comparison(res: Comparison, warmup: int, ref: str, source: str, args: argparse.Namespace) -> None:
    print(f"Сверка с эталоном ({ref}): EMA {args.fast}/{args.slow}, засев {args.ema_seed}, прогрев {warmup}")
    print(f"  сигналы эталона: {source}")
    print(f"  баров у нас: {res.ours_bars}, у эталона в том же интервале: {res.tv_bars}, общих: {res.common}")
    if res.only_ours or res.only_tv:
        print(f"  есть только у нас: {len(res.only_ours)}, только у эталона: {len(res.only_tv)}")
        for t in (res.only_ours[:5] + res.only_tv[:5])[:10]:
            print(f"    {fmt_time(t)}")
    if res.common < res.ours_bars:
        print("  ВНИМАНИЕ: эталон покрывает не весь интервал наших данных")
    print(f"  свечи с другими ценами: {len(res.candle_diffs)}")
    for line in res.candle_diffs[:10]:
        print(f"    {line}")

    def num(x: float) -> str:
        if np.isnan(x):
            return "-"
        return f"{x:.6f}" if x == 0 or x >= 1e-3 else f"{x:.1e}"

    for k, length in (("fast", args.fast), ("slow", args.slow)):
        w, a = res.ema_diff[k]
        print(f"  EMA {length}: макс. разница в прогреве {num(w)}, после прогрева {num(a)}")
    print(f"  сигналы в прогреве: баров с расхождением {res.mismatches_warmup} (ожидаемо, засев)")
    print(
        f"  сигналы после прогрева: у нас {res.signals_after}, "
        f"баров с расхождением {len(res.mismatches_after)}"
    )
    for line in res.mismatches_after:
        print(f"    {line}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Сверка solution.py с эталоном с длинной историей")
    ref = p.add_mutually_exclusive_group(required=True)
    ref.add_argument("--tv", type=Path, help="CSV из Export chart data TradingView")
    ref.add_argument("--bybit-history", type=int, metavar="N", help="эталон из N баров Bybit, например 3000")
    p.add_argument("--raw", type=Path, default=Path("examples/bybit_spot_BTCUSDT_1m.json"))
    p.add_argument("--fast", type=int, default=9)
    p.add_argument("--slow", type=int, default=21)
    p.add_argument("--warmup", type=int, default=None, help="по умолчанию 10 * slow, как в solution.py")
    p.add_argument("--ema-seed", choices=["sma", "first"], default="sma")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--base-url", default=BYBIT_BASE_URL)
    args = p.parse_args(argv)
    warmup = 10 * args.slow if args.warmup is None else args.warmup

    payload = json.loads(args.raw.read_text(encoding="utf-8"))
    check_payload(payload)
    ours, _ = clean_klines(payload)
    ours = add_signals(ours, args.fast, args.slow, args.ema_seed)
    if args.tv:
        name = f"TradingView, {args.tv.name}"
        reference, source = load_tv_csv(args.tv)
    else:
        name = f"Bybit, {args.bybit_history} баров"
        reference, source = asyncio.run(
            load_bybit_history(int(ours["ts"].iloc[-1]), args.bybit_history, args)
        )

    res = compare(ours, reference, warmup)
    print_comparison(res, warmup, name, source, args)
    return 1 if res.mismatches_after or res.common == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
