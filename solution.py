#!/usr/bin/env python3
"""Стратегия EMA Crossover из TradingView (Pine Script v5) на асинхронном Python.

1. Асинхронно (httpx) загружает последние 1000 минутных свечей BTCUSDT
   со спотового REST API Bybit (/v5/market/kline) одним запросом.
2. Чистит данные (clean_klines): порядок, дубли, битые и незакрытая свеча, пропуски.
3. Считает EMA 9/21 и пересечения так же, как ta.ema / ta.crossover / ta.crossunder.

Решения и известные расхождения с TradingView описаны в README.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

BYBIT_BASE_URL = "https://api.bybit.com"
KLINE_PATH = "/v5/market/kline"
BAR_MS = 60_000  # таймфрейм 1 минута


class BybitError(RuntimeError):
    """Bybit ответил ошибкой (retCode != 0) или неожиданной структурой."""


# --------------------------------------------------------------------------- загрузка


async def fetch_klines(
    client: httpx.AsyncClient,
    symbol: str = "BTCUSDT",
    limit: int = 1000,
    attempts: int = 3,
) -> dict:
    """Один запрос /v5/market/kline (spot, 1m), пагинация не нужна.

    Повторяем только сетевые сбои и 5xx. Ошибки самого API (неверный символ и т.п.)
    Bybit отдаёт с HTTP 200 и retCode != 0, поэтому retCode проверяем отдельно.
    """
    params = {"category": "spot", "symbol": symbol, "interval": "1", "limit": limit}
    for attempt in range(1, attempts + 1):
        try:
            resp = await client.get(KLINE_PATH, params=params)
            resp.raise_for_status()
            break
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            retryable = isinstance(exc, httpx.TransportError) or exc.response.status_code >= 500
            if not retryable or attempt == attempts:
                raise
            await asyncio.sleep(attempt)

    payload = resp.json()
    if payload.get("retCode") != 0:
        raise BybitError(f"retCode={payload.get('retCode')} retMsg={payload.get('retMsg')!r}")
    if not isinstance((payload.get("result") or {}).get("list"), list):
        raise BybitError("в ответе нет result.list")
    return payload


# --------------------------------------------------------------------------- очистка


@dataclass
class CleaningReport:
    received: int = 0
    descending: bool = False  # пришли ли свечи от новых к старым
    malformed: int = 0  # строку не удалось разобрать (не хватает полей, не число)
    unclosed: int = 0  # свеча ещё не закрылась по времени
    broken: int = 0  # цена <= 0 или не число, high < low, open/close вне [low, high]
    duplicates: int = 0  # повтор startTime
    gaps: int = 0  # сколько раз между соседними барами больше минуты
    missing_minutes: int = 0  # сколько минут пропущено всего
    kept: int = 0


def clean_klines(payload: dict, now_ms: int | None = None) -> tuple[pd.DataFrame, CleaningReport]:
    """Превращает result.list Bybit в таблицу закрытых свечей по возрастанию времени.

    Что делаем и почему:
    - Bybit отдаёт числа строками: [startTime, open, high, low, close, volume, turnover].
      Строку, которую нельзя разобрать, выбрасываем.
    - Незакрытая свеча. Первая в ответе обычно текущая минута, её close ещё меняется.
      Сигнал считается строго по закрытию бара, поэтому такую свечу отбрасываем.
      Закрытость проверяем по времени сервера Bybit (поле time в ответе), а не по
      локальным часам, у которых может быть сдвиг: startTime + 60 с > now -> не закрыта.
    - Битые свечи (цена <= 0 или не число, high < low, open/close вне [low, high],
      время не кратно минуте) выбрасываем, а не чиним: настоящих цен мы не знаем,
      а придуманная цена может дать ложное пересечение EMA.
    - Порядок. Bybit отдаёт от новых к старым, EMA считается от старых к новым,
      поэтому сортируем по startTime.
    - Дубли по startTime: оставляем первый. Битые убираем раньше дублей, чтобы битая
      копия не вытеснила нормальную.
    - Пропуски минут не заполняем, только считаем. TradingView рисует бар только там,
      где были сделки, и считает EMA по существующим барам. Если дорисовать пустые
      минуты, у нас появятся бары, которых нет у TV, и EMA разойдётся с графиком.
    """
    report = CleaningReport()
    raw_rows = payload["result"]["list"]
    report.received = len(raw_rows)
    if now_ms is None:
        now_ms = int(payload.get("time") or time.time() * 1000)

    rows = []
    for raw in raw_rows:
        try:
            ts = int(raw[0])
            o, h, low, c, v = (float(x) for x in raw[1:6])
        except (TypeError, ValueError, IndexError):
            report.malformed += 1
            continue
        rows.append((ts, o, h, low, c, v))

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    report.descending = len(df) > 1 and bool(df["ts"].is_monotonic_decreasing)

    unclosed = df["ts"] + BAR_MS > now_ms
    report.unclosed = int(unclosed.sum())
    df = df[~unclosed]

    prices = df[["open", "high", "low", "close"]]
    broken = (
        ~np.isfinite(df[["open", "high", "low", "close", "volume"]]).all(axis=1)
        | (prices <= 0).any(axis=1)
        | (df["volume"] < 0)
        | (df["high"] < df["low"])
        | (df["open"] > df["high"])
        | (df["open"] < df["low"])
        | (df["close"] > df["high"])
        | (df["close"] < df["low"])
        | (df["ts"] % BAR_MS != 0)
    )
    report.broken = int(broken.sum())
    df = df[~broken]

    df = df.sort_values("ts", kind="stable")
    dup = df["ts"].duplicated(keep="first")
    report.duplicates = int(dup.sum())
    df = df[~dup].reset_index(drop=True)

    step = df["ts"].diff()
    report.gaps = int((step > BAR_MS).sum())
    report.missing_minutes = int((step // BAR_MS - 1).clip(lower=0).sum())
    report.kept = len(df)
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df, report


# --------------------------------------------------------------------------- индикаторы


def ema(values, length: int, seed: str = "sma") -> np.ndarray:
    """EMA как ta.ema в Pine v5: alpha = 2 / (length + 1),
    ema[i] = alpha * src[i] + (1 - alpha) * ema[i - 1].

    seed="sma" (по умолчанию): первые length - 1 значений na, на баре length - 1
    значение равно SMA первых length закрытий. Так ведёт себя встроенная ta.ema
    на графике: EMA(21) появляется только с 21-го бара истории.
    seed="first": засев первым значением источника, как в примере pine_ema
    из справки Pine. Оставлено для сверки.

    От засева зависят только первые бары: ошибка засева умножается на
    (1 - alpha) на каждом баре и затухает. Поэтому в начале данных есть прогрев
    без торговли (см. README).
    """
    if length < 1:
        raise ValueError("length должен быть >= 1")
    src = np.asarray(values, dtype=float)
    out = np.full(len(src), np.nan)
    alpha = 2.0 / (length + 1)
    if seed == "sma":
        if len(src) < length:
            return out
        start = length - 1
        out[start] = src[:length].mean()
    elif seed == "first":
        if len(src) == 0:
            return out
        start = 0
        out[0] = src[0]
    else:
        raise ValueError(f"неизвестный seed: {seed}")
    for i in range(start + 1, len(src)):
        out[i] = alpha * src[i] + (1 - alpha) * out[i - 1]
    return out


def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    """ta.crossover: a > b на текущем баре и a <= b на предыдущем.
    Сравнение с na даёт false, как в Pine: на первом баре EMA пересечения нет."""
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    """ta.crossunder: a < b на текущем баре и a >= b на предыдущем."""
    return (a < b) & (a.shift(1) >= b.shift(1))


def add_signals(df: pd.DataFrame, fast: int, slow: int, seed: str = "sma") -> pd.DataFrame:
    """EMA и сигналы по Close каждого бара. Сигнал бара T известен только после его
    закрытия, исполнять его можно не раньше Open бара T + 1 (это делает backtest)."""
    out = df.copy()
    close = out["close"].to_numpy(dtype=float)
    out["ema_fast"] = ema(close, fast, seed)
    out["ema_slow"] = ema(close, slow, seed)
    out["buy"] = crossover(out["ema_fast"], out["ema_slow"])
    out["sell"] = crossunder(out["ema_fast"], out["ema_slow"])
    return out


# --------------------------------------------------------------------------- вывод


def fmt_time(t: pd.Timestamp) -> str:
    return t.strftime("%Y-%m-%d %H:%M")


def print_cleaning(df: pd.DataFrame, report: CleaningReport, symbol: str) -> None:
    print(f"Bybit spot {symbol} 1m: получено свечей {report.received}")
    print("Очистка данных:")
    order = "от новых к старым, отсортировано по времени" if report.descending else "по возрастанию"
    print(f"  порядок в ответе:            {order}")
    print(f"  незакрытая текущая свеча:    {report.unclosed} (отброшена)")
    print(f"  битые бары:                  {report.broken} (удалены)")
    print(f"  дубли:                       {report.duplicates} (удалены)")
    print(f"  нераспознанные строки:       {report.malformed} (удалены)")
    print(f"  пропуски минут:              {report.gaps} дыр, {report.missing_minutes} мин (не заполняем)")
    print(f"  осталось баров:              {report.kept}", end="")
    if report.kept:
        print(f", {fmt_time(df['time'].iloc[0])} .. {fmt_time(df['time'].iloc[-1])} UTC")
    else:
        print()


def print_signals(df: pd.DataFrame, args: argparse.Namespace) -> None:
    in_warmup = df.index < args.warmup
    buys, sells = df["buy"], df["sell"]
    print(f"\nИндикаторы: EMA {args.fast} / EMA {args.slow}, засев {args.ema_seed}, прогрев {args.warmup} баров")
    if len(df) > args.warmup:
        print(f"  торгуем с бара {fmt_time(df['time'].iloc[args.warmup])} UTC")
    else:
        print("  баров меньше, чем прогрев: сделок не будет")
    print(
        f"  сигналы в прогреве (не торгуем, могут отличаться от TV): "
        f"BUY {int(buys[in_warmup].sum())}, SELL {int(sells[in_warmup].sum())}"
    )
    print(f"  сигналы после прогрева: BUY {int(buys[~in_warmup].sum())}, SELL {int(sells[~in_warmup].sum())}")


def export_csv(df: pd.DataFrame, path: Path, warmup: int) -> None:
    """Бары с EMA и сигналами: для сверки с экспортом графика TradingView бар за баром."""
    out = df[["time", "open", "high", "low", "close", "volume", "ema_fast", "ema_slow", "buy", "sell"]].copy()
    out["warmup"] = out.index < warmup
    out.to_csv(path, index=False)
    print(f"  бары с EMA и сигналами выгружены в {path}")


# --------------------------------------------------------------------------- запуск


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EMA Crossover (Pine Script v5) на данных Bybit spot")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--limit", type=int, default=1000, help="сколько свечей запросить (1..1000)")
    p.add_argument("--base-url", default=BYBIT_BASE_URL, help="например https://api.bytick.com")
    p.add_argument("--fast", type=int, default=9, help="Fast EMA")
    p.add_argument("--slow", type=int, default=21, help="Slow EMA")
    p.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="сколько первых баров не торговать, пока EMA не забудет засев (по умолчанию 10 * slow)",
    )
    p.add_argument("--ema-seed", choices=["sma", "first"], default="sma", help="чем засевать EMA")
    p.add_argument("--from-file", type=Path, help="взять сохранённый ответ Bybit (JSON) вместо запроса")
    p.add_argument("--save-raw", type=Path, help="сохранить сырой ответ Bybit в JSON")
    p.add_argument("--csv", type=Path, help="выгрузить бары с EMA и сигналами в CSV")
    args = p.parse_args(argv)
    if not 1 <= args.limit <= 1000:
        p.error("--limit должен быть от 1 до 1000")
    if args.fast < 1 or args.slow < 1:
        p.error("длины EMA должны быть >= 1")
    if args.warmup is None:
        args.warmup = 10 * args.slow
    if args.warmup < 0:
        p.error("--warmup не может быть отрицательным")
    return args


async def load_payload(args: argparse.Namespace) -> dict:
    if args.from_file:
        return json.loads(args.from_file.read_text(encoding="utf-8"))
    async with httpx.AsyncClient(base_url=args.base_url, timeout=10.0) as client:
        payload = await fetch_klines(client, args.symbol, args.limit)
    if args.save_raw:
        args.save_raw.write_text(json.dumps(payload), encoding="utf-8")
    return payload


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = await load_payload(args)
    except (httpx.HTTPError, BybitError) as exc:
        print(f"Не удалось загрузить свечи с Bybit: {exc!r}", file=sys.stderr)
        return 1

    df, report = clean_klines(payload)
    print_cleaning(df, report, args.symbol)

    df = add_signals(df, args.fast, args.slow, args.ema_seed)
    print_signals(df, args)
    if args.csv:
        export_csv(df, args.csv, args.warmup)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
