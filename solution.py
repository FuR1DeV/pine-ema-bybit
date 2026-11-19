#!/usr/bin/env python3
"""Стратегия EMA Crossover из TradingView (Pine Script v5) на асинхронном Python.

1. Асинхронно (httpx) загружает последние 1000 минутных свечей BTCUSDT
   со спотового REST API Bybit (/v5/market/kline) одним запросом.
2. Чистит данные (clean_klines): порядок, дубли, битые и незакрытая свеча, пропуски.

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


# --------------------------------------------------------------------------- запуск


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EMA Crossover (Pine Script v5) на данных Bybit spot")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--limit", type=int, default=1000, help="сколько свечей запросить (1..1000)")
    p.add_argument("--base-url", default=BYBIT_BASE_URL, help="например https://api.bytick.com")
    p.add_argument("--from-file", type=Path, help="взять сохранённый ответ Bybit (JSON) вместо запроса")
    p.add_argument("--save-raw", type=Path, help="сохранить сырой ответ Bybit в JSON")
    args = p.parse_args(argv)
    if not 1 <= args.limit <= 1000:
        p.error("--limit должен быть от 1 до 1000")
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
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
