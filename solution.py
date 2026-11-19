#!/usr/bin/env python3
"""Стратегия EMA Crossover из TradingView (Pine Script v5) на асинхронном Python.

1. Асинхронно (httpx) загружает последние 1000 минутных свечей BTCUSDT
   со спотового REST API Bybit (/v5/market/kline) одним запросом.
2. Чистит данные (clean_klines): порядок, дубли, битые и незакрытая свеча, пропуски.
3. Считает EMA 9/21 и пересечения так же, как ta.ema / ta.crossover / ta.crossunder.
4. Прогоняет Spot Long-only: сигнал по Close бара T, сделка по Open бара T + 1,
   вход 1000 USDT, комиссия 0.1% на вход и 0.18% на выход.
5. Печатает сводку очистки, закрытые сделки и итоговую статистику.

Запуск: python solution.py (параметры: python solution.py --help).
Решения и известные расхождения с TradingView описаны в README.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

BYBIT_BASE_URL = "https://api.bybit.com"
KLINE_PATH = "/v5/market/kline"
BAR_MS = 60_000  # таймфрейм 1 минута

ORDER_USDT = 1000.0  # фиксированный размер входа
FEE_ENTRY = 0.001  # 0.1% от суммы покупки
FEE_EXIT = 0.0018  # 0.18% от суммы продажи
SLIPPAGE_LIMIT = 0.005  # предупреждаем, если исполнение дальше 0.5% от сигнальной Close


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

    try:
        payload = resp.json()
    except ValueError as exc:  # например, HTML-страница вместо JSON
        raise BybitError(f"ответ не JSON: {resp.text[:200]!r}") from exc
    check_payload(payload)
    return payload


def check_payload(payload: dict) -> None:
    """retCode == 0 и есть result.list. Проверяем и живой ответ, и файл из --from-file."""
    if not isinstance(payload, dict) or payload.get("retCode") != 0:
        code, msg = (payload.get("retCode"), payload.get("retMsg")) if isinstance(payload, dict) else (None, None)
        raise BybitError(f"retCode={code} retMsg={msg!r}")
    if not isinstance((payload.get("result") or {}).get("list"), list):
        raise BybitError("в ответе нет result.list")


# --------------------------------------------------------------------------- очистка


@dataclass
class CleaningReport:
    received: int = 0
    order: str = ""  # в каком порядке пришли свечи
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
    if len(df) < 2 or df["ts"].is_monotonic_increasing:
        report.order = "по возрастанию времени"
    elif df["ts"].is_monotonic_decreasing:
        report.order = "от новых к старым, отсортировано по времени"
    else:
        report.order = "вперемешку, отсортировано по времени"

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


# --------------------------------------------------------------------------- исполнение


def trade_pnl(
    entry_price: float,
    exit_price: float,
    order_usdt: float = ORDER_USDT,
    fee_entry: float = FEE_ENTRY,
    fee_exit: float = FEE_EXIT,
) -> tuple[float, float, float, float, float]:
    """Комиссии берём от фактической суммы каждой сделки.

    Вход: тратим order_usdt, комиссия входа order_usdt * fee_entry удерживается
    из покупки, поэтому монет получаем qty = (order_usdt - комиссия) / entry_price.
    Выход: продаём все qty, выручка qty * exit_price, комиссия выхода выручка * fee_exit.
    Чистый PnL = выручка - комиссия выхода - order_usdt; в процентах от order_usdt.

    Возвращает (qty, комиссия входа, комиссия выхода, PnL USDT, PnL %).
    """
    fee_in = order_usdt * fee_entry
    qty = (order_usdt - fee_in) / entry_price
    proceeds = qty * exit_price
    fee_out = proceeds * fee_exit
    pnl = proceeds - fee_out - order_usdt
    return qty, fee_in, fee_out, pnl, pnl / order_usdt * 100


@dataclass
class Trade:
    entry_time: pd.Timestamp  # бар T + 1 после сигнала BUY
    entry_signal_price: float  # Close бара T
    entry_price: float  # Open бара T + 1
    exit_time: pd.Timestamp
    exit_signal_price: float
    exit_price: float
    qty: float
    fee_entry: float
    fee_exit: float
    pnl_usdt: float
    pnl_pct: float
    entry_slippage: float  # цена исполнения / сигнальная Close - 1
    exit_slippage: float
    forced_exit: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    warmup_signals: int = 0  # сигналы в зоне прогрева, не торгуем
    ignored_signals: int = 0  # BUY в позиции или SELL без позиции
    slippage_warnings: int = 0
    last_bar_note: str | None = None  # сигнал на последнем баре, исполнить нечем


def backtest(
    df: pd.DataFrame,
    warmup: int = 0,
    order_usdt: float = ORDER_USDT,
    fee_entry: float = FEE_ENTRY,
    fee_exit: float = FEE_EXIT,
    slippage_limit: float = SLIPPAGE_LIMIT,
    warn: Callable[[str], None] = print,
) -> BacktestResult:
    """Spot Long-only: BUY открывает позицию на order_usdt, SELL продаёт всё в кэш.

    Защита от look-ahead: сигнал бара T считается по его Close и известен только
    после закрытия бара, поэтому сделка идёт по Open бара T + 1. На шаге i цикл
    смотрит только на сигнал бара i и цену открытия бара i + 1.
    - BUY в позиции и SELL без позиции игнорируются (позиция одна, шорта нет).
    - Сигналы первых warmup баров не торгуем: там EMA ещё помнит засев.
    - Сигнал на последнем баре исполнить нечем, бара T + 1 нет. BUY пропускаем,
      SELL закрывает позицию по Close этого бара вместе с принудительным закрытием.
    - Если на конец данных позиция открыта, закрываем её по последней цене (Close
      последнего закрытого бара) и помечаем сделку.
    - Если исполнение дальше slippage_limit от сигнальной Close, печатаем
      предупреждение и помечаем сделку.
    """
    res = BacktestResult()
    n = len(df)
    times = df["time"].tolist()
    ts = df["ts"].to_numpy()
    opens = df["open"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    buys = df["buy"].to_numpy(dtype=bool)
    sells = df["sell"].to_numpy(dtype=bool)

    def check_slippage(signal_i: int, exec_i: int, side: str) -> float:
        slip = opens[exec_i] / closes[signal_i] - 1
        if abs(slip) > slippage_limit:
            res.slippage_warnings += 1
            warn(
                f"ПРЕДУПРЕЖДЕНИЕ: проскальзывание {side} {slip:+.2%}: "
                f"сигнал {fmt_time(times[signal_i])} Close {closes[signal_i]:.2f}, "
                f"исполнение {fmt_time(times[exec_i])} Open {opens[exec_i]:.2f}"
            )
        return slip

    def close_position(pos: tuple[int, int, float], exit_signal_i: int, exit_i: int, forced: bool) -> Trade:
        sig_in, in_i, slip_in = pos
        exit_price = closes[exit_i] if forced else opens[exit_i]
        slip_out = 0.0 if forced else check_slippage(exit_signal_i, exit_i, "выхода")
        qty, fee_in, fee_out, pnl, pct = trade_pnl(opens[in_i], exit_price, order_usdt, fee_entry, fee_exit)
        notes = []
        if abs(slip_in) > slippage_limit:
            notes.append(f"проскальзывание входа {slip_in:+.2%}")
        if abs(slip_out) > slippage_limit:
            notes.append(f"проскальзывание выхода {slip_out:+.2%}")
        if ts[in_i] - ts[sig_in] > BAR_MS:
            notes.append("вход после пропуска минут")
        if not forced and ts[exit_i] - ts[exit_signal_i] > BAR_MS:
            notes.append("выход после пропуска минут")
        if forced:
            notes.append("закрыта принудительно по последней цене")
        return Trade(
            entry_time=times[in_i],
            entry_signal_price=closes[sig_in],
            entry_price=opens[in_i],
            exit_time=times[exit_i],
            exit_signal_price=closes[exit_signal_i],
            exit_price=exit_price,
            qty=qty,
            fee_entry=fee_in,
            fee_exit=fee_out,
            pnl_usdt=pnl,
            pnl_pct=pct,
            entry_slippage=slip_in,
            exit_slippage=slip_out,
            forced_exit=forced,
            notes=notes,
        )

    start = min(warmup, n)
    res.warmup_signals = int(buys[:start].sum() + sells[:start].sum())
    position: tuple[int, int, float] | None = None  # (бар сигнала, бар входа, проскальзывание входа)

    for i in range(start, n):
        if buys[i]:
            if position is not None:
                res.ignored_signals += 1
            elif i + 1 == n:
                res.last_bar_note = f"BUY на последнем баре {fmt_time(times[i])}: бара T+1 нет, вход не выполнен"
            else:
                position = (i, i + 1, check_slippage(i, i + 1, "входа"))
        elif sells[i]:
            if position is None:
                res.ignored_signals += 1
            elif i + 1 == n:
                res.last_bar_note = (
                    f"SELL на последнем баре {fmt_time(times[i])}: бара T+1 нет, "
                    "позиция закрыта по Close этого бара"
                )
            else:
                res.trades.append(close_position(position, i, i + 1, forced=False))
                position = None

    if position is not None:
        res.trades.append(close_position(position, n - 1, n - 1, forced=True))
    return res


# --------------------------------------------------------------------------- вывод


def fmt_time(t: pd.Timestamp) -> str:
    return t.strftime("%Y-%m-%d %H:%M")


def print_cleaning(df: pd.DataFrame, report: CleaningReport, symbol: str) -> None:
    print(f"Bybit spot {symbol} 1m: получено свечей {report.received}")
    print("Очистка данных:")
    print(f"  порядок в ответе:            {report.order}")
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


def print_trades(res: BacktestResult) -> None:
    print("Закрытые сделки (время UTC; вход и выход по Open бара, следующего за сигналом):")
    if not res.trades:
        print("  сделок нет")
        return
    print(
        f"  {'#':>3}  {'вход':16}  {'цена':>10}  {'выход':16}  {'цена':>10}  "
        f"{'PnL USDT':>9}  {'PnL %':>7}  {'проск. вх/вых':>15}  отметки"
    )
    for k, t in enumerate(res.trades, 1):
        slip = f"{t.entry_slippage:+.2%}/" + ("-" if t.forced_exit else f"{t.exit_slippage:+.2%}")
        print(
            f"  {k:>3}  {fmt_time(t.entry_time)}  {t.entry_price:>10.2f}  {fmt_time(t.exit_time)}  "
            f"{t.exit_price:>10.2f}  {t.pnl_usdt:>+9.2f}  {t.pnl_pct:>+7.2f}  {slip:>15}  "
            f"{', '.join(t.notes) or '-'}"
        )


def print_summary(res: BacktestResult, order_usdt: float = ORDER_USDT) -> None:
    trades = res.trades
    total = sum(t.pnl_usdt for t in trades)
    fees = sum(t.fee_entry + t.fee_exit for t in trades)
    wins = sum(t.pnl_usdt > 0 for t in trades)
    forced = sum(t.forced_exit for t in trades)
    print("\nИтого:")
    print(f"  сделок: {len(trades)} (из них закрыта принудительно в конце данных: {forced})")
    if trades:
        print(f"  Winrate: {wins / len(trades):.1%} ({wins} из {len(trades)} с чистым PnL > 0)")
    else:
        print("  Winrate: нет сделок")
    print(f"  суммарный чистый PnL: {total:+.2f} USDT (каждая сделка на {order_usdt:.0f} USDT)")
    print(f"  в том числе комиссии: {fees:.2f} USDT")
    print(f"  предупреждений о проскальзывании > {SLIPPAGE_LIMIT:.1%}: {res.slippage_warnings}")
    print(f"  проигнорировано сигналов (BUY в позиции, SELL без позиции): {res.ignored_signals}")
    if res.last_bar_note:
        print(f"  {res.last_bar_note}")


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
        try:
            payload = json.loads(args.from_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BybitError(f"не удалось прочитать {args.from_file}: {exc}") from exc
        check_payload(payload)
        return payload
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

    print()
    res = backtest(df, args.warmup)
    print_trades(res)
    print_summary(res)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
