"""Загрузка и очистка свечей. Сеть не нужна: ответы Bybit синтетические."""

import asyncio

import httpx
import pytest

import solution
from solution import BAR_MS, BybitError, clean_klines, fetch_klines

T0 = 1_699_999_980_000  # кратно минуте


def bar(i: int, o="100", h="101", low="99", c="100.5") -> list[str]:
    return [str(T0 + i * BAR_MS), o, h, low, c, "1.5", "150"]


def payload(rows: list, now_ms: int) -> dict:
    return {"retCode": 0, "retMsg": "OK", "result": {"category": "spot", "list": rows}, "time": now_ms}


def test_clean_dirty_response():
    rows = [
        bar(7),  # текущая минута, ещё не закрыта
        bar(6),
        bar(5),
        bar(5),  # дубль
        # бара 4 нет вовсе
        bar(3, o="0"),  # нулевая цена
        bar(2, h="98"),  # high < low
        bar(1),
        bar(0),
        ["abc"],  # не разбирается
    ]
    now = T0 + 7 * BAR_MS + 30_000
    df, rep = clean_klines(payload(rows, now))

    assert rep.received == 9
    assert rep.order.startswith("от новых к старым")
    assert rep.unclosed == 1
    assert rep.duplicates == 1
    assert rep.broken == 2
    assert rep.malformed == 1
    assert list(df["ts"]) == [T0, T0 + BAR_MS, T0 + 5 * BAR_MS, T0 + 6 * BAR_MS]
    assert rep.kept == 4
    # бары 2 и 3 выброшены как битые, бара 4 не было: одна дыра в 3 минуты
    assert rep.gaps == 1
    assert rep.missing_minutes == 3


def test_closed_candle_is_kept_at_exact_boundary():
    # свеча 0 закрывается ровно в T0 + 60 с
    _, rep = clean_klines(payload([bar(0)], now_ms=T0 + BAR_MS))
    assert rep.unclosed == 0 and rep.kept == 1


def test_uses_server_time_from_response():
    _, rep = clean_klines(payload([bar(1), bar(0)], now_ms=T0 + BAR_MS + 1))
    assert rep.unclosed == 1


def test_broken_copy_does_not_push_out_valid_duplicate():
    rows = [bar(1), bar(0, o="0"), bar(0)]
    df, rep = clean_klines(payload(rows, now_ms=T0 + 10 * BAR_MS))
    assert rep.broken == 1
    assert rep.duplicates == 0
    assert list(df["open"]) == [100.0, 100.0]


def test_open_outside_range_and_nan_are_broken():
    rows = [bar(2, o="102"), bar(1, c="nan"), bar(0)]
    _, rep = clean_klines(payload(rows, now_ms=T0 + 10 * BAR_MS))
    assert rep.broken == 2 and rep.kept == 1


def run_fetch(handler) -> dict:
    async def go():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url="https://bybit.test") as client:
            return await fetch_klines(client)

    return asyncio.run(go())


def test_fetch_sends_spot_1m_params():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.url.params)
        return httpx.Response(200, json=payload([bar(0)], T0 + BAR_MS))

    data = run_fetch(handler)
    assert seen == {"category": "spot", "symbol": "BTCUSDT", "interval": "1", "limit": "1000"}
    assert data["result"]["list"] == [bar(0)]


def test_fetch_raises_on_ret_code():
    def handler(request):
        return httpx.Response(200, json={"retCode": 10001, "retMsg": "params error", "result": {}})

    with pytest.raises(BybitError, match="10001"):
        run_fetch(handler)


def test_fetch_retries_5xx(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr(solution.asyncio, "sleep", no_sleep)
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(502)
        return httpx.Response(200, json=payload([bar(0)], T0 + BAR_MS))

    run_fetch(handler)
    assert len(calls) == 2


def test_fetch_does_not_retry_4xx():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(403)

    with pytest.raises(httpx.HTTPStatusError):
        run_fetch(handler)
    assert len(calls) == 1


def test_fetch_non_json_is_bybit_error():
    def handler(request):
        return httpx.Response(200, text="<html>blocked</html>")

    with pytest.raises(BybitError, match="не JSON"):
        run_fetch(handler)


def test_mixed_order_is_reported():
    _, rep = clean_klines(payload([bar(1), bar(0), bar(2)], now_ms=T0 + 10 * BAR_MS))
    assert rep.order.startswith("вперемешку")
