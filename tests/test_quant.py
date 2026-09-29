import hashlib
import hmac
import json
import math
import threading
import time
import urllib.request

import pytest

from quant import mexc as mx
from quant.backtest import run
from quant.engine import Config, Engine, round_price
from quant.strategy import Params, decide, ema, indicators, size

STEP = 900
DETAIL = {"symbol": "BTC_USDT", "contractSize": 0.0001, "minVol": 1, "volUnit": 1, "priceUnit": 0.1}


def candles_from(closes, t0=1_700_000_000):
    out, prev = [], closes[0]
    for i, c in enumerate(closes):
        out.append((t0 + i * STEP, prev, max(prev, c) * 1.001, min(prev, c) * 0.999, c, 10.0))
        prev = c
    return out


def wave(n=900):
    """Yavaş trendler + gürültü: kesişim üretir."""
    return [100 * (1 + 0.25 * math.sin(i / 60)) + 0.6 * math.sin(i * 1.7) for i in range(n)]


# ------------------------------------------------------------------ imza
def test_sign_matches_official_scheme():
    assert mx.query_string({"symbol": "BTC_USDT", "a": 1, "z": None}) == "a=1&symbol=BTC_USDT"
    want = hmac.new(b"sec", b"key1700000000000a=1&symbol=BTC_USDT", hashlib.sha256).hexdigest()
    assert mx.sign("key", "sec", "1700000000000", "a=1&symbol=BTC_USDT") == want


class FakeResp:
    def __init__(self, data, code=200):
        self.status_code, self._d, self.text = code, data, json.dumps(data)

    def json(self):
        return self._d


class FakeSession:
    def __init__(self, data):
        self.data, self.calls = data, []

    def request(self, method, url, data=None, headers=None, timeout=None):
        self.calls.append((method, url, data, headers))
        return FakeResp({"success": True, "code": 0, "data": self.data})


def test_post_signs_exact_body_and_drops_none():
    s = FakeSession({"orderId": 42})
    api = mx.Mexc("k", "s", session=s)
    assert api.market_order("BTC_USDT", mx.OPEN_LONG, 3, 60000, 3, stop_loss=59000) == 42
    method, url, body, h = s.calls[0]
    assert method == "POST" and url.endswith("/api/v1/private/order/create")
    assert "positionId" not in body and json.loads(body)["stopLossPrice"] == 59000
    assert h["Signature"] == mx.sign("k", "s", h["Request-Time"], body)


def test_get_signs_sorted_query():
    s = FakeSession([{"symbol": "BTC_USDT", "positionType": 2, "holdVol": 5, "holdAvgPrice": 100, "positionId": 7}])
    api = mx.Mexc("k", "s", session=s)
    p = api.positions("BTC_USDT")
    assert p[0]["side"] == "short" and p[0]["vol"] == 5
    _, url, _, h = s.calls[0]
    assert url.endswith("open_positions?symbol=BTC_USDT")
    assert h["Signature"] == mx.sign("k", "s", h["Request-Time"], "symbol=BTC_USDT")


def test_api_error_raises():
    class S(FakeSession):
        def request(self, *a, **k):
            return FakeResp({"success": False, "code": 602, "message": "Signature verification failed"})
    with pytest.raises(mx.MexcError, match="602"):
        mx.Mexc("k", "s", session=S(None)).equity()


# ------------------------------------------------------------------ strateji
def test_ema_and_cross_signal():
    assert ema([1, 1, 1], 5) == [1, 1, 1]
    p = Params(fast=3, slow=6, trend=0, atr=3)
    c = candles_from([10] * 20 + [9] * 5 + [12, 13, 14])
    ind = indicators(c, p)
    acts = [decide(c, ind, i, None, p).action for i in range(len(c))]
    assert "open_long" in acts and "open_short" in acts


def test_trailing_stop_closes_long():
    p = Params(fast=3, slow=6, trend=0, atr=3, stop_atr=2, trail_atr=1)
    c = candles_from([10] * 20 + [11, 12, 13, 14, 15, 13.5])
    ind = indicators(c, p)
    pos = {"side": "long", "entry": 11, "peak": 11, "atr0": ind.atr[20]}
    for i in range(20, len(c) - 1):
        assert decide(c, ind, i, pos, p).action == "hold"
    d = decide(c, ind, len(c) - 1, pos, p)
    assert d.action == "close" and d.reason == "iz süren stop"


def test_size_respects_risk_margin_and_minimum():
    # 1000 USDT, %1 risk = 10 USDT, stop 1000 uzakta → 0.01 BTC = 100 kontrat (0.0001 BTC)
    assert size(1000, 60000, 59000, DETAIL, 3, 1.0, 100) == 100
    # teminat sınırı: %5 * 3x * 1000 / 60000 = 0.0025 BTC = 25 kontrat
    assert size(1000, 60000, 59000, DETAIL, 3, 1.0, 5) == 25
    assert size(5, 60000, 59000, {**DETAIL, "minVol": 10}, 3, 1.0, 100) == 0


def test_backtest_runs_with_costs():
    c = candles_from(wave(3000))
    p = Params(fast=10, slow=30, trend=0, atr=14)
    free = run(c, p, fee_bps=0, slip_bps=0)
    paid = run(c, p, fee_bps=5, slip_bps=5)
    assert free["trades"] > 3 and free["trades"] == paid["trades"]
    assert paid["end_equity"] < free["end_equity"]
    assert 0 <= paid["max_dd_pct"] < 100


def test_round_price():
    assert round_price(59123.456, DETAIL) == 59123.5
    assert round_price(0.123456, {"priceUnit": 0.001}) == 0.123


# ------------------------------------------------------------------ motor (paper, sahte borsa)
class FakeApi:
    def __init__(self, closes):
        now = int(time.time()) // STEP * STEP
        self.c = candles_from(closes, now - len(closes) * STEP)

    def klines(self, symbol, interval, start, end):
        return [x for x in self.c if start <= x[0] <= end]

    def detail(self, symbol):
        return DETAIL

    def ticker(self, symbol):
        return self.c[-1][4]


def engine(tmp_path, closes, **kw):
    cfg = Config(symbols=["BTC_USDT"], params=Params(fast=3, slow=6, trend=0, atr=3), state_dir=str(tmp_path),
                 paper_equity=1000, max_margin_pct=100, **kw)
    return Engine(cfg, api=FakeApi(closes))


def test_engine_opens_paper_long_and_persists(tmp_path):
    e = engine(tmp_path, [10] * 30 + [9] * 5 + [12])
    e.process("BTC_USDT")
    pos = e.s["positions"]["BTC_USDT"]
    assert pos["side"] == "long" and pos["vol"] > 0 and pos["stop"] < 12
    assert e.s["paper"]["positions"]["BTC_USDT"]["vol"] == pos["vol"]
    e.save()
    again = engine(tmp_path, [10] * 40)
    assert again.s["positions"]["BTC_USDT"]["vol"] == pos["vol"]


def test_paused_engine_skips_entries_and_close_all(tmp_path):
    e = engine(tmp_path, [10] * 30 + [9] * 5 + [12])
    e.s["paused"] = True
    e.process("BTC_USDT")
    assert not e.s["positions"]
    e.s["paused"] = False
    e.process("BTC_USDT")
    assert e.s["positions"]
    e.command("close_all")
    e._commands()
    assert not e.s["positions"] and e.s["paused"] and e.s["trades"][-1]["reason"] == "panelden kapat"


def test_daily_loss_halts(tmp_path):
    e = engine(tmp_path, [10] * 40, max_daily_loss_pct=3)
    e.snapshot()
    e.s["paper"]["cash"] = 960
    e.snapshot()
    assert e.s["halted"]


def test_config_validation():
    with pytest.raises(ValueError):
        Config(leverage=50).validate()
    with pytest.raises(ValueError):
        Config(interval="5m").validate()


def test_panel_requires_token(tmp_path):
    from http.server import ThreadingHTTPServer
    from quant.panel import make_handler
    e = engine(tmp_path, [10] * 40)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(e, "gizli"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    try:
        assert b"Quant" in urllib.request.urlopen(base + "/").read()
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(base + "/api/status")
        assert err.value.code == 401
        req = urllib.request.Request(base + "/api/status", headers={"Authorization": "Bearer gizli"})
        assert json.loads(urllib.request.urlopen(req).read())["live"] is False
        req = urllib.request.Request(base + "/api/cmd", data=b'{"cmd":"pause"}', method="POST",
                                     headers={"Authorization": "Bearer gizli"})
        urllib.request.urlopen(req)
        assert e.cmds.get_nowait() == "pause"
    finally:
        srv.shutdown()
