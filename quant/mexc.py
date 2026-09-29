"""MEXC vadeli (contract) REST istemcisi.

İmza (resmî demo: github.com/mexcdevelop/mexc-api-demo, futures/signer):
  Signature = HMAC-SHA256(secret, apiKey + Request-Time + paramString), küçük harf hex
  GET/DELETE: paramString = anahtara göre sıralı "k=v&k2=v2" (URL kodlaması yok)
  POST:       paramString = gönderilen JSON gövdesinin birebir aynısı
Başlıklar: ApiKey, Request-Time (ms), Signature, Recv-Window, Content-Type: application/json
"""
import hashlib
import hmac
import json
import time
from urllib.parse import urlencode

import requests

PUBLIC_BASE = "https://contract.mexc.com"   # mum/kontrat verisi (research/stock_futures.py ile doğrulandı)
PRIVATE_BASE = "https://api.mexc.com"       # resmî entegrasyon rehberindeki OPEN-API adresi

INTERVALS = {"Min1": 60, "Min5": 300, "Min15": 900, "Min30": 1800, "Min60": 3600,
             "Hour4": 14400, "Hour8": 28800, "Day1": 86400}

# Emir sabitleri
OPEN_LONG, CLOSE_SHORT, OPEN_SHORT, CLOSE_LONG = 1, 2, 3, 4
MARKET = 5
ISOLATED, CROSS = 1, 2


class MexcError(RuntimeError):
    pass


def query_string(params):
    return "&".join(f"{k}={params[k]}" for k in sorted(params) if params[k] is not None)


def sign(key, secret, req_time, param_string):
    return hmac.new(secret.encode(), f"{key}{req_time}{param_string}".encode(), hashlib.sha256).hexdigest()


class Mexc:
    def __init__(self, key="", secret="", public_base=PUBLIC_BASE, private_base=PRIVATE_BASE,
                 recv_window=10000, session=None):
        self.key, self.secret = key, secret
        self.public_base, self.private_base = public_base.rstrip("/"), private_base.rstrip("/")
        self.recv_window = recv_window
        self.http = session or requests.Session()
        self._details = {}

    # ------------------------------------------------------------ taşıma
    def _unwrap(self, r):
        try:
            j = r.json()
        except ValueError:
            raise MexcError(f"HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code >= 400 or not j.get("success", False):
            raise MexcError(f"{j.get('code')} {j.get('message')}")
        return j.get("data")

    def public(self, path, params=None, tries=3):
        for i in range(tries):
            try:
                r = self.http.get(self.public_base + path, params=params, timeout=20)
                if r.status_code == 429:
                    time.sleep(2 * (i + 1))
                    continue
                return self._unwrap(r)
            except requests.RequestException as e:
                if i == tries - 1:
                    raise MexcError(f"ağ hatası: {e}")
                time.sleep(2 * (i + 1))
        raise MexcError("istek limiti (429)")

    def private(self, method, path, params=None):
        if not (self.key and self.secret):
            raise MexcError("MEXC_API_KEY / MEXC_API_SECRET tanımlı değil")
        params = {k: v for k, v in (params or {}).items() if v is not None}
        ts = str(int(time.time() * 1000))
        url = self.private_base + path
        if method == "POST":
            body = json.dumps(params, separators=(",", ":"))
            sig, data = sign(self.key, self.secret, ts, body), body
        else:
            sig, data = sign(self.key, self.secret, ts, query_string(params)), None
            if params:
                url += "?" + urlencode(sorted(params.items()))
        headers = {"ApiKey": self.key, "Request-Time": ts, "Signature": sig,
                   "Recv-Window": str(self.recv_window), "Content-Type": "application/json"}
        try:
            r = self.http.request(method, url, data=data, headers=headers, timeout=20)
        except requests.RequestException as e:
            raise MexcError(f"ağ hatası: {e}")
        return self._unwrap(r)

    # ------------------------------------------------------------ piyasa (herkese açık)
    def detail(self, symbol):
        """contractSize (1 kontrat = kaç coin), minVol, volUnit, priceUnit, maxLeverage."""
        if symbol not in self._details:
            d = self.public("/api/v1/contract/detail", {"symbol": symbol})
            if isinstance(d, list):
                d = next((x for x in d if x.get("symbol") == symbol), None)
            if not d:
                raise MexcError(f"kontrat bulunamadı: {symbol}")
            self._details[symbol] = d
        return self._details[symbol]

    def klines(self, symbol, interval, start, end):
        """[(t_açılış_sn, o, h, l, c, hacim)] — start/end saniye."""
        step = INTERVALS[interval]
        out, t = {}, start
        while t < end:
            t2 = min(end, t + 1900 * step)
            d = self.public(f"/api/v1/contract/kline/{symbol}", {"interval": interval, "start": t, "end": t2}) or {}
            for i, ts in enumerate(d.get("time") or []):
                out[int(ts)] = (int(ts), float(d["open"][i]), float(d["high"][i]), float(d["low"][i]),
                                float(d["close"][i]), float(d["vol"][i]))
            t = t2
            if t < end:
                time.sleep(0.15)
        return [out[k] for k in sorted(out)]

    def ticker(self, symbol):
        return float(self.public("/api/v1/contract/ticker", {"symbol": symbol})["lastPrice"])

    # ------------------------------------------------------------ hesap (imzalı)
    def equity(self, currency="USDT"):
        for a in self.private("GET", "/api/v1/private/account/assets") or []:
            if a.get("currency") == currency:
                return float(a.get("equity") or 0), float(a.get("availableBalance") or 0)
        return 0.0, 0.0

    def positions(self, symbol=None):
        """[{symbol, side: 'long'|'short', vol, entry, positionId, leverage, openType}]"""
        rows = self.private("GET", "/api/v1/private/position/open_positions", {"symbol": symbol}) or []
        out = []
        for p in rows:
            vol = float(p.get("holdVol") or 0)
            if vol <= 0:
                continue
            out.append({"symbol": p["symbol"], "side": "long" if int(p["positionType"]) == 1 else "short",
                        "vol": vol, "entry": float(p.get("holdAvgPrice") or p.get("openAvgPrice") or 0),
                        "positionId": p.get("positionId"), "leverage": p.get("leverage"),
                        "openType": p.get("openType")})
        return out

    def market_order(self, symbol, side, vol, price, leverage, open_type=ISOLATED, stop_loss=None,
                     position_id=None, tag=None):
        body = {"symbol": symbol, "price": price, "vol": vol, "leverage": leverage, "side": side,
                "type": MARKET, "openType": open_type, "positionId": position_id,
                "stopLossPrice": stop_loss, "externalOid": tag}
        d = self.private("POST", "/api/v1/private/order/create", body)
        return d.get("orderId") if isinstance(d, dict) else d

    def cancel_stops(self, symbol):
        """Pozisyon kapandıktan sonra borsada asılı kalan SL/TP emirlerini temizler."""
        return self.private("POST", "/api/v1/private/stoporder/cancel_all", {"symbol": symbol})
