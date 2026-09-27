"""Veri: Helius (cüzdan işlem geçmişi) + GeckoTerminal (günlük token fiyatları). Yalnız OKUR, hiçbir işlem göndermez.

Ortam değişkenleri: HELIUS_API_KEY (zorunlu), MASTBOUND_CACHE (varsayılan ./cache)
"""
import json
import os
import time

import requests

from .analysis import DAY, Series, Trade

SOL = "So11111111111111111111111111111111111111112"
USD_MINTS = {"EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": "USDC",
             "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB": "USDT"}
HELIUS_HOSTS = ["https://api-mainnet.helius-rpc.com/v0", "https://api.helius.xyz/v0"]
LAST = {"status": None, "error": None}
GT = "https://api.geckoterminal.com/api/v2"
CACHE = os.environ.get("MASTBOUND_CACHE", "cache")
_last_gt = [0.0]


def _get(url, params=None, gt=False, tries=4):
    for i in range(tries):
        if gt:                                   # GeckoTerminal: ~30 istek/dk
            wait = 2.1 - (time.time() - _last_gt[0])
            if wait > 0:
                time.sleep(wait)
            _last_gt[0] = time.time()
        try:
            r = requests.get(url, params=params, timeout=30, headers={"Accept": "application/json"})
            LAST["status"], LAST["error"] = r.status_code, None
            if r.status_code == 404:
                return None
            if r.status_code in (401, 403):
                LAST["error"] = r.text[:200]
                return None
            if r.status_code == 429:
                time.sleep(5 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            LAST["status"], LAST["error"] = None, str(e)[:200]
            time.sleep(2 * (i + 1))
    return None


class DataError(RuntimeError):
    pass


def fetch_transactions(address, api_key, max_pages=10):
    """Cüzdanın SWAP işlemleri (en yeniden eskiye), Helius ayrıştırılmış biçimde."""
    out, before, host = [], None, None
    for _ in range(max_pages):
        params = {"api-key": api_key, "type": "SWAP", "limit": 100}
        if before:
            params["before"] = before
        page = None
        for h in ([host] if host else HELIUS_HOSTS):
            page = _get(f"{h}/addresses/{address}/transactions", params)
            if page is not None:
                host = h
                break
        if page is None:
            if not out and LAST["status"] not in (200, 404):
                raise DataError(f"Helius yanıtı: {LAST['status']} {LAST['error'] or ''}".strip())
            break
        if not page:
            break
        out += page
        before = page[-1].get("signature")
        if len(page) < 100:
            break
    return out


def to_trades(address, txs, sol_prices):
    """Her işlemde cüzdanın net token değişimlerinden alım/satım çıkarır.
    Karşı taraf SOL ya da USD stablecoin ise USD değeri hesaplanır; diğer (token↔token) işlemler atlanır."""
    trades = []
    sol_series = sol_prices if isinstance(sol_prices, Series) else Series(sol_prices)
    for tx in txs:
        ts = tx.get("timestamp")
        if not ts:
            continue
        delta = {}
        for tt in tx.get("tokenTransfers") or []:
            amt = float(tt.get("tokenAmount") or 0)
            m = tt.get("mint")
            if tt.get("toUserAccount") == address:
                delta[m] = delta.get(m, 0.0) + amt
            if tt.get("fromUserAccount") == address:
                delta[m] = delta.get(m, 0.0) - amt
        sol = delta.pop(SOL, 0.0)
        for nt in tx.get("nativeTransfers") or []:
            amt = float(nt.get("amount") or 0) / 1e9
            if nt.get("toUserAccount") == address:
                sol += amt
            if nt.get("fromUserAccount") == address:
                sol -= amt
        usd_q = sum(delta.pop(m, 0.0) for m in list(delta) if m in USD_MINTS)
        others = {m: v for m, v in delta.items() if abs(v) > 0}
        if len(others) != 1:
            continue
        mint, qty = next(iter(others.items()))
        solp = sol_series.at_or_before(ts, 2 * DAY)
        quote_usd = usd_q + (sol * solp if solp else 0.0)
        if qty < 0 and quote_usd > 0:
            trades.append(Trade(ts, mint, "sell", -qty, quote_usd))
        elif qty > 0 and quote_usd < 0:
            trades.append(Trade(ts, mint, "buy", qty, -quote_usd))
    return trades


def _cache_path(name):
    os.makedirs(CACHE, exist_ok=True)
    return os.path.join(CACHE, name)


def _ohlcv(pool, token_side, tf):
    o = _get(f"{GT}/networks/solana/pools/{pool}/ohlcv/{tf}",
             {"limit": 1000, "currency": "usd", "token": token_side}, gt=True)
    return {int(row[0]): float(row[4]) for row in ((o or {}).get("data") or {}).get("attributes", {}).get("ohlcv_list", [])}


def prices(mint, since_ts=None, max_age_h=1):
    """{unix_saniye: kapanış USD}: son ~41 gün saatlik, daha eskisi (gerekirse) günlük. Önbellekli."""
    p = _cache_path(f"px2_{mint}.json")
    if os.path.exists(p) and time.time() - os.path.getmtime(p) < max_age_h * 3600:
        return {int(k): v for k, v in json.load(open(p)).items()}
    j = _get(f"{GT}/networks/solana/tokens/{mint}/pools", {"page": 1}, gt=True)
    pools = (j or {}).get("data") or []
    series = {}
    if pools:
        best = max(pools, key=lambda d: float(d["attributes"].get("reserve_in_usd") or 0))
        side = "base" if best["relationships"]["base_token"]["data"]["id"].endswith(mint) else "quote"
        pool = best["attributes"]["address"]
        hourly = _ohlcv(pool, side, "hour")
        first_hour = min(hourly) if hourly else time.time()
        if since_ts is None or since_ts < first_hour:
            series.update({k: v for k, v in _ohlcv(pool, side, "day").items() if k < first_hour})
        series.update(hourly)
    json.dump(series, open(p, "w"))
    return series


daily_prices = prices          # geriye dönük ad


def symbol_of(mint):
    p = _cache_path("symbols.json")
    cache = json.load(open(p)) if os.path.exists(p) else {}
    if mint not in cache:
        j = _get(f"{GT}/networks/solana/tokens/{mint}", gt=True)
        cache[mint] = (((j or {}).get("data") or {}).get("attributes") or {}).get("symbol") or mint[:6]
        json.dump(cache, open(p, "w"))
    return cache[mint]


def wallet_report(address, api_key, max_tokens=15, days=None):
    """days: yalnız son N günün işlemleri (kademe sınırı); None = tüm geçmiş."""
    from .analysis import analyze
    sol_px = prices(SOL)
    trades = to_trades(address, fetch_transactions(address, api_key), sol_px)
    if days:
        trades = [t for t in trades if t.ts >= time.time() - days * DAY]
    # en çok işlem yapılan token'lar (istek sınırı için)
    counts = {}
    for t in trades:
        counts[t.mint] = counts.get(t.mint, 0) + 1
    keep = sorted(counts, key=counts.get, reverse=True)[:max_tokens]
    oldest = {}
    for t in trades:
        oldest[t.mint] = min(oldest.get(t.mint, t.ts), t.ts)
    px = {m: prices(m, oldest[m]) for m in keep}
    trades = [t for t in trades if t.mint in px]
    for t in trades:
        t.symbol = symbol_of(t.mint)
    return analyze(trades, px)
