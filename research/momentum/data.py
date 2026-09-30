"""USDT perpetual 1dk OHLCV + funding indirme (ccxt, public API; borsa config.EXCHANGE) ve resample."""
import os
import time

import pandas as pd

import config as C

OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def symbol(coin):
    return f"{coin}/{C.QUOTE}:{C.QUOTE}"


def _exchange():
    import ccxt
    return getattr(ccxt, C.EXCHANGE)({"enableRateLimit": True,
                                      "options": {"defaultType": "swap"}})


def _path(coin, kind):
    return os.path.join(C.DATA_DIR, f"{C.EXCHANGE}_{coin}_{kind}.parquet")


def _fetch_retry(fn, *args, retries=5, **kw):
    for i in range(retries):
        try:
            return fn(*args, **kw)
        except Exception as e:  # ağ hataları: üstel bekleme
            if i == retries - 1:
                raise
            wait = 2 ** (i + 1)
            print(f"  hata ({type(e).__name__}: {str(e)[:120]}), {wait}s sonra tekrar")
            time.sleep(wait)


def download_ohlcv(coin, start, end, ex=None):
    """1dk mumları indirir, parquet'e ekleyerek kaydeder (kaldığı yerden devam eder)."""
    ex = ex or _exchange()
    os.makedirs(C.DATA_DIR, exist_ok=True)
    path = _path(coin, "1m")
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)

    old = pd.read_parquet(path) if os.path.exists(path) else None
    since = start_ms
    if old is not None and len(old):
        first = int(old.index[0].timestamp() * 1000)
        last = int(old.index[-1].timestamp() * 1000)
        if first <= start_ms:
            since = last + 60_000
        else:
            old = None  # istenen başlangıç daha erken; baştan indir
    if since >= end_ms:
        return old

    rows, n, since_req = [], 0, since
    while since < end_ms:
        batch = _fetch_retry(ex.fetch_ohlcv, symbol(coin), "1m", since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        since = batch[-1][0] + 60_000
        n += 1
        if n % 50 == 0:
            print(f"  {coin}: {pd.Timestamp(since, unit='ms', tz='UTC')}")
    if rows and rows[0][0] > since_req + 60_000 * 60 * 24:
        print(f"  UYARI: {C.EXCHANGE} {coin} verisi ancak "
              f"{pd.Timestamp(rows[0][0], unit='ms', tz='UTC')} tarihinden başlıyor")
    df = pd.DataFrame(rows, columns=["ts"] + OHLCV_COLS)
    df = df[df.ts < end_ms]
    df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
    if old is not None:
        df = pd.concat([old, df])
    df = df[~df.index.duplicated(keep="last")].sort_index().astype(float)
    df.to_parquet(path)
    return df


def download_funding(coin, start, end, ex=None):
    ex = ex or _exchange()
    since = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    rows = []
    while since < end_ms:
        batch = _fetch_retry(ex.fetch_funding_rate_history, symbol(coin), since=since, limit=1000)
        if not batch:
            break
        rows.extend((b["timestamp"], b["fundingRate"]) for b in batch)
        nxt = batch[-1]["timestamp"] + 1
        if nxt <= since:
            break
        since = nxt
    s = pd.Series({pd.Timestamp(t, unit="ms", tz="UTC"): float(r) for t, r in rows if t < end_ms},
                  name="rate", dtype=float).sort_index()
    s.to_frame().to_parquet(_path(coin, "funding"))
    return s


# ------------------------------------------------ data.binance.vision (resmi toplu arşiv)
VISION = "https://data.binance.vision/data/futures/um"
VISION_DIR = "vision2"          # v2: taker_buy sütunu dahil
# Binance'te farklı adla listelenen kontratlar (fiyat ölçeği strateji için önemsiz)
VISION_SYMBOL = {"PEPE": "1000PEPE", "BONK": "1000BONK", "SHIB": "1000SHIB", "FLOKI": "1000FLOKI"}


def _vision_csv(url):
    import io
    import urllib.error
    import urllib.request
    import zipfile
    for i in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                raw = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if i == 3:
                raise
            time.sleep(2 ** (i + 1))
        except Exception:
            if i == 3:
                raise
            time.sleep(2 ** (i + 1))
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]), header=None)
    return df[pd.to_numeric(df[0], errors="coerce").notna()].astype(float)  # başlık satırını at


def _ms(x):
    x = pd.Series(x).astype("int64")
    return x.where(x < 10 ** 14, x // 1000)          # mikro saniye gelirse ms'ye çevir


def download_vision(coin, start, end):
    """Aylık zip'ler; henüz yayınlanmamış ay için günlük zip'ler. Kaydeder ve döndürür."""
    os.makedirs(os.path.join(C.DATA_DIR, VISION_DIR), exist_ok=True)
    sym = f"{VISION_SYMBOL.get(coin, coin)}{C.QUOTE}"
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    parts = []
    for month in pd.period_range(start.tz_localize(None), (end - pd.Timedelta(minutes=1)).tz_localize(None), freq="M"):
        cache = os.path.join(C.DATA_DIR, VISION_DIR, f"{sym}-1m-{month}.parquet")
        if os.path.exists(cache):
            parts.append(pd.read_parquet(cache))
            continue
        df = _vision_csv(f"{VISION}/monthly/klines/{sym}/1m/{sym}-1m-{month}.zip")
        complete = df is not None
        if df is None:                                  # ay dosyası yok -> günlük dosyalar
            # aylık dosya sadece son 1-2 ay için henüz yayınlanmamış olabilir; daha eski eksik ay
            # = coin o tarihte listelenmemiş → günlük dosyaları tek tek denemeye gerek yok
            if month < (end - pd.Timedelta(days=62)).tz_localize(None).to_period("M"):
                continue
            days = pd.date_range(month.start_time, month.end_time.normalize(), freq="D")
            daily = [_vision_csv(f"{VISION}/daily/klines/{sym}/1m/{sym}-1m-{d:%Y-%m-%d}.zip")
                     for d in days if d < end.tz_localize(None)]
            daily = [d for d in daily if d is not None]
            if not daily:
                continue
            df = pd.concat(daily)
        df = pd.DataFrame({"ts": _ms(df[0]).to_numpy(), "open": df[1].to_numpy(),
                           "high": df[2].to_numpy(), "low": df[3].to_numpy(),
                           "close": df[4].to_numpy(), "volume": df[5].to_numpy(),
                           "taker_buy": df[9].to_numpy()})   # agresif alıcı hacmi (order flow)
        df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
        print(f"  {coin} {month}: {len(df):,} mum")
        if complete:
            df.to_parquet(cache)
        parts.append(df)
    if not parts:
        raise ValueError(f"{coin}: arşivde veri yok")
    df = pd.concat(parts)
    df = df[(df.index >= start) & (df.index < end)]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    os.makedirs(C.DATA_DIR, exist_ok=True)
    df.to_parquet(_path(coin, "1m"))
    return df


def download_funding_vision(coin, start, end):
    sym = f"{VISION_SYMBOL.get(coin, coin)}{C.QUOTE}"
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    rows, missing = [], []
    for month in pd.period_range(start.tz_localize(None), (end - pd.Timedelta(minutes=1)).tz_localize(None), freq="M"):
        df = _vision_csv(f"{VISION}/monthly/fundingRate/{sym}/{sym}-fundingRate-{month}.zip")
        if df is None:
            missing.append(str(month))
            continue
        rows.append(pd.Series(df[2].to_numpy(), index=pd.to_datetime(_ms(df[0]), unit="ms", utc=True)))
    if missing:
        print(f"  {coin} funding arşivde yok: {missing} (bu aylar ihmal)")
    s = pd.concat(rows).sort_index() if rows else pd.Series(dtype=float)
    s = s[(s.index >= start) & (s.index < end)].rename("rate")
    # funding anları tam saat başıdır; ms kaymasını yuvarla
    s.index = s.index.round("min").rename(None)
    s.to_frame().to_parquet(_path(coin, "funding"))
    return s, missing


def load_1m(coin):
    return pd.read_parquet(_path(coin, "1m"))


def load_funding(coin):
    p = _path(coin, "funding")
    return pd.read_parquet(p)["rate"] if os.path.exists(p) else None


def resample(df1m, rule):
    """1dk veriden üst zaman dilimi. Index = mumun AÇILIŞ zamanı; kapanış = index + rule."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    if "taker_buy" in df1m.columns:
        agg["taker_buy"] = "sum"
    out = df1m.resample(rule, label="left", closed="left").agg(agg)
    return out.dropna(subset=["open"])


def build_timeframes(df1m):
    return {tf: resample(df1m, tf) for tf in ["5min", "15min", "1h", "4h"]}
