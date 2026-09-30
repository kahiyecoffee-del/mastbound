"""SADECE TEST İÇİN sentetik 1dk veri (ağ erişimi yokken pipeline'ı doğrulamak için).
Rejim değiştiren (trend/yatay) GBM; gerçek piyasa sonucu DEĞİLDİR."""
import numpy as np
import pandas as pd

BASE = {"BTC": 60000, "ETH": 3000, "SOL": 150, "BNB": 600, "XRP": 0.6, "DOGE": 0.15,
        "AVAX": 30, "LINK": 15}


def make_1m(coin, start, end, seed=0, drift_scale=4e-5):
    rng = np.random.default_rng(seed + sum(map(ord, coin)))
    idx = pd.date_range(start, end, freq="1min", inclusive="left", tz="UTC")
    n = len(idx)
    regime_len = 60 * 24 * 3
    drift = np.repeat(rng.normal(0, 1, n // regime_len + 1) * drift_scale, regime_len)[:n]
    vol = np.repeat(rng.uniform(4e-4, 1.2e-3, n // regime_len + 1), regime_len)[:n]
    r = drift + vol * rng.standard_t(4, n) / np.sqrt(2)
    close = BASE.get(coin, 100) * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, 1, (2, n))) * vol * close * 0.6
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": (v := rng.uniform(1, 10, n)),
                         # taker alış oranı getiriyle ilişkili (gerçek piyasadaki gibi)
                         "taker_buy": v * np.clip(0.5 + 0.35 * r / vol + rng.normal(0, 0.1, n),
                                                  0.02, 0.98)}, index=idx)


def make_funding(start, end, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(pd.Timestamp(start).ceil("8h"), end, freq="8h",
                        inclusive="left")
    return pd.Series(rng.normal(1e-4, 5e-5, len(idx)), index=idx, name="rate")
