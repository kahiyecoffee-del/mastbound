"""Coin bazında kurulum/giriş tespiti ve işlem çıkış simülasyonu.

Bir işlemin çıkışı bakiyeden bağımsızdır (R bazlı), bu yüzden her coin için tüm aday
işlemler önceden hesaplanır; tek-pozisyon ve bakiye kuralı engine.py'de uygulanır.
"""
import numpy as np
import pandas as pd

import config as C
from indicators import macd, psar

MIN = np.int64(60_000_000_000)  # 1 dakika, ns


def trend_filter(tfs, tf):
    """Her üst zaman dilimi mumu için filtre sonucu, KAPANIŞ zamanına göre indekslenmiş."""
    df = tfs[tf]
    m = macd(df["close"], C.MACD_FAST, C.MACD_SLOW, C.MACD_SIGNAL)
    ok = pd.Series(False, index=df.index)
    if C.FILTER_MACD_ABOVE_ZERO:
        ok |= m["macd"] > 0
    if C.FILTER_MACD_ABOVE_SIGNAL:
        ok |= m["macd"] > m["signal"]
    out = pd.DataFrame({"ok": ok, "macd": m["macd"], "signal": m["signal"],
                        "candle_open": df.index})
    out.index = df.index + pd.Timedelta(tf)  # yalnızca kapanmış mum: kapanış zamanı
    return out


def find_setups(tfs, start, end):
    m15 = tfs[C.BREAKOUT_TF]
    prior_high = m15["high"].shift(1).rolling(C.BREAKOUT_LOOKBACK).max()
    lows = m15["low"] if C.STOP_INCLUDES_BREAKOUT_CANDLE else m15["low"].shift(1)
    stop = lows.rolling(C.STOP_LOOKBACK).min()
    s = pd.DataFrame({"close": m15["close"], "prior_high": prior_high, "stop": stop})
    s["T"] = m15.index + pd.Timedelta(C.BREAKOUT_TF)          # 15dk kapanış anı
    s = s[(s["close"] > s["prior_high"]) & (s["T"] >= start) & (s["T"] < end)]
    s = s[s["close"] > s["stop"]]
    s = s.reset_index(names="candle_open").sort_values("T")
    s["filter_ok"] = True
    for tf in C.FILTER_TIMEFRAMES:
        f = trend_filter(tfs, tf).add_prefix(f"{tf}_")
        s = pd.merge_asof(s, f, left_on="T", right_index=True, direction="backward",
                          allow_exact_matches=True)                 # kapanış <= T
        s["filter_ok"] &= s[f"{tf}_ok"].fillna(False).astype(bool)
    return s


class Minute:
    """1dk dizileri + PSAR, hızlı döngüler için numpy olarak."""

    def __init__(self, df1m):
        self.index = df1m.index.as_unit("ns")
        self.t = self.index.asi8
        self.o = df1m["open"].to_numpy(float)
        self.h = df1m["high"].to_numpy(float)
        self.l = df1m["low"].to_numpy(float)
        self.c = df1m["close"].to_numpy(float)
        self.sar, self.trend = psar(self.h, self.l, C.PSAR_STEP, C.PSAR_MAX)

    def ts(self, i):
        return self.index[i]


def find_entry(mm, T, stop):
    """T'den sonra PSAR yukarı→aşağı→yukarı. Dönüş: (giriş_idx | None, sebep, detay)."""
    i = int(np.searchsorted(mm.t, T.value))
    deadline = T.value + np.int64(C.ENTRY_TIMEOUT_MIN) * MIN
    state, info = 0, {}
    n = len(mm.t)
    while i < n and mm.t[i] + MIN <= deadline:
        if C.CANCEL_IF_BELOW_STOP and mm.l[i] < stop:
            return None, "below_stop", info
        prev = mm.trend[i - 1] if i > 0 else mm.trend[i]
        if state == 0 and prev == 1 and mm.trend[i] == -1:
            state, info["flip_down"] = 1, i
        elif state == 1 and prev == -1 and mm.trend[i] == 1:
            info["flip_up"] = i
            e = i + 1
            if e >= n:
                return None, "no_data", info
            if mm.o[e] <= stop:
                return None, "below_stop", info
            return e, "entry", info
        i += 1
    return None, ("timeout" if i < n else "no_data"), info


def simulate_exit(mm, e, stop, breakeven):
    entry = mm.o[e] * (1 + C.SLIPPAGE)
    R = entry - stop
    tp = entry + C.TP_R * R
    be_level = entry + C.BREAKEVEN_R * R
    cur, moved, touched1r, be_time = stop, False, False, None
    bar_ns = np.int64(C.BREAKEVEN_TF_MIN) * MIN
    n = len(mm.t)
    for k in range(e, n):
        if mm.l[k] <= cur:
            raw = cur if k == e else min(mm.o[k], cur)   # açılış boşluğu: açılıştan dolum
            return dict(exit_idx=k, exit_raw=raw, exit=raw * (1 - C.SLIPPAGE),
                        reason="BE" if moved else "SL", touched_1r=touched1r, be_idx=be_time)
        if mm.h[k] >= be_level:
            touched1r = True
        if mm.h[k] >= tp:
            return dict(exit_idx=k, exit_raw=tp, exit=tp * (1 - C.SLIPPAGE), reason="TP",
                        touched_1r=True, be_idx=be_time)
        # 5dk mum kapanışı = o 5dk'nın son 1dk mumunun kapanışı
        if breakeven and not moved and (mm.t[k] + MIN) % bar_ns == 0 and mm.c[k] > be_level:
            cur, moved, be_time = entry, True, k
    k = n - 1
    return dict(exit_idx=k, exit_raw=mm.c[k], exit=mm.c[k] * (1 - C.SLIPPAGE), reason="END",
                touched_1r=touched1r, be_idx=be_time)


def funding_cost(mm, funding, e, k):
    """Pozisyon açıkken geçen funding anları için sum(rate * fiyat) (qty ile çarpılır)."""
    if funding is None or not len(funding):
        return 0.0
    t0, t1 = mm.t[e], mm.t[k]
    ft = funding.index.as_unit("ns").asi8
    sel = (ft > t0) & (ft <= t1)
    if not sel.any():
        return 0.0
    idx = np.searchsorted(mm.t, ft[sel])
    idx = np.clip(idx, 0, len(mm.t) - 1)
    return float(np.sum(funding.to_numpy()[sel] * mm.o[idx]))


def liquidation_price(entry):
    return entry * (1 - 1 / C.LEVERAGE) / (1 - C.MAINT_MARGIN_RATE)


def coin_candidates(coin, df1m, tfs, funding, start, end):
    """Bir coin için tüm aday işlemler (BE'li ve BE'siz çıkışlarla) + kurulum istatistikleri."""
    mm = Minute(df1m)
    setups = find_setups(tfs, start, end)
    stats = {"breakouts": len(setups), "filter_pass": int(setups["filter_ok"].sum()),
             "entry": 0, "timeout": 0, "below_stop": 0, "no_data": 0, "skipped_pending": 0}
    cands, busy_until = [], np.int64(0)
    for row in setups[setups["filter_ok"]].to_dict("records"):
        T, stop = row["T"], row["stop"]
        if T.value < busy_until:                 # bu coinde bekleyen kurulum var
            stats["skipped_pending"] += 1
            continue
        e, why, info = find_entry(mm, T, stop)
        stats[why] += 1
        if e is None:
            end_i = min(int(np.searchsorted(mm.t, T.value)) + C.ENTRY_TIMEOUT_MIN, len(mm.t) - 1)
            busy_until = mm.t[end_i]
            continue
        busy_until = mm.t[e]
        entry = mm.o[e] * (1 + C.SLIPPAGE)
        c = dict(coin=coin, setup_time=T, breakout_candle=row["candle_open"],
                 breakout_close=row["close"], prior_high=row["prior_high"], stop=stop,
                 entry_idx=e, entry_time=mm.ts(e), entry_raw=mm.o[e], entry=entry,
                 R=entry - stop, tp=entry + C.TP_R * (entry - stop),
                 liq=liquidation_price(entry),
                 flip_down_time=mm.ts(info["flip_down"]), flip_up_time=mm.ts(info["flip_up"]),
                 flip_down_sar=mm.sar[info["flip_down"]], flip_up_sar=mm.sar[info["flip_up"]])
        for tf in C.FILTER_TIMEFRAMES:
            c[f"{tf}_candle"] = row[f"{tf}_candle_open"]
            c[f"{tf}_macd"] = row[f"{tf}_macd"]
            c[f"{tf}_signal"] = row[f"{tf}_signal"]
        for be in (True, False):
            x = simulate_exit(mm, e, stop, be)
            x["exit_time"] = mm.ts(x["exit_idx"])
            x["be_time"] = mm.ts(x["be_idx"]) if x["be_idx"] is not None else None
            x["funding_px"] = funding_cost(mm, funding, e, x["exit_idx"])
            c["be" if be else "nobe"] = x
        cands.append(c)
    stats["entry"] = len(cands)
    return cands, stats, mm
