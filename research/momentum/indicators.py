import numpy as np
import pandas as pd


def macd(close, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def psar(high, low, step=0.02, max_af=0.2):
    """Wilder Parabolic SAR. Dönüş: (sar, trend) — trend +1 yükseliş, -1 düşüş.
    trend[i] i. mumun kapanışında bilinen durumdur (dönüş mum içinde tetiklenir)."""
    high = np.asarray(high, float)
    low = np.asarray(low, float)
    n = len(high)
    sar = np.full(n, np.nan)
    trend = np.zeros(n, dtype=np.int8)
    if n < 2:
        return sar, trend
    up = high[1] >= high[0]
    af = step
    ep = high[0] if up else low[0]
    s = low[0] if up else high[0]
    sar[0], trend[0] = s, 1 if up else -1
    for i in range(1, n):
        s = s + af * (ep - s)
        if up:
            s = min(s, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if low[i] < s:                       # yükseliş -> düşüş
                up, s, ep, af = False, ep, low[i], step
            elif high[i] > ep:
                ep, af = high[i], min(af + step, max_af)
        else:
            s = max(s, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if high[i] > s:                      # düşüş -> yükseliş
                up, s, ep, af = True, ep, high[i], step
            elif low[i] < ep:
                ep, af = low[i], min(af + step, max_af)
        sar[i], trend[i] = s, 1 if up else -1
    return sar, trend
