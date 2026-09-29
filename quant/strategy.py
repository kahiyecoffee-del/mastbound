"""EMA kesişimi + ATR stop stratejisi. Saf mantık: ağ yok, backtest ve canlı motor aynı kodu kullanır.

Giriş (yalnız kapanmış mumda):
  LONG  : hızlı EMA yavaş EMA'yı yukarı keser ve kapanış > trend EMA'sı
  SHORT : hızlı EMA aşağı keser ve kapanış < trend EMA'sı (allow_short açıksa)
Çıkış:
  - ters kesişim
  - iz süren stop: max(giriş - stop_atr*ATR, en_yüksek_kapanış - trail_atr*ATR)  (short için simetrik)
  - borsada duran sabit stop (giriş - stop_atr*ATR) mum içi korumayı sağlar; bot kapalı olsa bile çalışır
"""
from dataclasses import dataclass


@dataclass
class Params:
    fast: int = 20
    slow: int = 50
    trend: int = 200          # 0 = trend filtresi yok
    atr: int = 14
    stop_atr: float = 2.0
    trail_atr: float = 3.0
    allow_short: bool = True

    @property
    def warmup(self):
        return max(self.fast, self.slow, self.trend, self.atr) + 2


def ema(values, n):
    out, k, e = [], 2 / (n + 1), None
    for v in values:
        e = v if e is None else e + k * (v - e)
        out.append(e)
    return out


def atr(highs, lows, closes, n):
    """Wilder ATR."""
    out, a = [], None
    for i in range(len(closes)):
        tr = highs[i] - lows[i] if i == 0 else max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]),
                                                     abs(lows[i] - closes[i - 1]))
        a = tr if a is None else (a * (n - 1) + tr) / n
        out.append(a)
    return out


@dataclass
class Indicators:
    fast: list
    slow: list
    trend: list
    atr: list


def indicators(candles, p):
    c = [x[4] for x in candles]
    return Indicators(ema(c, p.fast), ema(c, p.slow), ema(c, p.trend) if p.trend else [None] * len(c),
                      atr([x[2] for x in candles], [x[3] for x in candles], c, p.atr))


@dataclass
class Decision:
    action: str               # hold | open_long | open_short | close
    reason: str = ""
    stop: float = 0.0         # açılışta: borsaya konacak sabit stop / açıkken: güncel iz süren stop


def trail_stop(pos, atr_now, p):
    """pos: {side, entry, peak} — peak = girişten beri lehimize en iyi kapanış."""
    if pos["side"] == "long":
        return max(pos["entry"] - p.stop_atr * pos.get("atr0", atr_now), pos["peak"] - p.trail_atr * atr_now)
    return min(pos["entry"] + p.stop_atr * pos.get("atr0", atr_now), pos["peak"] + p.trail_atr * atr_now)


def decide(candles, ind, i, pos, p):
    """i numaralı (kapanmış) mumdaki karar. pos None veya {side, entry, peak, atr0}; peak'i günceller."""
    if i < p.warmup:
        return Decision("hold", "ısınma")
    close, a = candles[i][4], ind.atr[i]
    up = ind.fast[i - 1] <= ind.slow[i - 1] and ind.fast[i] > ind.slow[i]
    down = ind.fast[i - 1] >= ind.slow[i - 1] and ind.fast[i] < ind.slow[i]
    tr = ind.trend[i]

    if pos:
        long = pos["side"] == "long"
        pos["peak"] = max(pos["peak"], close) if long else min(pos["peak"], close)
        stop = trail_stop(pos, a, p)
        if (long and down) or (not long and up):
            return Decision("close", "ters kesişim", stop)
        if (long and close < stop) or (not long and close > stop):
            return Decision("close", "iz süren stop", stop)
        return Decision("hold", "", stop)

    if up and (tr is None or close > tr):
        return Decision("open_long", "EMA yukarı kesişim", close - p.stop_atr * a)
    if down and p.allow_short and (tr is None or close < tr):
        return Decision("open_short", "EMA aşağı kesişim", close + p.stop_atr * a)
    return Decision("hold")


def size(equity, price, stop, detail, leverage, risk_pct, max_margin_pct):
    """Kontrat adedi: stopta kaybedilen = equity*risk_pct%; teminat equity*max_margin_pct%'yi aşmaz.

    detail: MEXC kontrat bilgisi (contractSize, minVol, volUnit). Yetersizse 0 döner.
    """
    cs = float(detail.get("contractSize") or 1)
    unit = float(detail.get("volUnit") or 1)
    min_vol = float(detail.get("minVol") or unit)
    dist = abs(price - stop)
    if dist <= 0 or price <= 0 or equity <= 0:
        return 0
    coins = equity * risk_pct / 100 / dist
    coins = min(coins, equity * max_margin_pct / 100 * leverage / price)
    vol = int(coins / cs / unit) * unit
    if vol < min_vol:
        return 0
    return int(vol) if float(vol).is_integer() else round(vol, 8)
