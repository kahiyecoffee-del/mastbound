"""QUANT KATMANI — işlem analizi + uyarlamalı (adaptive) kurallar.

Aynı kod hem botta hem backtest'te (research_quant.py) çalışır; bota yalnız backtest'te
walk-forward olarak sonucu iyileştirdiği görülen kurallar açılır (config: QUANT_*).

Kurallar (her biri ayrı açılıp kapatılır):
  * ÖZSERMAYE FİLTRESİ  : bakiye son N kapanan işlemdeki bakiye ortalamasının altındaysa
                          (strateji "kötü dönemde") risk × low_mult; tekrar üstüne çıkınca normal.
                          mode "dd": bakiye zirveden %X'ten fazla düşükse risk × low_mult.
  * COİN FİLTRESİ       : bir coin+strateji çiftinin son N işleminin ortalama R'si eşiğin altındaysa
                          o çift `cooldown_days` gün kapatılır (geçmişi sıfırlanır).
Hiçbir kural riski taban değerin (config.RISK) üstüne çıkarmaz — yalnız azaltır/durdurur.
"""
from collections import deque

import numpy as np
import pandas as pd


class Adaptive:
    def __init__(self, eq_mode=None, eq_n=30, eq_dd=0.15, low_mult=0.5,
                 coin_filter=False, coin_n=12, coin_min=8, coin_thr=-0.3, cooldown_days=30):
        self.eq_mode, self.eq_n, self.eq_dd, self.low_mult = eq_mode, eq_n, eq_dd, low_mult
        self.coin_filter, self.coin_n, self.coin_min = coin_filter, coin_n, coin_min
        self.coin_thr, self.cooldown = coin_thr, pd.Timedelta(days=cooldown_days)
        self.bals = deque(maxlen=max(eq_n, 1))
        self.peak = None
        self.hist = {}                   # (coin, strat) -> deque[R]
        self.block = {}                  # (coin, strat) -> Timestamp (bu zamana kadar kapalı)

    # ---------------------------------------------------------------- olaylar
    def on_close(self, t_exit, coin, strat, r_mult, balance):
        self.bals.append(balance)
        self.peak = balance if self.peak is None else max(self.peak, balance)
        if not self.coin_filter:
            return
        k = (coin, strat)
        h = self.hist.setdefault(k, deque(maxlen=self.coin_n))
        h.append(float(r_mult))
        if len(h) >= self.coin_min and np.mean(h) < self.coin_thr:
            self.block[k] = pd.Timestamp(t_exit) + self.cooldown
            h.clear()

    # ---------------------------------------------------------------- kararlar
    def risk_mult(self, balance):
        if self.eq_mode == "sma" and len(self.bals) >= self.eq_n:
            return self.low_mult if balance < np.mean(self.bals) else 1.0
        if self.eq_mode == "dd" and self.peak:
            return self.low_mult if balance < self.peak * (1 - self.eq_dd) else 1.0
        return 1.0

    def blocked(self, coin, strat, now):
        until = self.block.get((coin, strat))
        return until is not None and pd.Timestamp(now) < until

    # ---------------------------------------------------------------- kalıcılık (bot durumu)
    def to_dict(self):
        return {"bals": list(self.bals), "peak": self.peak,
                "hist": {f"{c}|{s}": list(v) for (c, s), v in self.hist.items()},
                "block": {f"{c}|{s}": str(v) for (c, s), v in self.block.items()}}

    def load(self, d):
        if not d:
            return self
        self.bals.extend(d.get("bals", []))
        self.peak = d.get("peak")
        for k, v in d.get("hist", {}).items():
            self.hist[tuple(k.split("|"))] = deque(v, maxlen=self.coin_n)
        for k, v in d.get("block", {}).items():
            self.block[tuple(k.split("|"))] = pd.Timestamp(v)
        return self


def r_multiple(pnl, qty, entry, stop):
    risk = qty * abs(entry - stop)
    return pnl / risk if risk > 0 else 0.0


# -------------------------------------------------------------------- işlem analizi
def _stats(g):
    p = g["pnl_usdt"].to_numpy(float)
    r = g["R"].to_numpy(float) if "R" in g else np.full(len(p), np.nan)
    gl = -p[p < 0].sum()
    return pd.Series({"işlem": len(p), "kazanma%": (p > 0).mean() * 100 if len(p) else np.nan,
                      "ort.R": np.nanmean(r) if len(r) else np.nan,
                      "kâr faktörü": p[p > 0].sum() / gl if gl > 0 else np.inf,
                      "net pnl": p.sum()})


# Backtest beklentileri (research_quant.py, 2021-09.2026, 18 coin, 50/50 düzen): SON 50 İŞLEMİN
# kayan değerinin %10–%90 aralığı. Canlı son-50 değeri bunun dışındaysa uyarı verilir.
# (tüm dönem: EMA 2664 işlem, kazanma %37.0, ort.R +0.031 | DON 158 işlem, kazanma %44.9, ort.R +0.116)
EXPECTED_N = 50
EXPECTED = {"EMA": {"kazanma%": (28, 46), "ort.R": (-0.17, 0.22)},
            "DON": {"kazanma%": (40, 52), "ort.R": (0.00, 0.22)}}


def analyze(trades, adaptive=None, now=None):
    """trades: botun trades.csv'si (DataFrame). Dönen: rapor metni."""
    if trades is None or not len(trades):
        return "QUANT RAPORU: henüz kapanan işlem yok."
    t = trades.copy()
    if "R" not in t and {"giris", "stop_ilk", "adet"} <= set(t.columns):
        t["R"] = [r_multiple(p, q, e, s) for p, q, e, s in zip(t["pnl_usdt"], t["adet"], t["giris"], t["stop_ilk"])]
    lines = [f"QUANT RAPORU — {pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC",
             f"Toplam: {len(t)} işlem, net {t['pnl_usdt'].sum():+.2f} USDT", ""]
    by = "strateji" if "strateji" in t else None
    if by:
        s = t.groupby(by).apply(_stats, include_groups=False)
        lines += ["Strateji bazında:", s.round(2).to_string(), ""]
        for strat in s.index:
            exp = EXPECTED.get(strat)
            g = t[t[by] == strat].tail(EXPECTED_N)
            if not exp:
                continue
            if len(g) < EXPECTED_N:
                lines.append(f"  {strat}: backtest karşılaştırması için {EXPECTED_N} işlem gerekli ({len(g)} var)")
                continue
            last = _stats(g)
            for k, (lo, hi) in exp.items():
                v = last[k]
                ok = lo <= v <= hi
                lines.append(f"  {'✓' if ok else '⚠'} {strat} son {EXPECTED_N} işlem {k} = {v:.2f} "
                             f"(backtest %10–%90 aralığı {lo}–{hi})" + ("" if ok else " — canlı/backtest farkını incele"))
    lines += ["Coin bazında:", t.groupby("coin").apply(_stats, include_groups=False)
              .sort_values("net pnl").round(2).to_string(), ""]
    if "yon" in t:
        lines += ["Yön bazında:", t.groupby("yon").apply(_stats, include_groups=False).round(2).to_string(), ""]
    if "R" in t and len(t) >= 10:
        roll = t["R"].rolling(20, min_periods=10).mean().iloc[-1]
        lines.append(f"Son 20 işlemin ortalama R'si: {roll:+.2f}")
    if adaptive is not None:
        lines.append(f"Özsermaye filtresi risk çarpanı: {adaptive.risk_mult(t['bakiye'].iloc[-1]) if 'bakiye' in t else 1.0:g}")
        now_ts = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
        act = {f"{c}/{s}": f"{u:%Y-%m-%d}" for (c, s), u in adaptive.block.items() if adaptive.blocked(c, s, now_ts)}
        lines.append(f"Kapalı coin/strateji: {act or 'yok'}")
    return "\n".join(lines)
