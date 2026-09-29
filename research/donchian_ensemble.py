"""Donchian TOPLULUĞU + volatilite hedefli boyut (Zarattini, Pagani, Barbon 2025 "Catching Crypto Trends")
ve funding "çöküş filtresi" (Schmeling, Schrimpf, Todorov "Crypto Carry") — mevcut tek-dönem Donchian'la kıyas.

Veri: MEXC USDT vadeli GÜNLÜK mumlar + funding geçmişi (herkese açık API). Emir göndermez.
Kurallar günlük kapanışta hesaplanır, ertesi gün uygulanır (geleceğe bakma yok). Maliyet: ciro × (taker+kayma),
funding: pozisyon yönüne göre ödenir/alınır.

  python research/donchian_ensemble.py            # indir + tüm varyantlar → results/donchian_ensemble/report.txt
  python research/donchian_ensemble.py --synthetic   # kod testi (sahte veri)
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import requests

MEXC = "https://contract.mexc.com/api/v1/contract"
COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "AVAX", "LINK", "ADA", "DOT", "LTC", "TRX", "ATOM", "NEAR",
         "UNI", "SUI", "APT", "OP"]                      # canlı botun listesi
OUT = "results/donchian_ensemble"
COST = 0.0005 + 0.0002          # taker %0.05 + kayma %0.02 (repo varsayımı), ciro başına
LOOKBACKS = [5, 10, 20, 30, 60, 90, 150, 250, 360]
PERIODS = [("2021-01-01", "2023-12-31"), ("2024-01-01", "2025-09-23"), ("2025-09-24", "2026-09-24")]


# ------------------------------------------------------------------ veri
def get(url, params=None, tries=5):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=30, headers={"User-Agent": "research"})
            if r.status_code == 429:
                time.sleep(3 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            err = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"{url}: {err}")


def daily(coin, start="2019-01-01"):
    sym, rows = f"{coin}_USDT", []
    t, end = int(pd.Timestamp(start, tz="UTC").timestamp()), int(time.time())
    while t < end:
        t2 = min(end, t + 1900 * 86400)
        d = get(f"{MEXC}/kline/{sym}", {"interval": "Day1", "start": t, "end": t2}).get("data") or {}
        if d.get("time"):
            rows.append(pd.DataFrame({k: d[k] for k in ("time", "open", "high", "low", "close")}))
        t = t2
        time.sleep(0.2)
    if not rows:
        return None
    df = pd.concat(rows).drop_duplicates("time").sort_values("time")
    df.index = pd.to_datetime(df.pop("time"), unit="s", utc=True).normalize()
    return df.astype(float)


def funding(coin):
    rows, page = [], 1
    while page <= 80:
        d = get(f"{MEXC}/funding_rate/history", {"symbol": f"{coin}_USDT", "page_num": page, "page_size": 100}).get("data") or {}
        rows += d.get("resultList") or []
        if page >= int(d.get("totalPage") or 1):
            break
        page += 1
        time.sleep(0.15)
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series({pd.to_datetime(r["settleTime"], unit="ms", utc=True): float(r["fundingRate"]) for r in rows})
    return s.sort_index().resample("1D").sum()        # günlük toplam funding oranı


def load(cache):
    os.makedirs(cache, exist_ok=True)
    O, H, L, C, F = {}, {}, {}, {}, {}
    for c in COINS:
        p = os.path.join(cache, f"{c}.pkl")
        if os.path.exists(p):
            df, f = pd.read_pickle(p)
        else:
            df = daily(c)
            if df is None:
                print(f"{c}: veri yok, atlandı")
                continue
            f = funding(c)
            pd.to_pickle((df, f), p)
        O[c], H[c], L[c], C[c], F[c] = df["open"], df["high"], df["low"], df["close"], f
        print(f"{c}: {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}, funding {len(f)} gün")
    frame = lambda d: pd.DataFrame(d).sort_index()
    C_ = frame(C)
    return frame(O).reindex(C_.index), frame(H).reindex(C_.index), frame(L).reindex(C_.index), C_, \
        frame(F).reindex(C_.index).fillna(0.0)


def synthetic():
    idx = pd.date_range("2020-01-01", "2026-09-24", freq="1D", tz="UTC")
    rng = np.random.default_rng(0)
    C = {}
    for i, c in enumerate(COINS):
        r = rng.normal(0.0005, 0.04, len(idx)) + 0.002 * np.sin(np.arange(len(idx)) / (60 + 5 * i))
        C[c] = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
        C[c].iloc[: rng.integers(0, 400)] = np.nan
    C = pd.DataFrame(C)
    H, L = C * 1.02, C * 0.98
    F = pd.DataFrame(rng.normal(3e-4, 3e-4, C.shape), index=idx, columns=C.columns)
    return C.shift(1).fillna(C), H, L, C, F


# ------------------------------------------------------------------ sinyaller
def donchian_state(H, L, C, n, exit_frac=0.5, short=True):
    """Kırılım durumu (kapanışta): üst bant (önceki n gün en yüksek) kırılınca +1, alt bant kırılınca −1 (short açıksa).
    Çıkış: long için önceki n·exit_frac gün en düşüğü altına kapanış, short için tersi."""
    up = H.shift(1).rolling(n, min_periods=n).max()
    dn = L.shift(1).rolling(n, min_periods=n).min()
    m = max(2, int(n * exit_frac))
    ex_l = L.shift(1).rolling(m, min_periods=m).min()
    ex_s = H.shift(1).rolling(m, min_periods=m).max()
    st = np.zeros(C.shape)
    c, u, d, el, es = (x.to_numpy() for x in (C, up, dn, ex_l, ex_s))
    for j in range(C.shape[1]):
        s = 0
        for i in range(C.shape[0]):
            if np.isnan(c[i, j]) or np.isnan(u[i, j]):
                s = 0
            elif s == 1 and c[i, j] < el[i, j]:
                s = 0
            elif s == -1 and c[i, j] > es[i, j]:
                s = 0
            if s == 0 and not np.isnan(u[i, j]):
                if c[i, j] > u[i, j]:
                    s = 1
                elif short and c[i, j] < d[i, j]:
                    s = -1
            st[i, j] = s
    return pd.DataFrame(st, index=C.index, columns=C.columns)


def ensemble(H, L, C, lookbacks, short):
    return sum(donchian_state(H, L, C, n, short=short) for n in lookbacks) / len(lookbacks)


def vol(C, span=30):
    r = np.log(C).diff()
    return r.ewm(span=span, min_periods=20).std() * np.sqrt(365)


# ------------------------------------------------------------------ portföy
def backtest(sig, C, F, target_vol=0.25, gross_cap=2.0, coin_cap=1.0, fund_filter=None, btc_regime=False):
    """sig ∈ [-1,1] (kapanışta). Ağırlık = sig × (hedef vol / coin vol) / N, sonra portföy gerçekleşen vol'ü hedefe ölçeklenir;
    coin başı ve toplam kaldıraç tavanlı.
    Ertesi günün getirisine uygulanır. fund_filter: 7 günlük ort. günlük funding > eşik ise LONG ağırlık sıfırlanır."""
    s = sig.copy()
    if btc_regime and "BTC" in C:
        b = C["BTC"]
        up = (b > b.ewm(span=200, min_periods=150).mean()).astype(float)
        s = s.where(~((s > 0) & (up.to_numpy()[:, None] == 0)), 0.0)
        s = s.where(~((s < 0) & (up.to_numpy()[:, None] == 1)), 0.0)
    if fund_filter is not None:
        f7 = F.rolling(7, min_periods=3).mean()
        s = s.where(~((s > 0) & (f7 > fund_filter)), 0.0)
    v = vol(C)
    ret = C.pct_change().fillna(0.0)
    n = C.notna().sum(axis=1).clip(lower=1)
    raw = (s * (target_vol / v)).div(n, axis=0).fillna(0.0)          # coin başı eşit risk
    # portföy düzeyinde hedef volatilite: ham portföyün son 60 günlük gerçekleşen vol'üne göre ölçekle (geriye bakar)
    pr = (raw.shift(1).fillna(0.0) * ret).sum(axis=1)
    pvol = pr.rolling(60, min_periods=20).std() * np.sqrt(365)
    k = (target_vol / pvol).replace([np.inf, -np.inf], np.nan).clip(upper=10).fillna(1.0)
    w = raw.mul(k, axis=0).clip(-coin_cap, coin_cap)
    g = w.abs().sum(axis=1)
    w = w.mul((gross_cap / g).clip(upper=1.0).fillna(1.0), axis=0)
    wp = w.shift(1).fillna(0.0)                                   # kapanışta karar, ertesi gün uygulanır
    turn = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1).shift(1).fillna(0.0)
    daily_r = (wp * ret).sum(axis=1) - (wp * F).sum(axis=1) - turn * COST
    return daily_r, wp


def metrics(r):
    r = r.dropna()
    if len(r) < 30:
        return dict(n=len(r))
    eq = (1 + r).cumprod()
    yrs = len(r) / 365
    return dict(son=round(float(eq.iloc[-1] * 100), 1),
                cagr=round(float((eq.iloc[-1] ** (1 / yrs) - 1) * 100), 1),
                maxdd=round(float((eq / eq.cummax() - 1).min() * 100), 1),
                sharpe=round(float(r.mean() / r.std() * np.sqrt(365)), 2) if r.std() > 0 else None)


def report_one(name, r, wp, lines):
    m = metrics(r[r.index >= "2021-01-01"])
    per = []
    for a, b in PERIODS:
        pm = metrics(r[(r.index >= a) & (r.index <= b)])
        per.append(f"{pm.get('son', '-')}({pm.get('maxdd', '-')}%)")
    yr = r[r.index >= "2021-01-01"].groupby(r[r.index >= "2021-01-01"].index.year).apply(lambda x: (1 + x).prod() - 1)
    roll = (1 + r).rolling(365).apply(np.prod, raw=True).dropna() - 1
    roll = roll[roll.index >= "2022-01-01"][::30]
    lev = f" ort.kaldıraç {wp.abs().sum(axis=1)[wp.index >= '2021-01-01'].mean():.2f}" if wp is not None else ""
    lines.append(f"{name:<44} 100→{m.get('son')} CAGR %{m.get('cagr')} maxDD %{m.get('maxdd')} Sharpe {m.get('sharpe')}"
                 f" | dönemler {' / '.join(per)} | kârlı 12ay {int((roll > 0).sum())}/{len(roll)}{lev}")
    lines.append("      yıllar: " + "  ".join(f"{y}:{v * 100:+.0f}%" for y, v in yr.items()))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    O, H, L, C, F = synthetic() if a.synthetic else load(os.path.join(OUT, "cache"))
    lines = [f"DONCHIAN TOPLULUĞU araştırması — {pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M} UTC",
             f"coin {C.shape[1]}, gün {len(C)}, maliyet ciro×%{COST * 100:.2f} + funding; dönemler (100 ile ayrı başlar): "
             + " / ".join(f"{x}→{y}" for x, y in PERIODS) + " (son dönem = görülmemiş)", ""]

    lines.append("== A) TEMEL KIYAS (hedef vol %25, toplam kaldıraç ≤2x)")
    base_s = donchian_state(H, L, C, 20, short=True)
    report_one("Tek Donchian 20g L/S + BTC rejimi (≈ mevcut)", *backtest(base_s, C, F, btc_regime=True), lines)
    ens_ls = ensemble(H, L, C, LOOKBACKS, short=True)
    ens_lo = ensemble(H, L, C, LOOKBACKS, short=False)
    report_one("TOPLULUK 9 dönem long/short", *backtest(ens_ls, C, F), lines)
    report_one("TOPLULUK 9 dönem long/short + BTC rejimi", *backtest(ens_ls, C, F, btc_regime=True), lines)
    report_one("TOPLULUK 9 dönem sadece LONG (makaledeki)", *backtest(ens_lo, C, F), lines)

    lines += ["", "== B) SAĞLAMLIK — komşu ayarlar (TOPLULUK long/short)"]
    for tv in (0.15, 0.25, 0.35):
        for gc in (1.0, 2.0, 3.0):
            report_one(f"hedef vol %{tv * 100:.0f}, kaldıraç ≤{gc:g}x", *backtest(ens_ls, C, F, target_vol=tv, gross_cap=gc), lines)
    for name, lbs in (("kısa set 5-60", [5, 10, 20, 30, 60]), ("uzun set 30-360", [30, 60, 90, 150, 250, 360])):
        report_one(f"dönem seti: {name}", *backtest(ensemble(H, L, C, lbs, True), C, F), lines)

    lines += ["", "== C) FUNDING ÇÖKÜŞ FİLTRESİ (7g ort. günlük funding eşiği aşılınca LONG kapalı)"]
    for thr in (0.0003, 0.0006, 0.001):
        report_one(f"TOPLULUK L/S + funding filtresi >{thr * 100:.2f}%/gün", *backtest(ens_ls, C, F, fund_filter=thr), lines)

    lines += ["", "== D) COİN ÇIKARMA (TOPLULUK L/S, her seferinde 1 coin çıkarılır)"]
    base_m = metrics(backtest(ens_ls, C, F)[0][lambda x: x.index >= "2021-01-01"])
    better = 0
    for c in C.columns:
        keep = [x for x in C.columns if x != c]
        m = metrics(backtest(ens_ls[keep], C[keep], F[keep])[0][lambda x: x.index >= "2021-01-01"])
        better += (m.get("son") or 0) > 100
        lines.append(f"  -{c:<5} 100→{m.get('son')} maxDD %{m.get('maxdd')}")
    lines.append(f"  kârlı kalan: {better}/{C.shape[1]} (tam set 100→{base_m.get('son')})")

    lines += ["", "OKUMA: '≈ mevcut' satırı canlı Donchian'ın bu çerçevedeki yaklaşığıdır (iz süren ATR stop yerine kanal çıkışı).",
              "Karar ölçütü: son (görülmemiş) dönemde de kârlı, komşu ayarlar tutarlı, coin çıkarmada sağlam ve mevcut",
              "Donchian'dan açıkça iyi olmalı. Kaldıraç ≤2x bot sınırıyla aynı."]
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "report.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
