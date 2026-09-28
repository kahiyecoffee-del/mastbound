"""MEXC hisse vadelileri (stock futures) araştırması — üç strateji, ücretler dahil, emir GÖNDERMEZ.

  1) PRIM: ABD borsası kapalıyken (gece / hafta sonu) kontrat fiyatının son kapanıştan sapması,
     açılışta kapanıyor mu? Açılıştan önce sapmanın tersine gir, açılıştan sonra çık.
  2) FUNDING: hisse kontratlarında funding karşıtı L/S (en yüksek funding'e short, en düşüğe long).
  3) SEANS AÇILIŞI: 09:30 New York açılışında gece boşluğunu (gap) sönümleme / ilk 30 dk momentumu /
     açılış aralığı kırılımı (ORB).

Veri: MEXC contract API (kontrat 15 dk mumları + funding geçmişi) ve Yahoo Finance (gerçek hisse günlük
açılış/kapanış). Yalnız requests + pandas + numpy gerekir (momentum-bot venv'i yeterli).

  python research/stock_futures.py discover            # MEXC'deki hisse kontratlarını bul
  python research/stock_futures.py run                  # indir + 3 testi çalıştır → results/stock_futures/
  python research/stock_futures.py run --pairs NVDASTOCK_USDT=NVDA,TSLASTOCK_USDT=TSLA   # elle eşleştirme
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

MEXC = "https://contract.mexc.com/api/v1/contract"
YAHOO = "https://query2.finance.yahoo.com/v8/finance/chart"
NY = ZoneInfo("America/New_York")
UA = {"User-Agent": "Mozilla/5.0 (research script)"}
OUT = "results/stock_futures"
CACHE = os.path.join(OUT, "cache")

TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "GOOG", "META", "NFLX", "AMD", "INTC", "COIN", "MSTR",
           "HOOD", "PLTR", "CRCL", "SPY", "QQQ", "AVGO", "ORCL", "UBER", "BABA", "SMCI", "ARM", "MU", "DIS", "JPM",
           "V", "MA", "WMT", "COST", "LLY", "UNH", "XOM", "BA", "NKE", "PYPL", "SHOP", "SQ", "XYZ", "RIVN", "LCID",
           "GME", "AMC", "IWM", "DIA", "GLD", "SLV", "TSM", "ASML", "CRWD", "SNOW", "NIO", "IBM", "CSCO", "ADBE"]
SUFFIXES = ["STOCK", ""]
# MEXC'de tam şirket adıyla listelenen kontratlar (discover çıktısından)
ALIASES = {"NVIDIA_USDT": "NVDA", "TESLA_USDT": "TSLA", "COINBASE_USDT": "COIN", "ROBINHOOD_USDT": "HOOD",
           "SPY_USDT": "SPY"}

# Maliyet varsayımları (bps = 0.01%). MEXC güncel ücretleriyle değiştir: --taker-bps / --slip-bps
TAKER_BPS = 2.0      # taraf başına
SLIP_BPS = 2.0       # taraf başına kayma


# ------------------------------------------------------------------ indirme
def get(url, params=None, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=30)
            if r.status_code == 429:
                time.sleep(3 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            err = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"istek başarısız: {url} ({err})")


def contracts():
    return get(f"{MEXC}/detail").get("data") or []


def discover_pairs(items):
    """MEXC kontratlarından ABD hisse/ETF eşleşmeleri: {mexc_symbol: yahoo_ticker}"""
    pairs = {}
    syms = {c.get("symbol") for c in items}
    pairs.update({m: t for m, t in ALIASES.items() if m in syms})
    for c in items:
        sym = c.get("symbol", "")
        if not sym.endswith("_USDT"):
            continue
        base = sym[:-5].upper()
        name = f"{c.get('displayNameEn', '')} {c.get('displayName', '')}".upper()
        for t in TICKERS:
            if any(base == t + s for s in SUFFIXES if s) or (base == t and "STOCK" in name):
                pairs[sym] = t
                break
    return pairs


def cached(name, fn):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, name)
    if os.path.exists(p) and time.time() - os.path.getmtime(p) < 6 * 3600:
        return pd.read_pickle(p)
    df = fn()
    df.to_pickle(p)
    return df


def mexc_klines(sym, interval="Min15", days=400):
    """Kontrat mumları (UTC, index = mum açılışı)."""
    def fetch():
        step = {"Min15": 900, "Min60": 3600, "Day1": 86400}[interval]
        end = int(time.time())
        start = end - days * 86400
        rows = []
        t = start
        while t < end:
            t2 = min(end, t + 1900 * step)
            d = get(f"{MEXC}/kline/{sym}", {"interval": interval, "start": t, "end": t2}).get("data") or {}
            if d.get("time"):
                rows.append(pd.DataFrame({k: d[k] for k in ("time", "open", "high", "low", "close", "vol")}))
            t = t2
            time.sleep(0.15)
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "vol"])
        df = pd.concat(rows).drop_duplicates("time").sort_values("time")
        df.index = pd.to_datetime(df.pop("time"), unit="s", utc=True)
        return df.astype(float)
    return cached(f"k_{sym}_{interval}_{days}.pkl", fetch)


def mexc_funding(sym):
    def fetch():
        rows, page = [], 1
        while True:
            d = get(f"{MEXC}/funding_rate/history", {"symbol": sym, "page_num": page, "page_size": 100}).get("data") or {}
            rows += d.get("resultList") or []
            if page >= int(d.get("totalPage") or 1) or page > 60:
                break
            page += 1
            time.sleep(0.15)
        if not rows:
            return pd.Series(dtype=float)
        s = pd.Series({pd.to_datetime(r["settleTime"], unit="ms", utc=True): float(r["fundingRate"]) for r in rows})
        return s.sort_index()
    return cached(f"f_{sym}.pkl", fetch)


def yahoo_daily(ticker, rng="2y"):
    """Gerçek hisse günlük açılış/kapanış; index = New York işlem günü (date)."""
    def fetch():
        j = get(f"{YAHOO}/{ticker}", {"interval": "1d", "range": rng})
        r = j["chart"]["result"][0]
        q = r["indicators"]["quote"][0]
        df = pd.DataFrame({"open": q["open"], "close": q["close"]},
                          index=[datetime.fromtimestamp(t, NY).date() for t in r["timestamp"]])
        return df.dropna()
    return cached(f"y_{ticker}_{rng}.pkl", fetch)


# ------------------------------------------------------------------ yardımcılar
def session_times(day):
    """NY işlem günü → (açılış, kapanış) UTC"""
    o = datetime(day.year, day.month, day.day, 9, 30, tzinfo=NY).astimezone(timezone.utc)
    c = datetime(day.year, day.month, day.day, 16, 0, tzinfo=NY).astimezone(timezone.utc)
    return pd.Timestamp(o), pd.Timestamp(c)


class Px:
    """t anındaki kontrat fiyatı = t'den önce kapanmış son 15 dk mumunun kapanışı."""

    def __init__(self, k, bar=pd.Timedelta("15min")):
        self.end = (k.index + bar).values
        self.close = k["close"].to_numpy()
        self.k, self.bar = k, bar

    def at(self, t, max_age=pd.Timedelta("2h")):
        i = np.searchsorted(self.end, np.datetime64(t.tz_convert(None) if t.tzinfo else t), side="right") - 1
        if i < 0:
            return np.nan
        if pd.Timestamp(self.end[i], tz="UTC") < t - max_age:
            return np.nan
        return float(self.close[i])

    def window(self, a, b):
        return self.k[(self.k.index >= a) & (self.k.index + self.bar <= b)]


def stats(r, cost_bps):
    """r: işlem başı brüt getiriler (oran). Net = brüt − maliyet."""
    r = pd.Series(r, dtype=float).dropna()
    if len(r) == 0:
        return dict(n=0)
    net = r - cost_bps / 1e4
    half = len(net) // 2
    t = net.mean() / (net.std(ddof=1) / np.sqrt(len(net))) if len(net) > 2 and net.std() > 0 else np.nan
    return dict(n=len(net), brut_bps=round(r.mean() * 1e4, 1), net_bps=round(net.mean() * 1e4, 1),
                kazanma=round((net > 0).mean() * 100, 1), t=round(t, 2),
                ilk_yari_bps=round(net.iloc[:half].mean() * 1e4, 1) if half else np.nan,
                son_yari_bps=round(net.iloc[half:].mean() * 1e4, 1) if half else np.nan,
                toplam_pct=round(net.sum() * 100, 1))


RT = lambda: 2 * (TAKER_BPS + SLIP_BPS)     # gidiş-dönüş maliyet (bps)


# ------------------------------------------------------------------ 1) PRİM (kapalı piyasa sapması)
def test_premium(px, yd):
    """Her (kapanış D → sonraki açılış D') için: açılıştan önce kontratın son kapanışa göre primi.
    Kural: prim > X → short, < −X → long; açılıştan sonra kapat. Hafta sonu ayrıca raporlanır."""
    days = list(yd.index)
    rows = []
    for a, b in zip(days[:-1], days[1:]):
        _, close_a = session_times(a)
        open_b, _ = session_times(b)
        c_stock, o_stock = yd.loc[a, "close"], yd.loc[b, "open"]
        rec = dict(day=b, weekend=(b - a).days > 1, gap=o_stock / c_stock - 1)
        for lbl, t in (("pre4h", open_b - pd.Timedelta("4h")), ("pre1h", open_b - pd.Timedelta("1h")),
                       ("pre15", open_b - pd.Timedelta("15min"))):
            p = px.at(t)
            rec[lbl] = p
            rec[f"prem_{lbl}"] = p / c_stock - 1 if p == p else np.nan
        for lbl, t in (("post15", open_b + pd.Timedelta("15min")), ("post30", open_b + pd.Timedelta("30min")),
                       ("post60", open_b + pd.Timedelta("60min"))):
            rec[lbl] = px.at(t, max_age=pd.Timedelta("20min"))
        rows.append(rec)
    return pd.DataFrame(rows)


def premium_rules(df_all):
    out = []
    for entry in ("pre4h", "pre1h", "pre15"):
        for exit_ in ("post15", "post30", "post60"):
            for X in (0.005, 0.01, 0.02, 0.03):
                for wk in (None, True):
                    d = df_all if wk is None else df_all[df_all["weekend"]]
                    prem, e, x = d[f"prem_{entry}"], d[entry], d[exit_]
                    sig = np.where(prem > X, -1, np.where(prem < -X, 1, 0))
                    r = pd.Series(sig * (x / e - 1), index=d.index)[sig != 0]
                    out.append(dict(giris=entry, cikis=exit_, esik_pct=X * 100, sadece_haftasonu=bool(wk),
                                    **stats(r, RT())))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ 2) FUNDING karşıtı L/S
def test_funding(k_daily, fund, k=2, lookback=14, every=3):
    """k_daily: {sym: günlük kapanış}, fund: {sym: funding serisi}. Her `every` günde bir:
    son `lookback` günün ortalama funding'i en yüksek k → short, en düşük k → long. Eşit ağırlık."""
    px = pd.DataFrame(k_daily).dropna(how="all")
    fd = pd.DataFrame({s: f.resample("1D").sum() for s, f in fund.items() if len(f)}).reindex(px.index).fillna(0.0)
    if px.shape[1] < 2 * k + 1:
        return None, None
    ret = px.pct_change()
    avg = fd.rolling(lookback, min_periods=lookback // 2).mean()
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    cur = pd.Series(0.0, index=px.columns)
    for i, t in enumerate(px.index):
        if i % every == 0 and avg.loc[t].notna().sum() >= 2 * k + 1:
            a = avg.loc[t].dropna().sort_values()
            cur = pd.Series(0.0, index=px.columns)
            cur[a.index[:k]] = 1 / (2 * k)
            cur[a.index[-k:]] = -1 / (2 * k)
        w.loc[t] = cur
    wp = w.shift(1).fillna(0.0)
    turn = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1)
    daily = (wp * ret.fillna(0)).sum(axis=1) - (wp * fd).sum(axis=1) - turn * (TAKER_BPS + SLIP_BPS) / 1e4
    eq = (1 + daily).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    yr = daily.groupby(daily.index.tz_convert(None).to_period("M")).sum()
    summary = dict(gun=len(daily), toplam_pct=round(float(eq.iloc[-1] - 1) * 100, 1), maxdd_pct=round(float(dd) * 100, 1),
                   yillik_sharpe=round(float(daily.mean() / daily.std() * np.sqrt(365)), 2) if daily.std() > 0 else None,
                   karli_ay=f"{(yr > 0).sum()}/{len(yr)}",
                   funding_katki_pct=round(float(-(wp * fd).sum(axis=1).sum()) * 100, 1))
    return summary, fd


# ------------------------------------------------------------------ 3) SEANS AÇILIŞI
def test_open(px, yd):
    """Kontrat 15 dk mumlarıyla (gerçek hissenin seans günleri üzerinden):
    gap = kontratın açılıştaki fiyatı / önceki seans kapanışındaki fiyatı − 1
    a) GAP SÖNÜMLEME: |gap| > g → gap'in tersine gir (açılış), 60 dk / seans sonunda çık
    b) İLK 30 DK MOMENTUM: 09:30–10:00 yönünde 10:00'da gir, seans sonunda çık
    c) ORB: 09:30–10:00 aralığı; sonra ilk kırılımda gir, karşı uç stop, seans sonunda çık"""
    days = list(yd.index)
    rows = []
    for a, b in zip(days[:-1], days[1:]):
        _, ca = session_times(a)
        ob, cb = session_times(b)
        p_prev, p_open = px.at(ca), px.at(ob)
        p30, p60, pclose = px.at(ob + pd.Timedelta("30min")), px.at(ob + pd.Timedelta("60min")), px.at(cb)
        if not all(v == v for v in (p_prev, p_open, p30, pclose)):
            continue
        rec = dict(day=b, gap=p_open / p_prev - 1, r_open_60=p60 / p_open - 1, r_open_close=pclose / p_open - 1,
                   r_30=p30 / p_open - 1, r_30_close=pclose / p30 - 1)
        w = px.window(ob, ob + pd.Timedelta("30min"))
        after = px.window(ob + pd.Timedelta("30min"), cb)
        rec["orb"] = np.nan
        if len(w) and len(after):
            hi, lo = w["high"].max(), w["low"].min()
            for _, bar in after.iterrows():
                if bar["high"] > hi:           # long kırılım; stop = lo
                    exit_ = lo if after.loc[bar.name:, "low"].min() <= lo else pclose
                    rec["orb"] = exit_ / hi - 1
                    break
                if bar["low"] < lo:
                    exit_ = hi if after.loc[bar.name:, "high"].max() >= hi else pclose
                    rec["orb"] = -(exit_ / lo - 1)
                    break
        rows.append(rec)
    return pd.DataFrame(rows)


def open_rules(d):
    out = []
    for g in (0.005, 0.01, 0.02):
        for hold in ("r_open_60", "r_open_close"):
            s = np.sign(d["gap"]) * (d["gap"].abs() > g)
            out.append(dict(kural=f"GAP SÖNÜMLEME |gap|>{g * 100:g}% tut={hold[7:]}", **stats(-s[s != 0] * d.loc[s != 0, hold], RT())))
            out.append(dict(kural=f"GAP DEVAM |gap|>{g * 100:g}% tut={hold[7:]}", **stats(s[s != 0] * d.loc[s != 0, hold], RT())))
    s = np.sign(d["r_30"])
    out.append(dict(kural="İLK 30DK MOMENTUM → kapanış", **stats(s * d["r_30_close"], RT())))
    out.append(dict(kural="İLK 30DK TERS → kapanış", **stats(-s * d["r_30_close"], RT())))
    out.append(dict(kural="ORB 30dk (stop karşı uç)", **stats(d["orb"], RT())))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ ana akış
def run(pairs, days):
    os.makedirs(OUT, exist_ok=True)
    lines = [f"MEXC HİSSE KONTRATLARI ARAŞTIRMASI — {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
             f"maliyet: taraf başı ücret {TAKER_BPS} bps + kayma {SLIP_BPS} bps → gidiş-dönüş {RT():g} bps",
             f"eşleşmeler ({len(pairs)}): " + ", ".join(f"{m}={y}" for m, y in pairs.items()), ""]
    prem_all, open_all, k_daily, fund = [], [], {}, {}
    for sym, tk in pairs.items():
        try:
            k = mexc_klines(sym, "Min15", days)
            yd = yahoo_daily(tk)
        except Exception as e:
            lines.append(f"{sym}/{tk}: veri alınamadı ({e})")
            continue
        if len(k) < 200:
            lines.append(f"{sym}: çok az kontrat verisi ({len(k)} mum) → atlandı")
            continue
        yd = yd[(pd.to_datetime(yd.index).tz_localize(NY) >= k.index[0].tz_convert(NY) - pd.Timedelta("1D"))]
        px = Px(k)
        p = test_premium(px, yd)
        p["sym"] = sym
        prem_all.append(p)
        o = test_open(px, yd)
        o["sym"] = sym
        open_all.append(o)
        k_daily[sym] = k["close"].resample("1D").last()
        try:
            fund[sym] = mexc_funding(sym)
        except Exception as e:
            lines.append(f"{sym}: funding alınamadı ({e})")
        lines.append(f"{sym:<18} {tk:<6} kontrat {k.index[0]:%Y-%m-%d} → {k.index[-1]:%Y-%m-%d} ({len(k)} mum), "
                     f"hisse günleri {len(yd)}, funding kaydı {len(fund.get(sym, []))}")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    if prem_all:
        P = pd.concat(prem_all).sort_values("day").reset_index(drop=True)
        P.to_csv(os.path.join(OUT, "premium_events.csv"), index=False)
        R = premium_rules(P)
        R.to_csv(os.path.join(OUT, "premium_rules.csv"), index=False)
        corr = P[["prem_pre1h", "gap"]].dropna()
        lines += ["", "=" * 100, "1) PRİM — borsa kapalıyken kontrat sapması açılışta kapanıyor mu?",
                  f"olay sayısı {len(P)}; açılıştan 1s önceki prim ile gerçek gap korelasyonu "
                  f"{corr.corr().iloc[0, 1]:.2f} (1'e yakın = kontrat gap'i doğru tahmin ediyor, fırsat az)",
                  "en iyi 12 kural (net bps'e göre, en az 20 işlem):",
                  R[R["n"] >= 20].sort_values("net_bps", ascending=False).head(12).to_string(index=False)]
    if fund:
        lines += ["", "=" * 100, "2) FUNDING — hisse kontratlarında funding karşıtı L/S"]
        fs = pd.DataFrame({s: dict(ort_bps=f.mean() * 1e4, maks_bps=f.max() * 1e4, min_bps=f.min() * 1e4, n=len(f))
                           for s, f in fund.items() if len(f)}).T.round(2)
        lines.append("funding özeti (settlement başına):\n" + fs.to_string())
        for kk in (1, 2, 3):
            for every in (1, 3, 7):
                s, _ = test_funding(k_daily, fund, k=kk, every=every)
                if s:
                    lines.append(f"k={kk} her {every} gün: {s}")
    if open_all:
        O = pd.concat(open_all).sort_values("day").reset_index(drop=True)
        O.to_csv(os.path.join(OUT, "open_events.csv"), index=False)
        R = open_rules(O)
        R.to_csv(os.path.join(OUT, "open_rules.csv"), index=False)
        lines += ["", "=" * 100, f"3) SEANS AÇILIŞI — {len(O)} gün×kontrat", R.to_string(index=False)]
    lines += ["", "OKUMA: net_bps = işlem başı ücret+kayma sonrası ortalama (1 bps = %0.01). Anlamlı sayılması için:",
              "net_bps > 0, t > 2, n ≥ 50 ve ilk_yarı ile son_yarı AYNI işaretli. Çok sayıda kural denendiği için tek",
              "bir iyi satır tesadüf olabilir — komşu eşiklerin de pozitif olması gerekir."]
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "report.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)


def main():
    global TAKER_BPS, SLIP_BPS
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["discover", "run"])
    ap.add_argument("--pairs", help="MEXC=YAHOO,... (elle eşleştirme)")
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--taker-bps", type=float, default=TAKER_BPS)
    ap.add_argument("--slip-bps", type=float, default=SLIP_BPS)
    a = ap.parse_args()
    TAKER_BPS, SLIP_BPS = a.taker_bps, a.slip_bps
    if a.pairs:
        pairs = dict(p.split("=") for p in a.pairs.split(","))
    else:
        items = contracts()
        pairs = discover_pairs(items)
        if a.cmd == "discover":
            stockish = [c.get("symbol") for c in items if "STOCK" in json.dumps(c).upper()]
            print(f"MEXC kontrat sayısı: {len(items)}")
            print(f"hisse eşleşmeleri ({len(pairs)}):")
            for m, y in pairs.items():
                print(f"  {m:<20} → {y}")
            print(f"\n'STOCK' geçen kontratlar ({len(stockish)}): {', '.join(stockish[:80])}")
            return
    if not pairs:
        sys.exit("hisse kontratı bulunamadı — önce 'discover' çalıştırıp --pairs ile elle verin")
    run(pairs, a.days)


if __name__ == "__main__":
    main()
