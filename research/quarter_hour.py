"""Çeyrek saat emir dengesizliği (Kim & Hansen 2026 "Quarter-Hour Effect"): saatin :00/:15/:30/:45 dakikalarındaki
alıcı-satıcı dengesizliği sonraki saatlerin getirisini tahmin ediyor mu? Bizim coinlerde, masraftan sonra?

Veri: Binance USDT-M vadeli 1 dakikalık mumlar (data.binance.vision, herkese açık arşiv). Her mumda taker alış hacmi var:
  dengesizlik = (alış − satış) / toplam = (2·taker_alış − hacim) / hacim
Her saat kapanışında (canlı botun çalıştığı an) üç özellik hesaplanır:
  QH   = saatin çeyrek dakikalarındaki (:00 :15 :30 :45) dengesizlik
  KONT = kontrol: çeyrek olmayan dakikalar (:07 :22 :37 :52) — etki gerçekten çeyreğe özgü mü?
  SAAT = saatin tüm 60 dakikasının dengesizliği (genel emir akışı)
Her biri coin başına son 30 günün ortalama/sapmasıyla z-skoruna çevrilir (yalnız geçmiş veri).
Getiri: saat kapanışından h saat sonrasına (h = 1, 4, 8, 12). Emir göndermez.

  python research/quarter_hour.py              # indir + analiz → results/quarter_hour/report.txt
  python research/quarter_hour.py --synthetic  # kod testi (sahte veri, içine küçük bir etki gömülü)
"""
import argparse
import io
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

URL = "https://data.binance.vision/data/futures/um/monthly/klines/{s}/1m/{s}-1m-{y}-{m:02d}.zip"
COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "AVAX", "LINK", "ADA", "DOT", "LTC", "TRX", "ATOM", "NEAR",
         "UNI", "SUI", "APT", "OP"]                      # canlı botun listesi
MONTHS = pd.period_range("2023-01", "2026-08", freq="M")
PERIODS = [("2023-01-01", "2024-12-31"), ("2025-01-01", "2025-08-31"), ("2025-09-01", "2026-08-31")]
HORIZONS = [1, 4, 8, 12]
COST_RT = 2 * (0.0005 + 0.0002)   # gidiş-dönüş: repo varsayımı (taker %0.05 + kayma %0.02) × 2 = %0.14
OUT = "results/quarter_hour"
FEATS = ["QH", "KONT", "SAAT"]


# ------------------------------------------------------------------ veri
def month(sym, p):
    for i in range(4):
        try:
            r = requests.get(URL.format(s=sym, y=p.year, m=p.month), timeout=120)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            df = pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=[0, 4, 5, 9])
            df = df[pd.to_numeric(df[0], errors="coerce").notna()].astype(float)   # başlık satırı varsa at
            df.columns = ["t", "close", "vol", "buy"]
            if df["t"].iloc[0] > 1e14:                   # mikro saniye biçimi
                df["t"] //= 1000
            return df
        except (requests.RequestException, zipfile.BadZipFile) as e:
            err = e
    raise RuntimeError(f"{sym} {p}: {err}")


def minutes(coin):
    sym = f"{coin}USDT"
    with ThreadPoolExecutor(8) as ex:
        parts = [d for d in ex.map(lambda p: month(sym, p), MONTHS) if d is not None]
    if not parts:
        return None
    return pd.concat(parts).drop_duplicates("t").sort_values("t")


def hourly(m):
    """Dakika verisinden saatlik özellikler + saat kapanış fiyatı. İndeks = saat BİTİŞ anı (sinyal bu anda bilinir)."""
    t = m["t"].to_numpy().astype(np.int64) // 60000
    mo = t % 60
    sv = 2 * m["buy"].to_numpy() - m["vol"].to_numpy()
    v = m["vol"].to_numpy()
    hour = t // 60
    d = pd.DataFrame({"h": hour, "sv": sv, "v": v,
                      "qsv": np.where(mo % 15 == 0, sv, 0.0), "qv": np.where(mo % 15 == 0, v, 0.0),
                      "ksv": np.where(mo % 15 == 7, sv, 0.0), "kv": np.where(mo % 15 == 7, v, 0.0),
                      "close": m["close"].to_numpy(), "n": 1})
    g = d.groupby("h").agg(sv=("sv", "sum"), v=("v", "sum"), qsv=("qsv", "sum"), qv=("qv", "sum"),
                           ksv=("ksv", "sum"), kv=("kv", "sum"), close=("close", "last"), n=("n", "sum"))
    g = g[g["n"] >= 55]
    idx = pd.to_datetime((g.index.to_numpy() + 1) * 3600, unit="s", utc=True)
    out = pd.DataFrame({"QH": (g["qsv"] / g["qv"]).to_numpy(), "KONT": (g["ksv"] / g["kv"]).to_numpy(),
                        "SAAT": (g["sv"] / g["v"]).to_numpy(), "close": g["close"].to_numpy()}, index=idx)
    full = pd.date_range(out.index[0], out.index[-1], freq="1h")
    return out.replace([np.inf, -np.inf], np.nan).reindex(full)


def load(cache):
    os.makedirs(cache, exist_ok=True)
    H = {}
    for c in COINS:
        p = os.path.join(cache, f"{c}.pkl")
        if os.path.exists(p):
            H[c] = pd.read_pickle(p)
        else:
            m = minutes(c)
            if m is None:
                print(f"{c}: veri yok, atlandı", flush=True)
                continue
            H[c] = hourly(m)
            del m
            H[c].to_pickle(p)
        print(f"{c}: {H[c].index[0]:%Y-%m-%d} → {H[c].index[-1]:%Y-%m-%d}, {H[c]['close'].notna().sum()} saat",
              flush=True)
    return H


def synthetic():
    """Sahte dakika verisi: QH dengesizliği sonraki 4 saatin getirisine küçük bir pozitif katkı yapar."""
    rng = np.random.default_rng(1)
    H = {}
    for c in COINS[:6]:
        n = 60 * 24 * 500
        t = (pd.Timestamp("2024-06-01", tz="UTC").value // 10**6) + np.arange(n) * 60000
        mo = (t // 60000) % 60
        imb = rng.normal(0, 0.15, n) + np.where(mo % 15 == 0, rng.normal(0, 0.3, n), 0)
        vol = rng.lognormal(3, 0.5, n) * np.where(mo % 15 == 0, 3, 1)
        buy = vol * (1 + np.clip(imb, -0.95, 0.95)) / 2
        qsig = pd.Series(np.where(mo % 15 == 0, imb, 0.0)).rolling(60).sum().shift(1).fillna(0).to_numpy()
        r = rng.normal(0, 0.0008, n) + 0.00004 * qsig / 4
        m = pd.DataFrame({"t": t.astype(float), "close": 100 * np.exp(np.cumsum(r)), "vol": vol, "buy": buy})
        H[c] = hourly(m)
    return H


# ------------------------------------------------------------------ analiz
def panel(H):
    rows = []
    for c, h in H.items():
        d = pd.DataFrame(index=h.index)
        for f in FEATS:
            mu = h[f].rolling(720, min_periods=240).mean()
            sd = h[f].rolling(720, min_periods=240).std()
            d[f] = (h[f] - mu) / sd
        for k in HORIZONS:
            d[f"r{k}"] = h["close"].shift(-k) / h["close"] - 1
        d["coin"] = c
        rows.append(d)
    return pd.concat(rows).dropna(subset=FEATS).sort_index()


def ic_table(P, lines):
    """Bilgi katsayısı (özellik z ↔ ileri getiri korelasyonu). t ≈ IC·√(N/h) (örtüşen getiri düzeltmesi)."""
    lines.append("== 1) TAHMİN GÜCÜ — IC (korelasyon, ×100) ve t; |t|>3 anlamlı sayılır")
    head = f"{'özellik':<6} {'ufuk':>4} | " + " | ".join(f"{a[:7]}→{b[:7]:<7}" for a, b in PERIODS) + " | tümü"
    lines.append(head)
    ics = {}
    for f in FEATS:
        for k in HORIZONS:
            cells = []
            for a, b in PERIODS + [(None, None)]:
                s = P if a is None else P.loc[a:b]
                s = s[[f, f"r{k}"]].dropna()
                ic = s[f].corr(s[f"r{k}"])
                t = ic * np.sqrt(len(s) / k)
                cells.append(f"{ic * 100:+5.2f} t{t:+5.1f}")
                ics[(f, k, a)] = ic
            lines.append(f"{f:<6} {k:>3}s | " + " | ".join(cells))
    lines.append("")
    return ics


def incremental(P, lines):
    """QH, genel saatlik akış (SAAT) ve kontrol dakikaları sabitken ek bilgi taşıyor mu? (çoklu regresyon, bps/1σ)"""
    lines.append("== 2) QH'NİN EK KATKISI — getiri = a + b1·QH + b2·KONT + b3·SAAT  (b: 1σ başına bps, t örtüşme düzeltmeli)")
    for k in HORIZONS:
        cells = []
        for a, b in PERIODS:
            s = P.loc[a:b][FEATS + [f"r{k}"]].dropna()
            X = np.column_stack([np.ones(len(s))] + [s[f].to_numpy() for f in FEATS])
            y = s[f"r{k}"].to_numpy()
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            res = y - X @ beta
            cov = np.linalg.inv(X.T @ X) * res.var() * k
            t = beta / np.sqrt(np.diag(cov))
            cells.append("  ".join(f"{f} {beta[i + 1] * 1e4:+.1f}(t{t[i + 1]:+.1f})" for i, f in enumerate(FEATS)))
        lines.append(f"{k:>3}s | " + " || ".join(cells))
    lines.append("")


def trade(P, feat, k, thr, sign):
    """Saat kapanışında |z|>eşik ise sign·yön ile gir, k saat tut; coin başına örtüşmesiz. Net = brüt − %0.14."""
    out = []
    for c, d in P.groupby("coin"):
        d = d[[feat, f"r{k}"]].dropna()
        z, r, ts = d[feat].to_numpy(), d[f"r{k}"].to_numpy(), d.index
        busy = None
        for i in range(len(d)):
            if busy is not None and ts[i] < busy:
                continue
            if abs(z[i]) > thr:
                g = sign * np.sign(z[i]) * r[i]
                out.append((ts[i], c, g, g - COST_RT))
                busy = ts[i] + pd.Timedelta(hours=k)
    return pd.DataFrame(out, columns=["t", "coin", "gross", "net"]).set_index("t").sort_index()


def trade_table(P, ics, lines):
    lines.append("== 3) İŞLEM KURALI (yön 1. dönemin IC işaretinden seçilir → 2. ve 3. dönem gerçek sınav)")
    lines.append(f"   maliyet gidiş-dönüş %{COST_RT * 100:.2f}; ort. = işlem başı bps; dönemler: n / brüt / net / t(net)")
    best = []
    for f in FEATS:
        for k in HORIZONS:
            sign = 1 if ics[(f, k, PERIODS[0][0])] >= 0 else -1
            for thr in (1.0, 1.5, 2.0):
                T = trade(P, f, k, thr, sign)
                cells = []
                for a, b in PERIODS:
                    s = T.loc[a:b]
                    t = s["net"].mean() / s["net"].std() * np.sqrt(len(s)) if len(s) > 2 else 0
                    cells.append(f"{len(s):>5} {s['gross'].mean() * 1e4:+6.1f} {s['net'].mean() * 1e4:+6.1f} t{t:+5.1f}")
                oos = T.loc[PERIODS[1][0]:]
                best.append((oos["net"].mean() if len(oos) else -1, f, k, thr))
                lines.append(f"{f:<5}{'+' if sign > 0 else '−'} {k:>2}s z>{thr:<3} | " + " | ".join(cells))
    lines.append("")
    return sorted(best, reverse=True)[:5]


def hour_of_day(P, lines):
    """QH etkisi hangi UTC saatlerinde güçlü? (4s ufuk, tüm dönem; IC×100)"""
    lines.append("== 4) UTC SAATİNE GÖRE QH IC (4 saat ufuk, ×100) — tek saatlerde yoğunlaşıyorsa daha dar bir kural kurulabilir")
    s = P[["QH", "r4"]].dropna()
    hrs = s.index.hour
    cells = [f"{h:02d}:{s[hrs == h]['QH'].corr(s[hrs == h]['r4']) * 100:+5.1f}" for h in range(24)]
    for i in range(0, 24, 8):
        lines.append("   " + "  ".join(cells[i:i + 8]))
    lines.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    H = synthetic() if a.synthetic else load(os.path.join(OUT, "cache"))
    P = panel(H)
    lines = [f"ÇEYREK SAAT EMİR DENGESİZLİĞİ araştırması — {pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M} UTC"
             + ("  [SENTETİK]" if a.synthetic else ""),
             f"coin {len(H)}, gözlem {len(P)} (coin×saat); dönemler: "
             + " / ".join(f"{x}→{y}" for x, y in PERIODS) + "  (son dönem = görülmemiş)", ""]
    ics = ic_table(P, lines)
    incremental(P, lines)
    best = trade_table(P, ics, lines)
    hour_of_day(P, lines)
    lines.append("EN İYİ 5 KURAL (2025+ net ort., bps): " + "; ".join(f"{f} {k}s z>{t} {n * 1e4:+.1f}" for n, f, k, t in best))
    lines.append("")
    lines.append("OKUMA: Anlamlı bir etki için QH'nin IC'si her dönemde aynı işaretli ve |t|>3 olmalı, KONT'tan açıkça güçlü")
    lines.append("olmalı (yoksa çeyreğe özgü değil, genel emir akışıdır) ve işlem kuralı 2025+ dönemde masraftan sonra kârlı kalmalı.")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(OUT, "report.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
