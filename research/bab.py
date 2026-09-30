"""Betting Against Beta (Frazzini & Pedersen 2014) kriptoda: BTC'ye göre düşük betalı coinler, riske göre yüksek betalılardan
daha iyi mi? Beta-nötr L/S (düşük beta long ×1/β, yüksek beta short ×1/β) masraf + funding sonrası kâr eder mi?

Evren: MEXC USDT vadeli, bugünkü 24s cirosuna göre ilk N kripto kontrat (hisse/emtia/stabil hariç). Her tarihte coin ancak
≥180 gün geçmişi ve son 30 gün ort. günlük cirosu ≥ MIN_TURNOVER ise dahil edilir (ileriye bakmadan).
UYARI: bugünün listesiyle seçim hayatta kalma yanlılığı taşır (batmış coinler yok) — bu yanlılık en çok yüksek betalı,
küçük coinleri iyi gösterir, yani BAB'ın aleyhine çalışır.

Beta (FP 2014'ün kriptoya uyarlaması): β = korelasyon(3 günlük log getiri, 180 gün) × σ_coin(60g) / σ_BTC(60g),
sonra β = 0.6·β + 0.4·1 (küçültme). Haftalık yeniden dengeleme (her 7 günde), sinyal t kapanışında → t+1'den uygulanır.

  python research/bab.py              # indir + analiz → results/bab/report.txt
  python research/bab.py --synthetic  # kod testi (sahte veri, düşük betaya küçük bir alfa gömülü)
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from donchian_ensemble import MEXC, funding, get  # noqa: E402

N_COINS = int(os.environ.get("BAB_N", 120))
MIN_TURNOVER = 2e6              # USDT/gün
COST = 0.0005 + 0.0002          # ciro başına (repo varsayımı)
REBAL = 7
PERIODS = [("2021-01-01", "2023-12-31"), ("2024-01-01", "2025-09-23"), ("2025-09-24", "2026-09-29")]
EXCLUDE = {"USDC", "FDUSD", "TUSD", "DAI", "USDE", "USD1", "PYUSD", "XAU", "XAUT", "PAXG", "XAG", "GOLD", "SILVER",
           "USOIL", "UKOIL", "WTI", "BRENT", "NVIDIA", "TESLA", "COINBASE", "ROBINHOOD", "SPY", "QQQ"}
OUT = "results/bab"


# ------------------------------------------------------------------ veri
def universe():
    t = get(f"{MEXC}/ticker").get("data") or []
    d = {c["symbol"]: c for c in (get(f"{MEXC}/detail").get("data") or [])}
    rows = []
    for x in t:
        s = x.get("symbol", "")
        if not s.endswith("_USDT"):
            continue
        base = s[:-5]
        name = str(d.get(s, {}).get("displayNameEn", "")).upper()
        if base in EXCLUDE or base.endswith("STOCK") or "STOCK" in name:
            continue
        rows.append((float(x.get("amount24") or 0), base))
    return [b for _, b in sorted(rows, reverse=True)[:N_COINS]]


def daily(coin, start="2019-06-01"):
    rows = []
    t, end = int(pd.Timestamp(start, tz="UTC").timestamp()), int(time.time())
    while t < end:
        t2 = min(end, t + 1900 * 86400)
        d = get(f"{MEXC}/kline/{coin}_USDT", {"interval": "Day1", "start": t, "end": t2}).get("data") or {}
        if d.get("time"):
            rows.append(pd.DataFrame({k: d[k] for k in ("time", "close", "amount")}))
        t = t2
        time.sleep(0.15)
    if not rows:
        return None
    df = pd.concat(rows).drop_duplicates("time").sort_values("time")
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("time"), unit="s", utc=True)).normalize()
    return df.astype(float)


def load(cache):
    os.makedirs(cache, exist_ok=True)
    coins = universe()
    if "BTC" not in coins:
        coins = ["BTC"] + coins
    C, A, F = {}, {}, {}
    for c in coins:
        p = os.path.join(cache, f"{c}.pkl")
        try:
            if os.path.exists(p):
                df, f = pd.read_pickle(p)
            else:
                df = daily(c)
                if df is None or len(df) < 200:
                    continue
                f = funding(c)
                pd.to_pickle((df, f), p)
        except Exception as e:                      # tek coin hatası testi durdurmasın
            print(f"{c}: atlandı ({e})", flush=True)
            continue
        C[c], A[c], F[c] = df["close"], df["amount"], f
    print(f"evren: {len(C)} coin yüklendi (istenen {len(coins)})", flush=True)
    C = pd.DataFrame(C).sort_index()
    return C, pd.DataFrame(A).reindex(C.index), pd.DataFrame(F).reindex(C.index).fillna(0.0)


def synthetic():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2020-06-01", "2026-09-29", freq="1D", tz="UTC")
    m = rng.normal(0.0005, 0.03, len(idx))
    C, A = {"BTC": pd.Series(100 * np.cumprod(1 + m), idx)}, {}
    for i in range(40):
        b = 0.5 + 2.0 * i / 39
        r = b * m + rng.normal(0, 0.03, len(idx)) + 0.0006 * (1.5 - b)      # düşük betaya pozitif alfa
        s = pd.Series(100 * np.cumprod(1 + np.clip(r, -0.9, None)), idx)
        s.iloc[: rng.integers(0, 700)] = np.nan
        C[f"C{i}"] = s
    C = pd.DataFrame(C)
    A = pd.DataFrame(1e8, index=idx, columns=C.columns).where(C.notna())
    F = pd.DataFrame(rng.normal(1e-4, 1e-4, C.shape), index=idx, columns=C.columns)
    return C, A, F


# ------------------------------------------------------------------ analiz
def betas(R):
    lr = np.log1p(R)
    r3 = lr.rolling(3).sum()
    corr = r3.rolling(180, min_periods=150).corr(r3["BTC"])
    sd = lr.rolling(60, min_periods=50).std()
    b = corr.mul(sd.div(sd["BTC"], axis=0))
    return 0.6 * b + 0.4


def eligible(C, A):
    hist = C.notna().cumsum() >= 180
    liq = A.rolling(30, min_periods=20).mean() >= MIN_TURNOVER
    return hist & liq & C.notna()


def weights(B, E, kind, q=5):
    """Yeniden dengeleme günlerinde hedef ağırlıklar (BTC dahil değil; BTC yalnız 'piyasa')."""
    W = pd.DataFrame(0.0, index=B.index, columns=B.columns)
    days = B.index[::REBAL]
    for d in days:
        b = B.loc[d][E.loc[d]].drop("BTC", errors="ignore").dropna()
        if len(b) < 10:
            continue
        if kind == "bab":                                    # FP 2014: sıra ağırlıklı, her bacak β=1'e ölçeklenir
            z = b.rank()
            k = z - z.mean()
            wl, wh = (-k).clip(lower=0), k.clip(lower=0)
            wl, wh = wl / wl.sum(), wh / wh.sum()
            bl, bh = (wl * b).sum(), (wh * b).sum()
            W.loc[d, b.index] = wl / bl - wh / bh
        elif kind == "ls":                                   # dolar-nötr: en düşük beta dilimi long, en yüksek short
            lo, hi = b <= b.quantile(1 / q), b >= b.quantile(1 - 1 / q)
            W.loc[d, b.index[lo]] = 1 / lo.sum()
            W.loc[d, b.index[hi]] = -1 / hi.sum()
        else:                                                # "qK": K. beta dilimi long-only, eşit ağırlık
            k = int(kind[1:])
            lo_q, hi_q = b.quantile((k - 1) / q), b.quantile(k / q)
            sel = (b >= lo_q) & (b <= hi_q) if k == 1 else (b > lo_q) & (b <= hi_q)
            W.loc[d, b.index[sel]] = 1 / sel.sum()
    W = W.loc[days].reindex(B.index).ffill().fillna(0.0)
    return W


def run(W, R, F, cost=True):
    wp = W.shift(1).fillna(0.0)                              # t kapanışı sinyal → t+1 getiri
    gross = (wp * R.fillna(0.0)).sum(axis=1)
    fund = (wp * F).sum(axis=1)                              # long öder (+funding), short alır
    turn = (W - W.shift(1)).abs().sum(axis=1).shift(1).fillna(0.0)
    return gross - (fund + turn * COST if cost else 0.0), turn


def stats(r, btc):
    r = r.dropna()
    if len(r) < 30 or r.std() == 0:
        return "yetersiz veri"
    ann, vol = r.mean() * 365, r.std() * np.sqrt(365)
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    x = btc.reindex(r.index).fillna(0.0)
    X = np.column_stack([np.ones(len(r)), x])
    bt, *_ = np.linalg.lstsq(X, r.to_numpy(), rcond=None)
    res = r.to_numpy() - X @ bt
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * res.var())
    return (f"yıllık %{ann * 100:+6.1f} vol %{vol * 100:5.1f} Sharpe {ann / vol:+.2f} maxDD %{dd * 100:6.1f} "
            f"| BTC betası {bt[1]:+.2f} alfa %{bt[0] * 365 * 100:+6.1f}/yıl (t{bt[0] / se[0] if se[0] > 1e-9 else 0:+.1f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    C, A, F = synthetic() if a.synthetic else load(os.path.join(OUT, "cache"))
    R = C.pct_change(fill_method=None).clip(-0.9, 3)
    B = betas(R)
    E = eligible(C, A)
    btc = R["BTC"]
    n_el = E.drop(columns="BTC", errors="ignore").sum(axis=1)
    L = [f"BETTING AGAINST BETA (kripto) — {pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M} UTC" + ("  [SENTETİK]" if a.synthetic else ""),
         f"evren {C.shape[1]} coin; uygun coin sayısı (ort/min/max): {n_el[n_el > 0].mean():.0f}/{n_el[n_el > 0].min()}/{n_el.max()}",
         f"maliyet: ciro×%{COST * 100:.2f} + funding (MEXC funding geçmişi yalnız ~son 540 gün; öncesi 0). Haftalık denge.",
         "dönemler: " + " / ".join(f"{x}→{y}" for x, y in PERIODS) + " (son = görülmemiş)", ""]

    L.append("== 1) BETA DİLİMLERİ (long-only, eşit ağırlık, masrafsız) — FP'nin ana iddiası: beta arttıkça Sharpe düşer")
    for k in range(1, 6):
        r, _ = run(weights(B, E, f"q{k}"), R, F, cost=False)
        L.append(f"Q{k} {'(en düşük β)' if k == 1 else '(en yüksek β)' if k == 5 else '':<13} {stats(r.loc[PERIODS[0][0]:], btc)}")
    L.append(f"{'BTC':<16} {stats(btc.loc[PERIODS[0][0]:], btc)}")
    L.append("")

    L.append("== 2) STRATEJİLER (masraf + funding sonrası)")
    for name, kind in (("BAB beta-nötr (FP 2014)", "bab"), ("Dolar-nötr Q1 long / Q5 short", "ls")):
        W = weights(B, E, kind)
        r, turn = run(W, R, F)
        L.append(f"{name}: yıllık ciro {turn.loc[PERIODS[0][0]:].mean() * 365:.0f}x, ort. brüt kaldıraç "
                 f"{W.abs().sum(axis=1).loc[PERIODS[0][0]:].mean():.2f}x")
        L.append(f"   tümü      {stats(r.loc[PERIODS[0][0]:], btc)}")
        for x, y in PERIODS:
            L.append(f"   {x[:7]}→{y[:7]} {stats(r.loc[x:y], btc)}")
        yrs = r.loc[PERIODS[0][0]:].groupby(r.loc[PERIODS[0][0]:].index.year).apply(lambda s: (1 + s).prod() - 1)
        L.append("   yıllar: " + "  ".join(f"{y}:{v * 100:+.0f}%" for y, v in yrs.items()))
        rv = r.rolling(60, min_periods=20).std().shift(1) * np.sqrt(365)
        rs = r * (0.20 / rv).clip(upper=3).fillna(1.0)
        eq = (1 + rs.loc[PERIODS[0][0]:]).cumprod()
        L.append(f"   %20 hedef vol ile: 100→{eq.iloc[-1] * 100:.0f}  (canlı botla karşılaştırmak için)")
        L.append("")

    L.append("== 3) SAĞLAMLIK — BAB, beta penceresi ve dengeleme sıklığı")
    global REBAL
    base = REBAL
    for rb in (1, 7, 30):
        REBAL = rb
        r, _ = run(weights(B, E, "bab"), R, F)
        L.append(f"dengeleme {rb:>2} gün   {stats(r.loc[PERIODS[0][0]:], btc)}")
    REBAL = base
    L.append("")
    L.append("OKUMA: Q1→Q5 Sharpe/alfa düzenli düşüyorsa etki var. Strateji ancak her dönemde (özellikle görülmemiş son")
    L.append("dönemde) masraf+funding sonrası pozitif ve alfa t>2 ise canlıya aday; kısa taraf squeeze riski maxDD'de görünür.")
    txt = "\n".join(L)
    print(txt)
    open(os.path.join(OUT, "report.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
