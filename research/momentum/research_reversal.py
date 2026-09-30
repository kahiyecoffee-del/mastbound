"""REVERSAL (dönüş) stratejileri — 37 coin, 1 saatlik ve günlük mumlar, gerçek MEXC ücreti (taker %0.08 + kayma %0.02).

Aileler (sinyal kapanmış mumla, giriş SONRAKİ mumun açılışında; stop ve hedef aynı mumda ise önce STOP sayılır):
  R1 FLUSH   : 1s mum getirisi ≤ −k·σ (σ = son 20 gün 1s getiri std) [+ hacim ≥ v × 20 gün medyanı] → LONG;
               tersi (≥ +k·σ) → SHORT. Hedef: mumun kaybettiğinin `tp` kadarını geri alması; stop: mum dibi − 0.5 ATR;
               en fazla 24 saat.
  R2 Z24     : son 24 saat getirisi z ≤ −z0 → LONG, ≥ +z0 → SHORT; H saat tut, stop 3 ATR(1s).
  R3 XS      : her gün coinler arası: son `lb` gün en çok düşen n coine LONG, en çok yükselen n coine SHORT, `hold` gün
               (piyasa nötr, brüt 1x). Karşılaştırma için aynı kuralın momentum (ters işaret) hâli.
  R4 RSI2    : günlük; coin 200 gün ortalamasının üstünde ve RSI(2) < eşik → ertesi açılış LONG; kapanış 5 gün
               ortalamasını geçince çık (en fazla 10 gün). İsteğe bağlı BTC > 200g filtresi.
  R5 LİKİDASYON: R1 LONG + son 3 gün funding toplamı > eşik (kalabalık long'lar tasfiye olmuş).

Portföy: işlem başı notional = bakiye × f (varsayılan %20), aynı anda en fazla K=5, aynı coinde tek işlem.
Seçim yalnız 2021-23'e göre; 2024-25.09 ve SON 1 YIL örneklem dışıdır. Canlı funding kolu ile günlük korelasyon.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import config as C
import data
import research_combo as CB
import research_portfolio as RP
import research_strat3 as S3

FEE = 0.0008 + 0.0002          # taraf başı: taker + kayma
OUT = "results/reversal"
LINES = []


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


# ------------------------------------------------------------------ veri
def load(coins, A, Z):
    H, D, FU = {}, {}, {}
    dl = A - pd.Timedelta(days=40)
    for c in coins:
        try:
            m = data.download_vision(c, dl, Z)
        except Exception as e:  # noqa: BLE001
            print(f"[{c}] atlandı: {e}", flush=True)
            continue
        if len(m) < 60 * 24 * 90:
            continue
        H[c] = data.resample(m, "1h")
        D[c] = data.resample(m, "1D")
        del m
        try:
            f, _ = data.download_funding_vision(c, A, Z)
            f = pd.Series(np.asarray(f, float), index=pd.DatetimeIndex(f.index).tz_convert("UTC"))
            FU[c] = f.resample("1D").sum()
        except Exception:  # noqa: BLE001
            pass
        print(f"[{c}] {len(H[c]):,} saatlik mum", flush=True)
    return H, D, FU


def atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# ------------------------------------------------------------------ işlem simülasyonu (saatlik)
def run_trade(o, h, l, c, i, d, stop, tp, max_hold):
    """i: giriş mumu (açılışta). Dönen: (çıkış idx, net getiri, neden)."""
    e = o[i]
    n = len(o)
    for j in range(i, min(n, i + max_hold)):
        if d > 0:
            if l[j] <= stop:
                x = min(o[j], stop) if j > i else stop
                return j, (x / e - 1) - 2 * FEE, "SL"
            if tp is not None and h[j] >= tp:
                x = max(o[j], tp) if j > i else tp
                return j, (x / e - 1) - 2 * FEE, "TP"
        else:
            if h[j] >= stop:
                x = max(o[j], stop) if j > i else stop
                return j, (1 - x / e) - 2 * FEE, "SL"
            if tp is not None and l[j] <= tp:
                x = min(o[j], tp) if j > i else tp
                return j, (1 - x / e) - 2 * FEE, "TP"
    j = min(n, i + max_hold) - 1
    return j, d * (c[j] / e - 1) - 2 * FEE, "TIME"


def cands_hourly(H, FU, fam, p):
    res = []
    for coin, df in H.items():
        o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        v = df["volume"].to_numpy(float)
        t = df.index
        lr = np.log(df["close"]).diff()
        sig = lr.rolling(480, min_periods=240).std().shift(1).to_numpy()
        at = atr(df).to_numpy()
        vmed = df["volume"].rolling(480, min_periods=240).median().shift(1).to_numpy()
        r1 = lr.to_numpy()
        if fam == "Z24":
            r24 = np.log(df["close"]).diff(24).to_numpy()
            z = r24 / (sig * np.sqrt(24))
        fu3 = None
        if fam == "LIQ":
            f = FU.get(coin)
            if f is None:
                continue
            fd = f.rolling(3).sum()
            # günlük funding'i saatlere taşı: gün kapanışında bilinen değer ertesi gün kullanılır
            fu3 = fd.shift(1).reindex(t.floor("1D")).to_numpy()
        busy = -1
        for k in range(500, len(c) - 1):
            if k <= busy or not np.isfinite(sig[k]) or sig[k] <= 0:
                continue
            d = 0
            if fam in ("FLUSH", "LIQ"):
                if r1[k] <= -p["k"] * sig[k]:
                    d = 1
                elif r1[k] >= p["k"] * sig[k] and fam == "FLUSH":
                    d = -1
                if d and p.get("v", 0) and not (v[k] >= p["v"] * vmed[k]):
                    d = 0
                if d and fam == "LIQ" and not (np.isfinite(fu3[k]) and fu3[k] > p["fu"]):
                    d = 0
                if d and p["side"] != 0 and d != p["side"]:
                    d = 0
                if not d:
                    continue
                i = k + 1
                if d > 0:
                    stop = l[k] - 0.5 * at[k]
                    tp = c[k] + p["tp"] * (h[k] - c[k])
                    if tp <= o[i] * (1 + 2 * FEE) or stop >= o[i]:
                        continue
                else:
                    stop = h[k] + 0.5 * at[k]
                    tp = c[k] - p["tp"] * (c[k] - l[k])
                    if tp >= o[i] * (1 - 2 * FEE) or stop <= o[i]:
                        continue
                j, r, why = run_trade(o, h, l, c, i, d, stop, tp, 24)
            elif fam == "Z24":
                if not np.isfinite(z[k]):
                    continue
                if z[k] <= -p["z"]:
                    d = 1
                elif z[k] >= p["z"]:
                    d = -1
                if not d or (p["side"] != 0 and d != p["side"]):
                    continue
                i = k + 1
                stop = o[i] - d * 3 * at[k]
                j, r, why = run_trade(o, h, l, c, i, d, stop, None, p["H"])
            res.append(dict(coin=coin, t_in=t[i], t_out=t[j] + pd.Timedelta("1h"), d=d, ret=r, why=why))
            busy = j
    return res


def cands_rsi2(D, p):
    res = []
    btc = D["BTC"]["close"]
    btc_ok = (btc > btc.rolling(200).mean())
    for coin, df in D.items():
        o, c = df["open"].to_numpy(float), df["close"].to_numpy(float)
        cs = df["close"]
        dlt = cs.diff()
        up = dlt.clip(lower=0).ewm(alpha=1 / 2, adjust=False).mean()
        dn = (-dlt.clip(upper=0)).ewm(alpha=1 / 2, adjust=False).mean()
        rsi = (100 - 100 / (1 + up / dn.replace(0, np.nan))).to_numpy()
        s200 = cs.rolling(200).mean().to_numpy()
        s5 = cs.rolling(p.get("exit_n", 5)).mean().to_numpy()
        max_d = p.get("max_d", 10)
        bok = btc_ok.reindex(df.index).fillna(False).to_numpy()
        t = df.index
        k = 200
        while k < len(c) - 1:
            if c[k] > s200[k] and rsi[k] < p["th"] and (not p["btc"] or bok[k]):
                i = k + 1
                e = o[i]
                j = i
                while j < min(len(c) - 1, i + max_d):
                    if c[j] > s5[j]:
                        break
                    j += 1
                x = o[j + 1] if j + 1 < len(c) else c[j]
                res.append(dict(coin=coin, t_in=t[i], t_out=t[j] + pd.Timedelta("1D"), d=1,
                                ret=(x / e - 1) - 2 * FEE, why="EXIT"))
                k = j + 1
            else:
                k += 1
    return res


def xs_returns(D, lb, hold, n=5, sign=-1):
    """Coinler arası dönüş (sign=-1) / momentum (sign=+1): günlük getiri serisi (brüt 1x, maliyet dahil)."""
    Cl = pd.DataFrame({k: v["close"] for k, v in D.items()}).sort_index()
    r = Cl.pct_change(fill_method=None)
    past = Cl.pct_change(lb, fill_method=None)
    listed = Cl.notna() & Cl.shift(60).notna()
    w = pd.DataFrame(0.0, index=Cl.index, columns=Cl.columns)
    cur = pd.Series(0.0, index=Cl.columns)
    for k, day in enumerate(Cl.index):
        if k % hold == 0:
            s = past.loc[day].where(listed.loc[day]).dropna()
            cur = pd.Series(0.0, index=Cl.columns)
            if len(s) >= 2 * n:
                lo, hi = s.nsmallest(n).index, s.nlargest(n).index
                cur[lo] = -sign * 0.5 / n
                cur[hi] = sign * 0.5 / n
        w.loc[day] = cur
    turn = w.diff().abs().sum(axis=1).fillna(0.0)
    ret = (w.shift(1) * r.fillna(0.0)).sum(axis=1) - FEE * turn
    ret.index = ret.index.tz_convert("UTC") if ret.index.tz else ret.index.tz_localize("UTC")
    return ret


# ------------------------------------------------------------------ portföy
def portfolio(cands, A, Z, f=0.2, K=5):
    """İşlemleri giriş sırasıyla; aynı anda ≤ K, aynı coinde tek. Bakiye çıkışta güncellenir. Günlük getiri döner."""
    cands = sorted(cands, key=lambda x: x["t_in"])
    bal = 1.0
    open_ = []          # (t_out, coin, notional, ret)
    pnl_by_day = {}
    taken = []
    for tr in cands:
        if tr["t_in"] < A or tr["t_in"] >= Z:
            continue
        still = []
        for x in open_:
            if x[0] <= tr["t_in"]:
                bal += x[2] * x[3]
                pnl_by_day[x[0].floor("1D")] = pnl_by_day.get(x[0].floor("1D"), 0.0) + x[2] * x[3]
            else:
                still.append(x)
        open_ = still
        if len(open_) >= K or any(x[1] == tr["coin"] for x in open_) or bal <= 0:
            continue
        open_.append((tr["t_out"], tr["coin"], bal * f, tr["ret"]))
        taken.append(tr)
    for x in open_:
        pnl_by_day[x[0].floor("1D")] = pnl_by_day.get(x[0].floor("1D"), 0.0) + x[2] * x[3]
    days = pd.date_range(A, Z, freq="1D", inclusive="left")
    pnl = pd.Series(pnl_by_day).reindex(days).fillna(0.0)
    eq = 1.0 + pnl.cumsum()
    ret = eq.pct_change().fillna(eq.iloc[0] - 1.0)
    return ret, taken


def row(name, ret, taken, periods, fund_r):
    cells = []
    for pn, a, z in periods:
        s = RP.curve_stats(ret, a, z, start=100.0)
        tk = [x for x in taken if a <= x["t_in"] < z] if taken is not None else None
        n = len(tk) if tk is not None else 0
        avg = np.mean([x["ret"] for x in tk]) * 100 if tk else np.nan
        cells.append(f"{s['cagr']:7.1f} {s['dd']:5.1f} {n:5d} {avg:6.2f}")
    rr = ret.reindex(fund_r.index).fillna(0.0)
    cor = rr.corr(fund_r) if rr.std() > 0 else np.nan
    out(f"{name:48s} | " + " | ".join(cells) + f" | {cor:5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", default=CB.COINS18 + "," + S3.EXTRA)
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--val", default="2024-01-01")
    ap.add_argument("--test", default="2025-09-24")
    ap.add_argument("--end", default="2026-09-24")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    A, V, T_, Z = (pd.Timestamp(x, tz="UTC") for x in (args.start, args.val, args.test, args.end))
    periods = [("2021-23", A, V), ("2024-25.09", V, T_), ("SON 1 YIL", T_, Z)]
    t0 = time.time()
    H, D, FU = load(args.coins.split(","), A, Z)
    out(f"REVERSAL araştırması — {len(H)} coin, taraf başı maliyet %{FEE * 100:.2f}, yükleme {time.time() - t0:.0f} sn")
    ohlc = {k: v[["open", "high", "low", "close"]] for k, v in D.items()}
    fund_r = RP.funding_returns(ohlc, FU, RP.REAL[1] + C.SLIPPAGE)
    fund_r = fund_r[(fund_r.index >= A) & (fund_r.index < Z)]

    hdr = " | ".join(f"{pn:^26s}" for pn, _, _ in periods)
    sub = " | ".join(f"{'CAGR%':>7s} {'DD%':>5s} {'işlem':>5s} {'ort%':>6s}" for _ in periods)
    out(f"\n{'strateji (portföy: işlem başı %20 notional, K=5)':48s} | {hdr} | fundKor")
    out(f"{'':48s} | {sub} |")
    results = {}

    def add(name, cands, **kw):
        ret, taken = portfolio(cands, A, Z, **kw)
        results[name] = (ret, taken)
        row(name, ret, taken, periods, fund_r)

    for side, sn in ((1, "L"), (-1, "S"), (0, "L+S")):
        for k in (3, 4, 5):
            for v in (0, 3):
                for tp in (0.5, 1.0):
                    p = dict(k=k, v=v, tp=tp, side=side)
                    add(f"R1 FLUSH {sn} k{k} hacim{v} tp{tp}", cands_hourly(H, FU, "FLUSH", p))
    for side, sn in ((1, "L"), (-1, "S"), (0, "L+S")):
        for z in (2.5, 3.0, 4.0):
            for Hh in (6, 24, 72):
                add(f"R2 Z24 {sn} z{z} tut{Hh}s", cands_hourly(H, FU, "Z24", dict(z=z, H=Hh, side=side)))
    for th in (5, 10):
        for b in (False, True):
            add(f"R4 RSI2<{th} {'BTC filtreli' if b else ''}", cands_rsi2(D, dict(th=th, btc=b)))
    for k in (3, 4):
        for fu in (0.0, 0.0003):
            add(f"R5 LİKİDASYON k{k} funding3g>{fu}", cands_hourly(H, FU, "LIQ", dict(k=k, v=0, tp=1.0, side=1, fu=fu)))
    for lb, hold in ((1, 1), (3, 3), (7, 7)):
        for sign, nm in ((-1, "DÖNÜŞ"), (1, "momentum")):
            r = xs_returns(D, lb, hold, sign=sign)
            r = r[(r.index >= A) & (r.index < Z)]
            results[f"R3 XS {nm} {lb}g/{hold}g"] = (r, None)
            row(f"R3 XS {nm} {lb}g/{hold}g (piyasa nötr 1x)", r, None, periods, fund_r)
    row("-- canlı funding kolu (karşılaştırma)", fund_r, None, periods, fund_r)

    # yalnız 2021-23'e göre seçim → örneklem dışı
    def sharpe(r, a, z):
        x = r[(r.index >= a) & (r.index < z)]
        return x.mean() / x.std() * np.sqrt(365) if x.std() > 0 else -9
    ranked = sorted(results, key=lambda k: -sharpe(results[k][0], A, V))[:8]
    out("\nSEÇİM (yalnız 2021-23 Sharpe'ına göre ilk 8) → örneklem dışı Sharpe ve funding koluyla birleşim:")
    out(f"{'strateji':48s} {'IS Shrp':>7s} {'OOS1 Shrp':>9s} {'OOS2 Shrp':>9s} | "
        f"{'fund %50 + bu %50: CAGR/DD tüm dönem':>36s} | {'yalnız fund':>12s}")
    fs = RP.curve_stats(fund_r, A, Z, start=100.0)
    for k in ranked:
        r = results[k][0]
        comb = 0.5 * fund_r + 0.5 * r.reindex(fund_r.index).fillna(0.0)
        cs = RP.curve_stats(comb, A, Z, start=100.0)
        out(f"{k:48s} {sharpe(r, A, V):7.2f} {sharpe(r, V, T_):9.2f} {sharpe(r, T_, Z):9.2f} | "
            f"{cs['cagr']:14.1f} / {cs['dd']:5.1f}{'':14s} | {fs['cagr']:5.1f}/{fs['dd']:4.1f}")
    for k in ranked[:3]:
        tk = results[k][1]
        if tk:
            w = pd.Series([x["why"] for x in tk]).value_counts(normalize=True).mul(100).round(0).to_dict()
            yr = pd.Series(results[k][0]).groupby(results[k][0].index.year).apply(lambda s: (1 + s).prod() - 1)
            out(f"  {k}: çıkış nedenleri {w} | yıllık " + ", ".join(f"{y} {v * 100:+.0f}%" for y, v in yr.items()))
    with open(os.path.join(OUT, "summary.txt"), "w") as fh:
        fh.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
