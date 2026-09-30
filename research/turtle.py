"""TURTLE ve geliştirilmiş trend takibi — kripto dışı piyasalar (hisse endeksi, tahvil, metal, enerji, tarım, döviz).

Veri: Yahoo Finance günlük, temettü/bölünme düzeltilmiş ETF'ler (vadeli kontrat yerine; vadeli sürekli serilerdeki
vade geçiş boşlukları sahte trend üretir). Vadeli gibi davranması için pozisyonlar ARTIK getiri kazanır
(ETF getirisi − risksiz faiz, ^IRX); hesap bakiyesi risksiz faiz kazanır. Ücret: taraf başı 5 bps (notional).

İki aile:
  T) Olay bazlı orijinal Turtle: S1 20 gün kırılım (önceki S1 kârlıysa atla) / S2 55 gün; çıkış 10 / 20 gün
     karşı kırılım; 2N stop; birim = %1 bakiye / N; her 0.5N'de ek birim (en fazla 4); sınırlar: piyasa 4,
     küme 6, yön 12 birim; brüt notional ≤ 5× bakiye.
  M) Modern (Carver / Moskowitz-Ooi-Pedersen): sinyal [-1, 1], pozisyon = sinyal × hedef oynaklık / piyasa oynaklığı.
     Parametreler literatür varsayılanları — optimize EDİLMEDİ; 2019+ örneklem dışı olarak ayrıca raporlanır.

  python research/turtle.py            → results/turtle/report.txt
"""
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

YAHOO = "https://query2.finance.yahoo.com/v8/finance/chart"
MEXC = "https://contract.mexc.com/api/v1/contract"
NY = ZoneInfo("America/New_York")
UA = {"User-Agent": "Mozilla/5.0 (research script)"}
OUT = "results/turtle"
CACHE = os.path.join(OUT, "cache")

# ticker → küme
UNIVERSE = {
    "SPY": "hisse", "QQQ": "hisse", "IWM": "hisse", "EFA": "hisse", "EEM": "hisse", "EWJ": "hisse",
    "TLT": "tahvil", "IEF": "tahvil", "TIP": "tahvil",
    "GLD": "metal", "SLV": "metal", "DBB": "metal",
    "USO": "enerji", "UNG": "enerji",
    "DBA": "tarım",
    "FXE": "döviz", "FXY": "döviz", "FXA": "döviz", "FXF": "döviz",
}
COST = 0.0005
IS_END = "2018-12-31"
LINES = []


def out(s=""):
    print(s)
    LINES.append(s)


def get(url, params=None, tries=4):
    err = None
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


def yahoo(tk):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{tk.replace('^', '_')}.pkl")
    if os.path.exists(p) and time.time() - os.path.getmtime(p) < 12 * 3600:
        return pd.read_pickle(p)
    j = get(f"{YAHOO}/{tk}", {"period1": 631152000, "period2": int(time.time()), "interval": "1d",
                              "events": "div,split"})
    r = j["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    idx = pd.DatetimeIndex([datetime.fromtimestamp(t, NY).date() for t in r["timestamp"]])
    df = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close")}, index=idx, dtype=float)
    adj = (r["indicators"].get("adjclose") or [{}])[0].get("adjclose")
    if adj is not None:
        f = pd.Series(adj, index=idx, dtype=float) / df["close"]
        for k in ("open", "high", "low", "close"):
            df[k] = df[k] * f
    df = df[~df.index.duplicated(keep="last")].dropna()
    df = df[(df["high"] >= df["low"]) & (df["close"] > 0)]
    df.to_pickle(p)
    return df


def load():
    data = {}
    for tk in UNIVERSE:
        try:
            data[tk] = yahoo(tk)
        except Exception as e:  # noqa: BLE001
            out(f"  ! {tk} indirilemedi: {e}")
    cal = data["SPY"].index
    cal = cal[cal >= pd.Timestamp("2000-01-01")]
    px = {k: pd.DataFrame({tk: d[k].reindex(cal) for tk, d in data.items()}) for k in ("open", "high", "low", "close")}
    try:
        irx = yahoo("^IRX")["close"].reindex(cal).ffill().fillna(0.0)
    except Exception as e:  # noqa: BLE001
        out(f"  ! ^IRX indirilemedi ({e}) → risksiz faiz 0 kabul edildi")
        irx = pd.Series(0.0, index=cal)
    rf = (irx / 100.0 / 252.0).clip(lower=0.0)
    ext = {}
    for tk in ("BTC-USD",):
        try:
            ext[tk] = yahoo(tk)["close"]
        except Exception as e:  # noqa: BLE001
            out(f"  ! {tk} indirilemedi: {e}")
    return px, rf, ext, cal


# ------------------------------------------------------------------ istatistik
def stats(ret, rf):
    ret = ret.dropna()
    if len(ret) < 50:
        return None
    eq = (1 + ret).cumprod()
    yrs = len(ret) / 252.0
    ex = ret - rf.reindex(ret.index).fillna(0.0)
    vol = ret.std() * np.sqrt(252)
    sh = ex.mean() / ex.std() * np.sqrt(252) if ex.std() > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    return dict(cagr=eq.iloc[-1] ** (1 / yrs) - 1, vol=vol, sharpe=sh, dd=dd)


def norm(ret, rf, target=0.20):
    """Ex-post aynı oynaklığa (%20) ölçekle — yalnız varyantları karşılaştırmak için."""
    r = rf.reindex(ret.index).fillna(0.0)
    ex = ret - r
    v = ex.std() * np.sqrt(252)
    return r + ex * (target / v) if v > 0 else ret


def yearly(ret):
    return (1 + ret).groupby(ret.index.year).prod() - 1


# ------------------------------------------------------------------ M) modern çerçeve
def forecasts(C, kind):
    r = C.pct_change(fill_method=None)
    sig_d = r.ewm(span=35, min_periods=60).std()
    if kind == "tsmom":
        return np.sign(C / C.shift(252) - 1)
    if kind.startswith("breakout"):
        f = 0
        ns = [20, 40, 80, 160, 320] if kind == "breakout_ens" else [55]
        for n in ns:
            hi, lo = C.rolling(n, min_periods=n).max(), C.rolling(n, min_periods=n).min()
            raw = ((C - (hi + lo) / 2) / (hi - lo).replace(0, np.nan)) * 2
            f = f + raw.ewm(span=max(2, n // 4)).mean()
        return (f / len(ns)).clip(-1, 1)
    if kind == "ewmac_ens":
        f = 0
        for fast in (8, 16, 32, 64):
            raw = (C.ewm(span=fast).mean() - C.ewm(span=4 * fast).mean()) / (C * sig_d)
            scale = raw.abs().expanding(min_periods=250).mean()
            f = f + (raw / scale * 0.5).clip(-1, 1)
        return f / 4
    raise ValueError(kind)


def turtle_binary(C, entry=55, exit_=20):
    """Turtle giriş/çıkış mantığı (kapanışla): 55 gün zirve → long, 20 gün dip → çık; tersi short."""
    hi_e, lo_e = C.rolling(entry).max().shift(1), C.rolling(entry).min().shift(1)
    hi_x, lo_x = C.rolling(exit_).max().shift(1), C.rolling(exit_).min().shift(1)
    res = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    for tk in C.columns:
        c, he, le, hx, lx = (a[tk].to_numpy() for a in (C, hi_e, lo_e, hi_x, lo_x))
        pos, arr = 0, np.zeros(len(c))
        for t in range(len(c)):
            if np.isnan(c[t]) or np.isnan(he[t]):
                arr[t] = pos if not np.isnan(c[t]) else 0
                continue
            if pos > 0 and c[t] < lx[t]:
                pos = 0
            elif pos < 0 and c[t] > hx[t]:
                pos = 0
            if pos == 0:
                if c[t] > he[t]:
                    pos = 1
                elif c[t] < le[t]:
                    pos = -1
            arr[t] = pos
        res[tk] = arr
    return res


def run_modern(C, rf, F, target=0.20, idm=2.0, cluster=False, buffer=0.0, cost=COST, gross_cap=4.0):
    r = C.pct_change(fill_method=None)
    sig = (r.ewm(span=35, min_periods=60).std() * np.sqrt(252)).clip(lower=0.03)
    hist = C.notna().cumsum()
    active = (hist >= 260) & sig.notna() & C.notna()
    F = F.where(active, 0.0).fillna(0.0)
    if cluster:
        cl = pd.Series(UNIVERSE)[C.columns]
        share = pd.DataFrame(0.0, index=C.index, columns=C.columns)
        n_cl = pd.Series(0, index=C.index)
        for g in cl.unique():
            cols = cl.index[cl == g]
            n_cl += (active[cols].sum(axis=1) > 0).astype(int)
        for g in cl.unique():
            cols = cl.index[cl == g]
            n_in = active[cols].sum(axis=1).replace(0, np.nan)
            for c_ in cols:
                share[c_] = (1.0 / n_cl.replace(0, np.nan)) / n_in
        share = share.where(active, 0.0).fillna(0.0)
    else:
        n = active.sum(axis=1).replace(0, np.nan)
        share = active.astype(float).div(n, axis=0).fillna(0.0)
    base = (target / sig * idm * share).where(active, 0.0).fillna(0.0)
    W = F * base
    g = W.abs().sum(axis=1)
    W = W.mul((gross_cap / g).clip(upper=1.0).fillna(1.0), axis=0)
    if buffer > 0:
        Wv, Bv = W.to_numpy(), (base.to_numpy() * buffer)
        cur, res = np.zeros(Wv.shape[1]), np.zeros_like(Wv)
        for t in range(len(Wv)):
            dev = Wv[t] - cur
            move = np.abs(dev) > Bv[t]
            cur = np.where(move, Wv[t] - np.sign(dev) * Bv[t], cur)
            cur = np.where(base.to_numpy()[t] == 0, 0.0, cur)
            res[t] = cur
        W = pd.DataFrame(res, index=W.index, columns=W.columns)
    x = r.sub(rf, axis=0).fillna(0.0)
    turn = W.diff().abs().sum(axis=1).fillna(0.0)
    ret = rf + (W.shift(1) * x).sum(axis=1) - cost * turn
    first = active.any(axis=1).idxmax()
    ret = ret[ret.index >= first + pd.Timedelta(days=5)]
    return ret, turn.reindex(ret.index).mean() * 252, W


# ------------------------------------------------------------------ T) olay bazlı Turtle
def run_turtle(px, rf, risk=0.01, pyramid=True, skip_rule=True, chandelier=False, cost=COST, gross_cap=5.0,
               cap_market=4, cap_cluster=6, cap_dir=12):
    O, H, L, C = (px[k] for k in ("open", "high", "low", "close"))
    cols = list(C.columns)
    prevC_all = C.ffill().shift(1)
    tr = pd.concat([H - L, (H - prevC_all).abs(), (L - prevC_all).abs()]).groupby(level=0).max()
    tr = tr.reindex(C.index)[cols]
    N = tr.ewm(alpha=1 / 20, min_periods=20).mean().shift(1)
    lv = {}
    for n in (10, 20, 55):
        lv[("hi", n)] = H.rolling(n, min_periods=n).max().shift(1).to_numpy()
        lv[("lo", n)] = L.rolling(n, min_periods=n).min().shift(1).to_numpy()
    O_, H_, L_, C_, N_ = (a.to_numpy() for a in (O, H, L, C, N))
    rfv = rf.to_numpy()
    clus = [UNIVERSE[c] for c in cols]
    T, K = C_.shape
    E = 1.0
    eq = np.full(T, np.nan)
    pos = [None] * K           # dict(dir, qty, units, sys, stop, last, Ne, ext, tpnl)
    last_c = np.full(K, np.nan)
    last_win = np.zeros(K, bool)
    trades = []
    started = False
    for t in range(T):
        pnl = 0.0
        E0 = E
        for i in range(K):
            if np.isnan(C_[t, i]):
                continue
            o, h, l, c, n = O_[t, i], H_[t, i], L_[t, i], C_[t, i], N_[t, i]
            pc = last_c[i]
            p = pos[i]
            if p is not None:
                d = p["dir"]
                pnl -= p["qty"] * d * pc * rfv[t]
                exit_lvl = p["stop"]
                if not chandelier:
                    ch = lv[("lo", 10 if p["sys"] == 1 else 20)][t, i] if d > 0 else \
                        lv[("hi", 10 if p["sys"] == 1 else 20)][t, i]
                    if not np.isnan(ch):
                        exit_lvl = max(exit_lvl, ch) if d > 0 else min(exit_lvl, ch)
                hit = (l <= exit_lvl) if d > 0 else (h >= exit_lvl)
                if hit:
                    xp = min(o, exit_lvl) if d > 0 else max(o, exit_lvl)
                    g = p["qty"] * d * (xp - pc) - cost * p["qty"] * xp
                    pnl += g
                    p["tpnl"] += g
                    trades.append(p["tpnl"])
                    if p["sys"] == 1:
                        last_win[i] = p["tpnl"] > 0
                    pos[i] = None
                else:
                    g = p["qty"] * d * (c - pc)
                    pnl += g
                    p["tpnl"] += g
                    if pyramid:
                        while p["units"] < cap_market:
                            lvl = p["last"] + 0.5 * p["Ne"] * d
                            if not ((h >= lvl) if d > 0 else (l <= lvl)):
                                break
                            if not _room(pos, clus, i, d, cap_cluster, cap_dir, C_[t], E0, gross_cap, p["uq"] * c):
                                break
                            fp = max(o, lvl) if d > 0 else min(o, lvl)
                            g = p["uq"] * d * (c - fp) - cost * p["uq"] * fp
                            pnl += g
                            p["tpnl"] += g
                            p["qty"] += p["uq"]
                            p["units"] += 1
                            p["last"] = fp
                            p["stop"] = fp - 2 * p["Ne"] * d
                    if chandelier:
                        p["ext"] = max(p["ext"], h) if d > 0 else min(p["ext"], l)
                        if not np.isnan(n):
                            tr_ = p["ext"] - 3 * n * d
                            p["stop"] = max(p["stop"], tr_) if d > 0 else min(p["stop"], tr_)
            elif not np.isnan(n) and n > 0 and not np.isnan(lv[("hi", 55)][t, i]):
                sig = None
                for sysn, ln in ((1, 20), (2, 55)):
                    if sysn == 1 and skip_rule and last_win[i]:
                        # önceki S1 kârlıysa bu S1 sinyali atlanır; bayrak sıfırlanır (bir sonraki S1 alınır)
                        if h > lv[("hi", 20)][t, i] or l < lv[("lo", 20)][t, i]:
                            last_win[i] = False
                        continue
                    up, dn = h > lv[("hi", ln)][t, i], l < lv[("lo", ln)][t, i]
                    if up and not dn:
                        sig = (sysn, 1, lv[("hi", ln)][t, i])
                    elif dn and not up:
                        sig = (sysn, -1, lv[("lo", ln)][t, i])
                    if sig:
                        break
                if sig:
                    sysn, d, lvl = sig
                    uq = risk * E0 / n
                    if _room(pos, clus, i, d, cap_cluster, cap_dir, C_[t], E0, gross_cap, uq * c):
                        fp = max(o, lvl) if d > 0 else min(o, lvl)
                        st = fp - 2 * n * d
                        p = dict(dir=d, qty=uq, uq=uq, units=1, sys=sysn, stop=st, last=fp, Ne=n, ext=fp, tpnl=0.0)
                        if (l <= st) if d > 0 else (h >= st):   # aynı gün stop — temkinli say
                            g = uq * d * (st - fp) - cost * uq * (fp + st)
                            pnl += g
                            trades.append(g)
                            if sysn == 1:
                                last_win[i] = False
                        else:
                            g = uq * d * (c - fp) - cost * uq * fp
                            p["tpnl"] = g
                            pnl += g
                            pos[i] = p
                            if not pyramid:
                                p["units"] = cap_market
            last_c[i] = c
            started = True
        E = E0 + pnl + E0 * rfv[t]
        if E <= 0:
            E = 1e-9
        eq[t] = E if started else np.nan
    eqs = pd.Series(eq, index=C.index).dropna()
    ret = eqs.pct_change().dropna()
    first = N.notna().any(axis=1).idxmax()
    return ret[ret.index > first], np.array(trades)


def _room(pos, clus, i, d, cap_cluster, cap_dir, prices, E, gross_cap, add_notional):
    u_cl = u_dir = 0
    gross = add_notional
    for j, q in enumerate(pos):
        if q is None:
            continue
        if q["dir"] == d:
            u_dir += min(q["units"], 4)
            if clus[j] == clus[i]:
                u_cl += min(q["units"], 4)
        pj = prices[j]
        if not np.isnan(pj):
            gross += q["qty"] * pj
    return u_cl < cap_cluster and u_dir < cap_dir and gross <= gross_cap * E


# ------------------------------------------------------------------ rapor
def mexc_matches():
    try:
        items = get(f"{MEXC}/detail").get("data") or []
    except Exception as e:  # noqa: BLE001
        return [f"(MEXC listesi alınamadı: {e})"]
    keys = ["GOLD", "XAU", "SILVER", "XAG", "OIL", "WTI", "BRENT", "NATGAS", "COPPER", "SPX", "SP500", "NAS", "NDX",
            "US30", "DOW", "DAX", "NIKKEI", "EUR", "JPY", "GBP", "AUD", "CHF", "SPY", "QQQ", "GLD", "SLV", "TLT",
            "IWM", "PAXG", "XAUT"]
    res = []
    for c in items:
        s = c.get("symbol", "")
        name = f"{c.get('displayNameEn', '')}".upper()
        base = s.split("_")[0].upper()
        if any(k == base or base.startswith(k) or k in name.split() for k in keys):
            res.append(f"{s:22s} {c.get('displayNameEn', '')}  maxLev={c.get('maxLeverage')}  "
                       f"taker={c.get('takerFeeRate')}")
    return res or ["(eşleşen kontrat yok)"]


def row(name, ret, rf, extra=""):
    s = stats(ret, rf)
    si = stats(ret[ret.index <= IS_END], rf)
    so = stats(ret[ret.index > IS_END], rf)
    nr = stats(norm(ret, rf), rf)
    y = yearly(ret)
    last = (1 + ret[ret.index > ret.index[-1] - pd.Timedelta(days=365)]).prod() - 1
    out(f"{name:44s} {s['cagr']:7.1%} {s['vol']:6.1%} {s['sharpe']:5.2f} {s['dd']:7.1%} | "
        f"{si['sharpe']:5.2f} {so['sharpe']:5.2f} {so['cagr']:7.1%} {so['dd']:7.1%} | {last:7.1%} {y.min():7.1%} | "
        f"{nr['cagr']:6.1%} {nr['dd']:7.1%} {extra}")


def main():
    os.makedirs(OUT, exist_ok=True)
    out("TURTLE ve GELİŞTİRİLMİŞ TREND TAKİBİ — ETF vekilleri, artık getiri + risksiz faiz, taraf başı 5 bps")
    px, rf, ext, cal = load()
    C = px["close"]
    out("\n0) VERİ")
    for tk in C.columns:
        s = C[tk].dropna()
        out(f"  {tk:5s} {UNIVERSE[tk]:7s} {s.index[0].date()} → {s.index[-1].date()}  ({len(s)} gün)")
    out(f"  Risksiz faiz ort. %{rf.mean() * 252 * 100:.2f}/yıl; örneklem içi ≤ {IS_END}, dışı sonrası")

    out("\n0b) MEXC'DE İŞLEM GÖREBİLECEK KARŞILIKLAR")
    for s in mexc_matches():
        out("  " + s)

    hdr = (f"{'':44s} {'CAGR':>7s} {'Oyn.':>6s} {'Shrp':>5s} {'MaksDD':>7s} | {'IS-S':>5s} {'OOS-S':>5s} "
           f"{'OOS-CAGR':>7s} {'OOS-DD':>7s} | {'Son12a':>7s} {'EnKötüY':>7s} | {'%20oynaklıkta CAGR/DD':>14s}")
    out("\n1) SONUÇLAR (IS = 2018'e kadar, OOS = 2019+ örneklem dışı; son iki sütun: aynı %20 oynaklığa ölçekli)")
    out(hdr)
    res = {}

    variants_T = [
        ("T1 Turtle orijinal (S1+S2, ekleme, 2N)", dict()),
        ("T2 Turtle, eklemesiz", dict(pyramid=False)),
        ("T3 Turtle, atlama kuralı yok", dict(skip_rule=False)),
        ("T4 Turtle, çıkış = 3N iz süren stop", dict(chandelier=True)),
        ("T5 Turtle, risk %0.5", dict(risk=0.005)),
    ]
    for name, kw in variants_T:
        ret, tr = run_turtle(px, rf, **kw)
        res[name] = ret
        row(name, ret, rf, f"işlem {len(tr)}, kazanan %{(tr > 0).mean() * 100:.0f}")

    Fb = turtle_binary(C)
    Fbe = forecasts(C, "breakout_ens")
    Few = forecasts(C, "ewmac_ens")
    Fts = forecasts(C, "tsmom")
    Fmix = (Fbe + Few) / 2
    variants_M = [
        ("M1 12 ay momentum (işaret) + oynaklık hedefi", dict(F=Fts)),
        ("M2 Turtle 55/20 + oynaklık hedefi", dict(F=Fb)),
        ("M3 kırılım ensemble 20-320 kademeli", dict(F=Fbe)),
        ("M4 EWMAC ensemble 8-64 kademeli", dict(F=Few)),
        ("M5 M3+M4 ortalama", dict(F=Fmix)),
        ("M6 M5 + küme bazlı risk", dict(F=Fmix, cluster=True)),
        ("M7 M6 + %10 tampon (az işlem)", dict(F=Fmix, cluster=True, buffer=0.1)),
        ("M8 M7, ücret 10 bps (hassasiyet)", dict(F=Fmix, cluster=True, buffer=0.1, cost=0.001)),
        ("M9 M7, yalnız long", dict(F=Fmix.clip(lower=0), cluster=True, buffer=0.1)),
    ]
    for name, kw in variants_M:
        F = kw.pop("F")
        ret, turn, _ = run_modern(C, rf, F, **kw)
        res[name] = ret
        row(name, ret, rf, f"yıllık ciro {turn:.0f}×")

    spy = C["SPY"].pct_change(fill_method=None)
    res["-- SPY al-tut"] = spy[spy.index >= res["M7 M6 + %10 tampon (az işlem)"].index[0]].fillna(0.0)
    row("-- SPY al-tut (karşılaştırma)", res["-- SPY al-tut"], rf)

    out("\n2) YILLIK GETİRİLER")
    keys = ["T1 Turtle orijinal (S1+S2, ekleme, 2N)", "M2 Turtle 55/20 + oynaklık hedefi", "M5 M3+M4 ortalama",
            "M7 M6 + %10 tampon (az işlem)", "-- SPY al-tut"]
    Y = pd.DataFrame({k[:3].strip(): yearly(res[k]) for k in keys})
    Y.columns = ["T1", "M2", "M5", "M7", "SPY"]
    out(Y.to_string(float_format=lambda v: f"{v:7.1%}"))

    out("\n3) KRİPTO İLE İLİŞKİ ve KRİZLER")
    btc = ext.get("BTC-USD")
    for k, lab in ((keys[0], "T1"), (keys[3], "M7")):
        r = res[k]
        m = (1 + r).resample("ME").prod() - 1
        line = f"  {lab}: SPY ile günlük kor. {r.corr(spy.reindex(r.index)):.2f}"
        if btc is not None:
            bm = btc.resample("ME").last().pct_change()
            bd = btc.reindex(r.index).ffill().pct_change(fill_method=None)
            line += (f" | BTC ile günlük kor. {r.corr(bd):.2f}, aylık {m.corr(bm.reindex(m.index)):.2f}")
        out(line)
        for a, b, nm in (("2008-01-01", "2009-03-31", "2008 krizi"), ("2020-02-15", "2020-04-30", "Covid"),
                         ("2022-01-01", "2022-12-31", "2022"), ("2025-01-01", "2026-12-31", "2025+")):
            rr = r[(r.index >= a) & (r.index <= b)]
            sp = spy[(spy.index >= a) & (spy.index <= b)].fillna(0)
            bb = ""
            if btc is not None:
                bs = btc[(btc.index >= a) & (btc.index <= b)]
                if len(bs) > 5:
                    bb = f", BTC {bs.iloc[-1] / bs.iloc[0] - 1:+.1%}"
            if len(rr):
                out(f"      {nm:11s} {lab} {(1 + rr).prod() - 1:+.1%}  (SPY {(1 + sp).prod() - 1:+.1%}{bb})")

    with open(os.path.join(OUT, "report.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    sys.exit(main())
