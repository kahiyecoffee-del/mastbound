"""15dk momentum — veri güdümlü kural keşfi (long + short, birkaç saat taşıma, 2x).

1) TETİKLEYİCİLER (kapanmış 15dk mumda): 20-mum kırılımı, EMA9/21 kesişimi, Supertrend dönüşü,
   Bollinger-Keltner sıkışma kırılımı, MACD histogram sıfır kesişimi, displacement mumu.
2) Her tetikleyici için ÖZELLİKLER (yalnızca geçmiş veri): 1s/4s trend, BTC trendi, ADX 15dk/1s,
   RSI, VWAP uzaklığı, göreli hacim, taker oranı (1 ve 4 mum), CVD eğimi, ATR%, BB genişliği
   yüzdeliği, seans, funding, önceki gün aralığındaki konum, 4s momentum.
3) ÇIKIŞLAR: stop k*ATR (1.5 / 2.5), TP 2R / 3R / iz süren stop; en fazla 6 saat taşıma.
4) KEŞİF (2021-2023): tetikleyici x yön x çıkış grubunda, 1 ve 2 özellik koşullu (tercile) kuralların
   net R beklentisi (gerçekçi maliyet sonrası). DOĞRULAMA (2024-2025.09) ile elenir,
   seçilen kurallar GÖRÜLMEMİŞ TEST (2025.09-2026.09) üzerinde portföy olarak raporlanır.
"""
import argparse
import itertools
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd

import config as C
import research as R
import research_15m as Q
import research_ls as L

Q15 = pd.Timedelta("15min")
HOLD_MIN = 6 * 60
EXITS = [(k, mode, rr) for k in (1.5, 2.5) for mode, rr in (("tp", 2.0), ("tp", 3.0), ("trail", 0.0))]
MIN_N_TRAIN, MIN_N_VAL = 300, 100


# ------------------------------------------------------------------ göstergeler (15dk)
def rma(s, n):
    return s.ewm(alpha=1 / n, adjust=False).mean()


def adx(df, n=14):
    up, dn = df["high"].diff(), -df["low"].diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = R.atr(df, n)
    pdi = 100 * rma(pd.Series(pdm, index=df.index), n) / a
    mdi = 100 * rma(pd.Series(mdm, index=df.index), n) / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return rma(dx.fillna(0), n)


def rsi(close, n=14):
    d = close.diff()
    g, l_ = rma(d.clip(lower=0), n), rma(-d.clip(upper=0), n)
    return 100 - 100 / (1 + g / l_.replace(0, np.nan))


def supertrend_dir(df, n=10, m=3.0):
    a = R.atr(df, n).to_numpy()
    hl2 = ((df["high"] + df["low"]) / 2).to_numpy()
    c = df["close"].to_numpy()
    up, dn = hl2 - m * a, hl2 + m * a
    d = np.ones(len(c), np.int8)
    fu, fd = up.copy(), dn.copy()
    for i in range(1, len(c)):
        fu[i] = max(up[i], fu[i - 1]) if c[i - 1] > fu[i - 1] else up[i]
        fd[i] = min(dn[i], fd[i - 1]) if c[i - 1] < fd[i - 1] else dn[i]
        d[i] = 1 if c[i] > fd[i - 1] else -1 if c[i] < fu[i - 1] else d[i - 1]
    return d


def htf_trend(coin, tf, T):
    h = coin.tfs[tf]
    e50, e200 = L.ema(h["close"], 50), L.ema(h["close"], 200)
    s = pd.Series(np.where((h["close"] > e50) & (e50 > e200), 1,
                           np.where((h["close"] < e50) & (e50 < e200), -1, 0)),
                  index=h.index + pd.Timedelta(tf))
    return pd.merge_asof(pd.DataFrame({"T": T}), s.rename("v").to_frame(), left_on="T",
                         right_index=True)["v"].fillna(0).to_numpy()


def build(coin, btc_coin):
    df = coin.tfs["15min"]
    f = Q.feats15(coin)                                   # atr, relvol, ratio, vwap, killzone, ...
    T = f["T"]
    cl, op, hi, lo, vol = (df[c] for c in ("close", "open", "high", "low", "volume"))
    atr = f["atr"]
    x = pd.DataFrame(index=df.index)
    x["T"] = T
    x["atr"] = atr
    x["trend1h"] = htf_trend(coin, "1h", T)
    x["trend4h"] = htf_trend(coin, "4h", T)
    x["btc1h"] = htf_trend(btc_coin, "1h", T) if btc_coin is not None else 0
    x["adx15"] = adx(df)
    a1 = adx(coin.tfs["1h"])
    a1.index = a1.index + pd.Timedelta("1h")
    x["adx1h"] = pd.merge_asof(pd.DataFrame({"T": T}), a1.rename("v").to_frame(), left_on="T",
                               right_index=True)["v"].to_numpy()
    x["rsi"] = rsi(cl)
    x["vwap_dist"] = (cl - f["vwap"]) / atr
    x["relvol"] = f["relvol"]
    x["taker1"] = f["ratio"]
    tb = df["taker_buy"]
    x["taker4"] = tb.rolling(4).sum() / vol.rolling(4).sum()
    x["cvd16"] = (2 * tb - vol).rolling(16).sum() / vol.rolling(16).sum()
    x["atr_pct"] = atr / cl * 100
    mid, sd = cl.rolling(20).mean(), cl.rolling(20).std()
    bbw = (4 * sd / mid)
    x["bbw_rank"] = bbw.rolling(96).rank(pct=True)
    hr = T.dt.hour
    x["seans"] = np.select([hr < 7, hr < 13, hr < 21], [0, 1, 2], 3)
    if coin.f_t is not None:
        fs = pd.Series(coin.f_v, index=pd.to_datetime(coin.f_t, utc=True).as_unit(T.dt.unit))
        x["funding"] = pd.merge_asof(pd.DataFrame({"T": T}), fs.rename("v").to_frame(),
                                     left_on="T", right_index=True)["v"].to_numpy()
    else:
        x["funding"] = 0.0
    rng = (f["pdh"] - f["pdl"]).replace(0, np.nan)
    x["gun_konum"] = (cl - f["pdl"]) / rng
    x["mom4h"] = (cl - cl.shift(16)) / atr
    # tetikleyiciler (+1 long, -1 short, 0 yok)
    hh, ll = hi.shift(1).rolling(20).max(), lo.shift(1).rolling(20).min()
    trig = {}
    trig["kirilim20"] = np.where(cl > hh, 1, np.where(cl < ll, -1, 0))
    e9, e21 = L.ema(cl, 9), L.ema(cl, 21)
    s = np.sign(e9 - e21)
    trig["ema9_21"] = np.where(s != s.shift(1), s, 0)
    st = pd.Series(supertrend_dir(df), index=df.index)
    trig["supertrend"] = np.where(st != st.shift(1), st, 0)
    kc_u, kc_l = L.ema(cl, 20) + 1.5 * atr, L.ema(cl, 20) - 1.5 * atr
    sq = ((mid + 2 * sd) < kc_u) & ((mid - 2 * sd) > kc_l)
    sq_prev = sq.shift(1).fillna(False).astype(bool)
    trig["sikisma"] = np.where(sq_prev & (cl > mid + 2 * sd), 1,
                               np.where(sq_prev & (cl < mid - 2 * sd), -1, 0))
    m = L.ema(cl, 12) - L.ema(cl, 26)
    hs = np.sign(m - L.ema(m, 9))
    trig["macd_hist"] = np.where(hs != hs.shift(1), hs, 0)
    body, rng_ = cl - op, (hi - lo).replace(0, np.nan)
    pos = (cl - lo) / rng_
    trig["displacement"] = np.where((body > 1.5 * atr) & (pos > 0.8) & (f["relvol"] > 2), 1,
                                    np.where((-body > 1.5 * atr) & (pos < 0.2) & (f["relvol"] > 2), -1, 0))
    for k_, v in trig.items():
        x["trig_" + k_] = pd.Series(v, index=df.index).fillna(0).astype(int)
    return x


FEATURES = ["trend1h", "trend4h", "btc1h", "adx15", "adx1h", "rsi", "vwap_dist", "relvol",
            "taker1", "taker4", "cvd16", "atr_pct", "bbw_rank", "seans", "funding", "gun_konum",
            "mom4h"]
CATEG = {"trend1h", "trend4h", "btc1h", "seans"}


# ------------------------------------------------------------------ toplu çıkış simülasyonu
@numba.njit(cache=True)
def batch(o, h, l, c, ent, dirs, atrv, k, mode, rr, hold, close_at, atr15, slip, taker, maker,
          liq_lev):
    n = len(ent)
    netR = np.empty(n)
    ex_i = np.empty(n, np.int64)
    ex_p = np.empty(n)
    rsn = np.empty(n, np.int64)
    N = len(o)
    for m in range(n):
        e, d = ent[m], dirs[m]
        entry = o[e] * (1 + d * slip)
        Rd = k * atrv[m]
        stop = entry - d * Rd
        if mode == 0:
            tp = entry + d * rr * Rd
        else:
            tp = np.inf if d == 1 else -np.inf
        liq = entry * (1 - d / liq_lev)
        dl = min(e + hold, N - 1)
        kk, raw, r = L.sim(o, h, l, c, e, d, stop, tp, k if mode == 1 else 0.0, close_at, atr15,
                           dl, liq)
        if r == 2:
            px, fo = raw, maker
        else:
            px, fo = raw * (1 - d * slip), taker
        netR[m] = (d * (px - entry) - taker * entry - fo * px) / Rd
        ex_i[m], ex_p[m], rsn[m] = kk, px, r
    return netR, ex_i, ex_p, rsn


# ------------------------------------------------------------------ ana akış
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--val", default="2024-01-01")
    ap.add_argument("--test", default="2025-09-24")
    ap.add_argument("--end", default="2026-09-24")
    ap.add_argument("--coins", default=",".join(C.COINS))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drift", type=float, default=0.0)
    ap.add_argument("--out", default="results/research_15m_v2")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    A, V, T_, E = (pd.Timestamp(x, tz="UTC") for x in (args.start, args.val, args.test, args.end))
    coins = R.load(args, A, E)
    btc = coins.get("BTC")
    t0 = time.time()

    # --- olaylar
    ev_parts = []
    for name, coin in coins.items():
        x = build(coin, btc)
        x = x[(x["T"] >= A) & (x["T"] < E)]
        for tg in [c for c in x.columns if c.startswith("trig_")]:
            sel = x[x[tg] != 0]
            if not len(sel):
                continue
            ev = sel[["T", "atr"] + FEATURES].copy()
            ev["dir"] = sel[tg].to_numpy()
            ev["tetik"] = tg[5:]
            ev["coin"] = name
            ev["e"] = np.searchsorted(coin.mm.t, ev["T"].dt.as_unit("ns").astype("int64").to_numpy())
            ev_parts.append(ev[ev["e"] < len(coin.mm.t) - 1])
        print(f"[{name}] özellikler hazır ({time.time() - t0:.0f}s)", flush=True)
    ev = pd.concat(ev_parts, ignore_index=True)
    ev["donem"] = np.where(ev["T"] < V, "kesif", np.where(ev["T"] < T_, "dogrulama", "TEST"))
    print(f"{len(ev):,} olay", flush=True)

    # --- her çıkış için net R
    for xi, (k, mode, rr) in enumerate(EXITS):
        col = f"x{xi}"
        ev[col] = np.nan
        ev[col + "_i"] = -1
        ev[col + "_p"] = np.nan
        ev[col + "_r"] = -1
        for name, coin in coins.items():
            msk = (ev["coin"] == name).to_numpy()
            if not msk.any():
                continue
            mm = coin.mm
            close_at, a15 = coin.htf("15min")
            nr, ei, ep, rs = batch(mm.o, mm.h, mm.l, mm.c, ev.loc[msk, "e"].to_numpy(np.int64),
                                   ev.loc[msk, "dir"].to_numpy(np.int64),
                                   ev.loc[msk, "atr"].to_numpy(float), k, 0 if mode == "tp" else 1,
                                   rr, HOLD_MIN, close_at, a15, C.SLIPPAGE, C.TAKER_FEE,
                                   Q.MAKER_FEE, 5.0)
            ev.loc[msk, col], ev.loc[msk, col + "_i"] = nr, ei
            ev.loc[msk, col + "_p"], ev.loc[msk, col + "_r"] = ep, rs
        print(f"çıkış {xi + 1}/{len(EXITS)} ({time.time() - t0:.0f}s)", flush=True)
    xname = {f"x{i}": f"stop {k}ATR " + (f"TP {rr:g}R" if m == "tp" else "iz") + " maks 6s"
             for i, (k, m, rr) in enumerate(EXITS)}

    # --- temel tablo: tetikleyici x yön x çıkış, dönem bazında ortalama net R
    base = []
    for (tg, d), g in ev.groupby(["tetik", "dir"]):
        for xc in xname:
            r = {"tetik": tg, "yön": "long" if d == 1 else "short", "çıkış": xname[xc]}
            for p in ("kesif", "dogrulama", "TEST"):
                v = g.loc[g.donem == p, xc]
                r[f"{p}_n"], r[f"{p}_ortR"] = len(v), v.mean()
            base.append(r)
    base = pd.DataFrame(base).sort_values("kesif_ortR", ascending=False)

    # --- kural keşfi (1 ve 2 koşullu tercile filtreleri)
    rules = []
    for (tg, d), g in ev.groupby(["tetik", "dir"]):
        tr = g[g.donem == "kesif"]
        if len(tr) < MIN_N_TRAIN:
            continue
        bins = {}
        for fcol in FEATURES:
            if fcol in CATEG:
                bins[fcol] = (g[fcol].fillna(-9).astype(int).to_numpy(), None)
            else:
                q = tr[fcol].quantile([1 / 3, 2 / 3]).to_numpy()
                if not np.all(np.isfinite(q)):
                    continue
                b = np.digitize(g[fcol].to_numpy(), q)
                b[~np.isfinite(g[fcol].to_numpy())] = -9
                bins[fcol] = (b, q)
        per = g["donem"].to_numpy()
        for xc in xname:
            y = g[xc].to_numpy()
            cands = [((f1, v1),) for f1 in bins for v1 in np.unique(bins[f1][0]) if v1 != -9]
            cands += [((f1, v1), (f2, v2)) for f1, f2 in itertools.combinations(bins, 2)
                      for v1 in np.unique(bins[f1][0]) if v1 != -9
                      for v2 in np.unique(bins[f2][0]) if v2 != -9]
            for cond in cands:
                msk = np.ones(len(g), bool)
                for fcol, v in cond:
                    msk &= bins[fcol][0] == v
                mt = msk & (per == "kesif")
                n_t = mt.sum()
                if n_t < MIN_N_TRAIN:
                    continue
                r_t = y[mt].mean()
                if r_t <= 0.05:
                    continue
                mv = msk & (per == "dogrulama")
                mx = msk & (per == "TEST")
                desc = " & ".join(
                    f"{fcol}={v}" if bins[fcol][1] is None else
                    f"{fcol}∈{['düşük', 'orta', 'yüksek'][v]}(<{bins[fcol][1][0]:.3g}|<{bins[fcol][1][1]:.3g})"
                    for fcol, v in cond)
                rules.append(dict(tetik=tg, yön="long" if d == 1 else "short", çıkış=xname[xc],
                                  xc=xc, kural=desc, cond=cond, dir=d,
                                  kesif_n=n_t, kesif_ortR=r_t, kesif_gun=n_t / ((V - A).days),
                                  dogrulama_n=int(mv.sum()), dogrulama_ortR=y[mv].mean() if mv.any() else np.nan,
                                  TEST_n=int(mx.sum()), TEST_ortR=y[mx].mean() if mx.any() else np.nan,
                                  q={fcol: bins[fcol][1] for fcol, _ in cond}))
    rules = pd.DataFrame(rules)
    print(f"{len(rules):,} aday kural keşifte R>0.05 ({time.time() - t0:.0f}s)", flush=True)
    ok = rules[(rules.dogrulama_n >= MIN_N_VAL) & (rules.dogrulama_ortR > 0.05)].copy()
    ok["skor"] = np.minimum(ok.kesif_ortR, ok.dogrulama_ortR)
    ok = ok.sort_values("skor", ascending=False)

    # --- nihai strateji: doğrulamayı geçen en iyi kurallar (her yön için en fazla 3, farklı tetik)
    chosen = []
    for d in (1, -1):
        seen = set()
        for _, r in ok[ok.dir == d].iterrows():
            key = (r.tetik, r.xc)
            if key in seen:
                continue
            seen.add(key)
            chosen.append(r)
            if len([c for c in chosen if c.dir == d]) >= 3:
                break
    chosen = pd.DataFrame(chosen)

    # --- portföy simülasyonu (risk %1, kaldıraç≤2x, maks 2 pozisyon)
    def cands_for(period):
        out = []
        for _, r in chosen.iterrows():
            g = ev[(ev.tetik == r.tetik) & (ev.dir == r.dir) & (ev.donem == period)]
            msk = np.ones(len(g), bool)
            for fcol, v in r.cond:
                q = r.q[fcol]
                b = g[fcol].fillna(-9).astype(int).to_numpy() if q is None else np.digitize(g[fcol].to_numpy(), q)
                msk &= b == v
            for _, e in g[msk].iterrows():
                coin = coins[e.coin]
                xi = int(e[r.xc + "_i"])
                entry = coin.mm.o[int(e.e)] * (1 + e.dir * C.SLIPPAGE)
                k = EXITS[int(r.xc[1:])][0]
                out.append(dict(coin=e.coin, dir=int(e.dir), setup_time=e["T"],
                                entry_time=coin.mm.index[int(e.e)], entry=entry,
                                stop=entry - e.dir * k * e.atr, exit_time=coin.mm.index[xi],
                                exit=e[r.xc + "_p"], reason=L.REASONS[int(e[r.xc + "_r"])],
                                fee_in=C.TAKER_FEE,
                                fee_out=Q.MAKER_FEE if int(e[r.xc + "_r"]) == 2 else C.TAKER_FEE,
                                funding_px=e.dir * R.funding_px(coin, int(e.e), xi),
                                kural=f"{r.tetik}/{r.yön}"))
        # aynı anda aynı coine birden fazla kural tetiklenirse ilki (portföy zaten engeller)
        return out

    pf = {}
    for p, a, z in (("kesif", A, V), ("dogrulama", V, T_), ("TEST", T_, E)):
        if not len(chosen):
            break
        tr, eq = Q.run_pf(cands_for(p), 0.01, 2.0, K=2)
        m = L.metrics(tr, eq, a, z)
        daily = pd.Series(dtype=float)
        if eq:
            s = pd.Series([b for _, b in eq], index=pd.DatetimeIndex([t for t, _ in eq]))
            s = s.resample("1D").last().ffill()
            s = pd.concat([pd.Series([100.0], index=[s.index[0] - pd.Timedelta("1D")]), s])
            daily = s.pct_change().dropna() * 100
        m.update(gun_ort_=daily.mean() if len(daily) else np.nan,
                 gun_medyan=daily.median() if len(daily) else np.nan,
                 gun_max=daily.max() if len(daily) else np.nan,
                 gun_pozitif_=(daily > 0).mean() * 100 if len(daily) else np.nan,
                 gun_8plus=int((daily >= 8).sum()) if len(daily) else 0,
                 islem_gun=m["n"] / (z - a).days)
        pf[p] = (m, tr, eq, daily)

    # --- günlük hedef / zarar limiti testi (+%8'de dur, -%4'te dur), farklı risk seviyeleri
    lim_rows = []
    if len(chosen):
        cache_c = {p: cands_for(p) for p in ("kesif", "dogrulama", "TEST")}
        for risk in (0.01, 0.02, 0.03):
            for lab, tg_, st_ in (("limitsiz", None, None), ("+%8 hedef / -%4 stop", 0.08, 0.04)):
                row = {"risk_%": risk * 100, "kural": lab}
                for p, a, z in (("kesif", A, V), ("dogrulama", V, T_), ("TEST", T_, E)):
                    st = {}
                    tr, eq = Q.run_pf(cache_c[p], risk, 2.0, K=2, day_target=tg_, day_stop=st_,
                                      stats=st)
                    m = L.metrics(tr, eq, a, z)
                    row[f"{p}_bakiye"] = m["final"]
                    row[f"{p}_maxDD_%"] = m["mdd"]
                    row[f"{p}_hedef_gun"] = st.get("hedef", 0)
                    row[f"{p}_stop_gun"] = st.get("zarar_limiti", 0)
                lim_rows.append(row)
    lim = pd.DataFrame(lim_rows)
    if len(lim):
        lim.to_csv(os.path.join(args.out, "gunluk_limit_testi.csv"), index=False, float_format="%.4g")

    # --- çıktılar
    ev_cols = ["tetik", "yön", "çıkış", "kesif_n", "kesif_ortR", "dogrulama_ortR", "TEST_ortR"]
    base.to_csv(os.path.join(args.out, "tetik_cikis_tablosu.csv"), index=False, float_format="%.4g")
    rcols = ["tetik", "yön", "çıkış", "kural", "kesif_n", "kesif_gun", "kesif_ortR",
             "dogrulama_n", "dogrulama_ortR", "TEST_n", "TEST_ortR"]
    ok[rcols].head(200).to_csv(os.path.join(args.out, "kurallar_dogrulanmis.csv"), index=False,
                               float_format="%.4g")
    if len(chosen) and pf:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, p in zip(axes, ("kesif", "dogrulama", "TEST")):
            _, _, eq, _ = pf[p]
            if eq:
                ax.step([eq[0][0]] + [t for t, _ in eq], [100] + [b for _, b in eq], where="post")
            ax.axhline(100, color="gray", ls="--", lw=0.8)
            ax.set_title(f"{p}: {pf[p][0]['final']:.1f} USDT")
            ax.grid(alpha=0.3)
            ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        fig.suptitle("Seçilen 15dk kural seti — risk %1/işlem, kaldıraç ≤2x, maks 2 pozisyon")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "equity.png"), dpi=120)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 4))
        dd = pd.concat([pf[p][3] for p in pf])
        ax.hist(dd.clip(-10, 10), bins=80, color="#2a6fdb")
        ax.axvline(8, color="#d64545", ls="--", label="%8 hedef")
        ax.set_title("Günlük getiri dağılımı (%) — tüm dönemler")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "gunluk_getiri_dagilimi.png"), dpi=120)
        plt.close(fig)
        pd.DataFrame(pf["TEST"][1]).to_csv(os.path.join(args.out, "trades_TEST.csv"), index=False,
                                           float_format="%.8g")

    txt = [f"15dk MOMENTUM KURAL KEŞFİ | keşif {A.date()}→{V.date()} | doğrulama →{T_.date()} | "
           f"TEST →{E.date()}",
           "VERİ: SENTETİK" if args.synthetic else "Veri: Binance USDT-M 1dk + taker-buy + funding",
           f"{len(ev):,} tetik olayı, {len(xname)} çıkış varyantı, {len(FEATURES)} özellik. "
           "Net R = komisyon + slippage sonrası (TP maker).", "",
           "1) Tetikleyici x yön x çıkış — FİLTRESİZ ortalama net R (en iyi 15):",
           base.head(15).round(3).to_string(index=False), "",
           f"2) Keşifte net R > 0.05 olan {len(rules):,} kuraldan doğrulamada da > 0.05 olan: {len(ok):,}",
           ok[rcols].head(20).round(3).to_string(index=False) if len(ok) else "(yok)", "",
           "3) SEÇİLEN KURAL SETİ:",
           chosen[rcols].round(3).to_string(index=False) if len(chosen) else "(doğrulamayı geçen kural yok)", ""]
    for p in pf:
        m = pf[p][0]
        txt.append(f"   {p:10s}: 100 → {m['final']:.1f} USDT | CAGR %{m['cagr']:.1f} | maks DD %{m['mdd']:.1f} | "
                   f"işlem/gün {m['islem_gun']:.2f} | isabet %{m['win']:.1f} | günlük ort %{m['gun_ort_']:.3f} "
                   f"medyan %{m['gun_medyan']:.3f} en iyi gün %{m['gun_max']:.2f} | pozitif gün %{m['gun_pozitif_']:.1f} | "
                   f"≥%8 gün sayısı {m['gun_8plus']}")
    if len(lim):
        txt += ["", "4) GÜNLÜK LİMİT TESTİ — gün içi gerçekleşen +%8'de ya da -%4'te o gün yeni işlem yok",
                "   (hedef_gun / stop_gun = limitin tetiklendiği gün sayısı; kaldıraç ≤2x, maks 2 pozisyon)",
                lim.round(2).to_string(index=False)]
    text = "\n".join(txt)
    with open(os.path.join(args.out, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
