"""ÜÇ PROBLEM İÇİN OPTİMİZASYON ALGORİTMASI — teşhisin (research_diagnose) bulduğu üç zayıflık:

 1) ÇIKIŞ: kazananlar zirveden ~1R geri veriyor, zarar edenlerin %34'ü önce +0.5R kârdaydı.
    Stop hâlâ kârın ana mekanizması; ama kâr büyüdükçe SIKILAŞIR (hepsi 1s mum kapanışında, bot gibi):
      BE   : en iyi kâr ≥ a R olunca stop = giriş + b R (başabaş + komisyon)
      T2   : en iyi kâr ≥ a R olunca iz mesafesi 3xATR → k2 xATR
      KİLİT: en iyi kâr ≥ a R olunca stop ≥ giriş + f × (en iyi kâr)   (kârın f'si kilitlenir)
    Climax order-flow stopu ve 48s süre aynen kalır. Aynı mantık DONCHIAN'a da (günlük ATR, günlük kapanış).
 2) DONCHIAN SHORT: 2 dönemde zararda → mevcut / short yok / sıkı ayı rejimi (BTC EMA50<EMA200 ve coin
    EMA200 altında) / yarım boyut.
 3) MALİYET: EMA'da stop mesafesi çok geniş (%5+) işlemler komisyon sonrası zararda → üst sınır filtresi.

ALGORİTMA (aşamalı ızgara arama, AŞIRI UYUMA karşı):
  Aşama 1 EMA çıkış ızgarası → Aşama 2 DONCHIAN çıkış × short ızgarası → Aşama 3 maliyet filtresi.
  Her aşamada seçim YALNIZ 2021-23 (eğitim) verisiyle yapılır (skor: CAGR / maxDD), önceki aşamanın seçimi
  sabitlenir. 2024-25.09 (doğrulama) ve SON 1 YIL (görülmemiş test) sonuçları yalnız raporlanır.
  Ayrıca "3 dönemde tutarlı" seçim (mevcuda oranla en kötü dönemi en iyi olan) — test verisi seçime
  katıldığı için iyimserdir, karşılaştırma içindir.
Sonuç: MEVCUT vs algoritmanın seçtiği sistem — dönemler, yıllar, gerçek (açık pozisyon dahil) maxDD.
"""
import argparse
import itertools
import os
import time

import numba
import numpy as np
import pandas as pd

import config as C
import research as R
import research_15m as Q
import research_combo as CB
import research_cvd as RC
import research_daily_detail as DD
import research_ema921 as E
import research_ls as L
import research_quant as RQ

REASONS = ["SL", "BE", "TP", "TIME", "END", "LIQ", "TRAIL", "QSTOP", "LOCK", "T2"]


@numba.njit(cache=True)
def sim_x(o, h, l, c, e, d, stop, close_at, atr_arr, sig, trail_k, kt, act_px, entry, deadline, liq,
          be_px, be_lock, t2_px, k2, lk_px, lk_frac):
    """research_cvd.sim_q (iz stop + climax) + kâr koruma: BE, T2 (dar iz), KİLİT. Güncellemeler htf kapanışında."""
    cur, kind, best = stop, 0, o[e]
    tight, best2 = False, 0.0
    n = len(o)
    for k in range(e, n):
        if k >= deadline:
            return k, o[k], 3
        if d == 1:
            eff = cur if cur > liq else liq
            if l[k] <= eff:
                raw = eff if k == e else min(o[k], eff)
                return k, raw, kind if cur > liq else 5
            if h[k] > best:
                best = h[k]
            if tight and h[k] > best2:
                best2 = h[k]
        else:
            eff = cur if cur < liq else liq
            if h[k] >= eff:
                raw = eff if k == e else max(o[k], eff)
                return k, raw, kind if cur < liq else 5
            if l[k] < best:
                best = l[k]
            if tight and l[k] < best2:
                best2 = l[k]
        j = close_at[k]
        if j >= 0:
            a = atr_arr[j]
            mfe = d * (best - entry)
            if not tight and sig[j] and mfe >= act_px:
                tight, best2 = True, c[k]
            kk, kd = trail_k, 6
            if t2_px > 0 and mfe >= t2_px:
                kk, kd = k2, 9
            new = best - d * kk * a
            if (d == 1 and new > cur) or (d == -1 and new < cur):
                cur, kind = new, kd
            if be_px > 0 and mfe >= be_px:
                new = entry + d * be_lock
                if (d == 1 and new > cur) or (d == -1 and new < cur):
                    cur, kind = new, 1
            if lk_px > 0 and mfe >= lk_px:
                new = entry + d * lk_frac * mfe
                if (d == 1 and new > cur) or (d == -1 and new < cur):
                    cur, kind = new, 8
            if tight:
                new2 = best2 - d * kt * a
                if (d == 1 and new2 > cur) or (d == -1 and new2 < cur):
                    cur, kind = new2, 7
    return n - 1, c[n - 1], 4


def params(v, R_):
    """v = (be, t2, lk) → sim_x kâr koruma argümanları (fiyat birimi)."""
    be, t2, lk = v
    return ((be[0] * R_, be[1] * R_) if be else (0.0, 0.0)) + ((t2[0] * R_, t2[1]) if t2 else (0.0, 0.0)) + \
        ((lk[0] * R_, lk[1]) if lk else (0.0, 0.0))


def vname(v):
    be, t2, lk = v
    s = [f"BE {be[0]:g}R→+{be[1]:g}R" if be else "", f"T2 {t2[0]:g}R→{t2[1]:g}xATR" if t2 else "",
         f"KİLİT {lk[0]:g}R→%{lk[1] * 100:g}" if lk else ""]
    s = [x for x in s if x]
    return " + ".join(s) if s else "MEVCUT ÇIKIŞ"


def ema_recorder(flows, variants, store):
    """Q.trade yerine: girişler aynen; her çıkış varyantı sim_x ile (climax 0.5 ATR, 1R, 48s korunur)."""
    orig = Q.trade

    def trade(coin, e, d, stop, T, limit_px=None, tp=None, trail_k=0.0, deadline=None, trail_tf="15min"):
        base = orig(coin, e, d, stop, T, limit_px=limit_px, tp=tp, trail_k=trail_k, deadline=deadline,
                    trail_tf=trail_tf)
        if not base:
            return base
        mm = coin.mm
        close_at, atr_arr = coin.htf("1h")
        entry = base["entry"]
        R_ = abs(entry - stop)
        liq = entry * (1 - d / Q.LIQ_LEV)
        start = int(np.searchsorted(mm.t, base["entry_time"].value))
        dl = int(np.searchsorted(mm.t, (T + pd.Timedelta(hours=48)).value))
        sig = flows[d]["climax"]
        for key, v in variants.items():
            k, raw, r = sim_x(mm.o, mm.h, mm.l, mm.c, start, d, stop, close_at, atr_arr, sig, 3.0, 0.5, 1.0 * R_,
                              entry, dl, liq, *params(v, R_))
            store[key].append(dict(base, exit_time=mm.index[k], exit=raw * (1 - d * C.SLIPPAGE), reason=REASONS[r],
                                   fee_out=C.TAKER_FEE, funding_px=d * R.funding_px(coin, start, k)))
        return base
    return trade


def don_recorder(variants, store):
    """L.trade yerine (DONCHIAN): giriş aynen (piyasa), 2xATR günlük iz + varyantın kâr koruması."""
    orig = L.trade
    none = None

    def trade(coin, e, d, stop, T, tp=None, trail_tf=None, trail_k=0.0, deadline=None):
        nonlocal none
        base = orig(coin, e, d, stop, T, tp=tp, trail_tf=trail_tf, trail_k=trail_k, deadline=deadline)
        if not base:
            return base
        mm = coin.mm
        close_at, atr_arr = coin.htf(trail_tf or "1D")
        if none is None or len(none) != len(atr_arr):
            none = np.zeros(len(atr_arr), np.bool_)
        entry = base["entry"]
        R_ = abs(entry - stop)
        liq = entry * (1 - d / L.LEV) / (1 - d * C.MAINT_MARGIN_RATE)
        for key, v in variants.items():
            k, raw, r = sim_x(mm.o, mm.h, mm.l, mm.c, e, d, stop, close_at, atr_arr, none, trail_k, 0.0, 0.0,
                              entry, len(mm.o) + 1, liq, *params(v, R_))
            store[key].append(dict(base, exit_time=mm.index[k], exit=raw * (1 - d * C.SLIPPAGE), reason=REASONS[r],
                                   funding_px=d * R.funding_px(coin, e, k), strat="DONCHIAN",
                                   fee_in=C.TAKER_FEE, fee_out=C.TAKER_FEE))
        return base
    return trade


def evaluate(cs, periods):
    out = {}
    for lab, a, z in periods:
        tr, eq, _ = RQ.run_pf([c for c in cs if a <= c["entry_time"] < z])
        m = L.metrics(tr, eq, a, z)
        out[lab] = m
    return out


def score(m):
    """CAGR/maxDD (zararda olan her zaman kârda olanın altında kalır)."""
    return m["cagr"] / max(m["mdd"], 5.0) if m["cagr"] > 0 else m["cagr"] - 100.0


def rnet(t):
    risk = abs(t["entry"] - t["stop"])
    return ((t["exit"] - t["entry"]) * t["dir"] - t.get("fee_in", C.TAKER_FEE) * t["entry"]
            - t.get("fee_out", C.TAKER_FEE) * t["exit"] - t["funding_px"]) / risk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", default=CB.COINS18)
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--val", default="2024-01-01")
    ap.add_argument("--test", default="2025-09-24")
    ap.add_argument("--end", default="2026-09-24")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drift", type=float, default=0.0)
    ap.add_argument("--out", default="results/fix3")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    A, V, T_, Z = (pd.Timestamp(x, tz="UTC") for x in (args.start, args.val, args.test, args.end))
    periods = [("2021-23", A, V), ("2024-25.09", V, T_), ("SON 1 YIL", T_, Z)]
    PL = [p[0] for p in periods]
    years = [(str(y), pd.Timestamp(f"{y}-01-01", tz="UTC"), min(pd.Timestamp(f"{y + 1}-01-01", tz="UTC"), Z))
             for y in range(A.year, Z.year + 1) if pd.Timestamp(f"{y}-01-01", tz="UTC") < Z]
    C.COINS = args.coins.split(",")
    E.SIG_TF, E.TREND_TF, E.MAX_HOLD = "1h", "4h", pd.Timedelta(hours=168)

    # ---- ızgaralar
    ema_grid = list(itertools.product([None, (0.5, 0.1), (1.0, 0.2), (1.5, 0.5)],
                                      [None, (1.0, 1.5), (1.0, 2.0), (2.0, 1.5), (2.0, 2.0)],
                                      [None, (1.5, 0.5)]))
    don_grid = list(itertools.product([None, (1.0, 0.1), (1.5, 0.3)],
                                      [None, (2.0, 1.0), (3.0, 1.0), (2.0, 1.5)],
                                      [None]))
    short_modes = ["short mevcut", "short YOK", "short sıkı ayı rejimi", "short yarım boyut"]
    stop_caps = [None, 6.0, 5.0, 4.0]
    base_v = (None, None, None)

    t0 = time.time()
    coins = R.load(args, A, Z)
    btc = coins.get("BTC")
    ema = {v: [] for v in ema_grid}
    don = {v: [] for v in don_grid}
    bear = {}
    if btc is not None:
        bc = btc.tfs["1D"]["close"]
        bb = (L.ema(bc, 50) < L.ema(bc, 200))
        bear_btc = pd.Series(bb.to_numpy(), index=bc.index + pd.Timedelta("1D"))
    o_q, o_l = Q.trade, L.trade
    for n, c in coins.items():
        x = E.prep(c, btc)
        Q.trade = ema_recorder(RC.flow_signals(c), {v: v for v in ema_grid}, ema)
        try:
            E.signals(c, x, A, Z, "ikisi", "ema9_limit", "atr3", "iz3")
        finally:
            Q.trade = o_q
        feats = L.daily_feats(c, btc.tfs["1D"]["close"] if btc else None)
        L.trade = don_recorder({v: v for v in don_grid}, don)
        try:
            L.f_donchian(c, feats, A, Z, n=20, k=2.0, regime="btc")
        finally:
            L.trade = o_l
        cb = pd.Series((feats["close"] < feats["ema200"]).to_numpy(), index=feats["T"])
        for T, v in cb.items():
            bear[(n, T)] = bool(v) and (bool(bear_btc.asof(T) == True) if btc is not None else True)
        print(f"[{n}] hazır ({time.time() - t0:.0f}s)", flush=True)
    for v in ema_grid:
        ema[v] = [dict(t, strat="EMA") for t in ema[v]]

    def don_set(v, mode):
        out = []
        for t in don[v]:
            if t["dir"] == -1:
                if mode == "short YOK":
                    continue
                if mode == "short sıkı ayı rejimi" and not bear.get((t["coin"], t["setup_time"]), False):
                    continue
                if mode == "short yarım boyut":
                    t = dict(t, size_mult=0.5)
            out.append(t)
        return out

    def ema_set(v, cap):
        if cap is None:
            return ema[v]
        return [t for t in ema[v] if abs(t["entry"] - t["stop"]) / t["entry"] * 100 <= cap]

    def row(name, cs):
        m = evaluate(cs, periods)
        r = {"düzen": name}
        for lab in PL:
            r[f"{lab} son"], r[f"{lab} maxDD%"] = m[lab]["final"], m[lab]["mdd"]
        r["eğitim skoru"] = score(m[PL[0]])
        return r

    def stage(title, items):
        """items: [(ad, anahtar, adaylar)] — ilki MEVCUT (mevcut aşama tabanı)."""
        rows, keys = [], []
        for name, key, cs in items:
            r = row(name, cs)
            rows.append(r)
            keys.append(key)
            print(f"  {title} | {name}: " + " | ".join(f"{lab} {r[f'{lab} son']:.1f}" for lab in PL), flush=True)
        df = pd.DataFrame(rows)
        b0 = df.iloc[0]
        ratio = np.column_stack([df[f"{lab} son"] / b0[f"{lab} son"] for lab in PL])
        df["en kötü dönem oranı"] = ratio.min(axis=1)
        df["3 dönemde de iyi"] = np.where((ratio > 1.0).all(axis=1), "✅", "")
        i_is, i_rb = int(df["eğitim skoru"].to_numpy().argmax()), int(df["en kötü dönem oranı"].to_numpy().argmax())
        return df, (keys[i_is], df["düzen"].iloc[i_is]), (keys[i_rb], df["düzen"].iloc[i_rb])

    # ---- Aşama 1: EMA çıkışı (DONCHIAN mevcut)
    don0 = don_set(base_v, "short mevcut")
    print("AŞAMA 1 — EMA çıkış ızgarası", flush=True)
    items = [(vname(v), v, ema[v] + don0) for v in ema_grid]
    s1, b1_is, b1_rb = stage("A1", items)

    # ---- Aşama 2: DONCHIAN çıkış × short (EMA = aşama 1 seçimi)
    res2 = {}
    for tag, e_key in (("is", b1_is[0]), ("rb", b1_rb[0])):
        print(f"AŞAMA 2 ({tag}) — DONCHIAN çıkış × short ızgarası", flush=True)
        items = [(f"{vname(v)} | {mode}", (v, mode), ema[e_key] + don_set(v, mode))
                 for mode in short_modes for v in don_grid]
        items.sort(key=lambda it: it[1] != (base_v, "short mevcut"))
        res2[tag] = stage("A2", items)

    # ---- Aşama 3: maliyet (EMA stop mesafesi üst sınırı)
    res3 = {}
    for tag, e_key in (("is", b1_is[0]), ("rb", b1_rb[0])):
        dk = res2[tag][1 if tag == "is" else 2][0]
        print(f"AŞAMA 3 ({tag}) — EMA stop mesafesi sınırı", flush=True)
        items = [(f"stop ≤ %{cap:g}" if cap else "sınır yok", cap, ema_set(e_key, cap) + don_set(*dk))
                 for cap in stop_caps]
        res3[tag] = stage("A3", items)

    # ---- final karşılaştırma
    finals = [("MEVCUT (bot şu an)", ema[base_v] + don0)]
    for tag, lab in (("is", "ALGORİTMA (yalnız 2021-23 ile seçildi)"), ("rb", "3 DÖNEM TUTARLI SEÇİM (iyimser)")):
        e_key = b1_is[0] if tag == "is" else b1_rb[0]
        dk = res2[tag][1 if tag == "is" else 2][0]
        cap = res3[tag][1 if tag == "is" else 2][0]
        finals += [(f"{lab} — yalnız 1 (EMA çıkış)", ema[e_key] + don0),
                   (f"{lab} — 1+2 (+DONCHIAN)", ema[e_key] + don_set(*dk)),
                   (f"{lab} — 1+2+3 TAM", ema_set(e_key, cap) + don_set(*dk))]
    rows = []
    for name, cs in finals:
        r = row(name, cs)
        for lab, a, z in years:
            tr, eq, _ = RQ.run_pf([c for c in cs if a <= c["entry_time"] < z])
            r[f"{lab} %"] = L.metrics(tr, eq, a, z)["ret"]
        tr, eq, _ = RQ.run_pf(cs)
        m = L.metrics(tr, eq, A, Z)
        mtm = DD.mtm_curve(tr, coins, A, Z, 100.0)
        et = [t for t in tr if t["strat"] == "EMA"]
        dt = [t for t in tr if t["strat"] == "DONCHIAN"]
        re_, rd_ = np.array([rnet(t) for t in et]), np.array([rnet(t) for t in dt])
        r.update({"TÜM son": m["final"], "CAGR%": m["cagr"], "gerçek maxDD%": DD.dd(mtm),
                  "en kötü gün %": float(mtm.pct_change().min() * 100),
                  "EMA işlem": len(et), "EMA kazanma%": (re_ > 0).mean() * 100 if len(re_) else np.nan,
                  "EMA net R": re_.mean() if len(re_) else np.nan,
                  "DON işlem": len(dt), "DON net R": rd_.mean() if len(rd_) else np.nan})
        rows.append(r)
        print(f"FİNAL {name}: tüm {m['final']:.1f} CAGR {m['cagr']:.1f} gerçek DD {r['gerçek maxDD%']:.1f}", flush=True)
    fd = pd.DataFrame(rows)
    b0 = fd.iloc[0]
    fd["3 dönemde de iyi"] = np.where(np.column_stack([fd[f"{lab} son"] > b0[f"{lab} son"] for lab in PL]).all(axis=1),
                                      "✅", "")
    fd.to_csv(os.path.join(args.out, "final.csv"), index=False, float_format="%.4g")

    # çıkış nedeni dağılımı (tüm aday EMA işlemleri): mevcut vs seçilen
    def reasons(ts):
        d = pd.DataFrame({"reason": [t["reason"] for t in ts], "R": [rnet(t) for t in ts]})
        return d.groupby("reason")["R"].agg(["count", "mean"]).round(3)

    pcols = [f"{lab} {k}" for lab in PL for k in ("son", "maxDD%")]
    show = lambda df, n=None: (df.sort_values("eğitim skoru", ascending=False).head(n) if n else df)[
        ["düzen", "eğitim skoru"] + pcols + ["en kötü dönem oranı", "3 dönemde de iyi"]].round(2).to_string(index=False)
    for nm, df in (("asama1_ema_cikis", s1), ("asama2_is", res2["is"][0]), ("asama2_rb", res2["rb"][0]),
                   ("asama3_is", res3["is"][0]), ("asama3_rb", res3["rb"][0])):
        df.to_csv(os.path.join(args.out, f"{nm}.csv"), index=False, float_format="%.4g")
    fcols = ["düzen", "TÜM son", "CAGR%", "gerçek maxDD%", "en kötü gün %", "EMA işlem", "EMA kazanma%", "EMA net R",
             "DON işlem", "DON net R"] + pcols + ["3 dönemde de iyi"]
    txt = ["ÜÇ PROBLEM OPTİMİZASYONU — çıkış kâr koruma (EMA+DONCHIAN), DONCHIAN short, maliyet filtresi",
           "VERİ: SENTETİK" if args.synthetic else "Veri: Binance USDT-M 1dk + taker hacmi + funding (2021-01 → 2026-09)",
           "Portföy botla aynı (EMA %1 risk maks 3 poz + DONCHIAN %50), 18 coin, 100 USDT her dönem ayrı.",
           "Seçim: her aşamada YALNIZ 2021-23 skoruyla (CAGR/maxDD). 'en kötü dönem oranı' = min(dönem son / mevcut son).",
           "", "=== FİNAL KARŞILAŞTIRMA ===", fd[fcols].round(2).to_string(index=False), "",
           "Yıllık getiri %:", fd[["düzen"] + [f"{y[0]} %" for y in years]].round(1).to_string(index=False), "",
           f"Seçimler — eğitim: EMA [{b1_is[1]}] | DON [{res2['is'][1][1]}] | maliyet [{res3['is'][1][1]}]",
           f"Seçimler — tutarlı: EMA [{b1_rb[1]}] | DON [{res2['rb'][2][1]}] | maliyet [{res3['rb'][2][1]}]",
           "", "=== AŞAMA 1: EMA çıkış ızgarası (eğitim skoruna göre ilk 15) ===",
           "Taban:", show(s1.iloc[[0]]), "", show(s1, 15), "",
           f"Aşama 1'de 3 dönemde de mevcudu geçen varyant sayısı: {(s1['3 dönemde de iyi'] == '✅').sum()} / {len(s1)}",
           "", "=== AŞAMA 2 (eğitim seçimi üstüne): DONCHIAN çıkış × short (ilk 15) ===", show(res2["is"][0], 15), "",
           f"3 dönemde de iyi: {(res2['is'][0]['3 dönemde de iyi'] == '✅').sum()} / {len(res2['is'][0])}", "",
           "=== AŞAMA 3 (eğitim seçimi üstüne): EMA stop mesafesi sınırı ===", show(res3["is"][0]), "",
           "=== AŞAMA 3 (tutarlı seçim üstüne) ===", show(res3["rb"][0]), "",
           "EMA çıkış nedenleri (tüm aday işlemler, net R) — MEVCUT:", reasons(ema[base_v]).to_string(), "",
           f"EMA çıkış nedenleri — eğitim seçimi [{b1_is[1]}]:", reasons(ema[b1_is[0]]).to_string()]
    text = "\n".join(txt)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
