"""FUNDING KARŞITI L/S — KÂR AL (TP) ve AL-SAT varyantları.

Temel: son 14 gün ortalama funding'e göre en yüksek 5 short / en düşük 5 long, eşit ağırlık, 1x (canlıdaki kural).
Denenen (günlük OHLC ile, pozisyon bazında simülasyon; kâr al seviyesi gün içi en yüksek/en düşük ile kontrol edilir):
  * yenileme sıklığı: 7 (canlı) / 3 / 1 gün
  * sabit TP: pozisyon +%X kâra ulaşınca kapat (X = 5, 10, 15, 25); sonra
        - "bekle"   : o coin bir sonraki yenilemeye kadar boş kalır
        - "yeniden" : ertesi gün hâlâ listedeyse yeniden açılır (al-sat)
  * iz süren kâr koruma: pozisyon +%A kâra ulaştıktan sonra zirveden %B geri çekilirse kapat (A/B = 10/5, 15/7, 20/10)
Maliyet: her açılış/kapanışta %0.02 komisyon + %0.02 kayma; funding dahil.
Değerlendirme: 2021-23 / 2024-25.09 / SON 1 YIL, maxDD, bot v8 ile 1/3 pay birleşimi (aylık denge).
Seçim kuralı: yalnız üç dönemde de canlı kuraldan (7 gün, TP yok) iyi olan ve bota eklenince maxDD'yi bozmayanlar aday.
"""
import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd

import config as C
import research as R
import research_15m as Q
import research_combo as CB
import research_cvd as RC
import research_daily_detail as DD
import research_ema921 as E
import research_fix3 as F
import research_ls as L
import research_quant as RQ
import research_strat3 as S3

COST = 0.0002 + C.SLIPPAGE


def simulate(O, H, Lo, Cl, FU, sel_long, sel_short, every, tp=None, mode="bekle", trail=None):
    """Günlük döngü. sel_long/sel_short: gün → coin listesi (o günün kapanışında bilinen sıralama).
    Pozisyon t günü kapanışında açılır; t+1'den itibaren getiri. Dönen: günlük getiri serisi (eşit ağırlık, brüt 1)."""
    days = Cl.index
    cols = list(Cl.columns)
    ci = {c: i for i, c in enumerate(cols)}
    o, h, lo, c = (x.to_numpy(float) for x in (O, H, Lo, Cl))
    fu = FU.reindex(days).fillna(0.0).to_numpy(float)
    n = len(days)
    ret = np.zeros(n)
    pos = {}                                  # coin → [dir, entry, peak_gain]
    target = {}
    k = None
    for t in range(n):
        # 1) bugünün getirisi (dün kapanışta açık olanlar)
        if t > 0 and pos:
            w = 1.0 / (2 * k)
            r = 0.0
            for coin, p in list(pos.items()):
                j = ci[coin]
                d, e = p[0], p[1]
                prev = c[t - 1, j]
                if not np.isfinite(prev) or not np.isfinite(c[t, j]):
                    continue
                exit_px = None
                if tp is not None:
                    lvl = e * (1 + d * tp)
                    if (d == 1 and h[t, j] >= lvl) or (d == -1 and lo[t, j] <= lvl):
                        exit_px = max(lvl, o[t, j]) if d == 1 else min(lvl, o[t, j])   # açılış boşluğu lehimize
                if exit_px is None and trail is not None:
                    act, back = trail
                    best = p[2]                          # DÜNE kadarki en iyi kâr → bugünkü stop (gün içi sıra bilinmez)
                    if best >= act:
                        stop_px = e * (1 + d * (best - back))
                        if (d == 1 and lo[t, j] <= stop_px) or (d == -1 and h[t, j] >= stop_px):
                            exit_px = min(stop_px, o[t, j]) if d == 1 else max(stop_px, o[t, j])
                    if exit_px is None:
                        g_hi = d * ((h[t, j] if d == 1 else lo[t, j]) / e - 1)
                        p[2] = max(best, g_hi) if np.isfinite(g_hi) else best
                end = exit_px if exit_px is not None else c[t, j]
                r += w * d * (end / prev - 1) - w * d * fu[t, j]
                if exit_px is not None:
                    r -= w * COST
                    del pos[coin]
                    if mode == "bekle":
                        target.pop(coin, None)       # bir sonraki yenilemeye kadar boş
            ret[t] = r
        # 2) yenileme / yeniden açma (bugünün kapanışında)
        rebal = (t % every == 0)
        if rebal:
            L_, S_ = sel_long.get(days[t], []), sel_short.get(days[t], [])
            m = min(len(L_), len(S_))
            if m == 0:
                continue
            k = m
            new = {x: 1 for x in L_[:m]} | {x: -1 for x in S_[:m]}
            for coin in list(pos):
                if new.get(coin) != pos[coin][0]:
                    ret[t] -= COST / (2 * k)
                    del pos[coin]
            target = dict(new)
        elif mode == "yeniden" and k:
            L_, S_ = sel_long.get(days[t], []), sel_short.get(days[t], [])
            cur = {x: 1 for x in L_[:k]} | {x: -1 for x in S_[:k]}
            target = {x: d for x, d in target.items() if cur.get(x) == d} | \
                {x: d for x, d in cur.items() if x not in pos and x in target}
        for coin, d in target.items():
            if coin not in pos and np.isfinite(c[t, ci[coin]]):
                pos[coin] = [d, c[t, ci[coin]], 0.0]
                ret[t] -= COST / (2 * k)
    return pd.Series(ret, index=days)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", default=CB.COINS18)
    ap.add_argument("--extra", default=S3.EXTRA)
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--val", default="2024-01-01")
    ap.add_argument("--test", default="2025-09-24")
    ap.add_argument("--end", default="2026-09-24")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drift", type=float, default=0.0)
    ap.add_argument("--out", default="results/funding_tp")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    A, V, T_, Z = (pd.Timestamp(x, tz="UTC") for x in (args.start, args.val, args.test, args.end))
    periods = [("2021-23", A, V), ("2024-25.09", V, T_), ("SON 1 YIL", T_, Z)]
    PL = [p[0] for p in periods]
    years = [(str(y), pd.Timestamp(f"{y}-01-01", tz="UTC"), min(pd.Timestamp(f"{y + 1}-01-01", tz="UTC"), Z))
             for y in range(A.year, Z.year + 1) if pd.Timestamp(f"{y}-01-01", tz="UTC") < Z]
    bot_coins = args.coins.split(",")
    extra = [c for c in args.extra.split(",") if c and c not in bot_coins]
    C.COINS = bot_coins
    E.SIG_TF, E.TREND_TF, E.MAX_HOLD = "1h", "4h", pd.Timedelta(hours=168)
    t0 = time.time()

    def load_one(name):
        a = argparse.Namespace(**vars(args))
        a.coins = name
        return R.load(a, A, Z).get(name)

    btc = load_one("BTC")
    em, dn, daily, ohlc, funds = [], [], {}, {}, {}
    for name in ["BTC"] + [n for n in bot_coins if n != "BTC"] + extra:
        c = btc if name == "BTC" else load_one(name)
        if c is None:
            continue
        ohlc[name] = c.tfs["1D"][["open", "high", "low", "close"]].copy()
        if c.funding is not None and len(c.funding):
            f = pd.Series(np.asarray(c.funding, float), index=pd.DatetimeIndex(c.funding.index).tz_convert("UTC"))
            funds[name] = f.resample("1D").sum()
        if name in bot_coins:
            x = E.prep(c, btc)
            st = {S3.BASE_V: []}
            o = Q.trade
            Q.trade = F.ema_recorder(RC.flow_signals(c), {S3.BASE_V: S3.BASE_V}, st)
            try:
                E.signals(c, x, A, Z, "ikisi", "ema9_limit", "atr3", "iz3")
            finally:
                Q.trade = o
            for t in st[S3.BASE_V]:
                sp = abs(t["entry"] - t["stop"]) / t["entry"] * 100
                em.append(dict(t, strat="EMA", size_mult=0.5 if sp > 5.0 else 1.0))
            sd = {S3.DON_V7: []}
            ol = L.trade
            L.trade = F.don_recorder({S3.DON_V7: S3.DON_V7}, sd)
            try:
                L.f_donchian(c, L.daily_feats(c, btc.tfs["1D"]["close"]), A, Z, n=20, k=2.0, regime="btc")
            finally:
                L.trade = ol
            dn += sd[S3.DON_V7]
            daily[name] = S3.DailyOnly(c)
        print(f"[{name}] ({time.time() - t0:.0f}s)", flush=True)
    bot = []
    for t in em + dn:
        m_, tk = S3.MEXC.get(t["coin"], (0.0, 0.0002))
        maker_in = t["strat"] == "EMA" and t.get("maker_entry", False)
        bot.append(dict(t, fee_in=m_ if maker_in else tk, fee_out=tk))
    tr, eq, _ = RQ.run_pf(bot)
    bot_eq = DD.mtm_curve(tr, daily, A, Z, 100.0)
    idx = pd.date_range(A, Z, freq="1D", inclusive="left", tz="UTC")
    br = bot_eq.pct_change().fillna(0.0)
    br.index = br.index.tz_convert("UTC") if br.index.tz else br.index.tz_localize("UTC")
    br = br.reindex(idx).fillna(0.0)

    def panel(k):
        return pd.DataFrame({n: d[k] for n, d in ohlc.items()}).sort_index()
    O, H, Lo, Cl = panel("open"), panel("high"), panel("low"), panel("close")
    for X in (O, H, Lo):
        X.drop(X.index[X.index.duplicated()], inplace=True)
    Cl = Cl[~Cl.index.duplicated()]
    O, H, Lo = O.reindex(Cl.index), H.reindex(Cl.index), Lo.reindex(Cl.index)
    FU = pd.DataFrame(funds).reindex(Cl.index).fillna(0.0)
    listed = Cl.notna() & Cl.shift(30).notna()
    fr = FU.rolling(14).mean().where(listed)
    sel_long, sel_short = {}, {}
    for d in Cl.index:
        s = fr.loc[d].dropna()
        if len(s) >= 10:
            sel_long[d] = list(s.nsmallest(5).index)
            sel_short[d] = list(s.nlargest(5).index)

    variants = [("CANLI: 7 gün, TP yok", dict(every=7))]
    variants += [(f"{e} gün, TP yok", dict(every=e)) for e in (3, 1)]
    variants += [(f"{e} gün, TP yok", dict(every=e)) for e in (2, 4, 5)]
    for e, tp, mode in itertools.product((7, 3), (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50), ("bekle", "yeniden")):
        variants.append((f"{e} gün, TP %{tp * 100:g}, {mode}", dict(every=e, tp=tp, mode=mode)))
    for e, tp in itertools.product((2, 4, 5), (0.20, 0.25, 0.30)):
        variants.append((f"{e} gün, TP %{tp * 100:g}, yeniden", dict(every=e, tp=tp, mode="yeniden")))
    for e, (a_, b_) in itertools.product((7, 3), ((0.10, 0.05), (0.15, 0.07), (0.20, 0.10), (0.30, 0.15))):
        variants.append((f"{e} gün, iz: +%{a_ * 100:g} sonra %{b_ * 100:g} geri", dict(every=e, trail=(a_, b_))))

    def combo(r, w=0.33):
        out, vb, vs = [], 1 - w, w
        for d in idx:
            if d.day == 1:
                tot = vb + vs
                vb, vs = tot * (1 - w), tot * w
            vb *= 1 + br.loc[d]
            vs *= 1 + r.loc[d]
            out.append(vb + vs)
        return pd.Series(out, index=idx).pct_change().fillna(0.0)

    rows = []
    for name, kw in variants:
        r = simulate(O, H, Lo, Cl, FU, sel_long, sel_short, **kw).reindex(idx).fillna(0.0)
        row = {"varyant": name}
        for lab, a, z in periods:
            s = S3.stats(r, a, z)
            row[f"{lab} son"], row[f"{lab} maxDD%"] = s["final"], s["mdd"]
        s = S3.stats(r, A, Z)
        row.update({"TÜM son": s["final"], "CAGR%": s["cagr"], "maxDD%": s["mdd"]})
        for y, a, z in years:
            row[f"{y} %"] = S3.stats(r, a, z)["ret"]
        cr = combo(r)
        s = S3.stats(cr, A, Z)
        row.update({"bot+1/3 son": s["final"], "bot+1/3 CAGR%": s["cagr"], "bot+1/3 maxDD%": s["mdd"]})
        rows.append(row)
        print(f"{name}: {row['TÜM son']:.1f} | " + " | ".join(f"{lab} {row[f'{lab} son']:.1f}" for lab in PL) +
              f" | bot+1/3 {row['bot+1/3 son']:.1f} DD {row['bot+1/3 maxDD%']:.1f}", flush=True)
    df = pd.DataFrame(rows)
    b0 = df.iloc[0]
    df["3 dönemde de canlıdan iyi"] = np.where(
        np.column_stack([df[f"{lab} son"] > b0[f"{lab} son"] for lab in PL]).all(axis=1), "✅", "")
    df.to_csv(os.path.join(args.out, "sonuclar.csv"), index=False, float_format="%.4g")
    cols = ["varyant", "TÜM son", "CAGR%", "maxDD%"] + [f"{lab} {k}" for lab in PL for k in ("son", "maxDD%")] + \
        ["bot+1/3 son", "bot+1/3 CAGR%", "bot+1/3 maxDD%", "3 dönemde de canlıdan iyi"]
    txt = ["FUNDING KARŞITI — KÂR AL / AL-SAT VARYANTLARI (L14, 5+5 coin, 1x)",
           "VERİ: SENTETİK" if args.synthetic else f"{Cl.shape[1]} coin, Binance USDT-M günlük OHLC + funding",
           "Maliyet her açılış/kapanışta %0.04 (komisyon+kayma). 'bot+1/3' = bot v8 (2/3) + bu strateji (1/3).", "",
           df[cols].round(2).to_string(index=False), "",
           "Yıllık getiri %:", df[["varyant"] + [f"{y[0]} %" for y in years]].round(1).to_string(index=False)]
    text = "\n".join(txt)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
