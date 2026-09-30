"""Kullanıcı stratejisi: 1s MACD trendi + 15dk EMA9/21 kesişimi + geri çekilmede giriş (long/short).

LONG : 1s MACD yükselen trendde (kapanmış 1s mum) ve 15dk'da EMA9, EMA21'i yukarı keser →
       en fazla 8 mum (2 saat) geri çekilme beklenir → giriş. SHORT tam tersi.
Kurulum iptali: geri çekilme gelmeden EMA9/21 ters keserse ya da 1s trend bozulursa.
Varyantlar:
  macd   : "sinyal" (MACD>sinyal) | "sifir" (MACD>0) | "ikisi" (ikisi birden)
  giris  : "ema9_limit" | "ema21_limit" (değince limit emir, maker) |
           "ema9_onay" (EMA9'a değip üstünde kapanan mumdan sonra market)
  stop   : "swing" (son 5 mumun dibi/tepesi -0.1ATR) | "atr" (1.5 x ATR)
  cikis  : "tp2" (2R, maker) | "iz" (2xATR iz süren stop) | "ema_ters" (EMA9/21 ters kesişince)
  her durumda en fazla 12 saat taşıma.
Risk: işlem başı %1 / %2, kaldıraç ≤2x, maks 2 eşzamanlı pozisyon, GÜNDE MAKS 5 İŞLEM,
      günde 3 zararlı işlemden sonra o gün yeni işlem yok.
Dönemler: 2021-2023 (keşif), 2024-2025.09 (doğrulama), 2025.09-2026.09 (görülmemiş test).
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
import research as R
import research_15m as Q
import research_ls as L

SIG_TF, TREND_TF = "15min", "1h"          # --tf / --trend-tf ile değişir
WAIT_BARS = 8
MAX_HOLD = pd.Timedelta(hours=12)
MAX_PER_DAY, MAX_LOSSES_DAY = 5, 3


def prep(coin, btc=None):
    from research_15m_v2 import adx as _adx
    df = coin.tfs[SIG_TF]
    x = pd.DataFrame(index=df.index)
    for c in ("open", "high", "low", "close"):
        x[c] = df[c]
    x["T"] = df.index + pd.Timedelta(SIG_TF)
    x["e9"], x["e21"] = L.ema(df["close"], 9), L.ema(df["close"], 21)
    x["atr"] = R.atr(df)
    h = coin.tfs[TREND_TF]
    m = L.ema(h["close"], 12) - L.ema(h["close"], 26)
    sg = L.ema(m, 9)
    mm = pd.DataFrame({"m": m.to_numpy(), "s": sg.to_numpy()}, index=h.index + pd.Timedelta(TREND_TF))
    j = pd.merge_asof(x[["T"]], mm, left_on="T", right_index=True)
    x["macd"], x["sig"] = j["m"].to_numpy(), j["s"].to_numpy()
    x["adx"] = _adx(df).to_numpy()
    b = (btc or coin).tfs["1D"]["close"]
    up = pd.DataFrame({"u": (b > L.ema(b, 200)).astype(int).to_numpy()}, index=b.index + pd.Timedelta("1D"))
    x["btc_up"] = pd.merge_asof(x[["T"]], up, left_on="T", right_index=True)["u"].fillna(0).to_numpy()
    return x


def trend(x, mode):
    m, s = x["macd"].to_numpy(), x["sig"].to_numpy()
    with np.errstate(invalid="ignore"):
        if mode == "sinyal":
            up, dn = m > s, m < s
        elif mode == "sifir":
            up, dn = m > 0, m < 0
        else:
            up, dn = (m > s) & (m > 0), (m < s) & (m < 0)
    return np.where(up, 1, np.where(dn, -1, 0))


def signals(coin, x, a, b, macd_mode, entry, stop_mode, exit_mode, regime="none", adx_min=0.0):
    mm = coin.mm
    O, H, Lo, Cl = (x[c].to_numpy() for c in ("open", "high", "low", "close"))
    e9, e21, atr = x["e9"].to_numpy(), x["e21"].to_numpy(), x["atr"].to_numpy()
    tr = trend(x, macd_mode)
    T = x["T"]
    Tns = T.dt.as_unit("ns").astype("int64").to_numpy()
    adxv, bup = x["adx"].to_numpy(), x["btc_up"].to_numpy()
    rel = np.sign(e9 - e21)
    cross = np.zeros(len(Cl), int)
    cross[1:] = np.where((rel[1:] != rel[:-1]) & (rel[1:] != 0), rel[1:], 0)
    win = ((T >= a) & (T < b)).to_numpy()
    out = []
    n = len(Cl)
    for i in np.flatnonzero(win & (cross != 0)):
        d = cross[i]
        if tr[i] != d:
            continue
        if regime == "btc200" and ((d == 1 and bup[i] != 1) or (d == -1 and bup[i] == 1)):
            continue
        if adx_min and not (adxv[i] >= adx_min):
            continue
        for j in range(i + 1, min(i + 1 + WAIT_BARS, n)):
            # j. mum boyunca geçerlilik: j-1 kapanışındaki bilgiye göre
            if rel[j - 1] != d or tr[j - 1] != d:
                break
            s1 = int(np.searchsorted(mm.t, Tns[j - 1]))            # j. mumun ilk 1dk'sı
            s2 = int(np.searchsorted(mm.t, Tns[j]))
            if s2 <= s1:
                continue
            e = None
            limit_px = None
            if entry in ("ema9_limit", "ema21_limit"):
                lvl = e9[j - 1] if entry == "ema9_limit" else e21[j - 1]
                seg = mm.l[s1:s2] if d == 1 else mm.h[s1:s2]
                hit = np.flatnonzero(seg <= lvl) if d == 1 else np.flatnonzero(seg >= lvl)
                if len(hit):
                    e = s1 + hit[0]
                    limit_px = min(lvl, mm.o[e]) if d == 1 else max(lvl, mm.o[e])
                    ref = j - 1
            else:   # onay: j. mum EMA9'a değip onun doğru tarafında, trend yönünde kapanır
                touched = Lo[j] <= e9[j] if d == 1 else H[j] >= e9[j]
                ok = (Cl[j] > e9[j] and Cl[j] > O[j]) if d == 1 else (Cl[j] < e9[j] and Cl[j] < O[j])
                if touched and ok and rel[j] == d and tr[j] == d:
                    e = s2
                    ref = j
            if e is None:
                continue
            if e >= len(mm.o):
                break
            ent = limit_px if limit_px is not None else mm.o[e] * (1 + d * C.SLIPPAGE)
            if stop_mode == "swing":
                lo = max(0, ref - 4)
                stop = (Lo[lo:ref + 1].min() - 0.1 * atr[ref]) if d == 1 else (H[lo:ref + 1].max() + 0.1 * atr[ref])
            else:
                stop = ent - d * (float(stop_mode[3:]) if len(stop_mode) > 3 else 1.5) * atr[ref]
            if d * (ent - stop) <= ent * 0.001:          # çok dar / ters stop
                break
            R_ = abs(ent - stop)
            dl_t = x["T"].iloc[ref] + MAX_HOLD
            if exit_mode == "ema_ters":
                k = np.flatnonzero(rel[ref + 1:] == -d)
                if len(k):
                    dl_t = min(dl_t, T.iloc[ref + 1 + k[0]])          # ters kesişen mum kapanışı
            dl = int(np.searchsorted(mm.t, dl_t.value))
            kw = dict(limit_px=limit_px, deadline=dl, trail_tf=SIG_TF)
            if exit_mode == "mix":        # yarısı 2R'de, yarısı 3xATR iz süren stopla
                ta = Q.trade(coin, e, d, stop, T.iloc[ref], tp=ent + d * 2 * R_, **kw)
                tb = Q.trade(coin, e, d, stop, T.iloc[ref], trail_k=3.0, **kw)
                t = None
                if ta and tb:
                    t = dict(ta)
                    ea, eb = ta["exit"], tb["exit"]
                    t["exit"] = (ea + eb) / 2
                    t["fee_out"] = (ta["fee_out"] * ea + tb["fee_out"] * eb) / (ea + eb)
                    t["exit_time"] = max(ta["exit_time"], tb["exit_time"])
                    t["reason"] = f'{ta["reason"]}/{tb["reason"]}'
                    t["funding_px"] = (ta["funding_px"] + tb["funding_px"]) / 2
            else:
                t = Q.trade(coin, e, d, stop, T.iloc[ref],
                            tp=ent + d * float(exit_mode[2:]) * R_ if exit_mode.startswith("tp") else None,
                            trail_k=(2.0 if exit_mode == "iz" else float(exit_mode[2:]) if exit_mode.startswith("iz") else 0.0),
                            **kw)
            if t:
                out.append(t)
            break                       # her kesişim için tek giriş
    return out


def run_pf(cands, risk, lev=2.0, K=2, start_bal=100.0, day_rules=True, max_dir=0):
    order = {c: i for i, c in enumerate(C.COINS)}
    cands = sorted(cands, key=lambda c: (c["entry_time"], order.get(c["coin"], 99)))
    bal, open_, trades, eq = start_bal, [], [], []
    per_day, loss_day = {}, {}

    def realize(upto):
        nonlocal bal
        open_.sort(key=lambda p: p[0])
        while open_ and (upto is None or open_[0][0] <= upto):
            t_exit, pnl, rec = open_.pop(0)
            pnl = max(pnl, -bal)
            bal += pnl
            rec["pnl"], rec["balance"] = pnl, bal
            if pnl < 0:
                dd_ = t_exit.floor("1D")
                loss_day[dd_] = loss_day.get(dd_, 0) + 1
            trades.append(rec)
            eq.append((t_exit, bal))

    for c in cands:
        realize(c["entry_time"])
        if bal <= 0:
            break
        D = c["entry_time"].floor("1D")
        if day_rules and (per_day.get(D, 0) >= MAX_PER_DAY or loss_day.get(D, 0) >= MAX_LOSSES_DAY):
            continue
        if len(open_) >= K or any(p[2]["coin"] == c["coin"] for p in open_):
            continue
        if max_dir and sum(p[2]["dir"] == c["dir"] for p in open_) >= max_dir:
            continue
        stop_frac = abs(c["entry"] - c["stop"]) / c["entry"]
        notional = min(bal * risk / max(stop_frac, 1e-4), bal * lev / K)
        if notional < C.MIN_NOTIONAL.get(c["coin"], C.MIN_NOTIONAL_DEFAULT):
            continue
        qty = notional / c["entry"]
        fees = c["fee_in"] * notional + c["fee_out"] * qty * c["exit"]
        pnl = qty * c["dir"] * (c["exit"] - c["entry"]) - fees - qty * c["funding_px"]
        pnl = max(pnl, -notional / lev)
        per_day[D] = per_day.get(D, 0) + 1
        open_.append((c["exit_time"], pnl, dict(c, notional=notional, fees=fees,
                                                R=pnl / (qty * abs(c["entry"] - c["stop"])))))
    realize(None)
    return trades, eq


def per_coin(cands, sname, periods):
    """Coin bazında (portföy kısıtı olmadan) maliyet sonrası işlem başı R."""
    rows = []
    for coin in sorted({t["coin"] for t in cands}):
        r = {"strateji": sname, "coin": coin}
        for p, a, z in periods:
            ts = [t for t in cands if t["coin"] == coin and a <= t["entry_time"] < z]
            net = [(t["dir"] * (t["exit"] - t["entry"]) - t["fee_in"] * t["entry"] - t["fee_out"] * t["exit"])
                   / abs(t["entry"] - t["stop"]) for t in ts]
            r[f"{p}_n"] = len(ts)
            r[f"{p}_netR"] = float(np.mean(net)) if net else np.nan
            r[f"{p}_toplamR"] = float(np.sum(net)) if net else 0.0
            r[f"{p}_isabet_%"] = float(np.mean([x > 0 for x in net]) * 100) if net else np.nan
        rows.append(r)
    return rows


def day_stats(trades, a, z):
    days = pd.date_range(a, z, freq="1D", inclusive="left")
    cnt = pd.Series(0, index=days)
    for t in trades:
        D = t["entry_time"].floor("1D")
        if D in cnt.index:
            cnt[D] += 1
    return cnt.mean(), (cnt >= 3).mean() * 100, (cnt == 0).mean() * 100


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
    ap.add_argument("--out", default="results/ema921_pullback")
    ap.add_argument("--exits", default="tp2,iz,ema_ters", help="ör. tp3 ya da tp2,tp3,iz")
    ap.add_argument("--tf", default="15min", help="sinyal zaman dilimi (EMA9/21): 15min | 1h")
    ap.add_argument("--trend-tf", default="1h", help="MACD trend zaman dilimi: 1h | 4h")
    ap.add_argument("--max-hold-h", type=int, default=12)
    ap.add_argument("--macd", default="sinyal,sifir,ikisi")
    ap.add_argument("--entries", default="ema9_limit,ema21_limit,ema9_onay")
    ap.add_argument("--stops", default="swing,atr")
    ap.add_argument("--risks", default="1,2", help="işlem başı risk %%")
    ap.add_argument("--K", default="2", help="maks eşzamanlı pozisyon, ör. 2,3,4")
    ap.add_argument("--regimes", default="none", help="none | btc200 (BTC günlük EMA200 yönünde işlem)")
    ap.add_argument("--adx", default="0", help="1s ADX alt sınırı, ör. 0,20")
    ap.add_argument("--dircap", default="0", help="aynı yönde maks pozisyon (0=sınırsız), ör. 0,2")
    args = ap.parse_args()
    global SIG_TF, TREND_TF, MAX_HOLD
    SIG_TF, TREND_TF, MAX_HOLD = args.tf, args.trend_tf, pd.Timedelta(hours=args.max_hold_h)
    os.makedirs(args.out, exist_ok=True)
    A, V, T_, E = (pd.Timestamp(x, tz="UTC") for x in (args.start, args.val, args.test, args.end))
    periods = [("kesif", A, V), ("dogrulama", V, T_), ("TEST", T_, E)]
    years = [(str(y), pd.Timestamp(f"{y}-01-01", tz="UTC"), min(pd.Timestamp(f"{y + 1}-01-01", tz="UTC"), T_))
             for y in range(A.year, T_.year + 1) if pd.Timestamp(f"{y}-01-01", tz="UTC") < T_]
    coins = R.load(args, A, E)
    xs = {n: prep(c, coins.get("BTC")) for n, c in coins.items()}
    rows, keep = [], {}
    t0 = time.time()
    Ks = [int(k) for k in args.K.split(",")]
    per_coin_rows = []
    for macd_mode in args.macd.split(","):
        for entry in args.entries.split(","):
            for stop_mode in args.stops.split(","):
                for exit_mode, regime, adx_min in [(x_, r_, float(a_)) for x_ in args.exits.split(",")
                                                   for r_ in args.regimes.split(",") for a_ in args.adx.split(",")]:
                    sname = (f"MACD{TREND_TF}={macd_mode} giriş={entry} stop={stop_mode} çıkış={exit_mode}"
                             + (f" rejim={regime}" if regime != "none" else "")
                             + (f" ADX≥{adx_min:g}" if adx_min else ""))
                    cands = []
                    for n, c in coins.items():
                        cands += signals(c, xs[n], A, E, macd_mode, entry, stop_mode, exit_mode, regime, adx_min)
                    per_coin_rows.extend(per_coin(cands, sname, periods))
                    gR = np.array([t["dir"] * (t["exit"] - t["entry"]) / abs(t["entry"] - t["stop"]) for t in cands])
                    for risk, K, dcap in [(float(r) / 100, k, int(dc)) for r in args.risks.split(",") for k in Ks
                                          for dc in args.dircap.split(",")]:
                        kname = sname + (f" maksPoz={K}" if len(Ks) > 1 else "") + (f" aynıYönMaks={dcap}" if dcap else "")
                        row = {"strateji": kname, "risk_%": risk * 100, "K": K, "coin_sayisi": len(coins),
                               "aday": len(cands), "aday_gun": len(cands) / max((E - A).days, 1),
                               "brut_R_ort": gR.mean() if len(gR) else np.nan}
                        for p, a, z in periods:
                            cp = [t for t in cands if a <= t["entry_time"] < z]
                            tr, eq = run_pf(cp, risk, K=K, max_dir=dcap)
                            m = L.metrics(tr, eq, a, z)
                            mean_d, ge3, zero = day_stats(tr, a, z)
                            row.update({f"{p}_son": m["final"], f"{p}_CAGR_%": m["cagr"], f"{p}_maxDD_%": m["mdd"],
                                        f"{p}_islem": m["n"], f"{p}_isabet_%": m["win"],
                                        f"{p}_netR": np.mean([t["R"] for t in tr]) if tr else np.nan,
                                        f"{p}_islem_gun": mean_d, f"{p}_3plus_gun_%": ge3,
                                        f"{p}_long": sum(t["dir"] == 1 for t in tr),
                                        f"{p}_short": sum(t["dir"] == -1 for t in tr),
                                        f"{p}_long_pnl": sum(t["pnl"] for t in tr if t["dir"] == 1),
                                        f"{p}_short_pnl": sum(t["pnl"] for t in tr if t["dir"] == -1)})
                            keep[(kname, risk, p)] = (tr, eq)
                        for lab, a, z in years:
                            tr, eq = run_pf([t for t in cands if a <= t["entry_time"] < z], risk, K=K, max_dir=dcap)
                            row[f"{lab}_%"] = L.metrics(tr, eq, a, z)["ret"]
                        rows.append(row)
                    print(f"{sname}: aday {len(cands)}, brüt R {gR.mean() if len(gR) else float('nan'):+.3f} "
                          f"({time.time() - t0:.0f}s)", flush=True)

    lb = pd.DataFrame(rows)
    lb["karli_yil"] = (lb[[f"{y[0]}_%" for y in years]] > 0).sum(axis=1)
    lb = lb.sort_values("kesif_son", ascending=False)
    lb.to_csv(os.path.join(args.out, "tum_sonuclar.csv"), index=False, float_format="%.4g")
    cols = (["strateji", "risk_%", "brut_R_ort", "kesif_son", "kesif_maxDD_%", "kesif_islem_gun", "kesif_3plus_gun_%",
             "kesif_isabet_%", "dogrulama_son", "dogrulama_maxDD_%", "TEST_son", "TEST_maxDD_%", "TEST_islem_gun",
             "karli_yil"] + [f"{y[0]}_%" for y in years])
    fmt = lambda d: d[cols].round(2).to_string(index=False)
    ls_cols = ["strateji", "risk_%"] + [f"{p}_{k}" for p, _, _ in periods
                                        for k in ("long", "long_pnl", "short", "short_pnl")]
    core_x = next((x for x in args.exits.split(",") if x.startswith("tp")), args.exits.split(",")[0])
    core = lb[(lb.strateji == f"MACD{TREND_TF}=ikisi giriş=ema9_limit stop=swing çıkış={core_x}") & (lb["risk_%"] == 1)]
    top = lb[lb["risk_%"] == 1].head(5)
    both = lb[(lb.kesif_son > 100) & (lb.dogrulama_son > 100)].sort_values("dogrulama_son", ascending=False)
    if len(top):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        for ax, (p, a, z) in zip(axes, periods):
            for n in list(top.strateji)[:5]:
                tr, eq = keep[(n, 0.01, p)]
                if eq:
                    ax.step([eq[0][0]] + [t for t, _ in eq], [100] + [v for _, v in eq], where="post", lw=1.2,
                            label=n.replace(f"MACD{TREND_TF}=", "").replace(" giriş=", " | ").replace(" stop=", " | ")
                            .replace(" çıkış=", " | "))
            ax.axhline(100, color="gray", ls="--", lw=0.8)
            ax.set_title({"kesif": "Keşif 2021-23", "dogrulama": "Doğrulama 2024-25.09",
                          "TEST": "GÖRÜLMEMİŞ TEST 25.09-26.09"}[p])
            ax.grid(alpha=0.3)
            ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        axes[0].legend(fontsize=6)
        fig.suptitle(f"{TREND_TF} MACD + {SIG_TF} EMA9/21 geri çekilme — keşifte en iyi 5 (risk %1, 2x, günde maks 5 işlem)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "equity_top5.png"), dpi=120)
        plt.close(fig)
    pc = pd.DataFrame(per_coin_rows)
    pc.to_csv(os.path.join(args.out, "coin_bazli.csv"), index=False, float_format="%.4g")
    pc_txt = []
    for sn, g in pc.groupby("strateji"):
        g = g.sort_values("kesif_toplamR", ascending=False)
        pc_txt += [f"  {sn}", g.drop(columns="strateji").round(3).to_string(index=False), ""]
    txt = [f"{TREND_TF} MACD TREND + {SIG_TF} EMA9/21 KESİŞİM + GERİ ÇEKİLME (long/short)",
           "VERİ: SENTETİK" if args.synthetic else "Veri: Binance USDT-M 1dk + funding (data.binance.vision)",
           f"Kurallar: günde maks {MAX_PER_DAY} işlem, günde {MAX_LOSSES_DAY} zarardan sonra dur, maks 2 pozisyon, "
           f"kaldıraç ≤2x, en fazla {int(MAX_HOLD.total_seconds() // 3600)} saat taşıma. Limit giriş/TP maker "
           f"%{Q.MAKER_FEE * 100:g}, diğerleri taker %{C.TAKER_FEE * 100:g} + slippage %{C.SLIPPAGE * 100:g}.",
           "'son' = 100 USDT'den dönem sonu bakiye (her dönem ayrı başlar). brut_R_ort = maliyet öncesi işlem başı R.", "",
           f"A) Senin tarifin (MACD>sinyal ve >0, EMA9'a geri çekilmede limit, swing stop, çıkış={core_x}), risk %1:",
           fmt(core) if len(core) else "(yok)", "",
           "B) Keşif dönemine göre en iyi 10 (risk %1):", fmt(lb[lb["risk_%"] == 1].head(10)), "",
           f"C) Hem keşifte hem doğrulamada kârlı olanlar: {len(both)} adet",
           fmt(both.head(10)) if len(both) else "(yok)", "",
           "F) KOMPAKT KARŞILAŞTIRMA (tüm satırlar):",
           lb[["strateji", "risk_%", "K", "coin_sayisi", "aday_gun", "kesif_islem_gun", "TEST_islem_gun",
               "kesif_son", "kesif_maxDD_%", "dogrulama_son", "dogrulama_maxDD_%", "TEST_son", "TEST_maxDD_%",
               "kesif_isabet_%", "brut_R_ort"]].round(2).to_string(index=False), "",
           "G) COİN BAZLI (portföy kısıtı yok, maliyet sonrası R; netR>0 = coin kârlı):", *pc_txt,
           "E) LONG / SHORT ayrımı (işlem sayısı ve USDT kâr/zarar, 100 USDT başlangıç) — keşifte en iyi 10, risk %1:",
           lb[lb["risk_%"] == 1].head(10)[ls_cols].round(1).to_string(index=False), "",
           "D) Maliyet öncesi (brüt) işlem başı R — tüm varyantlar:",
           lb.groupby("strateji")["brut_R_ort"].first().sort_values(ascending=False).round(3).to_string()]
    text = "\n".join(txt)
    with open(os.path.join(args.out, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
