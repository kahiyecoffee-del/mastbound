"""PORTFÖY KURALLARI — gerçek MEXC API ücretleriyle (limit %0.06, piyasa %0.08): profesyonel risk kurallarının
bot v8'e (EMA + DONCHIAN, funding hariç) etkisi. Başlangıç 670 USDT (canlı hesap; minimum emir tutarları gerçekçi).

  A) temel: mevcut bot (EMA en fazla 3, DON bakiyenin %50'si 1x) — eski varsayım ücret vs gerçek ücret
  B) aynı anda daha çok EMA pozisyonu (4/5/6), işlem başı risk %1 sabit
  C) aynı yön tavanı: aynı yönde (long ya da short) açık pozisyon sayısı ≤ N (korelasyon riski)
  D) DONCHIAN riske göre boyut (%0.5/%1/%1.5 risk, en fazla 1–3 pozisyon) — şu an sabit %50 pay
  E) düşüş freni: bakiye zirveden %X düşünce yeni işlemlerin riski yarıya
  F) birleşik adaylar

Adaylar (sinyal + çıkış) botun v8 kurallarıyla bir kez üretilir; kurallar yalnız portföy katmanında değişir.

  python research_portfolio.py                 # Binance 1dk arşivi (data.binance.vision) → results/portfolio/
  python research_portfolio.py --synthetic     # kod testi
"""
import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd

import config as C
import research as R
import research_combo as CB
import research_cvd as RC
import research_daily_detail as DD
import research_ema921 as E
import research_fix3 as F
import research_ls as L
from research_quant import Adaptive, r_multiple

DON_V7 = ((1.0, 0.1), (3.0, 1.0), None)
BASE_V = (None, None, None)
REAL = (0.0006, 0.0008)          # hesabın gerçek MEXC API ücretleri (2026-09 dolumları)
OLD = (0.0002, 0.0005)           # önceki testlerin varsayımı


def run_pf2(cands, fees=REAL, risk=0.01, lev=2.0, k_ema=3, k_don=1, don_mode="alloc", don_alloc=0.5,
            don_lev=1.0, don_risk=0.01, dir_cap=None, brake=None, start_bal=670.0):
    """research_quant.run_pf + kurallar. brake=(dd_esik, carpan): kapanmış bakiye zirveden dd_esik kadar
    düşükken yeni işlemlerin boyutu × carpan. dir_cap: aynı yönde (EMA+DON) en fazla bu kadar açık pozisyon."""
    ad = Adaptive()
    order = {c: i for i, c in enumerate(C.COINS)}
    cands = sorted(cands, key=lambda c: (c["entry_time"], order.get(c["coin"], 99)))
    bal, peak, open_, trades, eq = start_bal, start_bal, [], [], []
    per_day, loss_day = {}, {}
    mk, tk = fees

    def realize(upto):
        nonlocal bal, peak
        open_.sort(key=lambda p: p[0])
        while open_ and (upto is None or open_[0][0] <= upto):
            t_exit, pnl, rec = open_.pop(0)
            pnl = max(pnl, -bal)
            bal += pnl
            peak = max(peak, bal)
            rec["pnl"], rec["balance"] = pnl, bal
            if pnl < 0 and rec["strat"] == "EMA":
                d_ = t_exit.floor("1D")
                loss_day[d_] = loss_day.get(d_, 0) + 1
            trades.append(rec)
            eq.append((t_exit, bal))

    for c in cands:
        realize(c["entry_time"])
        if bal <= 0:
            break
        s = c["strat"]
        cap = k_ema if s == "EMA" else k_don
        if sum(p[2]["strat"] == s for p in open_) >= cap or any(p[2]["coin"] == c["coin"] for p in open_):
            continue
        if dir_cap is not None and sum(p[2]["dir"] == c["dir"] for p in open_) >= dir_cap:
            continue
        D = c["entry_time"].floor("1D")
        if s == "EMA" and (per_day.get(D, 0) >= 5 or loss_day.get(D, 0) >= 3):
            continue
        m = 1.0
        if brake is not None and bal < peak * (1 - brake[0]):
            m = brake[1]
        sm = c.get("size_mult", 1.0)
        stop_frac = abs(c["entry"] - c["stop"]) / c["entry"]
        don_open = any(p[2]["strat"] == "DONCHIAN" for p in open_)
        if s == "DONCHIAN":
            if don_mode == "alloc":
                notional, lv = bal * don_alloc * don_lev * m, don_lev
            else:
                notional, lv = min(bal * don_risk * m / max(stop_frac, 1e-4), bal * don_lev / k_don), don_lev
        else:
            base = bal * (1 - don_alloc) if (don_mode == "alloc" and don_open) else bal
            notional, lv = min(base * risk * m * sm / max(stop_frac, 1e-4), base * lev * sm / k_ema), lev * max(sm, 1.0)
        if notional < C.MIN_NOTIONAL.get(c["coin"], C.MIN_NOTIONAL_DEFAULT):
            continue
        qty = notional / c["entry"]
        fee_in = mk if (s == "EMA" and c.get("maker_entry", False)) else tk
        pnl = qty * c["dir"] * (c["exit"] - c["entry"]) - fee_in * notional - tk * qty * c["exit"] - qty * c["funding_px"]
        pnl = max(pnl, -notional / lv)
        if s == "EMA":
            per_day[D] = per_day.get(D, 0) + 1
        open_.append((c["exit_time"], pnl, dict(c, notional=notional, qty=qty, risk_mult=m)))
    realize(None)
    return trades, eq


def scenarios():
    S = [("A0 mevcut bot — ESKİ ücret varsayımı (%0.02/%0.05)", dict(fees=OLD)),
         ("A1 mevcut bot — GERÇEK ücret (%0.06/%0.08)", dict())]
    S += [(f"B EMA en fazla {k} pozisyon", dict(k_ema=k)) for k in (4, 5, 6)]
    S += [(f"C aynı yön tavanı {n}", dict(dir_cap=n)) for n in (1, 2, 3)]
    S += [(f"C EMA 5 + aynı yön tavanı {n}", dict(k_ema=5, dir_cap=n)) for n in (2, 3)]
    for r_ in (0.005, 0.01, 0.015):
        for kd in (1, 2, 3):
            S.append((f"D DON risk %{r_ * 100:g}, en fazla {kd}", dict(don_mode="risk", don_risk=r_, k_don=kd)))
    S += [(f"E düşüş freni: zirveden %{th * 100:g} → risk ×{m:g}", dict(brake=(th, m)))
          for th, m in ((0.10, 0.5), (0.15, 0.5), (0.20, 0.5), (0.15, 0.25))]
    S += [("F EMA5 + yön≤3 + DON risk %1 ×2 + fren %15", dict(k_ema=5, dir_cap=3, don_mode="risk", don_risk=0.01,
                                                          k_don=2, brake=(0.15, 0.5))),
          ("F EMA4 + yön≤2 + DON risk %1 ×2", dict(k_ema=4, dir_cap=2, don_mode="risk", don_risk=0.01, k_don=2)),
          ("F EMA3 + DON risk %1 ×2 + fren %15", dict(don_mode="risk", don_risk=0.01, k_don=2, brake=(0.15, 0.5)))]
    return S


def build(args, A, Z, cache):
    """Botun v8 kurallarıyla aday işlemler + günlük kapanışlar (MTM için). Önbelleğe yalnız bunlar yazılır."""
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    import research_15m as Q
    C.COINS = args.coins.split(",")
    E.SIG_TF, E.TREND_TF, E.MAX_HOLD = "1h", "4h", pd.Timedelta(hours=168)
    t0 = time.time()
    coins = R.load(args, A, Z)
    btc = coins.get("BTC")
    ema, don = {BASE_V: []}, {DON_V7: []}
    o_q, o_l = Q.trade, L.trade
    for n, c in coins.items():
        x = E.prep(c, btc)
        Q.trade = F.ema_recorder(RC.flow_signals(c), {BASE_V: BASE_V}, ema)
        try:
            E.signals(c, x, A, Z, "ikisi", "ema9_limit", "atr3", "iz3")
        finally:
            Q.trade = o_q
        feats = L.daily_feats(c, btc.tfs["1D"]["close"] if btc else None)
        L.trade = F.don_recorder({DON_V7: DON_V7}, don)
        try:
            L.f_donchian(c, feats, A, Z, n=20, k=2.0, regime="btc")
        finally:
            L.trade = o_l
        print(f"[{n}] adaylar hazır ({time.time() - t0:.0f}s)", flush=True)
    em = []
    for t in ema[BASE_V]:
        sp = abs(t["entry"] - t["stop"]) / t["entry"] * 100
        em.append(dict(t, strat="EMA", size_mult=0.5 if sp > 5.0 else 1.0))
    cands = em + [dict(t, strat="DONCHIAN") for t in don[DON_V7]]
    closes = {n: c.tfs["1D"]["close"] for n, c in coins.items()}
    with open(cache, "wb") as f:
        pickle.dump((cands, closes), f)
    return cands, closes


def mtm_dd(trades, closes, a, z, start_bal):
    """research_daily_detail.mtm_curve ile aynı: açık pozisyonların gün kapanışı değerlemesi dahil bakiye."""
    days = pd.date_range(a, z, freq="1D", inclusive="left")
    real = pd.Series(0.0, index=days)
    unreal = pd.Series(0.0, index=days)
    for t in trades:
        close_day = t["exit_time"].floor("1D")
        real[real.index > close_day] += t["pnl"]
        held = days[(days >= t["entry_time"].floor("1D")) & (days <= close_day)]
        if len(held):
            px = closes[t["coin"]].reindex(held).ffill().to_numpy()
            unreal.loc[held] += t["qty"] * t["dir"] * (px - t["entry"])
    return start_bal + real + unreal


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
    ap.add_argument("--out", default="results/portfolio")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    A, V, T_, Z = (pd.Timestamp(x, tz="UTC") for x in (args.start, args.val, args.test, args.end))
    periods = [("2021-23", A, V), ("2024-25.09", V, T_), ("SON 1 YIL", T_, Z)]
    years = [(str(y), pd.Timestamp(f"{y}-01-01", tz="UTC"), min(pd.Timestamp(f"{y + 1}-01-01", tz="UTC"), Z))
             for y in range(A.year, Z.year + 1) if pd.Timestamp(f"{y}-01-01", tz="UTC") < Z]
    cache = os.path.join(args.out, "adaylar_syn.pkl" if args.synthetic else "adaylar.pkl")
    cands, closes = build(args, A, Z, cache)
    print(f"aday işlem: {len(cands)} (EMA {sum(c['strat'] == 'EMA' for c in cands)}, "
          f"DON {sum(c['strat'] == 'DONCHIAN' for c in cands)})", flush=True)

    rows = []
    for name, kw in scenarios():
        r = {"senaryo": name}
        tr, eq = run_pf2(cands, **kw)
        m = L.metrics(tr, eq, A, Z, start_bal=670.0)
        mtm = mtm_dd(tr, closes, A, Z, 670.0)
        r.update({"670→": m["final"], "CAGR%": m["cagr"], "gerçek maxDD%": DD.dd(mtm),
                  "en kötü gün %": float(mtm.pct_change().min() * 100), "işlem": len(tr)})
        for lab, a, z in periods:
            tp, ep = run_pf2([c for c in cands if a <= c["entry_time"] < z], **kw)
            mp = L.metrics(tp, ep, a, z, start_bal=670.0)
            r[f"{lab} 100→"], r[f"{lab} DD%"] = mp["final"] / 6.7, mp["mdd"]
        for lab, a, z in years:
            ty, ey = run_pf2([c for c in cands if a <= c["entry_time"] < z], **kw)
            r[f"{lab} %"] = L.metrics(ty, ey, a, z, start_bal=670.0)["ret"]
        rows.append(r)
        print(f"{name}: CAGR {m['cagr']:.1f}% gerçekDD {r['gerçek maxDD%']:.1f}%", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "sonuclar.csv"), index=False, float_format="%.4g")
    pc = [f"{lab} {k}" for lab, _, _ in periods for k in ("100→", "DD%")]
    txt = ["PORTFÖY KURALLARI — bot v8 (EMA + DONCHIAN; funding stratejisi hariç), 18 coin, "
           f"{args.start} → {args.end}, başlangıç 670 USDT",
           "VERİ: SENTETİK" if args.synthetic else "Veri: Binance USDT-M 1dk + taker hacmi + funding (data.binance.vision)",
           "Ücret: A0 hariç GERÇEK MEXC API (limit giriş %0.06, piyasa giriş/çıkış %0.08); kayma %0.02/dolum ve funding dahil.",
           "Dönem sütunları her dönem 100'den ayrı başlar (son 1 yıl = hiç görülmemiş). gerçek maxDD = açık pozisyonlar "
           "dahil saatlik.", "",
           df[["senaryo", "670→", "CAGR%", "gerçek maxDD%", "en kötü gün %", "işlem"] + pc].round(1).to_string(index=False),
           "", "Yıllık getiri %:", df[["senaryo"] + [f"{y[0]} %" for y in years]].round(1).to_string(index=False), "",
           "OKUMA: bir kural ancak (1) CAGR/gerçekDD mevcut bottan (A1) iyiyse, (2) SON 1 YIL'da da iyiyse ve (3) komşu",
           "ayarlarda (B/C/D/E içindeki sıralar) tutarlıysa canlıya aday."]
    text = "\n".join(txt)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
