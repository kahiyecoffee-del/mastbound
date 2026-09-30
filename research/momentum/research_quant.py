"""QUANT KATMANI testi — uyarlamalı kurallar 50/50 düzeni (EMA + DONCHIAN) iyileştiriyor mu?

Kurallar live_bot/quant.py'deki Adaptive sınıfıyla (botla AYNI kod) ve yalnız o ana kadar kapanmış
işlemlerin bilgisiyle uygulanır (ileriye bakma yok):
  * özsermaye filtresi "sma"  : bakiye son 30 kapanıştaki ortalamanın altındaysa risk × 0.5
  * özsermaye filtresi "dd"   : bakiye zirveden %15+ aşağıdaysa risk × 0.5
  * coin filtresi             : coin+strateji son 12 işlem ort. R < -0.3 (en az 8 işlem) → 30 gün kapalı
  * WALK-FORWARD STOP         : her çeyrek başında EMA stop çarpanı (2.5 / 3 / 3.5 ATR) son 12 ayın
                                sonucuna (getiri/maxDD) göre seçilir ve o çeyrek uygulanır
Hepsi mevcut sabit düzenle aynı raporda: dönemler, yıllar, gerçek (açık pozisyon dahil) maxDD.
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
import research_combo as CB
import research_daily_detail as DD
import research_ema921 as E
import research_ls as L
from live_bot.quant import Adaptive, r_multiple

STOPS = ("atr2.5", "atr3", "atr3.5")


def run_pf(cands, adaptive=None, risk=0.01, lev=2.0, k_ema=3, don_alloc=0.5, don_lev=1.0, start_bal=100.0):
    ad = adaptive or Adaptive()
    order = {c: i for i, c in enumerate(C.COINS)}
    cands = sorted(cands, key=lambda c: (c["entry_time"], order.get(c["coin"], 99)))
    bal, open_, trades, eq = start_bal, [], [], []
    per_day, loss_day = {}, {}
    skipped = {"coin": 0}

    def realize(upto):
        nonlocal bal
        open_.sort(key=lambda p: p[0])
        while open_ and (upto is None or open_[0][0] <= upto):
            t_exit, pnl, rec = open_.pop(0)
            pnl = max(pnl, -bal)
            bal += pnl
            rec["pnl"], rec["balance"] = pnl, bal
            if pnl < 0 and rec["strat"] == "EMA":
                d_ = t_exit.floor("1D")
                loss_day[d_] = loss_day.get(d_, 0) + 1
            ad.on_close(t_exit, rec["coin"], rec["strat"], r_multiple(pnl, rec["qty"], rec["entry"], rec["stop"]), bal)
            trades.append(rec)
            eq.append((t_exit, bal))

    for c in cands:
        realize(c["entry_time"])
        if bal <= 0:
            break
        s = c["strat"]
        cap = k_ema if s == "EMA" else 1
        if sum(p[2]["strat"] == s for p in open_) >= cap or any(p[2]["coin"] == c["coin"] for p in open_):
            continue
        D = c["entry_time"].floor("1D")
        if s == "EMA" and (per_day.get(D, 0) >= 5 or loss_day.get(D, 0) >= 3):
            continue
        if ad.blocked(c["coin"], s, c["entry_time"]):
            skipped["coin"] += 1
            continue
        m = ad.risk_mult(bal)
        sm = c.get("size_mult", 1.0)                 # kurulum kalitesine göre boyut çarpanı (A+ = büyük)
        if sm <= 0:
            continue
        stop_frac = abs(c["entry"] - c["stop"]) / c["entry"]
        if s == "DONCHIAN":
            notional, lv = bal * don_alloc * don_lev * m * sm, don_lev
        else:
            base = bal * (1 - don_alloc) if any(p[2]["strat"] == "DONCHIAN" for p in open_) else bal
            notional, lv = min(base * risk * m * sm / max(stop_frac, 1e-4), base * lev * sm / k_ema), lev * max(sm, 1.0)
        if notional < C.MIN_NOTIONAL.get(c["coin"], C.MIN_NOTIONAL_DEFAULT):
            continue
        qty = notional / c["entry"]
        fees = c.get("fee_in", C.TAKER_FEE) * notional + c.get("fee_out", C.TAKER_FEE) * qty * c["exit"]
        pnl = qty * c["dir"] * (c["exit"] - c["entry"]) - fees - qty * c["funding_px"]
        pnl = max(pnl, -notional / lv)
        if s == "EMA":
            per_day[D] = per_day.get(D, 0) + 1
        open_.append((c["exit_time"], pnl, dict(c, notional=notional, qty=qty, risk_mult=m)))
    realize(None)
    return trades, eq, skipped


def walk_forward(ema_by_stop, don, A, Z, train_days=365):
    """Her çeyrek: son `train_days` günde her stop varyantıyla portföyü çalıştır, getiri/maxDD'si en iyi
    olanı seç, o çeyreğin EMA adaylarını ondan al. İlk yıl (eğitim verisi yok) varsayılan atr3."""
    qs = pd.date_range(A, Z, freq="QS")
    if qs[0] != A:
        qs = qs.insert(0, A)
    chosen, out = [], []
    for i, q0 in enumerate(qs):
        q1 = qs[i + 1] if i + 1 < len(qs) else Z
        if q0 - A < pd.Timedelta(days=train_days):
            pick = "atr3"
        else:
            t0 = q0 - pd.Timedelta(days=train_days)
            scores = {}
            for sname, ema in ema_by_stop.items():
                cs = [c for c in ema + don if t0 <= c["entry_time"] < q0]
                tr, eq, _ = run_pf(cs)
                m = L.metrics(tr, eq, t0, q0)
                scores[sname] = m["ret"] / max(m["mdd"], 5.0)
            pick = max(scores, key=scores.get)
        chosen.append((f"{q0:%Y-%m}", pick))
        out += [c for c in ema_by_stop[pick] if q0 <= c["entry_time"] < q1]
    return out, chosen


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
    ap.add_argument("--out", default="results/quant")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    A, V, T_, Z = (pd.Timestamp(x, tz="UTC") for x in (args.start, args.val, args.test, args.end))
    periods = [("2021-23", A, V), ("2024-25.09", V, T_), ("TEST 25.09-26.09", T_, Z)]
    years = [(str(y), pd.Timestamp(f"{y}-01-01", tz="UTC"), min(pd.Timestamp(f"{y + 1}-01-01", tz="UTC"), Z))
             for y in range(A.year, Z.year + 1) if pd.Timestamp(f"{y}-01-01", tz="UTC") < Z]
    C.COINS = args.coins.split(",")
    E.SIG_TF, E.TREND_TF, E.MAX_HOLD = "1h", "4h", pd.Timedelta(hours=48)
    t0 = time.time()
    coins = R.load(args, A, Z)
    btc = coins.get("BTC")
    ema_by_stop, don = {s: [] for s in STOPS}, []
    for n, c in coins.items():
        x = E.prep(c, btc)
        for s in STOPS:
            ema_by_stop[s] += [dict(t, strat="EMA") for t in E.signals(c, x, A, Z, "ikisi", "ema9_limit", s, "iz3")]
        feats = L.daily_feats(c, btc.tfs["1D"]["close"] if btc else None)
        don += [dict(t, strat="DONCHIAN", fee_in=C.TAKER_FEE, fee_out=C.TAKER_FEE)
                for t in L.f_donchian(c, feats, A, Z, n=20, k=2.0, regime="btc")]
        print(f"[{n}] adaylar hazır ({time.time() - t0:.0f}s)", flush=True)
    wf, chosen = walk_forward(ema_by_stop, don, A, Z)
    print("walk-forward stop seçimleri:", chosen, flush=True)

    base = ema_by_stop["atr3"] + don
    configs = [
        ("MEVCUT (sabit, kural yok)", base, {}),
        ("Özsermaye filtresi SMA30 (risk ×0.5)", base, dict(eq_mode="sma")),
        ("Özsermaye filtresi DD%15 (risk ×0.5)", base, dict(eq_mode="dd")),
        ("Coin filtresi (12 işlem ort.R<-0.3 → 30g kapalı)", base, dict(coin_filter=True)),
        ("Walk-forward EMA stop", wf + don, {}),
        ("SMA30 + coin filtresi", base, dict(eq_mode="sma", coin_filter=True)),
        ("HEPSİ (SMA30 + coin + WF stop)", wf + don, dict(eq_mode="sma", coin_filter=True)),
    ]
    rows, curves, bench = [], {}, []
    for name, cs, kw in configs:
        row = {"düzen": name}
        for lab, a, z in periods:
            tr, eq, _ = run_pf([c for c in cs if a <= c["entry_time"] < z], Adaptive(**kw))
            m = L.metrics(tr, eq, a, z)
            row.update({f"{lab} son": m["final"], f"{lab} maxDD%": m["mdd"]})
        for lab, a, z in years:
            tr, eq, _ = run_pf([c for c in cs if a <= c["entry_time"] < z], Adaptive(**kw))
            row[f"{lab} %"] = L.metrics(tr, eq, a, z)["ret"]
        tr, eq, sk = run_pf(cs, Adaptive(**kw))
        m = L.metrics(tr, eq, A, Z)
        mtm = DD.mtm_curve(tr, coins, A, Z, 100.0) if tr else pd.Series([100.0])
        row.update({"TÜM DÖNEM son": m["final"], "CAGR%": m["cagr"], "gerçek maxDD%": DD.dd(mtm),
                    "en kötü gün %": float(mtm.pct_change().min() * 100) if len(mtm) > 1 else 0.0,
                    "işlem": m["n"], "yarım riskli işlem %": np.mean([t["risk_mult"] < 1 for t in tr]) * 100 if tr else 0,
                    "coin filtresiyle atlanan": sk["coin"]})
        row["getiri/DD"] = row["CAGR%"] / max(row["gerçek maxDD%"], 1e-9)
        if not rows:                      # sabit düzenin strateji istatistikleri → bot raporundaki beklenti aralıkları
            for s_ in ("EMA", "DONCHIAN"):
                g = [t for t in tr if t["strat"] == s_]
                r = np.array([r_multiple(t["pnl"], t["qty"], t["entry"], t["stop"]) for t in g])
                p = np.array([t["pnl"] for t in g])
                roll = pd.Series(r).rolling(50).mean()
                bench.append(f"  {s_}: {len(g)} işlem | kazanma %{(p > 0).mean() * 100:.1f} | ort.R {r.mean():+.3f} | "
                             f"50 işlemlik kayan ort.R %10-%90 aralığı {roll.quantile(0.1):+.2f} … {roll.quantile(0.9):+.2f} | "
                             f"50 işlemlik kayan kazanma% %10-%90: "
                             f"{pd.Series(p > 0).rolling(50).mean().quantile(0.1) * 100:.0f} … "
                             f"{pd.Series(p > 0).rolling(50).mean().quantile(0.9) * 100:.0f}")
        curves[name] = eq
        rows.append(row)
        print(f"{name}: son {m['final']:.1f}, gerçek DD {row['gerçek maxDD%']:.1f}", flush=True)
    lb = pd.DataFrame(rows)
    lb.to_csv(os.path.join(args.out, "sonuclar.csv"), index=False, float_format="%.4g")

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, eq in curves.items():
        if eq:
            ax.step([A] + [t for t, _ in eq], [100] + [b for _, b in eq], where="post", lw=1.3, label=name)
    for _, a, _ in periods[1:]:
        ax.axvline(a, color="black", ls=":", lw=1)
    ax.axhline(100, color="gray", ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.set_title("Quant katmanı — uyarlamalı kurallar (noktalı: doğrulama / görülmemiş test başlangıcı)")
    ax.set_ylabel("Bakiye (USDT, log)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "equity.png"), dpi=120)
    plt.close(fig)

    cols_f = ["düzen", "TÜM DÖNEM son", "CAGR%", "gerçek maxDD%", "getiri/DD", "en kötü gün %", "işlem",
              "yarım riskli işlem %", "coin filtresiyle atlanan"]
    cols_p = ["düzen"] + [f"{lab} {k}" for lab, _, _ in periods for k in ("son", "maxDD%")]
    cols_y = ["düzen"] + [f"{y[0]} %" for y in years]
    txt = ["QUANT KATMANI — uyarlamalı kurallar, EMA + DONCHIAN 50/50, 18 coin",
           "VERİ: SENTETİK" if args.synthetic else "Veri: Binance USDT-M 1dk + funding (data.binance.vision)",
           "Kurallar yalnız geçmiş (kapanmış) işlemlere bakar; her dönem 100 USDT ile ayrı başlar", "",
           "Tüm dönem:", lb[cols_f].round(2).to_string(index=False), "",
           "Dönemler:", lb[cols_p].round(1).to_string(index=False), "",
           "Yıllık getiri %:", lb[cols_y].round(1).to_string(index=False), "",
           "Sabit düzenin strateji istatistikleri (canlı karşılaştırma için beklenti aralıkları):", *bench, "",
           "Walk-forward EMA stop seçimleri (çeyrek → seçilen):",
           "  " + ", ".join(f"{q}:{s[3:]}" for q, s in chosen)]
    text = "\n".join(txt)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
