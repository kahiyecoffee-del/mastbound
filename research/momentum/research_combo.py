"""İki strateji tek bakiyede: EMA (4s MACD + 1s EMA9/21, 3xATR stop/iz) + günlük DONCHIAN L/S.

Portföy kuralları (tek bakiye, bileşik):
  * işlem başı %1 risk (her iki strateji), kaldıraç ≤2x, pozisyon başı notional ≤ bakiye*2/K_toplam
  * strateji başına maks pozisyon (EMA k_ema, Donchian k_don), aynı coinde tek pozisyon (hangi strateji olursa)
  * EMA için günde maks 5 işlem ve 3 zarardan sonra dur (Donchian günde en fazla birkaç sinyal üretir)
Karşılaştırma: yalnız EMA, yalnız Donchian, birlikte (3+3, 3+2). Dönemler: 2021-23 / 2024-25.09 / 2025.09-26.09.
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
import research_ema921 as E
import research_daily_detail as DD
import research_ls as L

COINS18 = "BTC,ETH,SOL,BNB,XRP,DOGE,AVAX,LINK,ADA,DOT,LTC,TRX,ATOM,NEAR,UNI,FIL,APT,OP"


def run_pf(cands, risk=0.01, k_ema=3, k_don=3, lev=2.0, start_bal=100.0, don_alloc=None, don_lev=1.0):
    """don_alloc verilirse: Donchian tek pozisyon, notional = bakiye*don_alloc*don_lev ("%80'i kullan");
    Donchian açıkken EMA boyutlandırması bakiyenin kalan (1-don_alloc) kısmı üzerinden yapılır."""
    if don_alloc:
        k_don = 1
    order = {c: i for i, c in enumerate(C.COINS)}
    cands = sorted(cands, key=lambda c: (c["entry_time"], order.get(c["coin"], 99)))
    k_tot = k_ema + k_don
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
        if cap <= 0 or sum(p[2]["strat"] == s for p in open_) >= cap:
            continue
        if any(p[2]["coin"] == c["coin"] for p in open_):
            continue
        D = c["entry_time"].floor("1D")
        if s == "EMA" and (per_day.get(D, 0) >= 5 or loss_day.get(D, 0) >= 3):
            continue
        stop_frac = abs(c["entry"] - c["stop"]) / c["entry"]
        if don_alloc and s == "DONCHIAN":
            notional = bal * don_alloc * don_lev
        elif don_alloc:
            base = bal * (1 - don_alloc) if any(p[2]["strat"] == "DONCHIAN" for p in open_) else bal
            notional = min(base * risk / max(stop_frac, 1e-4), base * lev / k_ema)
        else:
            notional = min(bal * risk / max(stop_frac, 1e-4), bal * lev / k_tot)
        if notional < C.MIN_NOTIONAL.get(c["coin"], C.MIN_NOTIONAL_DEFAULT):
            continue
        qty = notional / c["entry"]
        fees = c.get("fee_in", C.TAKER_FEE) * notional + c.get("fee_out", C.TAKER_FEE) * qty * c["exit"]
        pnl = qty * c["dir"] * (c["exit"] - c["entry"]) - fees - qty * c["funding_px"]
        pnl = max(pnl, -notional / (don_lev if (don_alloc and s == "DONCHIAN") else lev))
        if s == "EMA":
            per_day[D] = per_day.get(D, 0) + 1
        open_.append((c["exit_time"], pnl, dict(c, notional=notional, qty=qty)))
    realize(None)
    return trades, eq


def monthly(eq, a, z, start_bal=100.0):
    if not eq:
        return pd.Series(dtype=float)
    s = pd.Series([b for _, b in eq], index=pd.DatetimeIndex([t for t, _ in eq]))
    s = s[~s.index.duplicated(keep="last")]
    idx = pd.date_range(a, z, freq="ME")
    m = s.reindex(s.index.union(idx)).ffill().reindex(idx).fillna(start_bal)
    return m.pct_change().fillna(m.iloc[0] / start_bal - 1) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", default=COINS18)
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--val", default="2024-01-01")
    ap.add_argument("--test", default="2025-09-24")
    ap.add_argument("--end", default="2026-09-24")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drift", type=float, default=0.0)
    ap.add_argument("--out", default="results/combo_ema_donchian")
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
    cands = []
    for n, c in coins.items():
        x = E.prep(c, btc)
        for t in E.signals(c, x, A, Z, "ikisi", "ema9_limit", "atr3", "iz3"):
            cands.append(dict(t, strat="EMA"))
        feats = L.daily_feats(c, btc.tfs["1D"]["close"] if btc else None)
        for t in L.f_donchian(c, feats, A, Z, n=20, k=2.0, regime="btc"):
            cands.append(dict(t, strat="DONCHIAN", fee_in=C.TAKER_FEE, fee_out=C.TAKER_FEE))
        print(f"[{n}] adaylar hazır ({time.time() - t0:.0f}s)", flush=True)
    ema = [c for c in cands if c["strat"] == "EMA"]
    don = [c for c in cands if c["strat"] == "DONCHIAN"]
    print(f"EMA aday {len(ema)}, DONCHIAN aday {len(don)}", flush=True)

    configs = [("Yalnız EMA (maks 3)", ema, 3, 0, {}), ("Yalnız DONCHIAN (maks 3)", don, 0, 3, {}),
               ("BİRLİKTE EMA 3 + DONCHIAN 3", cands, 3, 3, {}), ("BİRLİKTE EMA 3 + DONCHIAN 2", cands, 3, 2, {}),
               ("DONCHIAN %80 bakiye 1x + EMA kalan", cands, 3, 1, dict(don_alloc=0.8, don_lev=1.0)),
               ("DONCHIAN %80 bakiye 2x + EMA kalan", cands, 3, 1, dict(don_alloc=0.8, don_lev=2.0)),
               ("DONCHIAN %50 bakiye 1x + EMA kalan", cands, 3, 1, dict(don_alloc=0.5, don_lev=1.0))]
    rows, curves, mret = [], {}, {}
    for name, cs, ke, kd, kw in configs:
        row = {"portföy": name}
        for lab, a, z in periods:
            tr, eq = run_pf([c for c in cs if a <= c["entry_time"] < z], k_ema=ke, k_don=kd, **kw)
            m = L.metrics(tr, eq, a, z)
            row.update({f"{lab} son": m["final"], f"{lab} maxDD%": m["mdd"], f"{lab} işlem": m["n"],
                        f"{lab} EMA/DON işlem": f'{sum(t["strat"] == "EMA" for t in tr)}/'
                                               f'{sum(t["strat"] == "DONCHIAN" for t in tr)}'})
        for lab, a, z in years:
            tr, eq = run_pf([c for c in cs if a <= c["entry_time"] < z], k_ema=ke, k_don=kd, **kw)
            row[f"{lab} %"] = L.metrics(tr, eq, a, z)["ret"]
        tr, eq = run_pf(cs, k_ema=ke, k_don=kd, **kw)
        m = L.metrics(tr, eq, A, Z)
        mtm = DD.mtm_curve(tr, coins, A, Z, 100.0) if tr else pd.Series([100.0])
        row.update({"TÜM DÖNEM son": m["final"], "TÜM DÖNEM maxDD%": m["mdd"],
                    "gerçek maxDD% (açık poz. dahil)": DD.dd(mtm), "CAGR%": m["cagr"],
                    "en kötü gün %": float(mtm.pct_change().min() * 100) if len(mtm) > 1 else 0.0})
        curves[name] = eq
        mret[name] = monthly(eq, A, Z)
        rows.append(row)
    lb = pd.DataFrame(rows)
    lb.to_csv(os.path.join(args.out, "sonuclar.csv"), index=False, float_format="%.4g")
    corr = np.corrcoef(mret["Yalnız EMA (maks 3)"], mret["Yalnız DONCHIAN (maks 3)"])[0, 1]

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, eq in curves.items():
        if eq:
            ax.step([A] + [t for t, _ in eq], [100] + [b for _, b in eq], where="post", lw=1.4, label=name)
    for _, a, _ in periods[1:]:
        ax.axvline(a, color="black", ls=":", lw=1)
    ax.axhline(100, color="gray", ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.set_title("EMA + DONCHIAN tek bakiye — tüm dönem (noktalı çizgiler: doğrulama ve görülmemiş test başlangıcı)")
    ax.set_ylabel("Bakiye (USDT, log)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "equity.png"), dpi=120)
    plt.close(fig)

    cols_p = ["portföy"] + [f"{lab} {k}" for lab, _, _ in periods for k in ("son", "maxDD%", "EMA/DON işlem")]
    cols_y = ["portföy"] + [f"{y[0]} %" for y in years] + ["TÜM DÖNEM son", "TÜM DÖNEM maxDD%",
                                                       "gerçek maxDD% (açık poz. dahil)", "en kötü gün %", "CAGR%"]
    txt = ["EMA (4s MACD + 1s EMA9/21, 3xATR) + DONCHIAN L/S (1G N=20, 2xATR iz, BTC EMA200 rejimi) — tek bakiye",
           "VERİ: SENTETİK" if args.synthetic else "Veri: Binance USDT-M 1dk + funding (data.binance.vision)",
           f"Coinler: {args.coins} | risk %1/işlem, kaldıraç ≤2x, aynı coinde tek pozisyon, 100 USDT her dönem ayrı başlar",
           "", "Dönemler:", lb[cols_p].round(1).to_string(index=False), "",
           "Yıllık getiri ve tüm dönem (tek hesap):", lb[cols_y].round(1).to_string(index=False), "",
           f"EMA ile DONCHIAN aylık getiri korelasyonu: {corr:+.2f} (0'a yakın/negatif = iyi çeşitlendirme)"]
    text = "\n".join(txt)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
