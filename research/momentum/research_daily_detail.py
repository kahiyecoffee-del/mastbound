"""Günlük L/S trend stratejileri — ayrıntılı rapor.

Stratejiler (research_ls.py'den, görülmemiş testi geçenler):
  EMA-KESİŞİM 1G 20/50, 3xATR iz süren stop
  DONCHIAN L/S 1G N=20, 2xATR iz süren stop, BTC EMA200 rejim filtresi
Değişkenler: işlem başı risk (%1/%2/%3), koruma kuralları, hesap büyüklüğü (100/1k/10k USDT).
Koruma kuralları:
  gunluk_haftalik : gün içi gerçekleşen -%4 → o gün, hafta içi -%8 → o hafta yeni işlem yok
  +dd_yarim       : ek olarak bakiye zirveden %10+ düşükken işlem başı risk yarıya iner
Drawdown hem kapanan işlemlere göre hem de açık pozisyonların günlük kapanışla
değerlemesine (mark-to-market) göre raporlanır.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
import research as R
import research_ls as L

STRATS = [
    ("EMA 9/21 (2ATR iz)", L.f_ema_cross, dict(fast=9, slow=21, k=2.0, flow=False)),
    ("EMA 9/21 (3ATR iz)", L.f_ema_cross, dict(fast=9, slow=21, k=3.0, flow=False)),
    ("EMA 20/50 (3ATR iz)", L.f_ema_cross, dict(fast=20, slow=50, k=3.0, flow=False)),
    ("DONCHIAN L/S 20 (2ATR iz, BTC rejim)", L.f_donchian, dict(n=20, k=2.0, regime="btc")),
]
RULES = ["limitsiz", "gunluk_haftalik", "gunluk_haftalik+dd_yarim"]


def run_pf(cands, risk, K=3, start_bal=100.0, rule="limitsiz", lev=L.LEV):
    order = {c: i for i, c in enumerate(C.COINS)}
    cands = sorted(cands, key=lambda c: (c["entry_time"], order.get(c["coin"], 99)))
    bal, peak, open_, trades, eq = start_bal, start_bal, [], [], []
    day, day_start, week, week_start = None, start_bal, None, start_bal
    skipped = {"limit": 0, "min_notional": 0, "dolu": 0}

    def realize(upto):
        nonlocal bal, peak
        open_.sort(key=lambda p: p[0])
        while open_ and (upto is None or open_[0][0] <= upto):
            t_exit, pnl, rec = open_.pop(0)
            pnl = max(pnl, -bal)
            bal += pnl
            peak = max(peak, bal)
            rec["pnl"], rec["balance"] = pnl, bal
            trades.append(rec)
            eq.append((t_exit, bal))

    for c in cands:
        D = c["entry_time"].floor("1D")
        W = D - pd.Timedelta(days=D.dayofweek)
        if D != day:
            realize(D)
            day, day_start = D, bal
        if W != week:
            realize(W)
            week, week_start = W, bal
        realize(c["entry_time"])
        if bal <= 0:
            break
        if rule != "limitsiz" and (bal / day_start - 1 <= -0.04 or bal / week_start - 1 <= -0.08):
            skipped["limit"] += 1
            continue
        if len(open_) >= K or any(p[2]["coin"] == c["coin"] for p in open_):
            skipped["dolu"] += 1
            continue
        r = risk
        if rule.endswith("dd_yarim") and bal < peak * 0.9:
            r = risk / 2
        stop_frac = abs(c["entry"] - c["stop"]) / c["entry"]
        notional = min(bal * r / max(stop_frac, 1e-4), bal * lev / K)
        if notional < C.MIN_NOTIONAL.get(c["coin"], C.MIN_NOTIONAL_DEFAULT):
            skipped["min_notional"] += 1
            continue
        qty = notional / c["entry"]
        fees = C.TAKER_FEE * (notional + qty * c["exit"])
        pnl = qty * c["dir"] * (c["exit"] - c["entry"]) - fees - qty * c["funding_px"]
        pnl = max(pnl, -notional / lev)
        rec = dict(c, notional=notional, qty=qty, fees=fees, balance_before=bal,
                   R=pnl / (qty * abs(c["entry"] - c["stop"])))
        open_.append((c["exit_time"], pnl, rec))
    realize(None)
    return trades, eq, skipped


def mtm_curve(trades, coins, a, z, start_bal):
    """Günlük mark-to-market bakiye: gerçekleşen + açık pozisyonların gün kapanışı değerlemesi."""
    days = pd.date_range(a, z, freq="1D", inclusive="left")
    closes = {n: c.tfs["1D"]["close"] for n, c in coins.items()}
    real = pd.Series(0.0, index=days)
    unreal = pd.Series(0.0, index=days)
    for t in trades:
        close_day = t["exit_time"].floor("1D")
        real[real.index > close_day] += t["pnl"]          # ertesi gün başından itibaren
        cl = closes[t["coin"]]
        held = days[(days >= t["entry_time"].floor("1D")) & (days <= close_day)]
        if len(held):
            px = cl.reindex(held).ffill().to_numpy()
            u = t["qty"] * t["dir"] * (px - t["entry"])
            unreal.loc[held] += u
    eqs = start_bal + real + unreal
    return eqs


def dd(series):
    s = np.asarray(series, float)
    pk = np.maximum.accumulate(s)
    return float(((pk - s) / pk).max() * 100) if len(s) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--test", default="2025-09-24")
    ap.add_argument("--end", default="2026-09-24")
    ap.add_argument("--coins", default=",".join(C.COINS))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drift", type=float, default=0.0)
    ap.add_argument("--out", default="results/daily_detail")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    A, T, E = (pd.Timestamp(x, tz="UTC") for x in (args.start, args.test, args.end))
    years = [(str(y), pd.Timestamp(f"{y}-01-01", tz="UTC"), min(pd.Timestamp(f"{y + 1}-01-01", tz="UTC"), T))
             for y in range(A.year, T.year + 1) if pd.Timestamp(f"{y}-01-01", tz="UTC") < T]
    coins = R.load(args, A, E)
    btc = coins["BTC"].tfs["1D"]["close"] if "BTC" in coins else None
    feats = {n: L.daily_feats(c, btc) for n, c in coins.items()}

    rows, keep = [], {}
    for sname, fn, kw in STRATS:
        cand = {"egitim": [], "TEST": []}
        for n, c in coins.items():
            cand["egitim"] += fn(c, feats[n], A, T, **kw)
            cand["TEST"] += fn(c, feats[n], T, E, **kw)
        for risk in (0.01, 0.02, 0.03):
            for rule in RULES:
                for bal0 in (100.0, 1000.0, 10000.0):
                    row = {"strateji": sname, "risk_%": risk * 100, "koruma": rule, "hesap": bal0}
                    for per, a, z in (("egitim", A, T), ("TEST", T, E)):
                        tr, eq, sk = run_pf(cand[per], risk, start_bal=bal0, rule=rule)
                        m = L.metrics(tr, eq, a, z, start_bal=bal0)
                        mtm = mtm_curve(tr, coins, a, z, bal0) if tr else pd.Series([bal0])
                        row.update({f"{per}_son": m["final"] / bal0 * 100,
                                    f"{per}_CAGR_%": m["cagr"], f"{per}_DD_kapanan_%": m["mdd"],
                                    f"{per}_DD_gercek_%": dd(mtm), f"{per}_islem": m["n"],
                                    f"{per}_isabet_%": m["win"], f"{per}_min_notional_atlanan": sk["min_notional"],
                                    f"{per}_limit_atlanan": sk["limit"]})
                        keep[(sname, risk, rule, bal0, per)] = (tr, eq, mtm)
                    for lab, a, z in years:
                        tr, eq, _ = run_pf([c for c in cand["egitim"] if a <= c["entry_time"] < z],
                                           risk, start_bal=bal0, rule=rule)
                        row[f"{lab}_%"] = L.metrics(tr, eq, a, z, start_bal=bal0)["ret"]
                    rows.append(row)
        print(f"{sname} tamam", flush=True)

    lb = pd.DataFrame(rows)
    lb.to_csv(os.path.join(args.out, "tum_sonuclar.csv"), index=False, float_format="%.4g")
    ycols = [f"{y[0]}_%" for y in years]
    main_cols = ["strateji", "risk_%", "koruma", "egitim_son", "egitim_CAGR_%", "egitim_DD_kapanan_%",
                 "egitim_DD_gercek_%", "egitim_islem"] + ycols + ["TEST_son", "TEST_DD_gercek_%", "TEST_islem"]
    k1000 = lb[lb.hesap == 1000.0]

    # grafikler: 1000 USDT hesap, her strateji için risk seviyeleri (koruma=gunluk_haftalik+dd_yarim)
    for per, a, z in (("egitim", A, T), ("TEST", T, E)):
        fig, axes = plt.subplots(1, len(STRATS), figsize=(6 * len(STRATS), 5), squeeze=False)
        for ax, (sname, _, _) in zip(axes[0], STRATS):
            for risk in (0.01, 0.02, 0.03):
                for rule, ls in (("limitsiz", ":"), ("gunluk_haftalik+dd_yarim", "-")):
                    _, _, mtm = keep[(sname, risk, rule, 1000.0, per)]
                    ax.plot(mtm.index, mtm / 10, ls=ls, lw=1.3,
                            label=f"risk %{risk * 100:g} {'korumalı' if ls == '-' else 'limitsiz'}")
            ax.axhline(100, color="gray", lw=0.8, ls="--")
            ax.set_title(sname, fontsize=10)
            ax.set_ylabel("Bakiye (başlangıç=100, günlük değerleme)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.suptitle(f"{'Eğitim 2021→2025.09' if per == 'egitim' else 'GÖRÜLMEMİŞ TEST 2025.09→2026.09'}")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"equity_{per}.png"), dpi=120)
        plt.close(fig)

    # aylık getiri tablosu — önerilen konfig (EMA, %2, korumalı, 1000 USDT), eğitim+test
    rec = ("EMA 9/21 (3ATR iz)", 0.02, "limitsiz", 1000.0)
    m1 = pd.concat([keep[rec + ("egitim",)][2], keep[rec + ("TEST",)][2] *
                    (keep[rec + ("egitim",)][2].iloc[-1] / 1000.0)])
    mon = m1.resample("ME").last().pct_change() * 100
    mon.iloc[0] = (m1.resample("ME").last().iloc[0] / 1000 - 1) * 100
    tab = pd.DataFrame({"yil": mon.index.year, "ay": mon.index.month, "r": mon.to_numpy()}) \
        .pivot(index="yil", columns="ay", values="r")
    tab.to_csv(os.path.join(args.out, "aylik_getiri_onerilen.csv"), float_format="%.2f")
    fig, ax = plt.subplots(figsize=(11, 3.8))
    im = ax.imshow(tab.to_numpy(), cmap="RdYlGn", vmin=-15, vmax=15, aspect="auto")
    ax.set_xticks(range(12), ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"])
    ax.set_yticks(range(len(tab)), tab.index)
    for (i, j), v in np.ndenumerate(tab.to_numpy()):
        if np.isfinite(v):
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7)
    ax.set_title(f"Aylık getiri % — {rec[0]}, risk %2, {rec[2]} (günlük değerleme)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "aylik_getiri_onerilen.png"), dpi=120)
    plt.close(fig)
    pd.DataFrame(keep[rec + ("TEST",)][0]).to_csv(os.path.join(args.out, "islemler_onerilen_TEST.csv"),
                                                  index=False, float_format="%.8g")
    pd.DataFrame(keep[rec + ("egitim",)][0]).to_csv(os.path.join(args.out, "islemler_onerilen_egitim.csv"),
                                                    index=False, float_format="%.8g")

    size = lb[(lb["risk_%"] == 2) & (lb.koruma == "gunluk_haftalik+dd_yarim")][
        ["strateji", "hesap", "egitim_son", "egitim_islem", "egitim_min_notional_atlanan", "TEST_son",
         "TEST_islem", "TEST_min_notional_atlanan"]]
    txt = [f"GÜNLÜK L/S TREND — AYRINTILI RAPOR | eğitim {A.date()}→{T.date()} | TEST {T.date()}→{E.date()}",
           "VERİ: SENTETİK" if args.synthetic else "Veri: Binance USDT-M 1dk + funding (data.binance.vision)",
           f"Kaldıraç ≤{L.LEV:g}x, maks 3 eşzamanlı pozisyon, ücret %{C.TAKER_FEE * 100:g} + slippage "
           f"%{C.SLIPPAGE * 100:g}. 'son' = başlangıç 100'e göre bitiş. DD_gercek = açık pozisyonlar "
           "günlük kapanışla değerlenmiş.", "",
           "1) 1000 USDT hesap — tüm risk / koruma kombinasyonları:",
           k1000[main_cols].round(1).to_string(index=False), "",
           "2) Hesap büyüklüğü etkisi (risk %2, korumalı): Binance min. pozisyon sınırı küçük hesapta işlem kaçırtır",
           size.round(1).to_string(index=False), "",
           f"3) Aylık getiri tablosu ({rec[0]}, risk %2, {rec[2]}, 1000 USDT) → aylik_getiri_onerilen.png/csv"]
    text = "\n".join(txt)
    with open(os.path.join(args.out, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
