"""EMA işlemleri için ORDER-FLOW TABANLI ÇIKIŞ: CVD / delta z / hacim z / fiyat & hacim verimliliği.

Mevcut çıkış: 3xATR (1s) iz süren stop, en fazla 48 saat.
Önerilen çıkış (kullanıcı fikri):
  * işlem trend boyunca açık kalır (koruma: mevcut 3xATR iz süren stop devam eder)
  * işlem en az `act` R kâra ulaştıktan sonra, kapanmış 1s mumda TREND TÜKENME / DÖNÜŞ sinyali gelirse
    stop anlık fiyatın `kt` × ATR altına çekilir (short'ta üstüne) ve fiyat lehe gittikçe
    o noktadan itibaren `kt` × ATR mesafeyle takip edilir; fiyat bunu kırarsa çıkılır.
Tükenme sinyalleri (long için; short tam tersi) — hepsi 1s mum, taker alış hacminden:
  delta = 2 × taker_alış − hacim,  CVD20 = son 20 mumun delta toplamı
  vol_z / delta_z = 50 mumluk z-skor,  ER10 = |Δfiyat 10| / Σ|Δfiyat| (fiyat verimliliği, Kaufman)
  VE = |kapanış − açılış| / hacim (hacim verimliliği) → ve_z
  "climax" : vol_z > 2 ve delta_z < −1.5                   (yüksek hacimli agresif satış)
  "absorb" : vol_z > 2 ve ve_z < −1                        (büyük hacim, küçük ilerleme = emilim)
  "div"    : kapanış 20 mumun zirvesinde ama CVD20 < 0 ve delta_z < −1   (fiyat↑ CVD↓ uyumsuzluk)
  "er"     : ER10 < 0.25 iken 3 mum önce ER10 > 0.5       (trend verimliliği çöktü)
  "any"    : herhangi biri
Varyantlar: sinyal × kt (0.5 / 1.0 ATR) × act (1R / 2R) × maks süre (48s / 7 gün).
Ayrıca KAYAN TP: TP = SL + gap×ATR (gap 4/5/6/9); stop ne kadar yukarı çekilirse TP de o kadar
yukarı taşınır — kâr esas olarak stopla alınır, TP yalnız en yüksek fiyatın (gap−3) ATR üstüne ani sıçramada dolar.
Girişler birebir aynı (4s MACD + 1s EMA9/21, 3xATR stop) — yalnız çıkış değişir. DONCHIAN aynen.
Seçim yalnız 2021-23'e göre; doğrulama ve son 1 yıl görülmemiş test ayrıca raporlanır.
"""
import argparse
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
import research_combo as CB
import research_daily_detail as DD
import research_ema921 as E
import research_ls as L
import research_quant as RQ

SIGNALS = ("climax", "absorb", "div", "er", "any")
REASONS = ["SL", "BE", "TP", "TIME", "END", "LIQ", "TRAIL", "QSTOP"]


def z(s, n=50):
    m, sd = s.rolling(n).mean(), s.rolling(n).std()
    return (s - m) / sd.replace(0, np.nan)


def flow_signals(coin):
    """1s mum başına (coin.tfs['1h'] satır sırasıyla) long-çıkış ve short-çıkış sinyal dizileri."""
    h = coin.tfs["1h"]
    c, o, v = h["close"], h["open"], h["volume"]
    delta = 2 * h["taker_buy"] - v
    vol_z, delta_z = z(v), z(delta)
    cvd20 = delta.rolling(20).sum()
    er = (c - c.shift(10)).abs() / c.diff().abs().rolling(10).sum()
    ve_z = z((c - o).abs() / v.replace(0, np.nan))
    hi20, lo20 = c.rolling(20).max(), c.rolling(20).min()
    out = {}
    for d in (1, -1):
        dz = -delta_z if d == 1 else delta_z               # long için satış baskısı pozitif
        s = {"climax": (vol_z > 2) & (dz > 1.5),
             "absorb": (vol_z > 2) & (ve_z < -1),
             "div": ((c >= hi20) & (cvd20 < 0) & (dz > 1)) if d == 1 else ((c <= lo20) & (cvd20 > 0) & (dz > 1)),
             "er": (er < 0.25) & (er.shift(3) > 0.5)}
        s["any"] = s["climax"] | s["absorb"] | s["div"] | s["er"]
        out[d] = {k: v_.fillna(False).to_numpy(np.bool_) for k, v_ in s.items()}
    return out


@numba.njit(cache=True)
def sim_q(o, h, l, c, e, d, stop, close_at, atr_arr, sig, trail_k, kt, act_px, entry, deadline, liq, tp_gap):
    """3xATR iz süren stop + (kâr ≥ act_px iken) tükenme sinyalinde stopu fiyatın kt×ATR yakınına çek.
    tp_gap > 0: TP = stop + tp_gap (long) — stop yukarı çekildikçe TP de aynı miktar yukarı taşınır."""
    cur, kind, best = stop, 0, o[e]
    use_tp = tp_gap > 0
    tp = cur + d * tp_gap
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
            if use_tp and h[k] >= tp and k > e:
                return k, max(tp, o[k]), 2
            if h[k] > best:
                best = h[k]
            if tight and h[k] > best2:
                best2 = h[k]
        else:
            eff = cur if cur < liq else liq
            if h[k] >= eff:
                raw = eff if k == e else max(o[k], eff)
                return k, raw, kind if cur < liq else 5
            if use_tp and l[k] <= tp and k > e:
                return k, min(tp, o[k]), 2
            if l[k] < best:
                best = l[k]
            if tight and l[k] < best2:
                best2 = l[k]
        j = close_at[k]
        if j >= 0:
            a = atr_arr[j]
            if not tight and sig[j] and d * (best - entry) >= act_px:
                tight, best2 = True, c[k]
            new = best - d * trail_k * a
            if (d == 1 and new > cur) or (d == -1 and new < cur):
                cur, kind = new, 6
            if tight:
                new2 = best2 - d * kt * a
                if (d == 1 and new2 > cur) or (d == -1 and new2 < cur):
                    cur, kind = new2, 7
            tp = cur + d * tp_gap                 # stop ne kadar çekildiyse TP de o kadar
    return n - 1, c[n - 1], 4


def make_recorder(flows, variants, store):
    """Q.trade yerine geçer: E.signals'ın girişlerini aynen kullanır, her varyantın çıkışını hesaplar."""
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
        dls = {h_: int(np.searchsorted(mm.t, (T + pd.Timedelta(hours=h_)).value)) for h_ in (48, 168)}
        none = np.zeros(len(atr_arr), np.bool_)
        for vn, (sname, kt, act, hold, gap) in variants.items():
            # tüm varyantlar (mevcut dahil) aynı simülatörle: dolum mumundan itibaren stop/iz kontrolü
            k, raw, r = sim_q(mm.o, mm.h, mm.l, mm.c, start, d, stop, close_at, atr_arr,
                              flows[d][sname] if sname else none, 3.0, kt, act * R_, entry, dls[hold], liq,
                              gap * R_ / 3.0 if gap else 0.0)
            tp_hit = REASONS[r] == "TP"               # TP limit emir: maker, kayma yok
            store[vn].append(dict(base, exit_time=mm.index[k], exit=raw if tp_hit else raw * (1 - d * C.SLIPPAGE),
                                  reason=REASONS[r], fee_out=Q.MAKER_FEE if tp_hit else C.TAKER_FEE,
                                  funding_px=d * R.funding_px(coin, start, k)))
        return base
    return trade


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
    ap.add_argument("--out", default="results/cvd_exit")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    A, V, T_, Z = (pd.Timestamp(x, tz="UTC") for x in (args.start, args.val, args.test, args.end))
    periods = [("2021-23", A, V), ("2024-25.09", V, T_), ("TEST 25.09-26.09", T_, Z)]
    years = [(str(y), pd.Timestamp(f"{y}-01-01", tz="UTC"), min(pd.Timestamp(f"{y + 1}-01-01", tz="UTC"), Z))
             for y in range(A.year, Z.year + 1) if pd.Timestamp(f"{y}-01-01", tz="UTC") < Z]
    C.COINS = args.coins.split(",")
    E.SIG_TF, E.TREND_TF, E.MAX_HOLD = "1h", "4h", pd.Timedelta(hours=168)   # girişleri etkilemez
    hl = lambda h_: "48s" if h_ == 48 else "7g"
    variants = {"MEVCUT: 3xATR iz, maks 48s": (None, 0, 0, 48, 0),
                "3xATR iz, maks 7 gün (sinyal yok)": (None, 0, 0, 168, 0)}
    for s in SIGNALS:
        for kt in (0.5, 1.0):
            for act in (1.0, 2.0):
                for hold in (48, 168):
                    variants[f"{s} | stop fiyat−{kt:g}ATR | ≥{act:g}R | maks {hl(hold)}"] = (s, kt, act, hold, 0)
    # SL ile birlikte yukarı çekilen TP: TP − SL = gap × ATR sabit. İz stop en yüksek fiyatın 3 ATR altında
    # olduğundan TP ≈ en yüksek fiyat + (gap − 3) ATR; başlangıçta TP = giriş + (gap − 3) ATR.
    for s in (None,) + SIGNALS:
        for gap in (4.0, 5.0, 6.0, 9.0):
            for hold in (48, 168):
                lab = "iz 3xATR" if s is None else f"{s} | stop fiyat−0.5ATR | ≥1R"
                variants[f"{lab} + TP=SL+{gap:g}ATR (SL ile yukarı) | maks {hl(hold)}"] = \
                    (s, 0.5, 1.0, hold, gap)
    t0 = time.time()
    coins = R.load(args, A, Z)
    btc = coins.get("BTC")
    store, don = {vn: [] for vn in variants}, []
    orig = Q.trade
    for n, c in coins.items():
        flows = flow_signals(c)
        Q.trade = make_recorder(flows, variants, store)
        try:
            E.signals(c, E.prep(c, btc), A, Z, "ikisi", "ema9_limit", "atr3", "iz3")
        finally:
            Q.trade = orig
        feats = L.daily_feats(c, btc.tfs["1D"]["close"] if btc else None)
        don += [dict(t, strat="DONCHIAN", fee_in=C.TAKER_FEE, fee_out=C.TAKER_FEE)
                for t in L.f_donchian(c, feats, A, Z, n=20, k=2.0, regime="btc")]
        rate = {k: float(np.mean(v_["any"])) for k, v_ in flows.items()}
        print(f"[{n}] hazır ({time.time() - t0:.0f}s) | 'any' sinyal oranı long {rate[1]:.3f} short {rate[-1]:.3f}",
              flush=True)
    for vn in store:
        store[vn] = [dict(t, strat="EMA") for t in store[vn]]

    rows = []
    for vn, ema in store.items():
        cs = ema + don
        row = {"çıkış": vn}
        for lab, a, z_ in periods:
            tr, eq, _ = RQ.run_pf([c for c in cs if a <= c["entry_time"] < z_])
            m = L.metrics(tr, eq, a, z_)
            row.update({f"{lab} son": m["final"], f"{lab} maxDD%": m["mdd"]})
        for lab, a, z_ in years:
            tr, eq, _ = RQ.run_pf([c for c in cs if a <= c["entry_time"] < z_])
            row[f"{lab} %"] = L.metrics(tr, eq, a, z_)["ret"]
        tr, eq, _ = RQ.run_pf(cs)
        m = L.metrics(tr, eq, A, Z)
        et = [t for t in tr if t["strat"] == "EMA"]
        r = np.array([(t["exit"] - t["entry"]) * t["dir"] / abs(t["entry"] - t["stop"]) for t in et])
        hold_h = np.array([(t["exit_time"] - t["entry_time"]).total_seconds() / 3600 for t in et])
        row.update({"TÜM son": m["final"], "CAGR%": m["cagr"], "maxDD%": m["mdd"],
                    "EMA ort.R": r.mean() if len(r) else np.nan, "EMA kazanma%": (r > 0).mean() * 100 if len(r) else np.nan,
                    "EMA ort. süre (s)": hold_h.mean() if len(hold_h) else np.nan,
                    "QSTOP çıkış %": np.mean([t["reason"] == "QSTOP" for t in et]) * 100 if et else 0,
                    "TP çıkış %": np.mean([t["reason"] == "TP" for t in et]) * 100 if et else 0})
        rows.append(row)
        print(f"{vn}: 2021-23 {row['2021-23 son']:.1f} | tüm {m['final']:.1f}", flush=True)
    lb = pd.DataFrame(rows)
    base_is = lb.loc[0, "2021-23 son"]
    lb = pd.concat([lb.iloc[:2], lb.iloc[2:].sort_values("2021-23 son", ascending=False)], ignore_index=True)
    lb.to_csv(os.path.join(args.out, "sonuclar.csv"), index=False, float_format="%.4g")

    # gerçek (açık pozisyon dahil) DD: mevcut + 2021-23'e göre en iyi 5 + ÜÇ DÖNEMDE DE mevcudu geçenler
    b0 = lb.iloc[0]
    beats = lb[(lb["2021-23 son"] > b0["2021-23 son"]) & (lb["2024-25.09 son"] > b0["2024-25.09 son"]) &
               (lb["TEST 25.09-26.09 son"] > b0["TEST 25.09-26.09 son"])]["çıkış"].tolist()
    top = list(dict.fromkeys(list(lb["çıkış"].iloc[:2]) + list(lb["çıkış"].iloc[2:7]) + beats))
    real, curves = {}, {}
    for vn in top:
        tr, eq, _ = RQ.run_pf(store[vn] + don)
        real[vn] = DD.dd(DD.mtm_curve(tr, coins, A, Z, 100.0))
        curves[vn] = eq
    lb["gerçek maxDD%"] = lb["çıkış"].map(real)

    fig, ax = plt.subplots(figsize=(12, 6))
    for vn, eq in curves.items():
        if eq:
            ax.step([A] + [t for t, _ in eq], [100] + [b for _, b in eq], where="post", lw=1.3, label=vn)
    for _, a, _ in periods[1:]:
        ax.axvline(a, color="black", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_title("EMA çıkışı: order-flow (CVD/delta/hacim z, verimlilik) tükenme stopu — 50/50 düzen")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "equity.png"), dpi=120)
    plt.close(fig)

    cols_f = ["çıkış", "2021-23 son", "2024-25.09 son", "TEST 25.09-26.09 son", "TÜM son", "CAGR%", "maxDD%",
              "gerçek maxDD%", "EMA ort.R", "EMA kazanma%", "EMA ort. süre (s)", "QSTOP çıkış %", "TP çıkış %"]
    cols_y = ["çıkış"] + [f"{y[0]} %" for y in years]
    txt = ["EMA ÇIKIŞI — ORDER-FLOW TÜKENME STOPU (CVD, delta z, hacim z, fiyat/hacim verimliliği)",
           "VERİ: SENTETİK" if args.synthetic else "Veri: Binance USDT-M 1dk + taker alış hacmi + funding",
           "Girişler aynı; DONCHIAN aynı; 50/50 düzen, 100 USDT her dönem ayrı. Sıralama yalnız 2021-23'e göre.",
           f"Mevcut çıkışın 2021-23 sonucu: {base_is:.1f}", "",
           lb[cols_f].round(2).to_string(index=False), "",
           "ÜÇ DÖNEMDE DE (2021-23, 2024-25.09, son 1 yıl) MEVCUT ÇIKIŞI GEÇENLER:",
           (lb[lb["çıkış"].isin(beats)][cols_f + [c for c in cols_y if c != "çıkış"]].round(2).to_string(index=False)
            if beats else "  (yok)"), "",
           "Yıllık getiri % (ilk 12 satır):", lb[cols_y].head(12).round(1).to_string(index=False)]
    text = "\n".join(txt)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
