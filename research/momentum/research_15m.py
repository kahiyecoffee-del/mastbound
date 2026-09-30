"""15 dakikalık gün içi momentum araştırması — long + short, günde birkaç işlem.

Aileler (sinyal kapanmış 15dk mumda, yürütme 1dk veride):
  MOMENTUM : 15dk N-mum kırılımı + göreli hacim + taker alış/satış oranı (order flow)
             + 1s EMA trendi + gün içi VWAP tarafı (+ isteğe bağlı kill zone)
  ICT-SWEEP: Asya aralığı / önceki gün yüksek-düşük likiditesinin süpürülmesi (sweep),
             ardından yapı kırılımı (MSS) ile ters yönde giriş
  ICT-FVG  : trend yönünde displacement mumu + Fair Value Gap; FVG'ye LİMİT emirle giriş
Tüm pozisyonlar en geç UTC gün sonunda kapanır (gün içi).

Maliyet modeli ("gerçekçi"): market emir = taker %0.05 + slippage %0.02;
limit giriş ve TP = maker %0.02, slippage yok; stop/iz/zaman çıkışı = taker + slippage.
Karşılaştırma için en iyi konfigler "hepsi taker" modeliyle de raporlanır.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import config as C
import research as R
import research_ls as L

MAKER_FEE = 0.0002
LIQ_LEV = 5.0
DAY = pd.Timedelta("1D")
Q = pd.Timedelta("15min")


# ------------------------------------------------------------------ işlem
def trade(coin, e, d, stop, T, limit_px=None, tp=None, trail_k=0.0, deadline=None, trail_tf="15min"):
    mm = coin.mm
    if e >= len(mm.o):
        return None
    if limit_px is None:
        entry, fee_in = mm.o[e] * (1 + d * C.SLIPPAGE), C.TAKER_FEE
    else:
        entry, fee_in = limit_px, MAKER_FEE
    if (d == 1 and stop >= entry) or (d == -1 and stop <= entry):
        return None
    liq = entry * (1 - d / LIQ_LEV)
    close_at, atr_arr = coin.htf(trail_tf)
    tpv = tp if tp is not None else (np.inf if d == 1 else -np.inf)
    dl = deadline if deadline is not None else len(mm.o) + 1
    start = e
    if limit_px is not None:
        # dolum mumu: dolumdan önce/sonra sırası bilinmez -> bu mumda sadece STOP kontrol edilir,
        # TP ancak sonraki mumdan itibaren geçerli (aksi halde ileriye bakma yanlılığı olur)
        if (d == 1 and mm.l[e] <= stop) or (d == -1 and mm.h[e] >= stop):
            k, raw, r = e, stop, 0
            start = None
        else:
            start = e + 1
            if start >= len(mm.o) or start >= dl:
                k = min(start, len(mm.o) - 1)
                raw, r, start = mm.o[k], 3, None
            elif (d == 1 and mm.o[start] <= stop) or (d == -1 and mm.o[start] >= stop):
                k, raw, r, start = start, mm.o[start], 0, None       # açılış boşluğu
    if start is not None:
        k, raw, r = L.sim(mm.o, mm.h, mm.l, mm.c, start, d, stop, tpv, trail_k, close_at,
                          atr_arr, dl, liq)
    reason = L.REASONS[r]
    if reason == "TP":
        exit_, fee_out = raw, MAKER_FEE
    else:
        exit_, fee_out = raw * (1 - d * C.SLIPPAGE), C.TAKER_FEE
    return dict(coin=coin.name, dir=d, setup_time=T, entry_time=mm.index[e], entry=entry,
                stop=stop, exit_time=mm.index[k], exit=exit_, reason=reason,
                fee_in=fee_in, fee_out=fee_out, maker_entry=limit_px is not None,
                funding_px=d * R.funding_px(coin, e, k))


def idx_at(coin, ts):
    return int(np.searchsorted(coin.mm.t, pd.Timestamp(ts).value))


# ------------------------------------------------------------------ 15dk özellikler
def feats15(coin):
    df = coin.tfs["15min"].copy()
    f = pd.DataFrame(index=df.index)
    for col in ("open", "high", "low", "close", "volume"):
        f[col] = df[col]
    f["T"] = df.index + Q
    f["atr"] = R.atr(df)
    f["relvol"] = df["volume"] / df["volume"].rolling(96).mean().shift(1)
    f["ratio"] = (df["taker_buy"] / df["volume"]).replace([np.inf, -np.inf], np.nan).fillna(0.5)
    day = df.index.floor("1D")
    tp_ = (df["high"] + df["low"] + df["close"]) / 3
    pv = (tp_ * df["volume"]).groupby(day).cumsum()
    vv = df["volume"].groupby(day).cumsum()
    f["vwap"] = pv / vv.replace(0, np.nan)
    h1 = coin.tfs["1h"]
    e20, e50 = L.ema(h1["close"], 20), L.ema(h1["close"], 50)
    tr = pd.Series(np.where((h1["close"] > e50) & (e20 > e50), 1,
                            np.where((h1["close"] < e50) & (e20 < e50), -1, 0)),
                   index=h1.index + pd.Timedelta("1h"), name="trend")
    f["trend"] = pd.merge_asof(f[["T"]], tr.to_frame(), left_on="T", right_index=True)["trend"] \
        .fillna(0).to_numpy()
    f["day"] = day
    hour_close = f["T"].dt.hour + f["T"].dt.minute / 60
    f["killzone"] = ((hour_close > 7) & (hour_close <= 16)).to_numpy()   # Londra + NY
    # Asya aralığı (00-07 UTC) ve önceki gün yüksek/düşük
    asia = df[df.index.hour < 7]
    ah = asia["high"].groupby(asia.index.floor("1D")).max()
    al = asia["low"].groupby(asia.index.floor("1D")).min()
    f["asia_high"] = f["day"].map(ah)
    f["asia_low"] = f["day"].map(al)
    d1 = coin.tfs["1D"]
    f["pdh"] = f["day"].map(d1["high"].shift(1))
    f["pdl"] = f["day"].map(d1["low"].shift(1))
    return f


def window_rows(f, a, b):
    return np.flatnonzero(((f["T"] >= a) & (f["T"] < b)).to_numpy())


# ------------------------------------------------------------------ aileler
def s_momentum(coin, f, a, b, n, k, exit_, kz, of):
    H, Lo, Cl = f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy()
    hh = f["high"].shift(1).rolling(n).max().to_numpy()
    ll = f["low"].shift(1).rolling(n).min().to_numpy()
    rv, ra, tr = f["relvol"].to_numpy(), f["ratio"].to_numpy(), f["trend"].to_numpy()
    vw, at, kzv = f["vwap"].to_numpy(), f["atr"].to_numpy(), f["killzone"].to_numpy()
    T = f["T"]
    out = []
    win = np.zeros(len(Cl), bool)
    win[window_rows(f, a, b)] = True
    base = win & (rv > 1.5) & (kzv if kz else True)
    with np.errstate(invalid="ignore"):
        lng = base & (Cl > hh) & (tr == 1) & (Cl > vw) & ((ra > 0.55) if of else True)
        sht = base & (Cl < ll) & (tr == -1) & (Cl < vw) & ((ra < 0.45) if of else True)
    for i in np.flatnonzero(lng | sht):
        d = 1 if lng[i] else -1
        Ti = T.iloc[i]
        e = idx_at(coin, Ti)
        if e >= len(coin.mm.o):
            continue
        ent = coin.mm.o[e] * (1 + d * C.SLIPPAGE)
        stop = ent - d * k * at[i]
        tp = ent + d * 2 * k * at[i] if exit_ == "tp2" else None
        t = trade(coin, e, d, stop, Ti, tp=tp, trail_k=2.0 if exit_ == "trail2" else 0.0,
                  deadline=idx_at(coin, Ti.floor("1D") + DAY))
        if t:
            out.append(t)
    return out


def s_sweep(coin, f, a, b, pool, tgt, of):
    H, Lo, Cl, Op = (f[c].to_numpy() for c in ("high", "low", "close", "open"))
    ph = f["asia_high" if pool == "asia" else "pdh"].to_numpy()
    pl = f["asia_low" if pool == "asia" else "pdl"].to_numpy()
    ra, at, kzv = f["ratio"].to_numpy(), f["atr"].to_numpy(), f["killzone"].to_numpy()
    T, day = f["T"], f["day"].to_numpy()
    used = set()
    out = []
    rows = window_rows(f, a, b)
    for i in rows:
        if not kzv[i] or np.isnan(ph[i]) or np.isnan(pl[i]):
            continue
        for d, swept in ((1, Lo[i] < pl[i] and Cl[i] > pl[i]), (-1, H[i] > ph[i] and Cl[i] < ph[i])):
            if not swept or (day[i], d) in used:
                continue
            # MSS: sonraki 4 mumda sweep mumunun yüksek/düşüğünün ötesinde kapanış
            for j in range(i + 1, min(i + 5, len(Cl))):
                if day[j] != day[i]:
                    break
                if (d == 1 and Cl[j] > H[i]) or (d == -1 and Cl[j] < Lo[i]):
                    if of and ((d == 1 and ra[j] < 0.55) or (d == -1 and ra[j] > 0.45)):
                        break
                    used.add((day[i], d))
                    Tj = T.iloc[j]
                    e = idx_at(coin, Tj)
                    if e >= len(coin.mm.o):
                        break
                    ext = Lo[i:j + 1].min() if d == 1 else H[i:j + 1].max()
                    stop = ext - d * 0.1 * at[i]
                    ent = coin.mm.o[e] * (1 + d * C.SLIPPAGE)
                    R_ = abs(ent - stop)
                    if R_ < ent * 0.001:
                        break
                    if tgt == "2R":
                        tp = ent + d * 2 * R_
                    else:                                   # karşı taraftaki likidite
                        tp = ph[i] if d == 1 else pl[i]
                        if d * (tp - ent) < R_:
                            break
                    t = trade(coin, e, d, stop, Tj, tp=tp, deadline=idx_at(coin, Tj.floor("1D") + DAY))
                    if t:
                        out.append(t)
                    break
    return out


def s_fvg(coin, f, a, b, lvl, rr, of):
    H, Lo, Cl, Op = (f[c].to_numpy() for c in ("high", "low", "close", "open"))
    ra, at, tr, kzv = (f[c].to_numpy() for c in ("ratio", "atr", "trend", "killzone"))
    T = f["T"]
    mm = coin.mm
    out = []
    for i in window_rows(f, a, b):
        if i < 3 or not kzv[i]:
            continue
        disp = Cl[i - 1] - Op[i - 1]
        if tr[i] == 1 and Lo[i] > H[i - 2] and disp > at[i]:
            d, top, bot, stop = 1, Lo[i], H[i - 2], Lo[i - 2] - 0.05 * at[i]
        elif tr[i] == -1 and H[i] < Lo[i - 2] and -disp > at[i]:
            d, top, bot, stop = -1, H[i], Lo[i - 2], H[i - 2] + 0.05 * at[i]
        else:
            continue
        if of and ((d == 1 and ra[i - 1] < 0.55) or (d == -1 and ra[i - 1] > 0.45)):
            continue
        level = top if lvl == "kenar" else (top + bot) / 2
        R_ = abs(level - stop)
        if R_ < level * 0.001:
            continue
        tp = level + d * rr * R_
        Ti = T.iloc[i]
        s = idx_at(coin, Ti)
        z = min(idx_at(coin, Ti + 8 * Q), idx_at(coin, Ti.floor("1D") + DAY) - 1)
        if z <= s:
            continue
        seg_l, seg_h = mm.l[s:z], mm.h[s:z]
        hit = np.flatnonzero(seg_l <= level) if d == 1 else np.flatnonzero(seg_h >= level)
        if not len(hit):
            continue
        k = hit[0]
        # hedef girişten önce görüldüyse iptal (ICT kuralı)
        pre = seg_h[:k] if d == 1 else seg_l[:k]
        if len(pre) and ((d == 1 and pre.max() >= tp) or (d == -1 and pre.min() <= tp)):
            continue
        e = s + k
        px = min(level, mm.o[e]) if d == 1 else max(level, mm.o[e])
        t = trade(coin, e, d, stop, Ti, limit_px=px, tp=px + d * rr * abs(px - stop),
                  deadline=idx_at(coin, Ti.floor("1D") + DAY))
        if t:
            out.append(t)
    return out


def signal_configs():
    cfgs = []
    for n in (12, 24):
        for k in (1.0, 1.5):
            for ex in ("tp2", "trail2"):
                for kz in (False, True):
                    for of in (False, True):
                        cfgs.append((f"MOMENTUM 15dk N={n} stop={k}ATR {ex}"
                                     + (" killzone" if kz else "") + (" +akış" if of else ""),
                                     s_momentum, dict(n=n, k=k, exit_=ex, kz=kz, of=of)))
    for pool in ("asia", "pdhl"):
        for tgt in ("2R", "likidite"):
            for of in (False, True):
                cfgs.append((f"ICT-SWEEP+MSS havuz={pool} hedef={tgt}" + (" +akış" if of else ""),
                             s_sweep, dict(pool=pool, tgt=tgt, of=of)))
    for lvl in ("kenar", "orta"):
        for rr in (2.0, 3.0):
            for of in (False, True):
                cfgs.append((f"ICT-FVG limit={lvl} TP={rr:g}R" + (" +akış" if of else ""),
                             s_fvg, dict(lvl=lvl, rr=rr, of=of)))
    return cfgs


# ------------------------------------------------------------------ portföy
def run_pf(cands, risk, lev, K=2, start_bal=100.0, all_taker=False, day_target=None,
           day_stop=None, stats=None):
    """day_target/day_stop: gün içi GERÇEKLEŞEN getiri (UTC gün başı bakiyesine göre) hedefe
    ya da zarar limitine ulaşınca o gün yeni işlem açılmaz (açık pozisyonlar kendi kuralıyla kapanır)."""
    order = {c: i for i, c in enumerate(C.COINS)}
    cands = sorted(cands, key=lambda c: (c["entry_time"], order.get(c["coin"], 99)))
    bal, open_, trades, eq = start_bal, [], [], []
    cur_day, day_start, halted = None, start_bal, set()

    def realize(upto):
        nonlocal bal
        open_.sort(key=lambda p: p[0])
        while open_ and (upto is None or open_[0][0] <= upto):
            t_exit, pnl, rec = open_.pop(0)
            pnl = max(pnl, -bal)
            bal += pnl
            rec["pnl"], rec["balance"] = pnl, bal
            trades.append(rec)
            eq.append((t_exit, bal))

    for c in cands:
        D = c["entry_time"].floor("1D")
        if D != cur_day:
            realize(D)
            cur_day, day_start = D, bal
        realize(c["entry_time"])
        if bal <= 0:
            break
        if day_target is not None or day_stop is not None:
            r = bal / day_start - 1 if day_start > 0 else 0
            if D in halted:
                continue
            if (day_target is not None and r >= day_target) or (day_stop is not None and r <= -day_stop):
                halted.add(D)
                if stats is not None:
                    stats["hedef" if r >= 0 else "zarar_limiti"] = stats.get(
                        "hedef" if r >= 0 else "zarar_limiti", 0) + 1
                continue
        if len(open_) >= K or any(p[2]["coin"] == c["coin"] for p in open_):
            continue
        stop_frac = abs(c["entry"] - c["stop"]) / c["entry"]
        notional = min(bal * risk / max(stop_frac, 1e-4), bal * lev / K)
        if notional < C.MIN_NOTIONAL.get(c["coin"], C.MIN_NOTIONAL_DEFAULT):
            continue
        qty = notional / c["entry"]
        fi, fo = (C.TAKER_FEE, C.TAKER_FEE) if all_taker else (c["fee_in"], c["fee_out"])
        fees = fi * notional + fo * qty * c["exit"]
        pnl = qty * c["dir"] * (c["exit"] - c["entry"]) - fees - qty * c["funding_px"]
        pnl = max(pnl, -notional / lev)
        rec = dict(c, notional=notional, fees=fees, R=pnl / (qty * abs(c["entry"] - c["stop"])))
        open_.append((c["exit_time"], pnl, rec))
    realize(None)
    return trades, eq


# ------------------------------------------------------------------ ana akış
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="2021-01-01,2022-01-01,2023-01-01,2024-01-01,2025-01-01,2025-09-24")
    ap.add_argument("--holdout-end", default="2026-09-24")
    ap.add_argument("--coins", default=",".join(C.COINS))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drift", type=float, default=0.0)
    ap.add_argument("--out", default="results/research_15m")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    b = [pd.Timestamp(x, tz="UTC") for x in args.train.split(",")]
    tr_a, tr_b, ho_b = b[0], b[-1], pd.Timestamp(args.holdout_end, tz="UTC")
    periods = [(f"{x:%Y-%m}", x, z) for x, z in zip(b[:-1], b[1:])]
    coins = R.load(args, tr_a, ho_b)
    feats = {n: feats15(c) for n, c in coins.items()}

    sizing = [(0.01, 2.0), (0.01, 5.0), (0.02, 5.0)]
    rows, keep = [], {}
    t0 = time.time()
    for si, (sname, fn, kw) in enumerate(signal_configs()):
        tr_c, ho_c = [], []
        for n, c in coins.items():
            tr_c += fn(c, feats[n], tr_a, tr_b, **kw)
            ho_c += fn(c, feats[n], tr_b, ho_b, **kw)
        # sizing'den bağımsız işlem kalitesi (1 birim, gerçekçi maliyet)
        gross_R = [c["dir"] * (c["exit"] - c["entry"]) / abs(c["entry"] - c["stop"]) for c in tr_c]
        for risk, lev in sizing:
            name = f"{sname} | risk %{risk * 100:g} | kaldıraç≤{lev:g}x"
            tr, eq = run_pf(tr_c, risk, lev)
            m = L.metrics(tr, eq, tr_a, tr_b)
            days = (tr_b - tr_a).days
            row = {"strateji": name, "sinyal": sname, "risk_%": risk * 100, "kaldirac": lev,
                   "egitim_bakiye": m["final"], "egitim_CAGR_%": m["cagr"],
                   "egitim_maxDD_%": m["mdd"], "egitim_calmar": m["calmar"], "islem": m["n"],
                   "islem_gun": m["n"] / days, "long": m["long_n"], "short": m["short_n"],
                   "isabet_%": m["win"], "PF": m["pf"],
                   "ortR_net": np.mean([t["R"] for t in tr]) if tr else np.nan,
                   "ortR_brut_aday": np.mean(gross_R) if gross_R else np.nan,
                   "aday": len(tr_c)}
            yr = []
            for lab, x, z in periods:
                pt, pe = run_pf([c for c in tr_c if x <= c["entry_time"] < z], risk, lev)
                yr.append(L.metrics(pt, pe, x, z)["ret"])
                row[f"{lab}_getiri_%"] = yr[-1]
            row["karli_yil"] = int((np.array(yr) > 0).sum())
            ht, he = run_pf(ho_c, risk, lev)
            hm = L.metrics(ht, he, tr_b, ho_b)
            row.update({"TEST_bakiye": hm["final"], "TEST_maxDD_%": hm["mdd"], "TEST_islem": hm["n"]})
            tt, te = run_pf(tr_c, risk, lev, all_taker=True)
            row["egitim_bakiye_hepsi_taker"] = L.metrics(tt, te, tr_a, tr_b)["final"]
            rows.append(row)
            keep[name] = (tr, eq, ht, he)
        print(f"{si + 1} sinyal, {time.time() - t0:.0f}s — {sname}: aday {len(tr_c)}, "
              f"ort brüt R {np.mean(gross_R) if gross_R else float('nan'):+.3f}", flush=True)

    lb = pd.DataFrame(rows).sort_values("egitim_calmar", ascending=False)
    lb.to_csv(os.path.join(args.out, "leaderboard.csv"), index=False, float_format="%.4g")
    ycols = [f"{p[0]}_getiri_%" for p in periods]
    cols = (["strateji", "egitim_bakiye", "egitim_CAGR_%", "egitim_maxDD_%", "islem_gun",
             "isabet_%", "ortR_net", "karli_yil"] + ycols +
            ["TEST_bakiye", "TEST_maxDD_%", "egitim_bakiye_hepsi_taker"])
    fmt = lambda d: d[cols].round(2).to_string(index=False)
    lb["aile"] = lb["sinyal"].str.split(" ").str[0]
    fam = lb.groupby("aile").head(1)
    grow = lb[(lb["egitim_maxDD_%"] <= 40) & (lb["islem"] >= 100)].sort_values(
        "egitim_CAGR_%", ascending=False)
    top = lb[lb["islem"] >= 100].head(5)
    if len(top):
        L.plot_curves({n: keep[n][1] for n in top["strateji"]},
                      os.path.join(args.out, "equity_train_top5.png"),
                      f"15dk gün içi — eğitim {tr_a.date()}→{tr_b.date()}: Calmar'a göre en iyi 5")
        L.plot_curves({n: keep[n][3] for n in top["strateji"]},
                      os.path.join(args.out, "equity_TEST_top5.png"),
                      f"15dk gün içi — GÖRÜLMEMİŞ TEST {tr_b.date()}→{ho_b.date()}")
        best = top.iloc[0]["strateji"]
        pd.DataFrame(keep[best][0]).to_csv(os.path.join(args.out, "best_trades_train.csv"),
                                           index=False, float_format="%.8g")
        pd.DataFrame(keep[best][2]).to_csv(os.path.join(args.out, "best_trades_TEST.csv"),
                                           index=False, float_format="%.8g")
    per_sig = (lb.groupby("sinyal")[["ortR_brut_aday", "aday"]].first()
               .sort_values("ortR_brut_aday", ascending=False))
    txt = [f"15dk GÜN İÇİ L/S | eğitim {tr_a.date()}→{tr_b.date()} | TEST {tr_b.date()}→{ho_b.date()}",
           "VERİ: SENTETİK" if args.synthetic else "Veri: Binance USDT-M 1dk + taker-buy + funding",
           f"Maliyet: market taker %{C.TAKER_FEE * 100:g}+slip %{C.SLIPPAGE * 100:g}; limit giriş/TP maker "
           f"%{MAKER_FEE * 100:g}. Maks 2 eşzamanlı pozisyon, tümü gün sonu kapanır.", "",
           "Sinyal kalitesi — adayların ortalama BRÜT R'si (maliyet öncesi, >0 = ham avantaj var):",
           per_sig.round(3).to_string(), "",
           "Calmar'a göre en iyi 15 (≥100 işlem):", fmt(lb[lb.islem >= 100].head(15)), "",
           "Büyüme: maks DD ≤ %40 içinde en yüksek CAGR (≥100 işlem):", fmt(grow.head(10)), "",
           "Her ailenin en iyisi:", fmt(fam)]
    text = "\n".join(txt)
    with open(os.path.join(args.out, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
