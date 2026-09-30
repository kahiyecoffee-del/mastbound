"""15dk KIRILIM → RETEST → DEVAM (yalnız LONG), kullanıcının kuralları. Gerçek MEXC API ücretleri.

Kurallar (15dk mum, yalnız kapanmış mumlar):
  1. EMA9 > EMA21
  2. EMA9 ve EMA21 yukarı eğimli (3 mum önceye göre yüksek)
  3. Sermaye akışı pozitif ve yükseliyor: son 8 mumun taker deltası (alış − satış) > 0 ve 4 mum öncesinden yüksek
  4. Kırılım: kapanış > önceki L mumun en yüksek seviyesi (swing high / direnç)
  5. Kırılım mumunda hacim > 1.5 × 20 mum ortalaması ve delta pozitif (alış baskın, delta/hacim > %10)
  6. Retest: sonraki W mum içinde fiyat kırılan seviyeye döner (low ≤ seviye × 1.002)
  7. Retest boyunca EMA9 > EMA21 ve kapanış ≥ EMA21; kapanış seviyenin %0.5'ten fazla altına inerse kurulum iptal
  8. Derinlik (ask tüketimi / bid'lerin yukarı taşınması): 2021–2026 için tarihsel orderbook yok → TEST EDİLEMEZ
  9. Giriş: retestten sonra 4 mum içinde, kapanış retest bölgesinin en yükseğini aşar ve delta pozitifse
     o mumun kapanışında piyasa emri. Stop: retest bölgesinin en düşüğü − 0.1×ATR(14).
Çıkış (1dk veriyle, aynı dakikada stop ve hedef → önce stop): TP 1.5R / 2R / 3R ya da 2×ATR iz süren stop; en fazla 24 saat.
Maliyet: giriş piyasa %0.08 + kayma %0.02; TP limit %0.06; stop/süre piyasa %0.08 + kayma %0.02.

  python research_15m_retest.py              # Binance 1dk arşivi → results/retest15/
  python research_15m_retest.py --synthetic  # kod testi
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import research as R
import research_combo as CB

TAKER, MAKER, SLIP = 0.0008, 0.0006, 0.0002
PERIODS = [("2021-23", "2021-01-01", "2024-01-01"), ("2024-25.09", "2024-01-01", "2025-09-24"),
           ("SON 1 YIL", "2025-09-24", "2026-09-24")]
BASE = dict(L=20, W=8, E=4, flow=True, volume=True, retest=True, slope=True, exit=("tp", 2.0))


def variants():
    V = []
    for L in (20, 48):
        for W in (4, 8):
            for ex in (("tp", 1.5), ("tp", 2.0), ("tp", 3.0), ("trail", 2.0)):
                V.append((f"kırılım {L} mum, retest ≤{W} mum, çıkış {ex[0]} {ex[1]:g}", dict(BASE, L=L, W=W, exit=ex)))
    V += [("KONTROL: 3 yok (sermaye akışı şartı kaldırıldı)", dict(BASE, flow=False)),
          ("KONTROL: 5 yok (hacim/delta şartı kaldırıldı)", dict(BASE, volume=False)),
          ("KONTROL: 2 yok (eğim şartı kaldırıldı)", dict(BASE, slope=False)),
          ("KONTROL: retest yok (kırılım mumunun kapanışında gir)", dict(BASE, retest=False))]
    return V


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def features(df):
    c, h, lo, v = df["close"], df["high"], df["low"], df["volume"]
    delta = 2 * df["taker_buy"] - v
    f = pd.DataFrame(index=df.index)
    f["o"], f["h"], f["l"], f["c"], f["v"], f["delta"] = df["open"], h, lo, c, v, delta
    f["e9"], f["e21"] = ema(c, 9), ema(c, 21)
    f["slope"] = (f["e9"] > f["e9"].shift(3)) & (f["e21"] > f["e21"].shift(3))
    cvd8 = delta.rolling(8).sum()
    f["flow"] = (cvd8 > 0) & (cvd8 > cvd8.shift(4))
    f["volok"] = (v > 1.5 * v.rolling(20).mean()) & (delta > 0.1 * v)
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    f["atr"] = tr.rolling(14).mean()
    for L in (20, 48):
        f[f"hh{L}"] = h.shift(1).rolling(L).max()
    return f


def setups(f, p):
    """Kurallara uyan girişler: [(giriş mumu idx, giriş fiyatı, stop)]"""
    n = len(f)
    c, h, lo, e9, e21 = (f[k].to_numpy(float) for k in ("c", "h", "l", "e9", "e21"))
    delta, atr = f["delta"].to_numpy(float), f["atr"].to_numpy(float)
    lvl = f[f"hh{p['L']}"].to_numpy(float)
    ok = (e9 > e21) & (c > lvl)
    if p["slope"]:
        ok &= f["slope"].to_numpy(bool)
    if p["flow"]:
        ok &= f["flow"].to_numpy(bool)
    if p["volume"]:
        ok &= f["volok"].to_numpy(bool)
    out, busy_until = [], -1
    for b in np.flatnonzero(ok):
        if b <= busy_until or not np.isfinite(atr[b]):
            continue
        L_ = lvl[b]
        if not p["retest"]:
            stop = min(lo[b], L_) - 0.1 * atr[b]
            out.append((b, c[b], stop))
            busy_until = b
            continue
        r = None
        for j in range(b + 1, min(b + 1 + p["W"], n)):
            if c[j] < L_ * 0.995 or e9[j] <= e21[j] or c[j] < e21[j]:
                break                                           # yapı bozuldu → kurulum iptal
            if lo[j] <= L_ * 1.002:
                r = j
                break
        if r is None:
            continue
        seg_hi, seg_lo = h[r], lo[r]
        for e in range(r + 1, min(r + 1 + p["E"], n)):
            if c[e] < L_ * 0.995 or e9[e] <= e21[e] or c[e] < e21[e]:
                break
            if c[e] > seg_hi and delta[e] > 0:
                stop = seg_lo - 0.1 * atr[e]
                if 0.0015 <= (c[e] - stop) / c[e] <= 0.03:
                    out.append((e, c[e], stop))
                    busy_until = e
                break
            seg_hi, seg_lo = max(seg_hi, h[e]), min(seg_lo, lo[e])
    return out


def exit_trade(mm, t_entry_ns, entry, stop, atr, ex):
    """1dk veriyle çıkış. Dönen: (net R, çıkış zamanı ns)."""
    i0 = np.searchsorted(mm.t, t_entry_ns)
    i1 = min(i0 + 24 * 60, len(mm.t))
    if i0 >= i1:
        return None
    hi, lo_ = mm.h[i0:i1], mm.l[i0:i1]
    fill = entry * (1 + SLIP)
    risk = fill - stop
    if risk <= 0:
        return None
    if ex[0] == "tp":
        tp = fill + ex[1] * risk
        s_hit = np.flatnonzero(lo_ <= stop)
        t_hit = np.flatnonzero(hi >= tp)
        si = s_hit[0] if len(s_hit) else 10 ** 9
        ti = t_hit[0] if len(t_hit) else 10 ** 9
        if si <= ti and si < 10 ** 9:
            px, fee_out, k = stop * (1 - SLIP), TAKER, si
        elif ti < 10 ** 9:
            px, fee_out, k = tp, MAKER, ti
        else:
            px, fee_out, k = mm.c[i1 - 1] * (1 - SLIP), TAKER, i1 - 1 - i0
    else:
        trail = np.maximum(stop, np.maximum.accumulate(hi) - ex[1] * atr)
        trail = np.r_[stop, trail[:-1]]                       # stop bir önceki dakikanın zirvesine göre
        s_hit = np.flatnonzero(lo_ <= trail)
        if len(s_hit):
            k = s_hit[0]
            px, fee_out = trail[k] * (1 - SLIP), TAKER
        else:
            k = i1 - 1 - i0
            px, fee_out = mm.c[i1 - 1] * (1 - SLIP), TAKER
    pnl = px - fill - TAKER * fill - fee_out * px
    return pnl / risk, mm.t[i0 + k]


def stats(df):
    if df.empty:
        return dict(n=0, win=np.nan, avgR=np.nan, pf=np.nan, sumR=0.0, eq=100.0, dd=0.0)
    r = df.sort_values("t_exit")["R"].to_numpy()
    eq = 100 * np.cumprod(1 + 0.01 * r)
    pk = np.maximum.accumulate(np.r_[100, eq])
    gl = -r[r < 0].sum()
    return dict(n=len(r), win=(r > 0).mean() * 100, avgR=r.mean(), pf=r[r > 0].sum() / gl if gl > 0 else np.inf,
                sumR=r.sum(), eq=eq[-1], dd=((pk - np.r_[100, eq]) / pk).max() * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", default=CB.COINS18)
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2026-09-24")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drift", type=float, default=0.0)
    ap.add_argument("--out", default="results/retest15")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    A, Z = pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC")
    V = variants()
    trades = {name: [] for name, _ in V}
    t0 = time.time()
    for name in args.coins.split(","):
        a = argparse.Namespace(**vars(args))
        a.coins = name
        c = R.load(a, A, Z).get(name)
        if c is None:
            continue
        df = c.tfs["15min"]
        df = df[(df.index >= A) & (df.index < Z)]
        f = features(df)
        close_ns = (f.index + pd.Timedelta("15min")).as_unit("ns").asi8
        atr = f["atr"].to_numpy(float)
        for vname, p in V:
            for e, px, stop in setups(f, p):
                res = exit_trade(c.mm, close_ns[e], px, stop, atr[e], p["exit"])
                if res is not None:
                    trades[vname].append(dict(coin=name, t_entry=f.index[e] + pd.Timedelta("15min"),
                                              t_exit=pd.Timestamp(res[1], tz="UTC"), R=res[0]))
        print(f"[{name}] tamam ({time.time() - t0:.0f}s)", flush=True)
        del c
    rows = []
    for vname, _ in V:
        df = pd.DataFrame(trades[vname], columns=["coin", "t_entry", "t_exit", "R"])
        s = stats(df)
        row = {"varyant": vname, "işlem": s["n"], "isabet%": s["win"], "ort. net R": s["avgR"], "PF": s["pf"],
               "%1 riskle 100→": s["eq"], "maxDD%": s["dd"]}
        for lab, a, z in PERIODS:
            sp = stats(df[(df["t_entry"] >= pd.Timestamp(a, tz="UTC")) & (df["t_entry"] < pd.Timestamp(z, tz="UTC"))])
            row[f"{lab} n"], row[f"{lab} ort.R"] = sp["n"], sp["avgR"]
        rows.append(row)
        df.to_csv(os.path.join(args.out, f"islemler_{len(rows):02d}.csv"), index=False)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.out, "sonuclar.csv"), index=False, float_format="%.4g")
    txt = ["15dk KIRILIM → RETEST → DEVAM (LONG) — kullanıcının kuralları, "
           f"{len(args.coins.split(','))} coin, {args.start} → {args.end}" + ("  [SENTETİK]" if args.synthetic else ""),
           "Ücret: giriş piyasa %0.08+kayma %0.02, TP limit %0.06, stop/süre piyasa %0.08+kayma %0.02. "
           "Madde 8 (derinlik) tarihsel veri olmadığı için test edilemedi.",
           "ort. net R = ücretler düşülmüş işlem başı sonuç (1R = stop mesafesi). '%1 riskle 100→' eşzamanlılık sınırı "
           "olmadan, işlem başı %1 riskle bileşik.", "",
           out.round(3).to_string(index=False), "",
           "OKUMA: ort. net R her dönemde > 0 (özellikle SON 1 YIL) ve komşu ayarlarda tutarlıysa strateji adaydır.",
           "KONTROL satırları ilgili şartın katkısını gösterir: şart kaldırılınca sonuç kötüleşmiyorsa o şart gereksizdir."]
    text = "\n".join(txt)
    with open(os.path.join(args.out, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
