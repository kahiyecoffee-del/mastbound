"""Long + short, günlük işlem stratejileri araştırması — öncelik SERMAYE KORUMA.

Farklar (research.py'ye göre):
  * Hem long hem short. Her sinyal kapanmış mumda üretilir, bir sonraki 1dk açılışta girilir.
  * Risk bazlı pozisyon: işlem başına bakiyenin %r'si riske edilir
      notional = bakiye * r / stop_mesafesi_%   (üst sınır: bakiye * kaldıraç / K)
  * Aynı anda en fazla K pozisyon (farklı coinlerde), tek bakiye, bileşik.
  * Order-flow: 1dk mumlardaki agresif alıcı (taker buy) hacmi oranı.
  * Seçim 2021-01 → 2025-09-24 verisinde; 2025-09-24 → 2026-09-24 hiç görülmemiş test.
  * Sıralama: önce maksimum drawdown sınırı, sonra Calmar (yıllık getiri / maks. DD).

  python research_ls.py --out results/research_ls
  python research_ls.py --synthetic --train 2021-01-01,2021-04-01,2021-07-01 --holdout-end 2021-09-01
"""
import argparse
import os
import time
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd

import config as C
import research as R

MIN = 60_000_000_000
DAY = 1440 * MIN
REASONS = ["SL", "BE", "TP", "TIME", "END", "LIQ", "TRAIL"]
LEV = 2.0
MAX_DD_OK = 30.0            # sermaye koruma eşiği (%)


# ------------------------------------------------------------------ yön-duyarlı çıkış
@numba.njit(cache=True)
def sim(o, h, l, c, e, d, stop, tp, trail_k, close_at, atr_arr, deadline, liq):
    cur = stop
    kind = 0
    best = o[e]
    n = len(o)
    for k in range(e, n):
        if k >= deadline:
            return k, o[k], 3
        if d == 1:
            eff = cur if cur > liq else liq
            if l[k] <= eff:
                raw = eff if k == e else min(o[k], eff)
                return k, raw, kind if cur > liq else 5
            if h[k] >= tp:
                return k, tp, 2
            if h[k] > best:
                best = h[k]
        else:
            eff = cur if cur < liq else liq
            if h[k] >= eff:
                raw = eff if k == e else max(o[k], eff)
                return k, raw, kind if cur < liq else 5
            if l[k] <= tp:
                return k, tp, 2
            if l[k] < best:
                best = l[k]
        j = close_at[k]
        if trail_k > 0 and j >= 0:
            new = best - d * trail_k * atr_arr[j]
            if (d == 1 and new > cur) or (d == -1 and new < cur):
                cur = new
                kind = 6
    return n - 1, c[n - 1], 4


def trade(coin, e, d, stop, T, tp=None, trail_tf=None, trail_k=0.0, deadline=None):
    mm = coin.mm
    if e >= len(mm.o):
        return None
    entry = mm.o[e] * (1 + d * C.SLIPPAGE)
    if (d == 1 and stop >= entry) or (d == -1 and stop <= entry):
        return None
    liq = entry * (1 - d / LEV) / (1 - d * C.MAINT_MARGIN_RATE)
    close_at, atr_arr = coin.htf(trail_tf or "1D")
    tpv = tp if tp is not None else (np.inf if d == 1 else -np.inf)
    k, raw, r = sim(mm.o, mm.h, mm.l, mm.c, e, d, stop, tpv, trail_k if trail_tf else 0.0,
                    close_at, atr_arr, deadline if deadline is not None else len(mm.o) + 1, liq)
    exit_ = raw * (1 - d * C.SLIPPAGE)
    return dict(coin=coin.name, dir=d, setup_time=T, entry_time=mm.index[e], entry=entry,
                stop=stop, exit_time=mm.index[k], exit=exit_, reason=REASONS[r],
                funding_px=d * R.funding_px(coin, e, k))


# ------------------------------------------------------------------ göstergeler
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def daily_feats(coin, btc_close=None):
    df = coin.tfs["1D"]
    f = pd.DataFrame(index=df.index)
    f["close"] = df["close"]
    f["high"], f["low"] = df["high"], df["low"]
    for n in (9, 10, 20, 21, 30, 50, 200):
        f[f"ema{n}"] = ema(df["close"], n)
    f["atr"] = R.atr(df)
    ratio = (df["taker_buy"] / df["volume"]).replace([np.inf, -np.inf], np.nan).fillna(0.5)
    f["flow"] = ratio.ewm(span=5, adjust=False).mean()
    if btc_close is not None:
        b = btc_close.reindex(df.index).ffill()
        f["btc_up"] = b > ema(b, 200)
    f["T"] = df.index + pd.Timedelta("1D")          # kapanış anı = sinyal anı
    return f


def entry_idx(coin, T):
    return int(np.searchsorted(coin.mm.t, T.value))


# ------------------------------------------------------------------ strateji aileleri
def f_ema_cross(coin, feats, a, b, fast, slow, k, flow):
    f = feats[(feats["T"] >= a) & (feats["T"] < b)]
    x = (f[f"ema{fast}"] > f[f"ema{slow}"]).astype(int)
    prev = (feats[f"ema{fast}"] > feats[f"ema{slow}"]).astype(int).shift(1).reindex(f.index)
    out = []
    for i, row in f[(x != prev) & prev.notna()].iterrows():
        d = 1 if row[f"ema{fast}"] > row[f"ema{slow}"] else -1
        if flow and ((d == 1 and row.flow < 0.5) or (d == -1 and row.flow > 0.5)):
            continue
        e = entry_idx(coin, row["T"])
        if e >= len(coin.mm.o):
            continue
        t = trade(coin, e, d, coin.mm.o[e] - d * k * row.atr, row["T"], trail_tf="1D", trail_k=k)
        if t:
            out.append(t)
    return out


def f_donchian(coin, feats, a, b, n, k, regime):
    hi = feats["high"].shift(1).rolling(n).max()
    lo = feats["low"].shift(1).rolling(n).min()
    sel = (feats["T"] >= a) & (feats["T"] < b)
    out = []
    for i in feats.index[sel]:
        row = feats.loc[i]
        up = row.btc_up if regime == "btc" else row.close > row.ema200
        d = 1 if (row.close > hi[i] and up) else -1 if (row.close < lo[i] and not up) else 0
        if not d:
            continue
        e = entry_idx(coin, row["T"])
        if e >= len(coin.mm.o):
            continue
        t = trade(coin, e, d, coin.mm.o[e] - d * k * row.atr, row["T"], trail_tf="1D", trail_k=k)
        if t:
            out.append(t)
    return out


def f_daily_bias(coin, feats, a, b, k, flow):
    """Gün içi: gün açılışında trend yönünde gir, gün sonunda (ya da stopta) kapat."""
    f = feats[(feats["T"] >= a) & (feats["T"] < b)]
    out = []
    for i, row in f.iterrows():
        d = 1 if row.close > row.ema20 > row.ema50 else -1 if row.close < row.ema20 < row.ema50 else 0
        if not d or (flow and ((d == 1 and row.flow < 0.5) or (d == -1 and row.flow > 0.5))):
            continue
        e = entry_idx(coin, row["T"])
        if e >= len(coin.mm.o):
            continue
        dl = entry_idx(coin, row["T"] + pd.Timedelta("1D"))
        t = trade(coin, e, d, coin.mm.o[e] - d * k * row.atr, row["T"], deadline=dl)
        if t:
            out.append(t)
    return out


def f_orb(coin, feats, a, b, m, filt, tp2):
    """Açılış aralığı kırılımı (UTC gün): ilk m dakikanın aralığı, ilk kırılımda gir, gün sonu çık."""
    mm = coin.mm
    days = pd.date_range(a, b, freq="1D", inclusive="left")
    prev = feats.set_index("T")
    out = []
    for D in days:
        i0 = int(np.searchsorted(mm.t, D.value))
        i1 = int(np.searchsorted(mm.t, D.value + m * MIN))
        i2 = int(np.searchsorted(mm.t, D.value + DAY))
        if i1 - i0 < m * 0.9 or i2 <= i1 + 1:
            continue
        rh, rl = mm.h[i0:i1].max(), mm.l[i0:i1].min()
        allow_l = allow_s = True
        if filt:
            if D not in prev.index:
                continue
            r = prev.loc[D]                          # dünün kapanışı (D anında kapanan gün)
            allow_l, allow_s = r.close > r.ema50, r.close < r.ema50
        cl = mm.c[i1:i2 - 1]
        up = np.flatnonzero(cl > rh)
        dn = np.flatnonzero(cl < rl)
        fu = up[0] if len(up) else 10 ** 9
        fd = dn[0] if len(dn) else 10 ** 9
        if fu == fd == 10 ** 9:
            continue
        d = 1 if fu < fd else -1
        if (d == 1 and not allow_l) or (d == -1 and not allow_s):
            continue
        e = i1 + min(fu, fd) + 1
        stop = rl if d == 1 else rh
        ent = mm.o[e] * (1 + d * C.SLIPPAGE)
        tp = ent + d * 2 * abs(ent - stop) if tp2 else None
        t = trade(coin, e, d, stop, D + m * pd.Timedelta(minutes=1), tp=tp, deadline=i2)
        if t:
            out.append(t)
    return out


def f_ema4h(coin, feats, a, b, k, flow):
    df = coin.tfs["4h"]
    e20, e50 = ema(df["close"], 20), ema(df["close"], 50)
    T = df.index + pd.Timedelta("4h")
    cross = (e20 > e50).astype(int).diff().fillna(0)
    daily = feats.set_index("T")[["close", "ema200", "flow"]]
    reg = pd.merge_asof(pd.DataFrame({"T": T}), daily, left_on="T", right_index=True)
    a4 = R.atr(df).to_numpy()
    out = []
    for j in np.flatnonzero(((T >= a) & (T < b)) & (cross.to_numpy() != 0)):
        d = 1 if cross.iloc[j] > 0 else -1
        r = reg.iloc[j]
        if pd.isna(r.ema200) or (d == 1 and r.close < r.ema200) or (d == -1 and r.close > r.ema200):
            continue
        if flow and ((d == 1 and r.flow < 0.5) or (d == -1 and r.flow > 0.5)):
            continue
        e = entry_idx(coin, T[j])
        if e >= len(coin.mm.o):
            continue
        t = trade(coin, e, d, coin.mm.o[e] - d * k * a4[j], T[j], trail_tf="4h", trail_k=k)
        if t:
            out.append(t)
    return out


def signal_configs():
    cfgs = []
    for fast, slow in [(10, 30), (20, 50), (50, 200)]:
        for k in (2.0, 3.0):
            for fl in (False, True):
                cfgs.append((f"EMA-KESİŞİM 1G {fast}/{slow} stop={k}ATR iz" + (" +akış" if fl else ""),
                             f_ema_cross, dict(fast=fast, slow=slow, k=k, flow=fl)))
    for n in (20, 55):
        for k in (2.0, 3.0):
            for rg in ("coin", "btc"):
                cfgs.append((f"DONCHIAN L/S 1G N={n} stop={k}ATR iz rejim={rg}EMA200",
                             f_donchian, dict(n=n, k=k, regime=rg)))
    for k in (1.0, 1.5):
        for fl in (False, True):
            cfgs.append((f"GÜNLÜK-YÖN (gün içi) EMA20/50 stop={k}ATR" + (" +akış" if fl else ""),
                         f_daily_bias, dict(k=k, flow=fl)))
    for m in (30, 60):
        for filt in (False, True):
            for tp2 in (False, True):
                cfgs.append((f"ORB (gün içi) {m}dk" + (" +EMA50" if filt else "")
                             + (" TP2R" if tp2 else " gün-sonu"), f_orb,
                             dict(m=m, filt=filt, tp2=tp2)))
    for k in (2.0, 3.0):
        for fl in (False, True):
            cfgs.append((f"EMA 4s 20/50 +EMA200(1G) stop={k}ATR iz" + (" +akış" if fl else ""),
                         f_ema4h, dict(k=k, flow=fl)))
    return cfgs


# ------------------------------------------------------------------ portföy
def run_pf(cands, risk, K, start_bal=100.0, fixed_notional=None):
    order = {c: i for i, c in enumerate(C.COINS)}
    cands = sorted(cands, key=lambda c: (c["entry_time"], order.get(c["coin"], 99)))
    bal, open_, trades, eq = start_bal, [], [], []

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
        realize(c["entry_time"])
        if bal <= 0:
            break
        if len(open_) >= K or any(p[2]["coin"] == c["coin"] for p in open_):
            continue
        stop_frac = abs(c["entry"] - c["stop"]) / c["entry"]
        if fixed_notional is not None:
            notional = bal * fixed_notional
        else:
            notional = min(bal * risk / max(stop_frac, 1e-4), bal * LEV / K)
        if notional < C.MIN_NOTIONAL.get(c["coin"], C.MIN_NOTIONAL_DEFAULT):
            continue
        qty = notional / c["entry"]
        fees = C.TAKER_FEE * (notional + qty * c["exit"])
        pnl = qty * c["dir"] * (c["exit"] - c["entry"]) - fees - qty * c["funding_px"]
        pnl = max(pnl, -notional / LEV)                   # isolated marj kaybı üst sınırı
        rec = dict(c, notional=notional, fees=fees, R=pnl / (qty * abs(c["entry"] - c["stop"])),
                   balance_before=bal)
        open_.append((c["exit_time"], pnl, rec))
    realize(None)
    return trades, eq


def metrics(trades, eq, a, b, start_bal=100.0):
    if not trades:
        return dict(final=start_bal, ret=0.0, cagr=0.0, mdd=0.0, calmar=0.0, n=0, win=np.nan,
                    pf=np.nan, long_n=0, short_n=0)
    bals = np.r_[start_bal, [x[1] for x in eq]]
    peak = np.maximum.accumulate(bals)
    mdd = ((peak - bals) / peak).max() * 100
    yrs = (b - a).days / 365.25
    final = bals[-1]
    cagr = ((final / start_bal) ** (1 / yrs) - 1) * 100 if final > 0 else -100.0
    pnl = np.array([t["pnl"] for t in trades])
    gl = -pnl[pnl < 0].sum()
    return dict(final=final, ret=(final / start_bal - 1) * 100, cagr=cagr, mdd=mdd,
                calmar=cagr / mdd if mdd > 0 else np.inf, n=len(trades),
                win=(pnl > 0).mean() * 100, pf=pnl[pnl > 0].sum() / gl if gl > 0 else np.inf,
                long_n=sum(t["dir"] == 1 for t in trades), short_n=sum(t["dir"] == -1 for t in trades))


def probe_bookdepth():
    """data.binance.vision'da tarihsel order book derinliği (bookDepth) ne zamandan beri var?"""
    res = {}
    for d in ["2021-06-01", "2022-06-01", "2023-01-15", "2023-06-01", "2024-01-15"]:
        url = (f"{R.data.VISION}/daily/bookDepth/BTCUSDT/BTCUSDT-bookDepth-{d}.zip")
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=20) as r:
                res[d] = r.status
        except Exception as e:
            res[d] = getattr(e, "code", type(e).__name__)
    return res


# ------------------------------------------------------------------ grafikler
def plot_curves(curves, path, title, start_bal=100.0):
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, eq in curves.items():
        if not eq:
            continue
        x = [eq[0][0]] + [t for t, _ in eq]
        y = [start_bal] + [v for _, v in eq]
        ax.step(x, y, where="post", lw=1.4, label=name[:80])
    ax.axhline(start_bal, color="gray", ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel("Bakiye (USDT, log)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_trades(coins, trades, path, title, start):
    names = [c for c in coins if any(t["coin"] == c for t in trades)]
    if not names:
        return
    fig, axes = plt.subplots(len(names), 1, figsize=(13, 2.6 * len(names)), squeeze=False)
    for ax, name in zip(axes[:, 0], names):
        s = coins[name].tfs["1D"]["close"]
        s = s[s.index >= start]
        ax.plot(s.index, s, color="gray", lw=0.8)
        for t in trades:
            if t["coin"] != name:
                continue
            col = "#2e9e5b" if t["pnl"] > 0 else "#d64545"
            ax.plot([t["entry_time"], t["exit_time"]], [t["entry"], t["exit"]], color=col, lw=2)
            ax.scatter([t["entry_time"]], [t["entry"]], s=14, zorder=3, color=col,
                       marker="^" if t["dir"] == 1 else "v")
        ax.set_ylabel(name)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0, 0].set_title(title + "  (▲ long, ▼ short; yeşil kâr, kırmızı zarar)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ ana akış
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="2021-01-01,2022-01-01,2023-01-01,2024-01-01,2025-01-01,2025-09-24")
    ap.add_argument("--holdout-end", default="2026-09-24")
    ap.add_argument("--coins", default=",".join(C.COINS))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drift", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/research_ls")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    bounds = [pd.Timestamp(x, tz="UTC") for x in args.train.split(",")]
    tr_a, tr_b = bounds[0], bounds[-1]
    ho_a, ho_b = tr_b, pd.Timestamp(args.holdout_end, tz="UTC")
    periods = [(f"{a:%Y-%m}", a, z) for a, z in zip(bounds[:-1], bounds[1:])]

    book = {} if args.synthetic else probe_bookdepth()
    coins = R.load(args, tr_a, ho_b)
    btc = coins["BTC"].tfs["1D"]["close"] if "BTC" in coins else None
    feats = {n: daily_feats(c, btc) for n, c in coins.items()}
    R.plot_market(coins, os.path.join(args.out, "market.png"), tr_a, [ho_a])

    sigs = signal_configs()
    if args.limit:
        sigs = sigs[::max(1, len(sigs) // args.limit)]
    sizing = [(0.005, 1), (0.01, 1), (0.02, 1), (0.005, 3), (0.01, 3), (0.02, 3)]
    rows, keep = [], {}
    t0 = time.time()
    for si, (sname, fn, kw) in enumerate(sigs):
        cands = {"train": [], "hold": []}
        for n, c in coins.items():
            cands["train"] += fn(c, feats[n], tr_a, tr_b, **kw)
            cands["hold"] += fn(c, feats[n], ho_a, ho_b, **kw)
        for risk, K in sizing:
            name = f"{sname} | risk %{risk * 100:g} | maks {K} poz."
            tr, eq = run_pf(cands["train"], risk, K)
            m = metrics(tr, eq, tr_a, tr_b)
            row = {"strateji": name, "sinyal": sname, "risk_%": risk * 100, "K": K,
                   "egitim_bakiye": m["final"], "egitim_CAGR_%": m["cagr"],
                   "egitim_maxDD_%": m["mdd"], "egitim_calmar": m["calmar"], "islem": m["n"],
                   "long": m["long_n"], "short": m["short_n"], "isabet_%": m["win"], "PF": m["pf"]}
            yr = []
            for lab, a, z in periods:
                pt, pe = run_pf([c for c in cands["train"] if a <= c["entry_time"] < z], risk, K)
                pm = metrics(pt, pe, a, z)
                row[f"{lab}_getiri_%"] = pm["ret"]
                yr.append(pm["ret"])
            row["karli_yil"] = int((np.array(yr) > 0).sum())
            row["en_kotu_yil_%"] = float(min(yr))
            ht, he = run_pf(cands["hold"], risk, K)
            hm = metrics(ht, he, ho_a, ho_b)
            row.update({"TEST_bakiye": hm["final"], "TEST_maxDD_%": hm["mdd"], "TEST_islem": hm["n"]})
            rows.append(row)
            keep[name] = (tr, eq, ht, he)
        print(f"{si + 1}/{len(sigs)} sinyal, {time.time() - t0:.0f}s — {sname}", flush=True)

    # benchmark: BTC al-tut 1x
    bt = []
    if "BTC" in coins:
        cb = coins["BTC"]
        for a, z in [(tr_a, tr_b), (ho_a, ho_b)]:
            t = trade(cb, entry_idx(cb, a), 1, 0.0, a, deadline=entry_idx(cb, z))
            bt.append(t)
        btr, beq = run_pf([bt[0]], 0, 1, fixed_notional=1.0)
        bht, bhe = run_pf([bt[1]], 0, 1, fixed_notional=1.0)
        bm, bhm = metrics(btr, beq, tr_a, tr_b), metrics(bht, bhe, ho_a, ho_b)
        rows.append({"strateji": "BENCHMARK BTC al-tut 1x", "sinyal": "benchmark",
                     "egitim_bakiye": bm["final"], "egitim_CAGR_%": bm["cagr"],
                     "egitim_maxDD_%": np.nan, "egitim_calmar": np.nan, "islem": 1,
                     "TEST_bakiye": bhm["final"], "TEST_maxDD_%": np.nan, "TEST_islem": 1})

    lb = pd.DataFrame(rows)
    ok = lb[(lb["egitim_maxDD_%"] <= MAX_DD_OK) & (lb["islem"] >= 30) &
            (lb["egitim_CAGR_%"] > 0)].sort_values("egitim_calmar", ascending=False)
    lb.sort_values("egitim_calmar", ascending=False).to_csv(
        os.path.join(args.out, "leaderboard.csv"), index=False, float_format="%.4g")

    ycols = [f"{p[0]}_getiri_%" for p in periods]
    cols = (["strateji", "egitim_bakiye", "egitim_CAGR_%", "egitim_maxDD_%", "egitim_calmar",
             "islem", "long", "short", "karli_yil"] + ycols + ["TEST_bakiye", "TEST_maxDD_%", "TEST_islem"])
    fmt = lambda d: d[cols].round(2).to_string(index=False)
    # her sinyal ailesinin en iyisi
    fam = lambda s: s.split(" ")[0]
    lb["aile"] = lb["sinyal"].map(fam)
    best_per_family = (lb[lb["islem"] >= 30].sort_values("egitim_calmar", ascending=False)
                       .groupby("aile").head(1))

    top = ok.head(5)
    if len(top):
        plot_curves({n: keep[n][1] for n in top["strateji"]},
                    os.path.join(args.out, "equity_train_top5.png"),
                    f"Eğitim {tr_a.date()}→{tr_b.date()}: DD≤%{MAX_DD_OK:g} içinde en iyi 5 (Calmar)")
        plot_curves({n: keep[n][3] for n in top["strateji"]},
                    os.path.join(args.out, "equity_TEST_top5.png"),
                    f"GÖRÜLMEMİŞ TEST {ho_a.date()}→{ho_b.date()}: aynı 5 strateji")
        best = top.iloc[0]["strateji"]
        plot_trades(coins, keep[best][0], os.path.join(args.out, "best_trades_train.png"),
                    f"En iyi: {best} — eğitim", tr_a)
        plot_trades(coins, keep[best][2], os.path.join(args.out, "best_trades_TEST.png"),
                    f"En iyi: {best} — görülmemiş test", ho_a)
        for tag, idx in [("train", 0), ("TEST", 2)]:
            pd.DataFrame(keep[best][idx]).to_csv(os.path.join(args.out, f"best_trades_{tag}.csv"),
                                                 index=False, float_format="%.8g")
    plot_curves({n: keep[n][1] for n in best_per_family["strateji"] if n in keep},
                os.path.join(args.out, "equity_family_best.png"),
                "Her ailenin eğitimde en iyisi (Calmar)")

    txt = [f"Eğitim: {tr_a.date()} → {tr_b.date()} | GÖRÜLMEMİŞ TEST: {ho_a.date()} → {ho_b.date()}",
           "VERİ: SENTETİK" if args.synthetic else "Veri: Binance USDT-M 1dk + taker-buy hacmi + funding (data.binance.vision)",
           f"{len(sigs)} sinyal x {len(sizing)} boyutlandırma = {len(sigs) * len(sizing)} konfig. "
           f"Kaldıraç tavanı {LEV}x, ücret %{C.TAKER_FEE * 100:g}, slippage %{C.SLIPPAGE * 100:g}.",
           f"Order book (bookDepth) arşiv yoklaması BTCUSDT: {book}", "",
           f"Sermaye koruma filtresi: eğitim maks. DD ≤ %{MAX_DD_OK:g}, ≥30 işlem, pozitif getiri → "
           f"{len(ok)} konfig geçti.", "",
           "Filtreyi geçen en iyi 15 (Calmar sırası):", fmt(ok.head(15)) if len(ok) else "(yok)", "",
           "Her ailenin en iyisi (filtre yok, Calmar):", fmt(best_per_family), "",
           "Benchmark:", lb[lb.sinyal == "benchmark"][["strateji", "egitim_bakiye", "egitim_CAGR_%",
                                                         "TEST_bakiye"]].round(2).to_string(index=False),
           "", "Eğitimde en yüksek bakiye (bilgi; risk göz ardı):",
           fmt(lb.sort_values("egitim_bakiye", ascending=False).head(8))]
    text = "\n".join(txt)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
