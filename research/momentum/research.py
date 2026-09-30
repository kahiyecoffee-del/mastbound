"""Momentum strateji araştırması: birçok varyantı aynı veride, aynı hesap kurallarıyla
(tek pozisyon, bileşik bakiye, 2x, ücret+slippage+funding) karşılaştırır.

Aşırı uyumu (overfitting) sınırlamak için dönem ikiye bölünür:
  IS  (in-sample)      : strateji SEÇİMİ bu dönemde yapılır
  OOS (out-of-sample)  : seçilen stratejinin görmediği dönemde sonucu
Her pencere 100 USDT ile ayrı başlar; "tam dönem" de ayrıca raporlanır.

  python research.py --out results/research            # gerçek veri (data.binance.vision)
  python research.py --synthetic --out /tmp/r          # sentetik veriyle hızlı test
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
import data
from engine import run_portfolio
from indicators import macd
from signals import Minute, find_entry, liquidation_price

MIN = 60_000_000_000
IS_END = "2026-05-24"
WARMUP_DAYS = 60
REASONS = ["SL", "BE", "TP", "TIME", "END", "LIQ", "TRAIL"]
INF = np.inf


# ------------------------------------------------------------------ çıkış simülasyonu
@numba.njit(cache=True)
def sim_exit(o, h, l, c, e, stop, tp, be_level, be_mask, trail_k, htf_close_at, htf_atr,
             deadline, liq, slip):
    entry = o[e] * (1 + slip)
    cur = stop
    kind = 0                       # 0 ilk stop, 1 BE, 6 trailing
    touched = False
    one_r = entry + (entry - stop) if stop > 0 else INF
    hh = o[e]
    be_idx = -1
    n = len(o)
    for k in range(e, n):
        if k >= deadline:                                    # zaman çıkışı: bu mumun açılışı
            return k, o[k], 3, be_idx, touched
        eff = cur if cur > liq else liq
        if l[k] <= eff:
            raw = eff if k == e else min(o[k], eff)
            r = kind if cur > liq else 5
            return k, raw, r, be_idx, touched
        if h[k] >= one_r:
            touched = True
        if h[k] >= tp:
            return k, tp, 2, be_idx, True
        if h[k] > hh:
            hh = h[k]
        if kind == 0 and be_mask[k] and c[k] > be_level:
            cur = entry
            kind = 1
            be_idx = k
        j = htf_close_at[k]
        if trail_k > 0 and j >= 0:
            new = hh - trail_k * htf_atr[j]
            if new > cur:
                cur = new
                kind = 6
    return n - 1, c[n - 1], 4, be_idx, touched


def atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


class Coin:
    def __init__(self, name, df1m, funding):
        self.name = name
        self.mm = Minute(df1m)
        self.funding = funding
        if funding is not None and len(funding):
            self.f_t = funding.index.as_unit("ns").asi8
            self.f_v = funding.to_numpy(float)
            self.f_idx = np.clip(np.searchsorted(self.mm.t, self.f_t), 0, len(self.mm.t) - 1)
        else:
            self.f_t = None
        self.tfs = data.build_timeframes(df1m)
        self.tfs["1D"] = data.resample(df1m, "1D")
        t = self.mm.t
        self.be_mask = ((t + MIN) % (5 * MIN)) == 0
        self._htf = {}

    def htf(self, tf):
        """(close_at[1dk idx] -> htf bar idx, atr dizisi) — sadece kapanmış mum kullanılır."""
        if tf not in self._htf:
            df = self.tfs[tf]
            close_ns = (df.index + pd.Timedelta(tf)).as_unit("ns").asi8
            k = np.searchsorted(self.mm.t, close_ns) - 1
            ok = (k >= 0) & (k < len(self.mm.t))
            close_at = np.full(len(self.mm.t), -1, np.int64)
            close_at[k[ok]] = np.arange(len(df))[ok]
            self._htf[tf] = (close_at, atr(df).to_numpy())
        return self._htf[tf]


def funding_px(coin, e, k):
    if coin.f_t is None:
        return 0.0
    a = np.searchsorted(coin.f_t, coin.mm.t[e], side="right")
    b = np.searchsorted(coin.f_t, coin.mm.t[k], side="right")
    if b <= a:
        return 0.0
    return float(np.sum(coin.f_v[a:b] * coin.mm.o[coin.f_idx[a:b]]))


def make_trade(coin, e, stop, T, tp_r=INF, be=False, trail_tf=None, trail_k=0.0,
               deadline=None):
    mm = coin.mm
    entry = mm.o[e] * (1 + C.SLIPPAGE)
    if stop >= entry:
        return None
    R = entry - stop if stop > 0 else entry * 0.05     # stopsuz stratejide nominal R
    tp = entry + tp_r * R if np.isfinite(tp_r) else INF
    be_level = entry + C.BREAKEVEN_R * R if be else INF
    if trail_tf:
        close_at, atr_arr = coin.htf(trail_tf)
    else:
        close_at, atr_arr = coin.htf("1h")
    liq = liquidation_price(entry)
    k, raw, r, be_idx, touched = sim_exit(
        mm.o, mm.h, mm.l, mm.c, e, stop, tp, be_level, coin.be_mask,
        trail_k if trail_tf else 0.0, close_at, atr_arr,
        deadline if deadline is not None else len(mm.o) + 1, liq, C.SLIPPAGE)
    x = dict(exit_idx=k, exit_time=mm.index[k], exit=raw * (1 - C.SLIPPAGE),
             reason=REASONS[r], be_time=mm.index[be_idx] if be_idx >= 0 else None,
             touched_1r=bool(touched), funding_px=funding_px(coin, e, k))
    # stopsuz stratejilerde likidasyonu sim_exit zaten uygular; portföy kontrolü için
    # stop'u likidasyonun üstünde gösteriyoruz
    return dict(coin=coin.name, setup_time=T, entry_time=mm.index[e], entry=entry,
                stop=stop if stop > 0 else liq * 1.0001, tp=tp, liq=liq, R=R, be=x, nobe=x)


# ------------------------------------------------------------------ strateji aileleri
def macd_flags(coin, tf):
    df = coin.tfs[tf]
    m = macd(df["close"])
    out = pd.DataFrame({"pos": m["macd"] > 0, "above": m["macd"] > m["signal"]})
    out.index = df.index + pd.Timedelta(tf)
    return out


def family_breakout(coin, start, end, tf, filt, stop_mode, entry_mode):
    """Orijinal yapı: kırılım + 4s/1s MACD filtresi (+ isteğe bağlı PSAR girişi).
    Dönen: setup listesi [(T, e, stop)], çıkış varyantları ayrı uygulanır."""
    df = coin.tfs[tf]
    prior_high = df["high"].shift(1).rolling(5).max()
    if stop_mode[0] == "swing":
        stop = df["low"].rolling(stop_mode[1]).min()
    else:
        stop = df["close"] - stop_mode[1] * atr(df)
    s = pd.DataFrame({"close": df["close"], "ph": prior_high, "stop": stop})
    s["T"] = df.index + pd.Timedelta(tf)
    s = s[(s.close > s.ph) & (s["T"] >= start) & (s["T"] < end) & (s.close > s.stop)]
    s = s.sort_values("T").reset_index(drop=True)
    ok = pd.Series(True, index=s.index)
    for htf in ["4h", "1h"]:
        f = pd.merge_asof(s[["T"]], macd_flags(coin, htf), left_on="T", right_index=True)
        cond = (f["pos"] | f["above"]) if filt == "or" else (f["pos"] & f["above"])
        ok &= cond.fillna(False).astype(bool).to_numpy()
    s = s[ok.to_numpy()]
    mm = coin.mm
    setups, busy = [], np.int64(0)
    for T, st in zip(s["T"], s["stop"]):
        if entry_mode == "psar":
            if T.value < busy:
                continue
            e, why, _ = find_entry(mm, T, st)
            if e is None:
                i0 = min(int(np.searchsorted(mm.t, T.value)) + C.ENTRY_TIMEOUT_MIN,
                         len(mm.t) - 1)
                busy = mm.t[i0]
                continue
            busy = mm.t[e]
        else:
            e = int(np.searchsorted(mm.t, T.value))
            if e >= len(mm.t) or mm.o[e] <= st:
                continue
        setups.append((T, e, st))
    return setups


def family_donchian(coin, start, end, tf, n, k):
    df = coin.tfs[tf]
    ema = df["close"].ewm(span=50, adjust=False).mean()
    hi = df["high"].shift(1).rolling(n).max()
    a = atr(df)
    T = df.index + pd.Timedelta(tf)
    sel = (df["close"] > hi) & (df["close"] > ema) & (T >= start) & (T < end)
    out = []
    for Ti, av in zip(T[sel.to_numpy()], a[sel].to_numpy()):
        e = int(np.searchsorted(coin.mm.t, Ti.value))
        if e < len(coin.mm.t):
            out.append((Ti, e, coin.mm.o[e] - k * av))
    return out


def rotation_schedule(coins, start, end, lookback, hold):
    """Her `hold` günde bir: son `lookback` gün getirisi en yüksek (ve > 0) coin tutulur."""
    closes = pd.DataFrame({c.name: c.tfs["1D"]["close"] for c in coins.values()})
    closes.index = closes.index + pd.Timedelta("1D")        # kapanış anı
    score = closes / closes.shift(lookback) - 1
    days = pd.date_range(start, end, freq=f"{hold}D", inclusive="left")
    sched = []
    for d in days:
        row = score.loc[:d].iloc[-1] if len(score.loc[:d]) else None
        pick = None
        if row is not None and row.notna().any() and row.max() > 0:
            pick = row.idxmax()
        if pick and sched and sched[-1][0] == pick and sched[-1][2] == d:
            sched[-1][2] = min(d + pd.Timedelta(days=hold), end)
        elif pick:
            sched.append([pick, d, min(d + pd.Timedelta(days=hold), end)])
    return sched


# ------------------------------------------------------------------ konfigürasyonlar
def build_configs():
    cfgs = []
    for tf in ["15min", "1h"]:
        for filt in ["or", "and"]:
            for sm in [("swing", 5), ("atr", 1.5), ("atr", 3.0)]:
                for em in ["psar", "now"]:
                    for ex in [("tp2_be", 2, True, 0), ("tp2", 2, False, 0),
                               ("tp3", 3, False, 0), ("trail3", INF, False, 3.0)]:
                        name = f"BRK {tf} filt={filt} stop={sm[0]}{sm[1]} giriş={em} çıkış={ex[0]}"
                        cfgs.append(dict(name=name, family="breakout",
                                         setup=("brk", tf, filt, sm, em),
                                         tp_r=ex[1], be=ex[2], trail_k=ex[3], trail_tf=tf))
    for tf in ["4h", "1D"]:
        for n in [20, 55]:
            for k in [2.0, 3.0, 4.0]:
                cfgs.append(dict(name=f"DONCHIAN {tf} N={n} chandelier={k}ATR",
                                 family="donchian", setup=("don", tf, n, k),
                                 tp_r=INF, be=False, trail_k=k, trail_tf=tf))
    for lb in [3, 7, 14, 30]:
        for hold in [1, 7]:
            for st in [0.0, 3.0]:
                cfgs.append(dict(name=f"ROTASYON lookback={lb}g tut={hold}g stop="
                                      f"{'yok' if not st else f'{st}ATR(1g)'}",
                                 family="rotation", setup=("rot", lb, hold), stop_k=st))
    cfgs.append(dict(name="BENCHMARK BTC al-tut (2x)", family="benchmark"))
    return cfgs


def candidates_for(cfg, coins, start, end, cache):
    fam = cfg["family"]
    out = []
    if fam in ("breakout", "donchian"):
        for coin in coins.values():
            key = (coin.name,) + cfg["setup"]
            if key not in cache:
                if fam == "breakout":
                    cache[key] = family_breakout(coin, start, end, *cfg["setup"][1:])
                else:
                    cache[key] = family_donchian(coin, start, end, *cfg["setup"][1:])
            for T, e, st in cache[key]:
                c = make_trade(coin, e, st, T, tp_r=cfg["tp_r"], be=cfg["be"],
                               trail_tf=cfg["trail_tf"] if cfg["trail_k"] else None,
                               trail_k=cfg["trail_k"])
                if c:
                    out.append(c)
    elif fam == "rotation":
        _, lb, hold = cfg["setup"]
        for name, d0, d1 in rotation_schedule(coins, start, end, lb, hold):
            coin = coins[name]
            e = int(np.searchsorted(coin.mm.t, d0.value))
            dl = int(np.searchsorted(coin.mm.t, d1.value))
            if e >= len(coin.mm.t):
                continue
            stop = 0.0
            if cfg["stop_k"]:
                close_at, a = coin.htf("1D")
                j = close_at[:e + 1][close_at[:e + 1] >= 0]
                stop = coin.mm.o[e] - cfg["stop_k"] * a[j[-1]] if len(j) else 0.0
            c = make_trade(coin, e, stop, d0, deadline=dl,
                           trail_tf="1D" if cfg["stop_k"] else None, trail_k=cfg["stop_k"])
            if c:
                out.append(c)
    else:
        coin = coins.get("BTC") or next(iter(coins.values()))
        for a, b in cfg["segments"]:                     # her pencerede ayrı al-tut
            e = int(np.searchsorted(coin.mm.t, a.value))
            dl = int(np.searchsorted(coin.mm.t, b.value))
            c = make_trade(coin, e, 0.0, a, deadline=dl)
            if c:
                out.append(c)
    return out


# ------------------------------------------------------------------ metrik
def summarize(trades):
    if not trades:
        return dict(final=C.INITIAL_BALANCE, ret=0.0, n=0, win=np.nan, pf=np.nan, mdd=0.0,
                    avgR=np.nan)
    df = pd.DataFrame(trades)
    eq = np.r_[C.INITIAL_BALANCE, df["balance"].to_numpy()]
    peak = np.maximum.accumulate(eq)
    gp, gl = df.pnl[df.pnl > 0].sum(), -df.pnl[df.pnl < 0].sum()
    return dict(final=eq[-1], ret=(eq[-1] / C.INITIAL_BALANCE - 1) * 100, n=len(df),
                win=(df.pnl > 0).mean() * 100, pf=gp / gl if gl > 0 else np.inf,
                mdd=((peak - eq) / peak).max() * 100, avgR=df["R_net"].mean())


def window(cands, a, b):
    return [c for c in cands if a <= c["entry_time"] < b]


# ------------------------------------------------------------------ grafikler
def plot_market(coins, path, start=None, vlines=None):
    fig, ax = plt.subplots(figsize=(12, 5))
    for c in coins.values():
        s = c.tfs["1D"]["close"]
        s = s[s.index >= (start or pd.Timestamp(C.START, tz="UTC"))]
        ax.plot(s.index, s / s.iloc[0] * 100, lw=1.3, label=c.name)
    ax.axhline(100, color="gray", ls="--", lw=0.8)
    for v in (vlines if vlines is not None else [pd.Timestamp(IS_END, tz="UTC")]):
        ax.axvline(v, color="black", ls=":", lw=1)
    ax.set_title("Piyasa: coinlerin normalize fiyatı (başlangıç=100); noktalı çizgi = IS/OOS ayrımı")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_equities(curves, path, title, vlines=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, tr in curves.items():
        if not tr:
            continue
        df = pd.DataFrame(tr)
        x = pd.concat([pd.Series([df["entry_time"].iloc[0]]), df["exit_time"]])
        ax.step(x, np.r_[C.INITIAL_BALANCE, df["balance"]], where="post", lw=1.4,
                label=name[:70])
    ax.axhline(C.INITIAL_BALANCE, color="gray", ls="--", lw=0.8)
    for v in (vlines if vlines is not None else [pd.Timestamp(IS_END, tz="UTC")]):
        ax.axvline(v, color="black", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel("Bakiye (USDT, log)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_trades_on_price(coins, trades, path, title, start=None):
    names = [c for c in coins if any(t["coin"] == c for t in trades)] or list(coins)
    fig, axes = plt.subplots(len(names), 1, figsize=(13, 2.6 * len(names)), squeeze=False)
    for ax, name in zip(axes[:, 0], names):
        s = coins[name].tfs["4h"]["close"]
        s = s[s.index >= (start or pd.Timestamp(C.START, tz="UTC"))]
        ax.plot(s.index, s, color="gray", lw=0.8)
        for t in trades:
            if t["coin"] != name:
                continue
            col = "#2e9e5b" if t["pnl"] > 0 else "#d64545"
            ax.plot([t["entry_time"], t["exit_time"]], [t["entry"], t["exit"]], color=col, lw=2)
            ax.scatter([t["entry_time"]], [t["entry"]], color=col, s=10, zorder=3)
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    axes[0, 0].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ ana akış
def load(args, start, end):
    coins = {}
    dl_start = start - pd.Timedelta(days=WARMUP_DAYS)
    for name in args.coins.split(","):
        if args.synthetic:
            import synthetic
            df = synthetic.make_1m(name, dl_start, end, seed=args.seed, drift_scale=args.drift)
            fund = synthetic.make_funding(start, end, seed=args.seed)
        else:
            try:
                df = data.download_vision(name, dl_start, end)
            except ValueError as e:
                print(f"[{name}] atlandı: {e}", flush=True)
                continue
            if len(df) < 60 * 24 * 90:
                print(f"[{name}] atlandı: 90 günden az veri", flush=True)
                continue
            try:
                fund, _ = data.download_funding_vision(name, start, end)
            except Exception:
                fund = None
        coins[name] = Coin(name, df, fund)
        print(f"[{name}] {len(df):,} mum yüklendi", flush=True)
    return coins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=C.START)
    ap.add_argument("--end", default=C.END)
    ap.add_argument("--is-end", default=IS_END)
    ap.add_argument("--coins", default=",".join(C.COINS))
    ap.add_argument("--balance", type=float, default=100.0)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drift", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0, help="test için ilk N konfig")
    ap.add_argument("--out", default="results/research")
    ap.add_argument("--periods", default="",
                    help="çok dönemli mod: virgüllü sınır tarihleri, ör. 2021-01-01,2022-01-01,...")
    args = ap.parse_args()
    if args.periods:
        return main_periods(args)
    C.INITIAL_BALANCE = args.balance
    os.makedirs(args.out, exist_ok=True)
    start, end = pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC")
    is_end = pd.Timestamp(args.is_end, tz="UTC")
    coins = load(args, start, end)
    plot_market(coins, os.path.join(args.out, "market.png"))

    cfgs = build_configs()
    for c in cfgs:
        c["segments"] = [(start, is_end), (is_end, end)]
    if args.limit:
        cfgs = cfgs[:args.limit] + [c for c in cfgs if c["family"] != "breakout"][:args.limit]
    rows, cache, all_trades = [], {}, {}
    t0 = time.time()
    for i, cfg in enumerate(cfgs):
        cands = candidates_for(cfg, coins, start, end, cache)
        res = {}
        for wname, a, b in [("IS", start, is_end), ("OOS", is_end, end), ("FULL", start, end)]:
            tr, _ = run_portfolio(window(cands, a, b), breakeven=True)
            res[wname] = summarize(tr)
            if wname == "FULL":
                all_trades[cfg["name"]] = tr
        row = {"strateji": cfg["name"], "aile": cfg["family"]}
        for w, m in res.items():
            row.update({f"{w}_bakiye": m["final"], f"{w}_getiri_%": m["ret"], f"{w}_islem": m["n"],
                        f"{w}_isabet_%": m["win"], f"{w}_PF": m["pf"], f"{w}_maxDD_%": m["mdd"],
                        f"{w}_ortR": m["avgR"]})
        rows.append(row)
        if i % 10 == 0:
            print(f"{i + 1}/{len(cfgs)} konfig, {time.time() - t0:.0f}s — {cfg['name']}: "
                  f"IS {res['IS']['final']:.1f} OOS {res['OOS']['final']:.1f}", flush=True)

    lb = pd.DataFrame(rows)
    elig = lb[lb["IS_islem"] >= 10]
    lb = lb.sort_values("IS_bakiye", ascending=False)
    lb.to_csv(os.path.join(args.out, "leaderboard.csv"), index=False, float_format="%.4g")

    top_is = elig.sort_values("IS_bakiye", ascending=False).head(5)
    best = top_is.iloc[0]["strateji"]
    bench = "BENCHMARK BTC al-tut (2x)"
    fam_best = elig.sort_values("IS_bakiye", ascending=False).groupby("aile").head(1)
    curves = {n: all_trades[n] for n in list(top_is["strateji"]) + [bench]
              if n in all_trades}
    plot_equities(curves, os.path.join(args.out, "equity_top_IS.png"),
                  "IS'de en iyi 5 strateji — tam dönem (noktalı çizgi sonrası = görülmemiş dönem)")
    plot_equities({n: all_trades[n] for n in fam_best["strateji"]},
                  os.path.join(args.out, "equity_family_best.png"),
                  "Her ailenin IS'de en iyisi — tam dönem")
    plot_trades_on_price(coins, all_trades[best], os.path.join(args.out, "best_trades.png"),
                         f"IS en iyisi: {best} — işlemler (yeşil kâr, kırmızı zarar)")
    pd.DataFrame(all_trades[best]).to_csv(os.path.join(args.out, "best_trades.csv"),
                                          index=False, float_format="%.8g")
    for w in ["IS", "OOS"]:
        tr, _ = run_portfolio(window(candidates_for(
            next(c for c in cfgs if c["name"] == best), coins, start, end, cache),
            start if w == "IS" else is_end, is_end if w == "IS" else end))
        pd.DataFrame(tr).to_csv(os.path.join(args.out, f"best_trades_{w}.csv"), index=False,
                                float_format="%.8g")

    cols = ["strateji", "IS_bakiye", "IS_islem", "IS_isabet_%", "IS_maxDD_%", "OOS_bakiye",
            "OOS_islem", "OOS_maxDD_%", "FULL_bakiye", "FULL_maxDD_%"]
    fmt = lambda d: d[cols].round(2).to_string(index=False)
    txt = [f"Dönem {start.date()} → {end.date()}, IS < {is_end.date()} ≤ OOS. "
           f"Her pencere {C.INITIAL_BALANCE} USDT ile başlar. {len(cfgs)} konfig.",
           "VERİ: SENTETİK" if args.synthetic else "Veri: Binance USDT-M 1dk (data.binance.vision)",
           "", "IS'de en iyi 5 (seçim buna göre):", fmt(top_is), "",
           "Her ailenin IS en iyisi:", fmt(fam_best), "",
           "Benchmark:", fmt(lb[lb.strateji == bench]), "",
           "Tam dönemde en iyi 10 (bilgi amaçlı; seçim için kullanmak aşırı uyum riski taşır):",
           fmt(lb.sort_values("FULL_bakiye", ascending=False).head(10)), "",
           "OOS'ta en iyi 10 (bilgi amaçlı):",
           fmt(lb.sort_values("OOS_bakiye", ascending=False).head(10)), "",
           f"IS ve OOS'un İKİSİNDE de kârlı konfig sayısı: "
           f"{int(((lb.IS_bakiye > C.INITIAL_BALANCE) & (lb.OOS_bakiye > C.INITIAL_BALANCE)).sum())}"
           f" / {len(lb)}",
           fmt(lb[(lb.IS_bakiye > C.INITIAL_BALANCE) & (lb.OOS_bakiye > C.INITIAL_BALANCE)]
               .sort_values("FULL_bakiye", ascending=False).head(15))]
    text = "\n".join(txt)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


def main_periods(args):
    """Her dönem (ör. her yıl) 100 USDT ile ayrı başlar; sağlamlık = kaç dönemde kârlı."""
    C.INITIAL_BALANCE = args.balance
    os.makedirs(args.out, exist_ok=True)
    b = [pd.Timestamp(x, tz="UTC") for x in args.periods.split(",")]
    start, end = b[0], b[-1]
    wins = [(f"{a:%Y-%m}", a, z) for a, z in zip(b[:-1], b[1:])]
    labels = [w[0] for w in wins]
    coins = load(args, start, end)
    plot_market(coins, os.path.join(args.out, "market.png"), start, b[1:-1])
    cfgs = build_configs()
    for c in cfgs:
        c["segments"] = [(a, z) for _, a, z in wins]
    if args.limit:
        cfgs = cfgs[:args.limit] + [c for c in cfgs if c["family"] != "breakout"][:args.limit]
    rows, cache, full_trades = [], {}, {}
    t0 = time.time()
    for i, cfg in enumerate(cfgs):
        cands = candidates_for(cfg, coins, start, end, cache)
        row = {"strateji": cfg["name"], "aile": cfg["family"]}
        rets = []
        for lab, a, z in wins:
            m = summarize(run_portfolio(window(cands, a, z))[0])
            row.update({f"{lab}_getiri_%": m["ret"], f"{lab}_islem": m["n"],
                        f"{lab}_maxDD_%": m["mdd"]})
            rets.append(m["ret"])
        tr, _ = run_portfolio(cands)
        full_trades[cfg["name"]] = tr
        m = summarize(tr)
        rets = np.array(rets)
        row.update({"karli_donem": int((rets > 0).sum()), "medyan_getiri_%": float(np.median(rets)),
                    "en_kotu_%": float(rets.min()), "ort_getiri_%": float(rets.mean()),
                    "toplam_islem": m["n"], "bilesik_bakiye": m["final"],
                    "bilesik_maxDD_%": m["mdd"], "isabet_%": m["win"], "PF": m["pf"],
                    "ortR_net": m["avgR"]})
        rows.append(row)
        if i % 10 == 0:
            print(f"{i + 1}/{len(cfgs)} konfig, {time.time() - t0:.0f}s — {cfg['name']}: "
                  + " ".join(f"{r:+.0f}%" for r in rets), flush=True)

    lb = pd.DataFrame(rows)
    lb["skor"] = lb["karli_donem"] * 1000 + lb["medyan_getiri_%"]
    lb = lb.sort_values(["karli_donem", "medyan_getiri_%"], ascending=False)
    lb.to_csv(os.path.join(args.out, "leaderboard.csv"), index=False, float_format="%.4g")
    elig = lb[lb["toplam_islem"] >= 20]
    top = elig.head(5)
    bench = "BENCHMARK BTC al-tut (2x)"
    vl = b[1:-1]
    plot_equities({n: full_trades[n] for n in list(top["strateji"]) + [bench]},
                  os.path.join(args.out, "equity_top_robust.png"),
                  "En sağlam 5 strateji — bileşik bakiye (noktalı çizgiler = dönem sınırları)", vl)
    fam_best = elig.groupby("aile").head(1)
    plot_equities({n: full_trades[n] for n in fam_best["strateji"]},
                  os.path.join(args.out, "equity_family_best.png"),
                  "Her ailenin en sağlam stratejisi — bileşik bakiye", vl)
    best = top.iloc[0]["strateji"]
    plot_trades_on_price(coins, full_trades[best], os.path.join(args.out, "best_trades.png"),
                         f"En sağlam: {best} — işlemler (yeşil kâr, kırmızı zarar)", start)
    pd.DataFrame(full_trades[best]).to_csv(os.path.join(args.out, "best_trades.csv"),
                                           index=False, float_format="%.8g")
    # yıl x strateji ısı haritası (en sağlam 25)
    h = elig.head(25)
    mat = h[[f"{l}_getiri_%" for l in labels]].to_numpy(float)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(np.clip(mat, -100, 100), cmap="RdYlGn", vmin=-100, vmax=100, aspect="auto")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(h)), [n[:60] for n in h["strateji"]], fontsize=7)
    for (r, c_), v in np.ndenumerate(mat):
        ax.text(c_, r, f"{v:+.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, label="dönem getirisi % (±100'de kırpılmış)")
    ax.set_title("En sağlam 25 strateji — dönem bazında getiri")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "heatmap.png"), dpi=120)
    plt.close(fig)

    cols = (["strateji", "karli_donem"] + [f"{l}_getiri_%" for l in labels] +
            ["medyan_getiri_%", "en_kotu_%", "toplam_islem", "bilesik_bakiye", "bilesik_maxDD_%"])
    fmt = lambda d: d[cols].round(1).to_string(index=False)
    dist = lb["karli_donem"].value_counts().sort_index(ascending=False)
    txt = [f"Dönemler: {', '.join(f'{a.date()}→{z.date()}' for _, a, z in wins)}. Her dönem "
           f"{C.INITIAL_BALANCE} USDT ile başlar; 'bilesik_bakiye' tüm dönemi tek hesapla gösterir.",
           "VERİ: SENTETİK" if args.synthetic else "Veri: Binance USDT-M 1dk (data.binance.vision)",
           f"{len(cfgs)} konfig. Sıralama: kârlı dönem sayısı, sonra medyan dönem getirisi.", "",
           "Kaç dönemde kârlı → konfig sayısı:", dist.to_string(), "",
           "En sağlam 15 (≥20 işlem):", fmt(elig.head(15)), "",
           "Her ailenin en sağlamı:", fmt(fam_best), "",
           "Benchmark:", fmt(lb[lb.strateji == bench]), "",
           "Bileşik bakiyede en iyi 10 (bilgi amaçlı):",
           fmt(lb.sort_values("bilesik_bakiye", ascending=False).head(10))]
    text = "\n".join(txt)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
