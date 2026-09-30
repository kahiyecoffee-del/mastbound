"""ÜÇÜNCÜ STRATEJİ ARAŞTIRMASI — mevcut bota (EMA + DONCHIAN, v8) EK olarak kârı artıracak ve düşüşü azaltacak,
ONUNLA DÜŞÜK KORELASYONLU bir getiri kaynağı.

Adaylar (hepsi günlük, az parametreli, literatürde bilinen aileler; sinyal gün kapanışında, pozisyon ertesi gün):
 KRİPTO (38 likit USDT-M perp, Binance 1dk'dan günlük; funding dahil):
  X1 kesitsel momentum L/S  : her hafta son N gün getirisine göre en iyi 5 coin long, en kötü 5 short (piyasa-nötr)
  X2 kesitsel momentum long : en iyi 5 long, yalnız BTC > EMA200 iken (değilse nakit)
  X3 funding karşıtı L/S    : her hafta 7g ort. funding en YÜKSEK 5 coin short, en DÜŞÜK 5 long (kalabalığın tersi + carry)
  X4 kısa vadeli dönüş L/S  : her gün son 1 günün en çok düşen 5 coini long, en çok yükseleni short
 HİSSE / EMTİA (Yahoo günlük; MEXC'de hisse ve altın perp'leri var):
  S1 zaman serisi trend     : SPY, QQQ, GLD, SLV, TLT — kapanış > SMA(N) ise long (değilse nakit), eşit risk
  S2 mega-cap momentum      : 8 büyük teknoloji hissesi; ayda bir son 6 ayın en iyi 3'ü long, yalnız SPY > SMA200
  S3 altın trendi           : GLD kapanış > SMA(N) → long, değilse nakit
Maliyet: kripto işlem başı %0.02 taker + %0.02 kayma (MEXC gerçek oranlara yakın), stres: %0.05 + %0.02;
hisse/emtia %0.05 + %0.02. Kaldıraç 1x (brüt maruziyet = sermaye).

DEĞERLENDİRME (kabul kriteri ikisi birden):
  a) tek başına: 2021-23, 2024-25.09 ve SON 1 YIL'ın ÜÇÜNDE DE kârlı
  b) bota eklenince (aylık yeniden dengelenen %20 / %33 sermaye payı): üç dönemde de getiri/maxDD oranı (Calmar)
     mevcut botunkinden iyi. Ayrıca botla günlük getiri korelasyonu ve botun en kötü 10 gününde ne yaptığı.
Parametre seçimi yalnız 2021-23 ile (her ailede 3 varyant); diğer dönemler görülmemiş test.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import config as C
import research as R
import research_15m as Q
import research_combo as CB
import research_cvd as RC
import research_daily_detail as DD
import research_ema921 as E
import research_fix3 as F
import research_ls as L
import research_quant as RQ

EXTRA = "ETC,BCH,XLM,AAVE,ICP,HBAR,INJ,SAND,MANA,AXS,EGLD,ALGO,VET,CRV,COMP,SNX,RUNE,EOS,THETA,ARB"
DON_V7 = ((1.0, 0.1), (3.0, 1.0), None)
BASE_V = (None, None, None)
MEXC = {"BTC": (0.0, 0.0), "ETH": (0.0, 0.0), "NEAR": (0.0, 0.0), "SOL": (0.0, 0.0001)}
TREND = "SPY,QQQ,GLD,SLV,TLT"
MEGA = "AAPL,MSFT,NVDA,AMZN,META,GOOGL,TSLA,AVGO"


class DailyOnly:
    def __init__(self, c):
        self.tfs = {"1D": c.tfs["1D"][["close"]].copy()}


def yahoo_daily(tickers, start):
    import yfinance as yf
    out = {}
    for t in tickers:
        try:
            df = yf.download(t, interval="1d", start=start, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            s = df["Close"].dropna()
            s.index = pd.DatetimeIndex(s.index).tz_localize("UTC") if s.index.tz is None else s.index.tz_convert("UTC")
            out[t] = s
            print(f"[{t}] {len(s)} gün", flush=True)
        except Exception as e:
            print(f"[{t}] alınamadı: {e}", flush=True)
    return pd.DataFrame(out)


def run_weights(W, ret, cost, fund=None):
    """W: t günü kapanışında belirlenen ağırlıklar → t+1 getirisine uygulanır. Döner: günlük net getiri serisi."""
    W = W.reindex(ret.index).fillna(0.0)
    Wp = W.shift(1).fillna(0.0)
    gross = (Wp * ret.fillna(0.0)).sum(axis=1)
    turn = (W - W.shift(1).fillna(0.0)).abs().sum(axis=1).shift(1).fillna(0.0)
    r = gross - cost * turn
    if fund is not None:
        r = r - (Wp * fund.reindex(ret.index).fillna(0.0)).sum(axis=1)     # long pozitif funding öder
    return r


def rank_weights(score, n_long, n_short, every, valid=None):
    """Her `every` günde bir: skoru en yüksek n_long long (+), en düşük n_short short (−), eşit ağırlık, brüt 1."""
    W = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    last = None
    for i, d in enumerate(score.index):
        if last is not None and i % every:
            W.loc[d] = last
            continue
        s = score.loc[d]
        if valid is not None:
            s = s[valid.loc[d].fillna(False)]
        s = s.dropna()
        w = pd.Series(0.0, index=score.columns)
        if len(s) >= n_long + n_short:
            k = n_long + n_short
            if n_long:
                w[s.nlargest(n_long).index] = 1.0 / k
            if n_short:
                w[s.nsmallest(n_short).index] = -1.0 / k
        W.loc[d] = w
        last = w
    return W


def stats(r, a, z):
    x = r[(r.index >= a) & (r.index < z)]
    if not len(x):
        return dict(final=100.0, ret=0.0, mdd=0.0, cagr=0.0, calmar=np.nan)
    eq = 100 * (1 + x).cumprod()
    pk = eq.cummax()
    mdd = float(((pk - eq) / pk).max() * 100)
    yrs = max((z - a).days / 365.25, 1e-9)
    fin = float(eq.iloc[-1])
    cagr = ((fin / 100) ** (1 / yrs) - 1) * 100 if fin > 0 else -100.0
    return dict(final=fin, ret=fin - 100, mdd=mdd, cagr=cagr, calmar=cagr / mdd if mdd > 0 else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", default=CB.COINS18)
    ap.add_argument("--extra", default=EXTRA)
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--val", default="2024-01-01")
    ap.add_argument("--test", default="2025-09-24")
    ap.add_argument("--end", default="2026-09-24")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--no-stocks", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drift", type=float, default=0.0)
    ap.add_argument("--out", default="results/strat3")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    A, V, T_, Z = (pd.Timestamp(x, tz="UTC") for x in (args.start, args.val, args.test, args.end))
    periods = [("2021-23", A, V), ("2024-25.09", V, T_), ("SON 1 YIL", T_, Z)]
    PL = [p[0] for p in periods]
    years = [(str(y), pd.Timestamp(f"{y}-01-01", tz="UTC"), min(pd.Timestamp(f"{y + 1}-01-01", tz="UTC"), Z))
             for y in range(A.year, Z.year + 1) if pd.Timestamp(f"{y}-01-01", tz="UTC") < Z]
    bot_coins = args.coins.split(",")
    extra = [c for c in args.extra.split(",") if c and c not in bot_coins]
    C.COINS = bot_coins
    E.SIG_TF, E.TREND_TF, E.MAX_HOLD = "1h", "4h", pd.Timedelta(hours=168)
    t0 = time.time()

    def load_one(name):
        a = argparse.Namespace(**vars(args))
        a.coins = name
        return R.load(a, A, Z).get(name)

    # ---- bot v8 günlük özsermaye + kripto günlük veriler
    btc = load_one("BTC")
    em, dn, daily, closes, funds = [], [], {}, {}, {}
    for name in ["BTC"] + [n for n in bot_coins if n != "BTC"] + extra:
        c = btc if name == "BTC" else load_one(name)
        if c is None:
            continue
        d1 = c.tfs["1D"]
        closes[name] = d1["close"]
        if c.funding is not None and len(c.funding):
            f = pd.Series(np.asarray(c.funding, float), index=pd.DatetimeIndex(c.funding.index).tz_convert("UTC"))
            funds[name] = f.resample("1D").sum()
        if name in bot_coins:
            x = E.prep(c, btc)
            st = {BASE_V: []}
            o = Q.trade
            Q.trade = F.ema_recorder(RC.flow_signals(c), {BASE_V: BASE_V}, st)
            try:
                E.signals(c, x, A, Z, "ikisi", "ema9_limit", "atr3", "iz3")
            finally:
                Q.trade = o
            for t in st[BASE_V]:
                sp = abs(t["entry"] - t["stop"]) / t["entry"] * 100
                em.append(dict(t, strat="EMA", size_mult=0.5 if sp > 5.0 else 1.0))
            sd = {DON_V7: []}
            ol = L.trade
            L.trade = F.don_recorder({DON_V7: DON_V7}, sd)
            try:
                L.f_donchian(c, L.daily_feats(c, btc.tfs["1D"]["close"]), A, Z, n=20, k=2.0, regime="btc")
            finally:
                L.trade = ol
            dn += sd[DON_V7]
            daily[name] = DailyOnly(c)
        print(f"[{name}] ({time.time() - t0:.0f}s)", flush=True)
    bot = []
    for t in em + dn:
        m_, tk = MEXC.get(t["coin"], (0.0, 0.0002))
        maker_in = t["strat"] == "EMA" and t.get("maker_entry", False)
        bot.append(dict(t, fee_in=m_ if maker_in else tk, fee_out=tk))
    tr, eq, _ = RQ.run_pf(bot)
    bot_eq = DD.mtm_curve(tr, daily, A, Z, 100.0)
    bot_r = bot_eq.pct_change().fillna(0.0)
    bot_r.index = bot_r.index.tz_convert("UTC") if bot_r.index.tz else bot_r.index.tz_localize("UTC")

    idx = pd.date_range(A, Z, freq="1D", inclusive="left", tz="UTC")
    P = pd.DataFrame(closes).sort_index()
    P = P[~P.index.duplicated()]
    Rk = P.pct_change(fill_method=None)
    listed = P.notna() & P.shift(30).notna()                      # en az 30 günlük geçmiş
    FU = pd.DataFrame(funds).reindex(P.index).fillna(0.0) if funds else None
    btc_up = (P["BTC"] > L.ema(P["BTC"], 200))
    kc, kc_s = 0.0002 + C.SLIPPAGE, 0.0005 + C.SLIPPAGE          # kripto maliyet (normal / stres)

    strat = {}                                                   # ad → (aile, getiri, stres getiri)

    def add(fam, name, W, cost, cost_s, fund):
        strat[name] = (fam, run_weights(W, Rk, cost, fund).reindex(idx).fillna(0.0),
                       run_weights(W, Rk, cost_s, fund).reindex(idx).fillna(0.0))

    for n in (14, 28, 56):
        mom = P.pct_change(n, fill_method=None).where(listed)
        add("X1 kesitsel momentum L/S", f"X1 mom L/S {n}g", rank_weights(mom, 5, 5, 7), kc, kc_s, FU)
        W2 = rank_weights(mom, 5, 0, 7).mul(btc_up.astype(float), axis=0)
        add("X2 kesitsel momentum long (BTC>EMA200)", f"X2 mom long {n}g", W2, kc, kc_s, FU)
    if FU is not None:
        for n in (3, 7, 14):
            fr = FU.rolling(n).mean().where(listed)
            add("X3 funding karşıtı L/S", f"X3 funding {n}g", rank_weights(-fr, 5, 5, 7), kc, kc_s, FU)
    for n in (1, 3, 5):
        rev = -P.pct_change(n, fill_method=None).where(listed)
        add("X4 kısa vade dönüş L/S", f"X4 dönüş {n}g", rank_weights(rev, 5, 5, 1), kc, kc_s, FU)

    if not args.no_stocks and not args.synthetic:
        S = yahoo_daily(TREND.split(",") + MEGA.split(","), "2019-06-01")
        S = S.reindex(pd.date_range(S.index.min(), Z, freq="1D", tz="UTC")).ffill()   # hafta sonu: getiri 0
        Rs = S.pct_change(fill_method=None)
        sc, sc_s = 0.0005 + C.SLIPPAGE, 0.001 + C.SLIPPAGE
        tt = [t for t in TREND.split(",") if t in S]
        vol = Rs[tt].rolling(60).std()
        for n in (50, 100, 200):
            on = (S[tt] > S[tt].rolling(n).mean()).astype(float)
            w = on / vol
            W = w.div((1 / vol).sum(axis=1), axis=0).fillna(0.0)          # eşit risk, toplam ≤ 1
            strat[f"S1 trend SMA{n}"] = ("S1 zaman serisi trend (SPY,QQQ,GLD,SLV,TLT)",
                                         run_weights(W, Rs[tt], sc).reindex(idx).fillna(0.0),
                                         run_weights(W, Rs[tt], sc_s).reindex(idx).fillna(0.0))
        mg = [t for t in MEGA.split(",") if t in S]
        if mg and "SPY" in S:
            spy_up = S["SPY"] > S["SPY"].rolling(200).mean()
            for n in (63, 126, 252):
                mom = S[mg].pct_change(n, fill_method=None)
                W = rank_weights(mom, 3, 0, 21).mul(spy_up.astype(float), axis=0)
                strat[f"S2 mega-cap mom {n}g"] = ("S2 mega-cap momentum (SPY>SMA200)",
                                                  run_weights(W, Rs[mg], sc).reindex(idx).fillna(0.0),
                                                  run_weights(W, Rs[mg], sc_s).reindex(idx).fillna(0.0))
        if "GLD" in S:
            for n in (50, 100, 200):
                W = (S[["GLD"]] > S[["GLD"]].rolling(n).mean()).astype(float)
                strat[f"S3 altın SMA{n}"] = ("S3 altın trendi",
                                             run_weights(W, Rs[["GLD"]], sc).reindex(idx).fillna(0.0),
                                             run_weights(W, Rs[["GLD"]], sc_s).reindex(idx).fillna(0.0))
        if "SPY" in S:
            strat["(referans) SPY al-tut"] = ("referans", Rs["SPY"].reindex(idx).fillna(0.0),
                                             Rs["SPY"].reindex(idx).fillna(0.0))

    br = bot_r.reindex(idx).fillna(0.0)
    worst = br.nsmallest(10).index

    def combo(r, w):
        """aylık yeniden dengelenen (1-w) bot + w strateji."""
        out, vb, vs = [], 1 - w, w
        for d in idx:
            if d.day == 1:
                tot = vb + vs
                vb, vs = tot * (1 - w), tot * w
            vb *= 1 + br.loc[d]
            vs *= 1 + r.loc[d]
            out.append(vb + vs)
        return pd.Series(out, index=idx).pct_change().fillna(0.0)

    base = {lab: stats(br, a, z) for lab, a, z in periods}
    base_all = stats(br, A, Z)
    rows = []
    for name, (fam, r, rs) in strat.items():
        row = {"aile": fam, "strateji": name}
        for lab, a, z in periods:
            s = stats(r, a, z)
            row[f"{lab} son"], row[f"{lab} maxDD%"] = s["final"], s["mdd"]
            row[f"{lab} stres son"] = stats(rs, a, z)["final"]
        s = stats(r, A, Z)
        row.update({"TÜM son": s["final"], "CAGR%": s["cagr"], "maxDD%": s["mdd"],
                    "botla korelasyon": float(np.corrcoef(r, br)[0, 1]),
                    "botun en kötü 10 gününde ort. %": float(r.loc[worst].mean() * 100)})
        for lab, a, z in years:
            row[f"{lab} %"] = stats(r, a, z)["ret"]
        row["tek başına 3 dönem +"] = "✓" if all(row[f"{lab} son"] > 100 for lab in PL) else ""
        for w in (0.2, 0.33):
            cr = combo(r, w)
            ok = []
            for lab, a, z in periods:
                s = stats(cr, a, z)
                row[f"%{w * 100:.0f} pay {lab} son"], row[f"%{w * 100:.0f} pay {lab} Calmar"] = s["final"], s["calmar"]
                ok.append(s["calmar"] > base[lab]["calmar"])
            s = stats(cr, A, Z)
            row[f"%{w * 100:.0f} pay TÜM son"], row[f"%{w * 100:.0f} pay maxDD%"] = s["final"], s["mdd"]
            row[f"%{w * 100:.0f} pay CAGR%"] = s["cagr"]
            row[f"%{w * 100:.0f} pay 3 dönem Calmar↑"] = "✓" if all(ok) else ""
        rows.append(row)
        print(f"{name}: tüm {row['TÜM son']:.1f} | " + " | ".join(f"{lab} {row[f'{lab} son']:.1f}" for lab in PL) +
              f" | kor {row['botla korelasyon']:+.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "sonuclar.csv"), index=False, float_format="%.4g")

    # her ailede 2021-23'e göre seçilen varyant (Calmar)
    pick = []
    for fam, g in df[df["aile"] != "referans"].groupby("aile"):
        cal = []
        for _, rr in g.iterrows():
            s = stats(strat[rr["strateji"]][1], A, V)
            cal.append(s["calmar"] if np.isfinite(s["calmar"]) else -1e9)
        pick.append(g.iloc[int(np.argmax(cal))]["strateji"])
    sel = df[df["strateji"].isin(pick)]
    accept = sel[(sel["tek başına 3 dönem +"] == "✓") &
                 ((sel["%20 pay 3 dönem Calmar↑"] == "✓") | (sel["%33 pay 3 dönem Calmar↑"] == "✓"))]

    solo = ["strateji", "TÜM son", "CAGR%", "maxDD%"] + [f"{lab} son" for lab in PL] + \
        [f"{lab} stres son" for lab in PL] + ["botla korelasyon", "botun en kötü 10 gününde ort. %", "tek başına 3 dönem +"]
    comb = ["strateji"] + [f"%{p} pay {k}" for p in (20, 33) for k in ["TÜM son", "CAGR%", "maxDD%"] +
                           [f"{lab} Calmar" for lab in PL] + ["3 dönem Calmar↑"]]
    txt = ["ÜÇÜNCÜ STRATEJİ ARAŞTIRMASI — bota ek, düşük korelasyonlu getiri kaynağı",
           "VERİ: SENTETİK" if args.synthetic else
           f"Kripto: {P.shape[1]} coin (Binance USDT-M, funding dahil) | Hisse/emtia: Yahoo günlük",
           "Bot v8 (MEXC gerçek komisyon) günlük özsermayesi referans. 100 USDT her dönem ayrı.",
           "BOT: " + " | ".join(f"{lab}: son {base[lab]['final']:.1f}, maxDD %{base[lab]['mdd']:.1f}, Calmar "
                                f"{base[lab]['calmar']:.2f}" for lab in PL) +
           f" | TÜM: {base_all['final']:.1f}, CAGR %{base_all['cagr']:.1f}, maxDD %{base_all['mdd']:.1f}", "",
           "=== TEK BAŞINA (normal maliyet; 'stres' = yüksek komisyon) ===",
           df[solo].round(2).to_string(index=False), "",
           "Yıllık getiri %:", df[["strateji"] + [f"{y[0]} %" for y in years]].round(1).to_string(index=False), "",
           "=== BOTA EKLENİNCE (aylık dengelenen %20 / %33 pay; Calmar = CAGR/maxDD; ✓ = üç dönemde de bottan iyi) ===",
           df[comb].round(2).to_string(index=False), "",
           "Her aileden YALNIZ 2021-23'e göre seçilen varyant: " + ", ".join(pick),
           "KABUL (tek başına 3 dönem + VE bota eklenince 3 dönem Calmar↑): " +
           (", ".join(accept["strateji"]) if len(accept) else "(yok)")]
    text = "\n".join(txt)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
