"""Orderbook scalp testi: ilk 5 seviyenin alış/satış dengesizliği (ve mikro-fiyat) sonraki 1–60 saniyeyi, gecikme ve
masraftan sonra kâr edecek kadar tahmin ediyor mu?

Veri: Tardis.dev ücretsiz günleri (her ayın 1'i, API anahtarı gerekmez) — Binance USDT-M vadeli `book_snapshot_5`
(her orderbook değişiminde ilk 5 seviye). Kitap 100 ms'lik ızgaraya indirgenir (her dilimin son hali).

Özellikler (t anında bilinen):
  IMB1  = (alış1 − satış1) / (alış1 + satış1)              en iyi seviye miktar dengesizliği
  IMB5  = aynısı ilk 5 seviyenin toplamıyla
  MICRO = (mikro-fiyat − orta fiyat) / orta, bps           mikro = (ask·bidQ + bid·askQ) / (bidQ + askQ)
Eşikler ileriye bakmadan: gün k'nın eşikleri aynı coinin bir önceki gününün dağılımından (yüzdelik) alınır.

İşlem modelleri (gecikme L: sinyal t'de, emir t+L'de):
  TAKER: t+L'de karşı fiyattan (ask/bid) gir, h sonra karşı fiyattan çık. Masraf: 2 × taker ücreti.
  MAKER: t+L'de kendi tarafın en iyi fiyatına limit koy; ancak fiyat içinden geçerse (long: ask ≤ limitin) dolmuş say
         (muhafazakâr), W saniye içinde dolmazsa iptal; dolduktan h sonra taker ile çık. Masraf: 0 + taker ücreti.
Sonuçlar işlem başına bps (brüt = masrafsız, net = ücret sonrası; spread geçişi her ikisinde de fiyata dahil).

  python research/orderbook_scalp.py              # indir + analiz → results/orderbook_scalp/report.txt
  python research/orderbook_scalp.py --synthetic  # kod testi (sahte kitap, içine küçük bir etki gömülü)
"""
import argparse
import gzip
import os
import time

import numpy as np
import pandas as pd
import requests

URL = "https://datasets.tardis.dev/v1/binance-futures/book_snapshot_5/{y}/{m:02d}/01/{s}.csv.gz"
SYMBOLS = os.environ.get("OB_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
MONTHS = pd.period_range(os.environ.get("OB_FROM", "2025-10"), os.environ.get("OB_TO", "2026-09"), freq="M")
GRID_MS = 100
HORIZONS_S = [1, 5, 30, 60]
LATENCIES_MS = [0, 100, 500, 1000]
QUANTS = [0.99, 0.995, 0.999]      # sinyal eşiği: özelliğin üst/alt yüzdeliği (bir önceki günden)
TAKER = 0.0002                    # MEXC vadeli taker ücreti %0.02 (tek yön); maker %0
MAKER_WAIT_S = 5
FEATS = ["IMB1", "IMB5", "MICRO"]
OUT = "results/orderbook_scalp"


# ------------------------------------------------------------------ veri
def download(sym, p, cache):
    path = os.path.join(cache, f"{sym}-{p}.csv.gz")
    if os.path.exists(path):
        return path
    url = URL.format(y=p.year, m=p.month, s=sym)
    for i in range(4):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                if r.status_code in (401, 403, 404):
                    print(f"{sym} {p}: HTTP {r.status_code} — atlandı", flush=True)
                    return None
                r.raise_for_status()
                with open(path + ".part", "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            os.replace(path + ".part", path)
            return path
        except requests.RequestException as e:
            print(f"{sym} {p}: {e} (deneme {i + 1})", flush=True)
            time.sleep(5 * (i + 1))
    return None


def grid(path):
    """book_snapshot_5 → 100 ms ızgara: bid/ask fiyatı + özellikler (float32)."""
    cols = ["local_timestamp"] + [f"{s}[{i}].{k}" for i in range(5) for s in ("asks", "bids") for k in ("price", "amount")]
    parts = []
    for ch in pd.read_csv(path, usecols=cols, chunksize=2_000_000, dtype="float64"):
        ch["g"] = (ch["local_timestamp"] // (GRID_MS * 1000)).astype("int64")
        parts.append(ch.groupby("g").last())
    d = pd.concat(parts)
    d = d[~d.index.duplicated(keep="last")]
    full = np.arange(d.index.min(), d.index.max() + 1)
    d = d.reindex(full).ffill()
    return features(d)


def features(d):
    a0, b0 = d["asks[0].price"].to_numpy(), d["bids[0].price"].to_numpy()
    aq0, bq0 = d["asks[0].amount"].to_numpy(), d["bids[0].amount"].to_numpy()
    aq5 = sum(d[f"asks[{i}].amount"].to_numpy() for i in range(5))
    bq5 = sum(d[f"bids[{i}].amount"].to_numpy() for i in range(5))
    mid = (a0 + b0) / 2
    micro = (a0 * bq0 + b0 * aq0) / (bq0 + aq0)
    ok = (a0 > b0) & np.isfinite(mid)
    out = pd.DataFrame({"ask": a0, "bid": b0,
                        "IMB1": (bq0 - aq0) / (bq0 + aq0), "IMB5": (bq5 - aq5) / (bq5 + aq5),
                        "MICRO": (micro - mid) / mid * 1e4}, index=d.index)
    return out[ok].astype("float32").reindex(d.index).ffill()


def synthetic_day(seed):
    """Sahte kitap: IMB1 sonraki ~2 saniyenin orta fiyat hareketine küçük katkı yapar."""
    rng = np.random.default_rng(seed)
    n = 864_000 // 4
    e = rng.normal(0, 0.14, n)
    imb = np.zeros(n)
    for i in range(1, n):                         # kalıcı dengesizlik (AR(1)), sonraki adımların getirisine katkı
        imb[i] = 0.99 * imb[i - 1] + e[i]
    imb = np.clip(imb, -0.95, 0.95)
    mid = 50000 * np.exp(np.cumsum(rng.normal(0, 0.00004, n) + 0.00002 * np.r_[0, imb[:-1]]))
    spread = 0.1
    bq0 = rng.lognormal(1, 0.5, n)
    aq0 = bq0 * (1 - imb) / (1 + imb)
    d = {"asks[0].price": mid + spread / 2, "bids[0].price": mid - spread / 2,
         "asks[0].amount": aq0, "bids[0].amount": bq0}
    for i in range(1, 5):
        d[f"asks[{i}].price"], d[f"bids[{i}].price"] = mid + spread / 2 + i * 0.1, mid - spread / 2 - i * 0.1
        d[f"asks[{i}].amount"], d[f"bids[{i}].amount"] = rng.lognormal(1, 0.5, n), rng.lognormal(1, 0.5, n)
    return features(pd.DataFrame(d, index=np.arange(n)))


# ------------------------------------------------------------------ analiz
def ic_rows(g, lat_steps, h_steps):
    """Özellik ↔ orta fiyat getirisi (t+L → t+L+h) korelasyonu, alt örneklenmiş (örtüşmeyi azaltmak için)."""
    mid = ((g["ask"] + g["bid"]) / 2).to_numpy("float64")
    out = {}
    for f in FEATS:
        x = g[f].to_numpy("float64")
        for L in lat_steps:
            for h in h_steps:
                step = max(h, 10)
                i = np.arange(0, len(mid) - L - h, step)
                r = mid[i + L + h] / mid[i + L] - 1
                m = np.isfinite(x[i]) & np.isfinite(r)
                out[(f, L, h)] = (np.corrcoef(x[i][m], r[m])[0, 1] if m.sum() > 100 else np.nan, m.sum())
    return out


def trades(g, feat, lo, hi, L, h, mode):
    """Eşik aşımında işlem; örtüşmesiz. Dönüş: brüt getiriler (spread dahil, ücret hariç)."""
    x = g[feat].to_numpy("float64")
    ask, bid = g["ask"].to_numpy("float64"), g["bid"].to_numpy("float64")
    n = len(x)
    sig = np.where(x >= hi, 1, np.where(x <= lo, -1, 0))
    idx = np.flatnonzero(sig)
    res, free_at, W = [], -1, MAKER_WAIT_S * 1000 // GRID_MS
    for i in idx:
        if i < free_at:
            continue
        s, e = sig[i], i + L
        if mode == "taker":
            x_ = e + h
            if x_ >= n:
                break
            r = (bid[x_] / ask[e] - 1) if s > 0 else (1 - ask[x_] / bid[e])
            res.append(r)
            free_at = x_
        else:
            if e + W + h >= n:
                break
            lim = bid[e] if s > 0 else ask[e]
            win = ask[e + 1:e + 1 + W] if s > 0 else bid[e + 1:e + 1 + W]
            hit = np.flatnonzero(win <= lim) if s > 0 else np.flatnonzero(win >= lim)
            if not len(hit):
                free_at = e + W
                continue
            f_ = e + 1 + hit[0]
            x_ = f_ + h
            r = (bid[x_] / lim - 1) if s > 0 else (1 - ask[x_] / lim)
            res.append(r)
            free_at = x_
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cache = os.path.join(OUT, "cache")
    os.makedirs(cache, exist_ok=True)

    lat_steps = [L // GRID_MS for L in LATENCIES_MS]
    h_steps = [h * 1000 // GRID_MS for h in HORIZONS_S]
    IC = {}                     # (sym, day) → ic dict
    T = {}                      # (feat, q, L, h, mode) → [brüt getiriler]
    spreads = {}
    days_used = []
    syms = ["SYN"] if a.synthetic else SYMBOLS
    months = pd.period_range("2026-01", "2026-04", freq="M") if a.synthetic else MONTHS
    for sym in syms:
        prev = None
        for k, p in enumerate(months):
            t0 = time.time()
            if a.synthetic:
                g = synthetic_day(k)
            else:
                path = download(sym, p, cache)
                if path is None:
                    continue
                size = os.path.getsize(path) / 1e6
                g = grid(path)
                os.remove(path)                      # diski doldurmayalım
            spreads.setdefault(sym, []).append(float(np.nanmedian((g["ask"] - g["bid"]) / g["bid"] * 1e4)))
            IC[(sym, str(p))] = ic_rows(g, lat_steps, h_steps)
            if prev is not None:                     # eşikler bir önceki günden → ileriye bakma yok
                for f in FEATS:
                    for q in QUANTS:
                        lo, hi = np.nanquantile(prev[f], 1 - q), np.nanquantile(prev[f], q)
                        for L in lat_steps:
                            for h in h_steps:
                                T.setdefault((f, q, L, h, "taker"), []).extend(trades(g, f, lo, hi, L, h, "taker"))
                                if L == lat_steps[1] or L == lat_steps[0]:
                                    T.setdefault((f, q, L, h, "maker"), []).extend(trades(g, f, lo, hi, L, h, "maker"))
                days_used.append(f"{sym} {p}")
            prev = g[FEATS].copy()
            del g
            print(f"{sym} {p}: tamam ({time.time() - t0:.0f} sn"
                  + ("" if a.synthetic else f", {size:.0f} MB") + ")", flush=True)

    L_ = [f"ORDERBOOK SCALP testi — {pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M} UTC" + ("  [SENTETİK]" if a.synthetic else ""),
          f"veri: Binance vadeli book_snapshot_5, 100 ms ızgara; işlem testi günleri: {len(days_used)} ({', '.join(days_used)})",
          f"ücret: taker %{TAKER * 100:.2f}/yön, maker %0; spread geçişi fiyata dahil. Eşik: önceki günün yüzdeliği.",
          "medyan spread (bps): " + ", ".join(f"{s} {np.mean(v):.2f}" for s, v in spreads.items()), ""]

    L_.append("== 1) TAHMİN GÜCÜ — IC ×100 (tüm coin-günlerin ortalaması), satır: özellik/gecikme, sütun: ufuk")
    L_.append(f"{'':<16}" + "".join(f"{h:>8}s" for h in HORIZONS_S))
    for f in FEATS:
        for L, Lms in zip(lat_steps, LATENCIES_MS):
            vals = []
            for h in h_steps:
                v = [d[(f, L, h)][0] for d in IC.values() if np.isfinite(d[(f, L, h)][0])]
                vals.append(f"{np.mean(v) * 100:+8.2f}" if v else "     nan")
            L_.append(f"{f:<6} gecikme {Lms:>4}ms" + "".join(f"{x:>9}" for x in vals))
    L_.append("")

    L_.append("== 2) İŞLEM KURALI — işlem başı bps: n / brüt (spread dahil, ücretsiz) / net (ücret sonrası) / t(net)")
    best = []
    for mode in ("taker", "maker"):
        L_.append(f"-- {mode.upper()}" + (f" (limit {MAKER_WAIT_S} sn bekler, içinden geçerse dolmuş sayılır)" if mode == "maker" else ""))
        for (f, q, L, h, md), r in sorted(T.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
            if md != mode or len(r) < 5:
                continue
            r = np.array(r)
            fee = 2 * TAKER if mode == "taker" else TAKER
            net = r - fee
            t = net.mean() / net.std() * np.sqrt(len(net)) if net.std() > 0 else 0
            L_.append(f"{f:<5} q{q:<6} gecikme {L * GRID_MS:>4}ms ufuk {h * GRID_MS // 1000:>3}s | {len(r):>6} "
                      f"{r.mean() * 1e4:+7.2f} {net.mean() * 1e4:+7.2f} t{t:+6.1f}")
            best.append((net.mean(), t, len(r), f, q, L * GRID_MS, h * GRID_MS // 1000, mode))
    L_.append("")
    best.sort(reverse=True)
    L_.append("EN İYİ 8 (net bps, t, n): " + "; ".join(
        f"{m} {f} q{q} {lat}ms {h}s {n_ * 1e4:+.2f} (t{t:+.1f}, n{c})" for n_, t, c, f, q, lat, h, m in best[:8]))
    L_.append("")
    L_.append("OKUMA: Scalp ancak 100–500 ms gecikmede net bps belirgin pozitif ve t>3 ise gerçekçi. Kaldıraç kazancı değil,")
    L_.append("oynaklığı büyütür; net negatifken kaldıraç sadece kaybı hızlandırır. Veri Binance; MEXC'de spread/derinlik daha zayıf.")
    txt = "\n".join(L_)
    print(txt)
    open(os.path.join(OUT, "report.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
