"""Backtest: canlı motorla aynı strateji; kararlar mum kapanışında, dolum bir sonraki mumun açılışında.

Borsadaki sabit stop mum içinde tetiklenir (gap varsa açılış fiyatından). Ücret + kayma her iki tarafta.

  python -m quant.backtest --symbol BTC_USDT --interval Min15 --days 180
  python -m quant.backtest --symbol ETH_USDT --fast 12 --slow 26 --trend 0 --no-short
"""
import argparse
import time
from dataclasses import asdict

from .strategy import Params, decide, indicators


def run(candles, p, equity=1000.0, risk_pct=1.0, leverage=3, max_margin_pct=25, fee_bps=2.0, slip_bps=2.0):
    ind = indicators(candles, p)
    cost = (fee_bps + slip_bps) / 1e4
    eq, peak_eq, max_dd = equity, equity, 0.0
    pos, trades, pending = None, [], None

    def close_pos(t, px, why):
        nonlocal eq, pos
        sgn = 1 if pos["side"] == "long" else -1
        pnl = sgn * (px - pos["entry"]) * pos["qty"] - cost * px * pos["qty"]
        eq += pnl
        trades.append({"side": pos["side"], "t_in": pos["t"], "t_out": t, "entry": pos["entry"], "exit": px,
                       "pnl": pnl, "reason": why})
        pos = None

    for i, (t, o, h, l, c, _) in enumerate(candles):
        # 1) önceki mumun kararı bu mumun açılışında uygulanır
        if pending:
            d, pending = pending, None
            if d.action == "close" and pos:
                close_pos(t, o, d.reason)
            elif d.action in ("open_long", "open_short") and not pos:
                side = "long" if d.action == "open_long" else "short"
                stop = d.stop + (o - candles[i - 1][4])          # stop mesafesi dolum fiyatına taşınır
                dist = abs(o - stop)
                qty = min(eq * risk_pct / 100 / dist, eq * max_margin_pct / 100 * leverage / o) if dist > 0 else 0
                if qty > 0:
                    eq -= cost * o * qty
                    pos = {"side": side, "entry": o, "qty": qty, "stop": stop, "peak": o, "t": t,
                           "atr0": ind.atr[i - 1]}
        # 2) borsadaki sabit stop mum içinde
        if pos:
            s = pos["stop"]
            if pos["side"] == "long" and l <= s:
                close_pos(t, min(o, s), "borsa stop")
            elif pos["side"] == "short" and h >= s:
                close_pos(t, max(o, s), "borsa stop")
        # 3) kapanışta karar
        d = decide(candles, ind, i, pos, p)
        if d.action != "hold":
            pending = d
        mark = eq + (((c - pos["entry"]) if pos["side"] == "long" else (pos["entry"] - c)) * pos["qty"] if pos else 0)
        peak_eq = max(peak_eq, mark)
        max_dd = max(max_dd, 1 - mark / peak_eq)

    if pos:
        close_pos(candles[-1][0], candles[-1][4], "test sonu")
    return summary(trades, equity, eq, max_dd)


def summary(trades, start, end, max_dd):
    wins = [x["pnl"] for x in trades if x["pnl"] > 0]
    losses = [-x["pnl"] for x in trades if x["pnl"] <= 0]
    return {"trades": len(trades), "win_rate": len(wins) / len(trades) if trades else 0.0,
            "return_pct": (end / start - 1) * 100, "max_dd_pct": max_dd * 100,
            "profit_factor": sum(wins) / sum(losses) if losses else float("inf") if wins else 0.0,
            "end_equity": end, "list": trades}


def main():
    ap = argparse.ArgumentParser(description="EMA+ATR backtest (MEXC vadeli mumları)")
    ap.add_argument("--symbol", default="BTC_USDT")
    ap.add_argument("--interval", default="Min15")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--fast", type=int, default=20)
    ap.add_argument("--slow", type=int, default=50)
    ap.add_argument("--trend", type=int, default=200)
    ap.add_argument("--atr", type=int, default=14)
    ap.add_argument("--stop-atr", type=float, default=2.0)
    ap.add_argument("--trail-atr", type=float, default=3.0)
    ap.add_argument("--no-short", action="store_true")
    ap.add_argument("--risk-pct", type=float, default=1.0)
    ap.add_argument("--leverage", type=int, default=3)
    ap.add_argument("--fee-bps", type=float, default=2.0, help="taraf başı taker ücreti (bps)")
    ap.add_argument("--slip-bps", type=float, default=2.0)
    ap.add_argument("--trades", action="store_true", help="işlemleri tek tek yaz")
    a = ap.parse_args()

    from .mexc import Mexc
    end = int(time.time())
    candles = Mexc().klines(a.symbol, a.interval, end - a.days * 86400, end)
    p = Params(a.fast, a.slow, a.trend, a.atr, a.stop_atr, a.trail_atr, not a.no_short)
    r = run(candles, p, risk_pct=a.risk_pct, leverage=a.leverage, fee_bps=a.fee_bps, slip_bps=a.slip_bps)
    print(f"{a.symbol} {a.interval} {a.days}g  mum={len(candles)}  {asdict(p)}")
    print(f"işlem={r['trades']}  kazanma={r['win_rate']:.0%}  getiri={r['return_pct']:+.2f}%  "
          f"maxDD={r['max_dd_pct']:.2f}%  PF={r['profit_factor']:.2f}")
    if a.trades:
        for x in r["list"]:
            print(f"  {time.strftime('%Y-%m-%d %H:%M', time.gmtime(x['t_in']))} {x['side']:5} "
                  f"{x['entry']:.4f} → {x['exit']:.4f}  {x['pnl']:+.2f}  {x['reason']}")


if __name__ == "__main__":
    main()
