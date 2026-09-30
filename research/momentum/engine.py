"""Tek bakiye, tek pozisyon portföy simülasyonu."""
import pandas as pd

import config as C

COIN_ORDER = {c: i for i, c in enumerate(C.COINS)}


def run_portfolio(candidates, breakeven=True):
    key = "be" if breakeven else "nobe"
    cands = sorted(candidates, key=lambda c: (c["entry_time"], COIN_ORDER.get(c["coin"], 99)))
    balance = C.INITIAL_BALANCE
    free_at = None                      # bir sonraki girişin yapılabileceği en erken 1dk açılışı
    trades, skipped = [], {"busy": 0, "liquidation": 0, "min_notional": 0, "bankrupt": 0}
    for c in cands:
        x = c[key]
        if free_at is not None and c["entry_time"] < free_at:
            skipped["busy"] += 1
            continue
        if c["stop"] <= c["liq"]:
            skipped["liquidation"] += 1
            continue
        if balance <= 0:
            skipped["bankrupt"] += 1
            continue
        notional = balance * C.LEVERAGE
        if notional < C.MIN_NOTIONAL.get(c["coin"], C.MIN_NOTIONAL_DEFAULT):
            skipped["min_notional"] += 1
            continue
        qty = notional / c["entry"]
        fee_in = notional * C.TAKER_FEE
        fee_out = qty * x["exit"] * C.TAKER_FEE
        funding = qty * x["funding_px"]
        gross = qty * (x["exit"] - c["entry"])
        pnl = gross - fee_in - fee_out - funding
        pnl = max(pnl, -balance)        # isolated marj: en fazla bakiye kadar kaybedilebilir
        risk_usdt = qty * c["R"]
        balance_before = balance
        balance += pnl
        # zaman çıkışı mumun AÇILIŞINDA olur; aynı mumda yeni giriş yapılabilir
        free_at = x["exit_time"] + (pd.Timedelta(0) if x["reason"] == "TIME"
                                    else pd.Timedelta(minutes=1))
        trades.append(dict(
            coin=c["coin"], setup_time=c["setup_time"], entry_time=c["entry_time"],
            entry=c["entry"], stop=c["stop"], tp=c["tp"], liq=c["liq"],
            exit_time=x["exit_time"], exit=x["exit"], reason=x["reason"],
            be_time=x["be_time"], touched_1r=x["touched_1r"],
            R_result=(x["exit"] - c["entry"]) / c["R"],
            R_net=pnl / risk_usdt, notional=notional, qty=qty,
            fees=fee_in + fee_out, funding=funding, pnl=pnl,
            balance_before=balance_before, balance=balance))
    return trades, skipped
