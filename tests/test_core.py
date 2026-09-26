from mastbound.analysis import DAY, HOUR, Trade, analyze, card_text
from mastbound.chain import SOL, to_trades

ME = "Me111111111111111111111111111111111111111"
TOK = "Tok11111111111111111111111111111111111111"
T0 = 20000 * DAY


def daily(prices, start=T0):
    return {start + i * DAY: p for i, p in enumerate(prices)}


def test_early_sell_panic_and_pnl():
    px = daily([1.0, 0.95, 0.9, 0.8, 1.0, 1.3, 1.6, 1.5] + [1.4] * 30)
    buy = Trade(T0 - 10 * DAY, TOK, "buy", 1000, 1000, "TOK")        # 1.0'dan aldı
    sell = Trade(T0 + 3 * DAY + 100, TOK, "sell", 1000, 800, "TOK")  # 0.8'den sattı
    r = analyze([buy, sell], {TOK: {**daily([1.0] * 10, T0 - 10 * DAY), **px}})
    assert r.early_sells == 1 and round(r.missed_usd) == 800         # (1.6-0.8)*1000
    assert r.panic_sells == 1
    assert round(r.realized_pnl) == -200 and r.closed == 1 and r.wins == 0
    assert "TOK" in card_text(ME, r)


def test_fomo_buy():
    px = daily([1.0, 1.2, 1.5, 1.4, 1.1, 0.9, 1.0] + [1.0] * 10)
    r = analyze([Trade(T0 + 3 * DAY, TOK, "buy", 100, 140, "TOK")], {TOK: px})
    assert r.fomo_buys == 1 and r.early_sells == 0


def test_calm_trader_scores_zero():
    px = daily([1.0] * 40)
    trades = [Trade(T0 + d * DAY, TOK, s, 10, 10) for d, s in ((5, "buy"), (10, "sell"), (12, "buy"), (15, "sell"))]
    r = analyze(trades, {TOK: px})
    assert r.score == 0 and r.measured == 4


def test_recent_trades_are_unmeasured_not_zero():
    px = daily([1.0] * 5)
    last = max(px)
    r = analyze([Trade(last - 1800, TOK, "sell", 10, 10), Trade(last - HOUR, TOK, "buy", 10, 10)], {TOK: px})
    assert r.unmeasured == 2 and r.score is None
    assert "yeterli veri yok" in card_text(ME, r)


def test_intraday_hourly_regret():
    # saatlik seri: satıştan 3 saat sonra fiyat iki katına çıkıyor, 12 saat veri var
    px = {T0 + h * HOUR: (2.0 if h >= 3 else 1.0) for h in range(0, 13)}
    r = analyze([Trade(T0 + 30 * 60, TOK, "sell", 100, 100, "TOK")], {TOK: px})
    assert r.early_sells == 1 and round(r.missed_usd) == 100


def test_to_trades_sol_sell():
    tx = {"timestamp": T0 + 50, "tokenTransfers": [
        {"fromUserAccount": ME, "toUserAccount": "pool", "mint": TOK, "tokenAmount": 500}],
        "nativeTransfers": [{"fromUserAccount": "pool", "toUserAccount": ME, "amount": 2_000_000_000}]}
    tr = to_trades(ME, [tx], {T0: 150.0})
    assert len(tr) == 1 and tr[0].side == "sell" and tr[0].amount == 500 and tr[0].usd == 300.0


def test_to_trades_usdc_buy_and_wsol():
    usdc = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    tx = {"timestamp": T0, "tokenTransfers": [
        {"fromUserAccount": ME, "toUserAccount": "pool", "mint": usdc, "tokenAmount": 50},
        {"fromUserAccount": "pool", "toUserAccount": ME, "mint": TOK, "tokenAmount": 1000}]}
    tr = to_trades(ME, [tx], {})
    assert tr[0].side == "buy" and tr[0].usd == 50
    tx2 = {"timestamp": T0 + 10, "tokenTransfers": [
        {"fromUserAccount": ME, "toUserAccount": "pool", "mint": SOL, "tokenAmount": 1},
        {"fromUserAccount": "pool", "toUserAccount": ME, "mint": TOK, "tokenAmount": 10}]}
    assert to_trades(ME, [tx2], {T0: 100.0})[0].usd == 100.0


def test_card_png():
    from mastbound.analysis import Report
    from mastbound.card import fmt_price, render
    png = render(ME, Report(trades=3, measured=3, score=50, worst=[(10.0, "TOK", 0.0000212, 0.00003)]))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert fmt_price(0.0000212) == "$0.0000212" and fmt_price(1.82) == "$1.82"
    assert render(ME, Report(trades=1, score=None))[:4] == b"\x89PNG"
