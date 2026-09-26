from mastbound.analysis import DAY, Trade, analyze, card_text
from mastbound.chain import SOL, to_trades

ME = "Me111111111111111111111111111111111111111"
TOK = "Tok11111111111111111111111111111111111111"


def series(start_day, prices):
    return {start_day + i: p for i, p in enumerate(prices)}


def test_early_sell_and_panic():
    d0 = 20000
    # fiyat 1.0'dan 0.8'e düşüyor (3 günde %20), kullanıcı 0.8'den satıyor, sonra 1.6'ya çıkıyor
    px = series(d0, [1.0, 0.95, 0.9, 0.8, 1.0, 1.3, 1.6, 1.5] + [1.4] * 30)
    t = Trade((d0 + 3) * DAY + 100, TOK, "sell", 1000, 800, "TOK")
    r = analyze([t], {TOK: px})
    assert r.early_sells == 1 and round(r.missed_usd) == 800      # (1.6-0.8)*1000
    assert r.panic_sells == 1
    assert r.score > 0 and "TOK" in card_text(ME, r)


def test_fomo_buy():
    d0 = 20000
    px = series(d0, [1.0, 1.2, 1.5, 1.4, 1.1, 0.9, 1.0] + [1.0] * 10)
    t = Trade((d0 + 3) * DAY, TOK, "buy", 100, 140, "TOK")          # 1.40'tan aldı, 3 gün önce 1.0
    r = analyze([t], {TOK: px})
    assert r.fomo_buys == 1 and r.early_sells == 0


def test_calm_trader_scores_zero():
    d0 = 20000
    px = series(d0, [1.0] * 40)
    trades = [Trade((d0 + 5) * DAY, TOK, "buy", 10, 10), Trade((d0 + 10) * DAY, TOK, "sell", 10, 10)]
    assert analyze(trades, {TOK: px}).score == 0


def test_to_trades_sol_sell():
    d = 20000
    tx = {"timestamp": d * DAY + 50, "tokenTransfers": [
        {"fromUserAccount": ME, "toUserAccount": "pool", "mint": TOK, "tokenAmount": 500}],
        "nativeTransfers": [{"fromUserAccount": "pool", "toUserAccount": ME, "amount": 2_000_000_000}]}
    tr = to_trades(ME, [tx], {d: 150.0})
    assert len(tr) == 1 and tr[0].side == "sell" and tr[0].amount == 500 and tr[0].usd == 300.0


def test_to_trades_usdc_buy_and_wsol():
    d = 20000
    usdc = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    tx = {"timestamp": d * DAY, "tokenTransfers": [
        {"fromUserAccount": ME, "toUserAccount": "pool", "mint": usdc, "tokenAmount": 50},
        {"fromUserAccount": "pool", "toUserAccount": ME, "mint": TOK, "tokenAmount": 1000}]}
    tr = to_trades(ME, [tx], {})
    assert tr[0].side == "buy" and tr[0].usd == 50
    tx2 = {"timestamp": d * DAY, "tokenTransfers": [
        {"fromUserAccount": ME, "toUserAccount": "pool", "mint": SOL, "tokenAmount": 1},
        {"fromUserAccount": "pool", "toUserAccount": ME, "mint": TOK, "tokenAmount": 10}]}
    assert to_trades(ME, [tx2], {d: 100.0})[0].usd == 100.0
