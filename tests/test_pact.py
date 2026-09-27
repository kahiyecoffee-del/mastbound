from mastbound import pact

DAY = pact.DAY


def test_storm_alert_once_per_day_then_kept():
    p = pact.new_pact("M", "WIF", 10, price=2.0, balance=100, now=0)
    assert pact.evaluate(p, 3600, 1.9, 100) is None                    # -5%: sakin
    assert "Storm alert" in pact.evaluate(p, 7200, 1.6, 100)           # -20%
    assert pact.evaluate(p, 7200 + 3600, 1.5, 100) is None             # aynı gün tekrar yok
    assert "Storm alert" in pact.evaluate(p, 7200 + DAY, 1.5, 100)
    assert "Pact kept" in pact.evaluate(p, 10 * DAY, 1.5, 100) and p["status"] == "kept"
    assert pact.evaluate(p, 11 * DAY, 1.0, 0) is None


def test_selling_breaks_pact():
    p = pact.new_pact("M", "BONK", 30, price=1.0, balance=100, now=0)
    assert pact.evaluate(p, DAY, 1.0, 60) is None                       # biraz sattı: tolerans
    assert "sirens won" in pact.evaluate(p, 2 * DAY, 1.0, 40) and p["status"] == "broken"


def test_summary():
    ps = [pact.new_pact("M", "WIF", 10, 1, 1, now=0), {**pact.new_pact("N", "BONK", 5, 1, 1, now=0), "status": "kept"}]
    lines = pact.summary(ps, now=DAY)
    assert lines[0].startswith("⛓ WIF — 9") and "kept" in lines[1]
