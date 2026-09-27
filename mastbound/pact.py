"""Ulysses Pact (v1, kontratsız): kullanıcı bir token'ı N gün satmayacağına söz verir; bot fırtınada hatırlatır.

Fon kilitlenmez (emanet yok, akıllı kontrat yok). Bot saatte bir bağlı cüzdanın bakiyesine ve fiyata bakar:
  - fiyat söz başından %DROP düştüyse → "fırtına" hatırlatması (günde en fazla bir)
  - bakiye başlangıcın yarısının altına indiyse → söz bozuldu
  - süre dolduysa → söz tutuldu
Gerçek kilit (kontrat + %0,15 ücret) denetimden sonra v2'de gelir.
"""
import time

DAY = 86400
DROP = 0.15
BROKEN_BELOW = 0.5          # bakiye başlangıcın bu oranının altına inerse söz bozulmuş sayılır
MAX_ACTIVE = {"free": 1, "standard": 3, "pro": 10}


def new_pact(mint, symbol, days, price, balance, now=None):
    now = int(time.time() if now is None else now)
    return {"mint": mint, "symbol": symbol, "start": now, "end": now + int(days * DAY), "days": days,
            "price": price, "balance": balance, "status": "active", "alerted": None}


def evaluate(p, now, price, balance):
    """Sözün durumunu günceller; kullanıcıya gidecek mesajı (ya da None) döner."""
    if p["status"] != "active":
        return None
    sym, until = p["symbol"], time.strftime("%b %d", time.gmtime(p["end"]))
    if balance is not None and balance < p["balance"] * BROKEN_BELOW:
        p["status"] = "broken"
        return (f"🧜 The sirens won this time — your {sym} balance dropped below half of what you tied to the mast.\n"
                "No judgment: run /regret on your wallet in a few weeks and see what the storm cost.")
    if now >= p["end"]:
        p["status"] = "kept"
        return f"⚓ Pact kept! You held {sym} through {p['days']} days without jumping ship. Iron hands."
    if price and p["price"] and price <= p["price"] * (1 - DROP) and (p.get("alerted") is None or now - p["alerted"] >= DAY):
        p["alerted"] = now
        down = (1 - price / p["price"]) * 100
        return (f"🌊 Storm alert: {sym} is down {down:.0f}% since your pact.\n"
                f"You promised yourself to hold until {until}. Your calm self tied you to the mast for moments "
                "exactly like this one. Breathe before you click sell.")
    return None


def summary(pacts, now=None):
    now = time.time() if now is None else now
    lines = []
    for p in pacts:
        if p["status"] == "active":
            left = max(0, (p["end"] - now) / DAY)
            lines.append(f"⛓ {p['symbol']} — {left:.0f} days left")
        else:
            lines.append(f"{'✅' if p['status'] == 'kept' else '💔'} {p['symbol']} — {p['status']}")
    return lines
