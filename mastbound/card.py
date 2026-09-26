"""Paylaşılabilir Pişmanlık Kartı (PNG, 1080×1350 — X/Instagram/Telegram için dikey).

Tema: gece denizi. Puan bir "fırtına ölçeri" (yarım daire gösterge + ibre) ile gösterilir; puana göre bir denizci
kişiliği (unvan + kısa cümle) ve çizilmiş bir tekne (sağlam gemi → kağıt tekne) verilir. Altta 2×2 sayı kutucukları
ve en büyük pişmanlıklar. Metin her zaman metin renginde; renk yalnız gösterge ve kâr/zarar yön işaretinde
(işaret + renk birlikte, renk tek başına anlam taşımaz).
"""
import io
import math
import os

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .analysis import MIN_SCORED

W, H = 1080, 1350
PAD = 64
SKY_TOP, SKY_BOTTOM = (13, 27, 48), (18, 58, 82)
INK = "#ffffff"
INK2 = "#cfe0ea"
INK3 = "#8fb0c2"
FOAM = (255, 255, 255)
GOOD, WARN, CRIT = "#2fd08a", "#fab219", "#ff6b6b"
GAUGE_TRACK = (255, 255, 255, 38)

PERSONAS = [  # (en yüksek puan, unvan, cümle)
    (15, "Iron Captain", "Sirens sing. You don't even blink."),
    (35, "Steady Sailor", "A little wobble, but you hold the wheel."),
    (55, "Wobbly Deckhand", "Half of your sells are the wind's idea."),
    (75, "Siren-Struck", "The sirens call and you jump ship."),
    (100, "Paper Boat", "One wave and you're gone. Tie yourself to the mast!"),
]

_FONT_DIRS = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu", "/usr/share/fonts/TTF"]


def _font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for d in _FONT_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


def compact_usd(v):
    a = abs(v)
    return f"${a / 1e6:.1f}M" if a >= 1e6 else f"${a / 1e3:.1f}K" if a >= 1e4 else f"${a:,.0f}"


def fmt_price(p):
    """0.0000212 gibi küçük fiyatlar bilimsel gösterim olmadan, 3 anlamlı basamakla."""
    if p <= 0:
        return "$0"
    dec = max(2, -int(math.floor(math.log10(p))) + 2)
    return f"${p:.{dec}f}"


def severity(score):
    return GOOD if score < 34 else WARN if score < 67 else CRIT


def persona(score):
    if score is None:
        return "Still at the Harbor", "Not enough voyages yet to judge your hands."
    for hi, title, line in PERSONAS:
        if score <= hi:
            return title, line
    return PERSONAS[-1][1:]


def _background():
    img = Image.new("RGB", (W, H))
    px = img.load()
    for y in range(H):
        t = y / (H - 1)
        c = tuple(int(SKY_TOP[i] + (SKY_BOTTOM[i] - SKY_TOP[i]) * t) for i in range(3))
        for x in range(W):
            px[x, y] = c
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    # yıldızlar (sabit tohum: her kartta aynı gökyüzü)
    seed = 7
    for _ in range(70):
        seed = (seed * 1103515245 + 12345) % 2 ** 31
        x = seed % W
        seed = (seed * 1103515245 + 12345) % 2 ** 31
        y = seed % 520
        r = 1 + (seed % 3 == 0)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, 90 + seed % 100))
    # ay
    d.ellipse([W - 250, 150, W - 170, 230], fill=(255, 244, 214, 230))
    d.ellipse([W - 228, 138, W - 150, 216], fill=SKY_TOP + (255,))
    return Image.alpha_composite(img.convert("RGBA"), layer)


def _waves(d, y0, amp, alpha, phase):
    pts = [(x, y0 + amp * math.sin(x / 55.0 + phase)) for x in range(0, W + 10, 10)]
    d.polygon(pts + [(W, H), (0, H)], fill=(20, 90, 120, alpha))
    d.line(pts, fill=FOAM + (min(255, alpha + 40),), width=3)


def _boat(d, cx, cy, score):
    """Puan düşükse direkli sağlam gemi, yüksekse kağıttan tekne."""
    if score is not None and score > 55:
        # kağıt tekne
        d.polygon([(cx - 90, cy), (cx + 90, cy), (cx + 55, cy + 42), (cx - 55, cy + 42)], fill=(245, 245, 240, 255))
        d.polygon([(cx - 90, cy), (cx, cy - 70), (cx + 90, cy)], fill=(225, 228, 230, 255))
        d.line([(cx, cy - 70), (cx, cy)], fill=(190, 195, 200, 255), width=3)
        return
    # sağlam gemi: gövde, direk, yelken, bayrak
    d.polygon([(cx - 100, cy), (cx + 100, cy), (cx + 70, cy + 45), (cx - 70, cy + 45)], fill=(140, 86, 52, 255))
    d.line([(cx - 95, cy + 10), (cx + 95, cy + 10)], fill=(110, 64, 38, 255), width=4)
    d.line([(cx, cy), (cx, cy - 150)], fill=(90, 58, 36, 255), width=7)
    d.polygon([(cx + 6, cy - 140), (cx + 85, cy - 30), (cx + 6, cy - 30)], fill=(250, 248, 240, 255))
    d.polygon([(cx - 6, cy - 120), (cx - 70, cy - 30), (cx - 6, cy - 30)], fill=(235, 232, 222, 255))
    d.polygon([(cx, cy - 150), (cx + 38, cy - 140), (cx, cy - 130)], fill=(57, 135, 229, 255))
    # direğe bağlanmış ip: Ulysses
    d.line([(cx - 12, cy - 70), (cx + 12, cy - 60)], fill=(230, 200, 140, 255), width=4)
    d.line([(cx - 12, cy - 55), (cx + 12, cy - 45)], fill=(230, 200, 140, 255), width=4)


def _gauge(d, cx, cy, radius, score):
    box = [cx - radius, cy - radius, cx + radius, cy + radius]
    d.arc(box, 180, 360, fill=GAUGE_TRACK, width=26)
    if score is None:
        return
    end = 180 + 180 * score / 100
    d.arc(box, 180, end, fill=severity(score), width=26)
    # ibre yerine yay üzerinde parlak bir işaret (sayıyla çakışmaz)
    ang = math.radians(end)
    mx, my = cx + radius * math.cos(ang), cy + radius * math.sin(ang)
    d.ellipse([mx - 22, my - 22, mx + 22, my + 22], fill=INK)
    d.ellipse([mx - 11, my - 11, mx + 11, my + 11], fill=severity(score))


def _tile(d, x, y, w, h, label, value, marker=None, marker_color=None):
    d.rounded_rectangle([x, y, x + w, y + h], radius=28, fill=(255, 255, 255, 26), outline=(255, 255, 255, 40), width=2)
    d.text((x + 28, y + 24), label, font=_font(26), fill=INK2)
    vx = x + 28
    if marker:
        d.text((vx, y + 66), marker, font=_font(50, True), fill=marker_color)
        vx += d.textlength(marker, font=_font(50, True)) + 10
    d.text((vx, y + 66), value, font=_font(50, True), fill=INK)


def render(addr, r):
    img = _background()
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    # başlık
    d.text((PAD, PAD), "⚓ MASTBOUND", font=_font(40, True), fill=INK)
    d.text((PAD, PAD + 54), "Regret Mirror", font=_font(28), fill=INK2)
    short = f"{addr[:4]}…{addr[-4:]}"
    sw = d.textlength(short, font=_font(26))
    d.rounded_rectangle([W - PAD - sw - 36, PAD + 4, W - PAD, PAD + 52], radius=24, fill=(255, 255, 255, 30))
    d.text((W - PAD - sw - 18, PAD + 13), short, font=_font(26), fill=INK2)

    # fırtına ölçeri + puan
    gx, gy, rad = 330, 470, 210
    _gauge(d, gx, gy, rad, r.score)
    big = "?" if r.score is None else str(r.score)
    bw = d.textlength(big, font=_font(120, True))
    d.text((gx - bw / 2, gy - 165), big, font=_font(120, True), fill=INK)
    lab = "Paper Hands Score"
    d.text((gx - d.textlength(lab, font=_font(26)) / 2, gy - 30), lab, font=_font(26), fill=INK2)

    # tekne + dalgalar (sağda)
    boat_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    bd = ImageDraw.Draw(boat_layer)
    _boat(bd, 850, 430, r.score)
    layer = Image.alpha_composite(layer, boat_layer)
    d = ImageDraw.Draw(layer)
    _waves(d, 480, 10, 110, 0.0)
    d.text((gx - rad - 28, gy + 20), "calm", font=_font(22), fill=INK3)
    d.text((gx + rad - 40, gy + 20), "paper", font=_font(22), fill=INK3)

    # kişilik
    title, line = persona(r.score)
    ty = 540
    d.text((PAD, ty), title, font=_font(56, True), fill=INK)
    d.text((PAD, ty + 72), line, font=_font(28), fill=INK2)
    if r.score is None:
        d.text((PAD, ty + 112), f"{r.measured} trades measurable so far — need {MIN_SCORED}.", font=_font(24), fill=INK3)

    # 2×2 kutucuk
    ky, gap = 720, 20
    tw, th = (W - 2 * PAD - gap) // 2, 140
    _tile(d, PAD, ky, tw, th, "Missed by selling early", compact_usd(r.missed_usd))
    if r.closed:
        up = r.realized_pnl >= 0
        _tile(d, PAD + tw + gap, ky, tw, th, f"Realized PnL · win {r.wins / r.closed * 100:.0f}%",
              compact_usd(r.realized_pnl), "▲" if up else "▼", GOOD if up else CRIT)
    else:
        _tile(d, PAD + tw + gap, ky, tw, th, "Realized PnL", "—")
    _tile(d, PAD, ky + th + gap, tw, th, "Panic sells", str(r.panic_sells))
    _tile(d, PAD + tw + gap, ky + th + gap, tw, th, "FOMO buys", str(r.fomo_buys))

    # en büyük pişmanlıklar
    ly = ky + 2 * th + gap + 36
    if r.worst:
        d.text((PAD, ly), "Biggest regrets", font=_font(30, True), fill=INK)
        for i, (miss, sym, p, hi, n) in enumerate(r.worst[:3]):
            yy = ly + 50 + i * 46
            d.text((PAD, yy), f"{sym[:12]}", font=_font(27, True), fill=INK)
            kez = f"{n}× " if n > 1 else ""
            d.text((PAD + 210, yy + 2), f"{kez}sold {fmt_price(p)} → later {fmt_price(hi)}", font=_font(25), fill=INK2)
            val = compact_usd(miss)
            d.text((W - PAD - d.textlength(val, font=_font(27, True)), yy), val, font=_font(27, True), fill=INK)
    else:
        d.text((PAD, ly), f"{r.trades} trades analyzed · {r.tokens} tokens", font=_font(28), fill=INK2)

    # alt dalgalar + alt bilgi
    _waves(d, H - 70, 8, 70, 1.3)
    foot = "Not financial advice — a measurement of past trades only."
    d.text((PAD, H - 52), foot, font=_font(22), fill=INK2)

    out = Image.alpha_composite(img, layer).convert("RGB")
    buf = io.BytesIO()
    out.save(buf, "PNG", optimize=True)
    return buf.getvalue()
