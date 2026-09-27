"""Paylaşılabilir Pişmanlık Kartı (PNG, 1080×1350 — X/Instagram/Telegram için dikey).

Tasarım: borsa "PnL paylaşım kartı" dili — koyu zemin, marka altını vurgusu, büyük rakamlar, sade etiketler,
alt şeritte QR ile "kendi cüzdanını ölç" çağrısı. Yazı tipleri projeyle gelir (Inter + Space Grotesk, OFL).
Renk anlamı tek başına taşımaz: kâr/zarar yönü hem renk hem üçgen işaretle gösterilir.
"""
import io
import math
import os
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .analysis import MIN_SCORED
from .logo import render_logo

W, H = 1080, 1350
PAD = 64
BG_TOP, BG_BOTTOM = (11, 14, 20), (8, 10, 14)
INK = (245, 247, 250)
INK2 = (160, 170, 184)
INK3 = (104, 114, 128)
LINE = (255, 255, 255, 22)
GOLD = (240, 196, 92)
GOLD_SOFT = (240, 196, 92, 40)
UP, DOWN = (46, 204, 138), (246, 70, 93)
WARN = (250, 178, 25)

PERSONAS = [  # (en yüksek puan, unvan, cümle)
    (15, "Iron Captain", "Sirens sing. You don't even blink."),
    (35, "Steady Sailor", "A little wobble, but you hold the wheel."),
    (55, "Wobbly Deckhand", "Half of your moves are the wind's idea."),
    (75, "Siren-Struck", "The sirens call and you jump ship."),
    (100, "Paper Boat", "One wave and you're gone. Tie yourself to the mast!"),
]

FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
_FACES = {"regular": "Inter-Regular.ttf", "medium": "Inter-Medium.ttf", "bold": "Inter-Bold.ttf",
          "black": "Inter-ExtraBold.ttf", "num": "SpaceGrotesk-Bold.ttf"}
_FALLBACK = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


@lru_cache(maxsize=64)
def _font(size, face="regular"):
    for p in [os.path.join(FONT_DIR, _FACES[face])] + _FALLBACK:
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
    return UP if score < 34 else WARN if score < 67 else DOWN


def persona(score):
    if score is None:
        return "Still at the Harbor", "Not enough voyages yet to judge your hands."
    for hi, title, line in PERSONAS:
        if score <= hi:
            return title, line
    return PERSONAS[-1][1:]


def _tracked(d, xy, text, font, fill, spacing):
    """Harf aralıklı yazı (etiketler için); genişliği döner."""
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=font, fill=fill)
        x += d.textlength(ch, font=font) + spacing
    return x - xy[0] - spacing


def _tracked_len(d, text, font, spacing):
    return sum(d.textlength(ch, font=font) for ch in text) + spacing * (len(text) - 1)


def _label(d, xy, text, fill=INK3, size=20):
    _tracked(d, xy, text.upper(), _font(size, "medium"), fill, 2.2)


@lru_cache(maxsize=1)
def _background():
    base = Image.linear_gradient("L").resize((W, H))
    top, bot = Image.new("RGB", (W, H), BG_TOP), Image.new("RGB", (W, H), BG_BOTTOM)
    img = Image.composite(bot, top, base).convert("RGBA")
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    g = ImageDraw.Draw(glow)
    g.ellipse([W - 620, -260, W + 260, 620], fill=(240, 196, 92, 58))       # altın ışık (sağ üst)
    g.ellipse([-420, 760, 420, 1520], fill=(40, 120, 200, 46))              # deniz mavisi (sol alt)
    glow = glow.filter(ImageFilter.GaussianBlur(150))
    img = Image.alpha_composite(img, glow)
    lines = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ld = ImageDraw.Draw(lines)
    for x in range(-H, W, 36):                                              # ince çapraz doku
        ld.line([(x, H), (x + H, 0)], fill=(255, 255, 255, 7), width=1)
    for k, a in enumerate((26, 18, 12)):                                    # alt kısımda dalga konturları
        y0 = 1128 + k * 20
        pts = [(x, y0 + 9 * math.sin(x / 70.0 + k * 1.7)) for x in range(0, W + 8, 8)]
        ld.line(pts, fill=(120, 170, 220, a), width=2)
    return Image.alpha_composite(img, lines)


@lru_cache(maxsize=4)
def _logo(size):
    return render_logo(size)


def _ring(layer, cx, cy, radius, score):
    """270°'lik halka gösterge, içinde logo madalyonu."""
    d = ImageDraw.Draw(layer)
    width = 22
    box = [cx - radius, cy - radius, cx + radius, cy + radius]
    start, sweep = 135, 270
    d.arc(box, start, start + sweep, fill=(255, 255, 255, 26), width=width)
    for k in range(0, 101, 10):                                             # ölçek çentikleri
        a = math.radians(start + sweep * k / 100)
        r1, r2 = radius + 12, radius + (26 if k % 50 == 0 else 20)
        d.line([(cx + r1 * math.cos(a), cy + r1 * math.sin(a)), (cx + r2 * math.cos(a), cy + r2 * math.sin(a))],
               fill=(255, 255, 255, 70), width=2)
    if score is not None:
        col = severity(score)
        end = start + sweep * max(score, 1) / 100
        glow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
        ImageDraw.Draw(glow).arc(box, start, end, fill=col + (150,), width=width + 16)
        layer.alpha_composite(glow.filter(ImageFilter.GaussianBlur(14)))
        d = ImageDraw.Draw(layer)
        d.arc(box, start, end, fill=col, width=width)
        for ang in (start, end):                                            # yuvarlak uçlar
            a = math.radians(ang)
            px, py = cx + (radius - width / 2) * math.cos(a), cy + (radius - width / 2) * math.sin(a)
            d.ellipse([px - width / 2, py - width / 2, px + width / 2, py + width / 2], fill=col)
    lg = _logo(int(radius * 1.32))
    layer.alpha_composite(lg, (int(cx - lg.width / 2), int(cy - lg.height / 2)))
    f = _font(18, "medium")
    for txt, ang in (("CALM", start), ("PAPER", start + sweep)):           # ölçek uçları
        a = math.radians(ang)
        tx, ty = cx + radius * math.cos(a), cy + radius * math.sin(a) + 24
        _tracked(d, (tx - _tracked_len(d, txt, f, 2) / 2, ty), txt, f, INK3, 2)


def _triangle(d, x, y, size, up, fill):
    if up:
        d.polygon([(x, y + size), (x + size, y + size), (x + size / 2, y)], fill=fill)
    else:
        d.polygon([(x, y), (x + size, y), (x + size / 2, y + size)], fill=fill)


def _stat(d, x, y, label, value, color=INK, arrow=None):
    _label(d, (x, y), label)
    vx = x
    if arrow is not None:
        _triangle(d, x, y + 58, 24, arrow, color)
        vx += 34
    d.text((vx, y + 34), value, font=_font(54, "num"), fill=color)


def _qr(data, size):
    try:
        import qrcode
    except ImportError:
        return None
    q = qrcode.QRCode(border=2, box_size=10, error_correction=qrcode.constants.ERROR_CORRECT_M)
    q.add_data(data)
    q.make(fit=True)
    return q.make_image(fill_color="black", back_color="white").get_image().convert("RGBA").resize(
        (size, size), Image.NEAREST)


def render(addr, r, bot=None):
    img = _background().copy()
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    # ── başlık
    layer.alpha_composite(_logo(76), (PAD, 52))
    _tracked(d, (PAD + 94, 56), "MASTBOUND", _font(36, "black"), INK, 3.5)
    _tracked(d, (PAD + 96, 102), "REGRET MIRROR", _font(18, "medium"), GOLD, 3.2)
    short = f"{addr[:4]}…{addr[-4:]}"
    f = _font(24, "medium")
    sw = d.textlength(short, font=f)
    d.rounded_rectangle([W - PAD - sw - 56, 66, W - PAD, 114], radius=24, fill=(255, 255, 255, 16),
                        outline=(255, 255, 255, 34), width=2)
    d.ellipse([W - PAD - sw - 38, 85, W - PAD - sw - 28, 95], fill=UP)            # "okundu" noktası
    d.text((W - PAD - sw - 18, 76), short, font=f, fill=INK2)

    # ── skor
    top = 196
    _label(d, (PAD, top), "Paper Hands Score", INK2, 22)
    big = "—" if r.score is None else str(r.score)
    col = INK if r.score is None else severity(r.score)
    fb = _font(210, "num")
    d.text((PAD - 8, top + 12), big, font=fb, fill=col)
    bw = d.textlength(big, font=fb)
    d.text((PAD + bw + 4, top + 176), "/100", font=_font(40, "bold"), fill=INK3)

    title, line = persona(r.score)
    ft = _font(30, "bold")
    tw = d.textlength(title, font=ft)
    py = top + 290
    d.rounded_rectangle([PAD, py, PAD + tw + 48, py + 58], radius=29, fill=GOLD_SOFT, outline=GOLD, width=2)
    d.text((PAD + 24, py + 11), title, font=ft, fill=GOLD)
    d.text((PAD, py + 80), line, font=_font(25), fill=INK2)
    if r.score is None:
        d.text((PAD, py + 118), f"{r.measured} trades measurable so far — need {MIN_SCORED}.",
               font=_font(22), fill=INK3)

    _ring(layer, 812, top + 200, 160, r.score)
    d = ImageDraw.Draw(layer)

    # ── ayraç + istatistikler (3 sütun × 2 satır)
    dy = 690
    d.line([(PAD, dy), (W - PAD, dy)], fill=LINE, width=2)
    cw = (W - 2 * PAD) // 3
    y1, y2 = dy + 36, dy + 170
    if r.closed:
        up = r.realized_pnl >= 0
        _stat(d, PAD, y1, "Realized PnL", compact_usd(r.realized_pnl), UP if up else DOWN, up)
        _stat(d, PAD + cw, y1, "Win rate", f"{r.wins / r.closed * 100:.0f}%")
    else:
        _stat(d, PAD, y1, "Realized PnL", "—")
        _stat(d, PAD + cw, y1, "Win rate", "—")
    _stat(d, PAD + 2 * cw, y1, "Missed early", compact_usd(r.missed_usd), DOWN if r.missed_usd >= 1 else INK)
    _stat(d, PAD, y2, "Panic sells", str(r.panic_sells))
    _stat(d, PAD + cw, y2, "FOMO buys", str(r.fomo_buys))
    _stat(d, PAD + 2 * cw, y2, "Trades scored", f"{r.measured}/{r.trades}")

    # ── en büyük pişmanlıklar
    ry = dy + 318
    d.line([(PAD, ry - 26), (W - PAD, ry - 26)], fill=LINE, width=2)
    if r.worst:
        _label(d, (PAD, ry), "Biggest regrets", INK2, 20)
        for i, (miss, sym, p, hi, n) in enumerate(r.worst[:2]):
            yy = ry + 38 + i * 44
            d.text((PAD, yy), sym[:10], font=_font(27, "bold"), fill=INK)
            times = f"{n}× " if n > 1 else ""
            d.text((PAD + 200, yy + 3), f"{times}sold {fmt_price(p)}  ·  peak {fmt_price(hi)}", font=_font(23),
                   fill=INK2)
            val = "−" + compact_usd(miss)
            d.text((W - PAD - d.textlength(val, font=_font(28, "num")), yy), val, font=_font(28, "num"), fill=DOWN)
    else:
        _label(d, (PAD, ry), "Clean exits" if r.sells else "No sells yet", INK2, 20)
        msg = ("No sell was followed by a 10%+ run within 30 days." if r.sells
               else "Nothing sold, nothing to regret — yet.")
        d.text((PAD, ry + 40), msg, font=_font(26, "medium"), fill=INK)

    # ── alt şerit: çağrı + QR
    fy = 1164
    d.rounded_rectangle([PAD, fy, W - PAD, fy + 122], radius=26, fill=(255, 255, 255, 14),
                        outline=(255, 255, 255, 30), width=2)
    tx = PAD + 30
    if bot:
        qr = _qr(f"https://t.me/{bot}", 94)
        if qr:
            layer.alpha_composite(qr, (W - PAD - 108, fy + 14))
        d.text((tx, fy + 24), "Mirror your own wallet", font=_font(30, "bold"), fill=INK)
        d.text((tx, fy + 68), f"@{bot}  ·  Telegram", font=_font(24, "medium"), fill=GOLD)
    else:
        d.text((tx, fy + 24), "Mastbound · Regret Mirror", font=_font(30, "bold"), fill=INK)
        d.text((tx, fy + 68), "Tie yourself to the mast.", font=_font(24, "medium"), fill=GOLD)
    foot = "Not financial advice · a measurement of past on-chain trades only"
    ff = _font(19)
    d.text(((W - d.textlength(foot, font=ff)) / 2, H - 44), foot, font=ff, fill=INK3)

    out = Image.alpha_composite(img, layer).convert("RGB")
    buf = io.BytesIO()
    out.save(buf, "PNG", optimize=True)
    return buf.getvalue()
