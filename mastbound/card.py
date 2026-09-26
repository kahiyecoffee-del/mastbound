"""Paylaşılabilir Pişmanlık Kartı (PNG, 1080×1350 — X/Instagram/Telegram için dikey).

Düzen: marka başlığı → kağıt el puanı (tek büyük sayı + şiddet göstergesi) → 2×2 sayı kutucukları →
en büyük pişmanlıklar → uyarı. Metin her zaman metin renginde; renk yalnız gösterge dolgusu ve
kâr/zarar yön işaretinde (işaret + renk birlikte, renk tek başına anlam taşımaz).
"""
import io
import math
import os

from PIL import Image, ImageDraw, ImageFont

from .analysis import MIN_SCORED

W, H = 1080, 1350
PAD = 72
SURFACE = "#1a1a19"
TILE = "#242422"
TRACK = "#33332f"
INK = "#ffffff"
INK2 = "#c3c2b7"
INK3 = "#8d8c83"
ACCENT = "#3987e5"
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"

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
    s = f"${a / 1e6:.1f}M" if a >= 1e6 else f"${a / 1e3:.1f}K" if a >= 1e4 else f"${a:,.0f}"
    return s


def fmt_price(p):
    """0.0000212 gibi küçük fiyatlar bilimsel gösterim olmadan, 3 anlamlı basamakla."""
    if p <= 0:
        return "$0"
    dec = max(2, -int(math.floor(math.log10(p))) + 2)
    return f"${p:.{dec}f}"


def severity(score):
    return GOOD if score < 34 else WARN if score < 67 else CRIT


def _tile(d, x, y, w, h, label, value, marker=None, marker_color=None):
    d.rounded_rectangle([x, y, x + w, y + h], radius=20, fill=TILE)
    d.text((x + 32, y + 30), label, font=_font(30), fill=INK2)
    vx = x + 32
    if marker:
        d.text((vx, y + 76), marker, font=_font(56, True), fill=marker_color)
        vx += d.textlength(marker, font=_font(56, True)) + 12
    d.text((vx, y + 76), value, font=_font(56, True), fill=INK)


def render(addr, r):
    img = Image.new("RGB", (W, H), SURFACE)
    d = ImageDraw.Draw(img)
    # başlık
    d.text((PAD, PAD), "MASTBOUND", font=_font(44, True), fill=INK)
    d.text((PAD, PAD + 60), "Regret Mirror", font=_font(32), fill=INK2)
    short = f"{addr[:4]}…{addr[-4:]}"
    d.text((W - PAD - d.textlength(short, font=_font(30)), PAD + 10), short, font=_font(30), fill=INK3)

    # kahraman sayı: kağıt el puanı
    y = 250
    d.text((PAD, y), "Paper Hands Score", font=_font(34), fill=INK2)
    if r.score is None:
        d.text((PAD, y + 50), "—", font=_font(150, True), fill=INK)
        d.text((PAD, y + 230), f"Not enough data: {r.measured} trades measurable, need {MIN_SCORED}",
               font=_font(28), fill=INK3)
    else:
        big = str(r.score)
        d.text((PAD, y + 40), big, font=_font(170, True), fill=INK)
        bx = PAD + d.textlength(big, font=_font(170, True)) + 16
        d.text((bx, y + 150), "/100", font=_font(48), fill=INK2)
        # şiddet göstergesi: aynı yerde iz, dolgu şiddeti taşır
        by, bw = y + 250, W - 2 * PAD
        d.rounded_rectangle([PAD, by, PAD + bw, by + 18], radius=9, fill=TRACK)
        fw = max(18, int(bw * r.score / 100))
        d.rounded_rectangle([PAD, by, PAD + fw, by + 18], radius=9, fill=severity(r.score))
        d.text((PAD, by + 30), "calm", font=_font(24), fill=INK3)
        d.text((PAD + bw - d.textlength("paper hands", font=_font(24)), by + 30), "paper hands", font=_font(24), fill=INK3)

    # 2×2 kutucuk
    ty, gap = 600, 24
    tw, th = (W - 2 * PAD - gap) // 2, 164
    _tile(d, PAD, ty, tw, th, "Missed by selling early", compact_usd(r.missed_usd))
    if r.closed:
        up = r.realized_pnl >= 0
        _tile(d, PAD + tw + gap, ty, tw, th, f"Realized PnL · win {r.wins / r.closed * 100:.0f}%",
              compact_usd(r.realized_pnl), "▲" if up else "▼", GOOD if up else CRIT)
    else:
        _tile(d, PAD + tw + gap, ty, tw, th, "Realized PnL", "—")
    _tile(d, PAD, ty + th + gap, tw, th, "Panic sells", str(r.panic_sells))
    _tile(d, PAD + tw + gap, ty + th + gap, tw, th, "FOMO buys", str(r.fomo_buys))

    # en büyük pişmanlıklar
    ly = ty + 2 * th + gap + 48
    if r.worst:
        d.text((PAD, ly), "Biggest regrets", font=_font(32, True), fill=INK)
        for i, (miss, sym, p, hi, n) in enumerate(r.worst[:3]):
            yy = ly + 56 + i * 52
            d.text((PAD, yy), f"{sym[:12]}", font=_font(30, True), fill=INK)
            kez = f"{n}× " if n > 1 else ""
            mid = f"{kez}sold {fmt_price(p)} → later {fmt_price(hi)}"
            d.text((PAD + 230, yy), mid, font=_font(28), fill=INK2)
            val = compact_usd(miss)
            d.text((W - PAD - d.textlength(val, font=_font(30, True)), yy), val, font=_font(30, True), fill=INK)
    else:
        d.text((PAD, ly), f"{r.trades} trades analyzed · {r.tokens} tokens", font=_font(30), fill=INK2)

    # alt bilgi
    foot = "Not financial advice — a measurement of past trades only."
    d.text((PAD, H - PAD - 34), foot, font=_font(24), fill=INK3)
    d.rectangle([0, H - 10, W, H], fill=ACCENT)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()
