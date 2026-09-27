"""Marka görselleri: X kapak (1500×500), profil resmi (640×640), tanıtım görseli (1200×675).

`python -m mastbound.brand` → assets/ klasörüne yazar. Kartla aynı görsel dil (koyu zemin, altın vurgu, Inter).
"""
import math
import os

from PIL import Image, ImageDraw, ImageFilter

from .card import GOLD, INK, INK2, INK3, UP, _font, _logo, _ring, _tracked

ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")


def _canvas(w, h, glows):
    base = Image.linear_gradient("L").resize((w, h))
    img = Image.composite(Image.new("RGB", (w, h), (8, 10, 14)), Image.new("RGB", (w, h), (12, 16, 23)),
                          base).convert("RGBA")
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    g = ImageDraw.Draw(glow)
    for box, col in glows:
        g.ellipse(box, fill=col)
    img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(max(w, h) // 9)))
    tex = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    td = ImageDraw.Draw(tex)
    for x in range(-h, w, 36):
        td.line([(x, h), (x + h, 0)], fill=(255, 255, 255, 7), width=1)
    for k, a in enumerate((30, 20, 12)):
        y0 = h - 70 + k * 18
        td.line([(x, y0 + 8 * math.sin(x / 70.0 + k * 1.7)) for x in range(0, w + 8, 8)],
                fill=(120, 170, 220, a), width=2)
    return Image.alpha_composite(img, tex)


def _ring_on(img, cx, cy, radius, score):
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))     # yarı saydam çizimler ayrı katmanda karışsın
    _ring(layer, cx, cy, radius, score)
    img.alpha_composite(layer)


def banner_x():
    w, h = 1500, 500
    img = _canvas(w, h, [((w - 700, -300, w + 300, 500), (240, 196, 92, 70)), ((-300, 200, 500, 900), (40, 120, 200, 50))])
    d = ImageDraw.Draw(img)
    # X profil resmi sol altı kapattığı için içerik ortada/sağda
    x0 = 430
    _tracked(d, (x0, 128), "MASTBOUND", _font(84, "black"), INK, 8)
    _tracked(d, (x0 + 4, 236), "TIE YOURSELF TO THE MAST", _font(26, "medium"), GOLD, 6)
    d.text((x0, 296), "Regret Mirror  ·  Paper Hands Score  ·  Ulysses Lock", font=_font(30, "medium"), fill=INK2)
    d.text((x0, 346), "Read-only. Your keys never leave your wallet.", font=_font(24), fill=INK3)
    _ring_on(img, 1345, 235, 105, 21)
    return img


def avatar():
    s = 640
    img = _canvas(s, s, [((-100, -100, s + 100, s + 100), (240, 196, 92, 60))])
    lg = _logo(460)
    img.alpha_composite(lg, ((s - 460) // 2, (s - 460) // 2))
    return img.convert("RGB")


def promo():
    w, h = 1200, 675
    img = _canvas(w, h, [((w - 600, -300, w + 200, 400), (240, 196, 92, 70)), ((-300, 300, 400, 1000), (40, 120, 200, 50))])
    d = ImageDraw.Draw(img)
    img.alpha_composite(_logo(72), (64, 56))
    _tracked(d, (156, 64), "MASTBOUND", _font(34, "black"), INK, 3.5)
    _tracked(d, (158, 106), "REGRET MIRROR", _font(17, "medium"), GOLD, 3)
    d.text((64, 200), "What did your", font=_font(64, "black"), fill=INK)
    d.text((64, 276), "paper hands cost you?", font=_font(64, "black"), fill=GOLD)
    y = 392
    for txt in ("Early sells, panic sells & FOMO buys", "Paper Hands Score from 0 to 100", "Shareable card in under 3 minutes"):
        d.ellipse([66, y + 12, 80, y + 26], fill=UP)
        d.text((98, y), txt, font=_font(28, "medium"), fill=INK2)
        y += 52
    d.text((64, 586), "Paste any Solana wallet · read-only · no connect, no keys", font=_font(22), fill=INK3)
    _ring_on(img, 975, 360, 130, 21)
    return img.convert("RGB")


def _wrap(d, text, font, width):
    lines = []
    for para in text.split("\n"):
        cur = ""
        for w in para.split():
            t = f"{cur} {w}".strip()
            if d.textlength(t, font=font) <= width:
                cur = t
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


def post_image(text, bot=None, headline=None):
    """Kanal/X gönderisi için markalı görsel (1200×675): ilk paragraf büyük başlık olarak."""
    w, h = 1200, 675
    img = _canvas(w, h, [((w - 600, -300, w + 200, 400), (240, 196, 92, 60)), ((-300, 300, 400, 1000), (40, 120, 200, 45))])
    d = ImageDraw.Draw(img)
    img.alpha_composite(_logo(60), (64, 52))
    _tracked(d, (140, 58), "MASTBOUND", _font(28, "black"), INK, 3)
    _tracked(d, (142, 94), "TIE YOURSELF TO THE MAST", _font(14, "medium"), GOLD, 2.6)
    head = (headline or text).replace("{bot}", f"@{bot}" if bot else "Mastbound").split("\n\n")[0]
    head = "".join(ch for ch in head if ord(ch) < 0x2000 or 0x2010 <= ord(ch) <= 0x206F).strip()   # emoji yok
    for size in (60, 52, 46, 40, 34):
        f = _font(size, "black")
        lines = _wrap(d, head, f, w - 160)
        if len(lines) * size * 1.25 <= 400:
            break
    y = 170 + (400 - len(lines) * size * 1.25) / 2
    d.rectangle([64, y + 6, 70, y + len(lines) * size * 1.25 - 14], fill=GOLD)
    for ln in lines:
        d.text((96, y), ln, font=f, fill=INK)
        y += size * 1.25
    foot = f"@{bot} on Telegram" if bot else "Regret Mirror · read-only · no keys"
    d.text((64, h - 70), foot, font=_font(24, "medium"), fill=GOLD)
    return img.convert("RGB")


def png_bytes(img):
    import io
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def main():
    os.makedirs(ASSETS, exist_ok=True)
    banner_x().convert("RGB").save(os.path.join(ASSETS, "banner_x.png"), optimize=True)
    avatar().save(os.path.join(ASSETS, "avatar.png"), optimize=True)
    promo().save(os.path.join(ASSETS, "promo.png"), optimize=True)
    print("assets/banner_x.png, assets/avatar.png, assets/promo.png")


if __name__ == "__main__":
    main()
