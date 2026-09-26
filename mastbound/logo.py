"""Mastbound token logosu: madeni para içinde direk + yelken + direğe dolanmış ip + dalga.

Çizim 4 kat büyük yapılıp küçültülür (kenar yumuşatma). `python -m mastbound.logo` → assets/logo.png (1024 px)
Bags / token görseli için kare PNG yeterli.
"""
import os

from PIL import Image, ImageDraw

NAVY_DARK, NAVY = (10, 24, 44), (22, 64, 96)
GOLD, GOLD_DARK = (244, 196, 88), (196, 146, 46)
SAIL, SAIL_SHADE = (250, 247, 238), (224, 220, 206)
WOOD = (120, 76, 44)
ROPE, ROPE_DARK = (236, 196, 120), (176, 132, 64)
SEA, FOAM = (57, 135, 229), (200, 230, 255)


def render_logo(size=1024):
    S = size * 4
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = S // 2
    # para: altın kenar + lacivert iç (dikey geçiş)
    d.ellipse([0, 0, S - 1, S - 1], fill=GOLD_DARK)
    d.ellipse([S * .02, S * .02, S * .98, S * .98], fill=GOLD)
    inner = [S * .08, S * .08, S * .92, S * .92]
    grad = Image.new("RGBA", (S, S))
    gd = ImageDraw.Draw(grad)
    for y in range(S):
        t = y / S
        gd.line([(0, y), (S, y)], fill=tuple(int(NAVY_DARK[i] + (NAVY[i] - NAVY_DARK[i]) * t) for i in range(3)) + (255,))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).ellipse(inner, fill=255)
    img.paste(grad, (0, 0), mask)
    d = ImageDraw.Draw(img)

    # deniz (alt kısım, para içinde)
    sea = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sea)
    import math
    top = S * .70
    pts = [(x, top + S * .018 * math.sin(x / (S * .06))) for x in range(0, S + 1, S // 200)]
    sd.polygon(pts + [(S, S), (0, S)], fill=SEA + (255,))
    sd.line(pts, fill=FOAM + (255,), width=int(S * .012))
    img.paste(sea, (0, 0), Image.composite(sea, Image.new("RGBA", (S, S)), mask).split()[3])
    d = ImageDraw.Draw(img)

    # gövde
    hull_y = S * .66
    d.polygon([(c - S * .22, hull_y), (c + S * .22, hull_y), (c + S * .15, hull_y + S * .085),
               (c - S * .15, hull_y + S * .085)], fill=WOOD)
    # direk
    mw = S * .022
    d.rectangle([c - mw, S * .20, c + mw, hull_y], fill=WOOD)
    # yelkenler
    d.polygon([(c + mw * 1.6, S * .24), (c + S * .20, S * .58), (c + mw * 1.6, S * .58)], fill=SAIL)
    d.polygon([(c - mw * 1.6, S * .30), (c - S * .15, S * .58), (c - mw * 1.6, S * .58)], fill=SAIL_SHADE)
    # bayrak
    d.polygon([(c + mw, S * .20), (c + S * .09, S * .225), (c + mw, S * .25)], fill=SEA)
    # direğe dolanmış ip (bağlı olma): direğin önünden geçen kıvrımlar + sarkan uç
    rw, rh, lw = S * .075, S * .03, int(S * .014)
    for k in range(4):
        y = S * (.38 + k * .045)
        box = [c - rw, y - rh, c + rw, y + rh]
        d.arc(box, 10, 170, fill=ROPE_DARK, width=lw + int(S * .006))
        d.arc(box, 10, 170, fill=ROPE, width=lw)
    ty = S * (.38 + 3 * .045) + rh * .8
    d.line([(c + rw * .7, ty), (c + rw * 1.05, ty + S * .07)], fill=ROPE_DARK, width=lw + int(S * .006))
    d.line([(c + rw * .7, ty), (c + rw * 1.05, ty + S * .07)], fill=ROPE, width=lw)
    d.ellipse([c + rw * 1.05 - lw, ty + S * .07 - lw, c + rw * 1.05 + lw, ty + S * .07 + lw], fill=ROPE)
    return img.resize((size, size), Image.LANCZOS)


def main():
    out = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    os.makedirs(out, exist_ok=True)
    render_logo(1024).save(os.path.join(out, "logo.png"), optimize=True)
    render_logo(256).save(os.path.join(out, "logo_256.png"), optimize=True)
    print("assets/logo.png, assets/logo_256.png")


if __name__ == "__main__":
    main()
