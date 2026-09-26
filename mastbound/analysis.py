"""Pişmanlık Aynası — saf hesaplar (ağ erişimi yok, test edilebilir).

Girdi: bir cüzdanın işlemleri (Trade listesi) ve her token için USD fiyat serisi {unix_saniye: kapanış}.
Seri karışık çözünürlükte olabilir (son günler saatlik, eskisi günlük); hesaplar zaman damgasıyla yapılır.

Tanımlar (hepsi yalnız o işlemden SONRAKİ fiyatla ölçülür):
  erken satış  : satıştan sonraki WINDOW içinde görülen en yüksek fiyat satış fiyatının en az %MIN_MISS üstüne
                 çıktıysa, kaçırılan = (en yüksek − satış fiyatı) × satılan miktar
  panik satış  : satıştan önceki 3 günde fiyat ≥ %PANIC_DROP düşmüş ve 7 gün içinde satış fiyatının üstüne dönmüş
  FOMO alım    : alımdan önceki 3 günde fiyat ≥ %FOMO_RISE yükselmiş ve 7 gün içinde alım fiyatının
                 %FOMO_FALL altına inmiş
  ölçülemedi   : işlemden sonra yeterli fiyat verisi yok (çok yeni işlem ya da fiyat bulunamadı)
Kâr-zarar: token başına ortalama maliyetle gerçekleşen kâr/zarar.
"""
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field

DAY = 86400
HOUR = 3600
WINDOW = 30 * DAY
MIN_AFTER = 2 * HOUR          # bir işlemi değerlendirmek için sonrasında en az bu kadar veri
MIN_MISS = 0.10
PANIC_DROP = 0.15
FOMO_RISE = 0.30
FOMO_FALL = 0.15
MIN_SCORED = 3                # puan için en az bu kadar ölçülebilen işlem


@dataclass
class Trade:
    ts: int            # unix saniye
    mint: str
    side: str          # "buy" | "sell"
    amount: float      # token adedi
    usd: float         # işlemin USD değeri
    symbol: str = ""

    @property
    def price(self):
        return self.usd / self.amount if self.amount else 0.0


@dataclass
class Report:
    trades: int = 0
    sells: int = 0
    buys: int = 0
    measured: int = 0
    unmeasured: int = 0
    missed_usd: float = 0.0
    early_sells: int = 0
    panic_sells: int = 0
    fomo_buys: int = 0
    realized_pnl: float = 0.0
    wins: int = 0
    closed: int = 0
    tokens: int = 0
    worst: list = field(default_factory=list)     # (kaçırılan $, sembol, satış fiyatı, sonraki zirve)
    score: object = None                          # int ya da None (yetersiz veri)

    def summary(self):
        return dict(self.__dict__)


class Series:
    """Zaman damgalı fiyat serisi; aralık sorguları ikili arama ile."""

    def __init__(self, points):
        items = sorted((int(k), float(v)) for k, v in (points or {}).items() if v)
        self.t = [a for a, _ in items]
        self.p = [b for _, b in items]

    def at_or_before(self, ts, max_gap=3 * DAY):
        i = bisect_right(self.t, ts) - 1
        return self.p[i] if i >= 0 and ts - self.t[i] <= max_gap else None

    def window(self, a, b):
        i, j = bisect_left(self.t, a), bisect_right(self.t, b)
        return self.p[i:j]

    def last_ts(self):
        return self.t[-1] if self.t else 0


def analyze(trades, prices):
    """trades: [Trade]; prices: {mint: {unix_saniye: fiyat}}"""
    r = Report(trades=len(trades))
    series = {m: Series(s) for m, s in prices.items()}
    cost = {}                                    # mint -> [adet, toplam maliyet $]
    for t in sorted(trades, key=lambda x: x.ts):
        s = series.get(t.mint) or Series({})
        after = s.window(t.ts + 1, t.ts + WINDOW)
        week = s.window(t.ts + 1, t.ts + 7 * DAY)
        enough = bool(after) and s.last_ts() - t.ts >= MIN_AFTER and t.price > 0
        before = s.at_or_before(t.ts - 3 * DAY)
        if t.side == "buy":
            r.buys += 1
            c = cost.setdefault(t.mint, [0.0, 0.0])
            c[0] += t.amount
            c[1] += t.usd
            if not enough:
                r.unmeasured += 1
                continue
            r.measured += 1
            if before and t.price >= before * (1 + FOMO_RISE) and week and min(week) <= t.price * (1 - FOMO_FALL):
                r.fomo_buys += 1
            continue
        r.sells += 1
        c = cost.get(t.mint)
        if c and c[0] > 0:
            avg = c[1] / c[0]
            q = min(t.amount, c[0])
            pnl = t.usd * (q / t.amount) - avg * q
            r.realized_pnl += pnl
            r.closed += 1
            r.wins += pnl > 0
            c[0] -= q
            c[1] -= avg * q
        if not enough:
            r.unmeasured += 1
            continue
        r.measured += 1
        hi = max(after)
        if hi >= t.price * (1 + MIN_MISS):
            miss = (hi - t.price) * t.amount
            r.missed_usd += miss
            r.early_sells += 1
            r.worst.append((miss, t.symbol or t.mint[:6], t.price, hi))
        if before and t.price <= before * (1 - PANIC_DROP) and week and max(week) > t.price:
            r.panic_sells += 1
    r.tokens = len({t.mint for t in trades})
    r.worst.sort(reverse=True)
    r.worst = r.worst[:3]
    r.score = paper_hands_score(r)
    return r


def paper_hands_score(r):
    """0 = taş gibi sakin, 100 = tamamen kağıt el; ölçülebilen işlem azsa None."""
    if r.measured < MIN_SCORED:
        return None
    emo = r.early_sells + r.panic_sells + r.fomo_buys
    return int(round(min(1.0, emo / r.measured * 1.5) * 100))


def card_text(addr, r):
    short = f"{addr[:4]}…{addr[-4:]}"
    lines = [f"⚓ MASTBOUND — Pişmanlık Aynası ({short})", "",
             f"İncelenen işlem: {r.trades} ({r.buys} alım, {r.sells} satış, {r.tokens} token)"]
    if r.closed:
        sign = "+" if r.realized_pnl >= 0 else "−"
        lines.append(f"Gerçekleşen kâr/zarar: {sign}${abs(r.realized_pnl):,.0f} | "
                     f"kazanma oranı %{r.wins / r.closed * 100:.0f} ({r.wins}/{r.closed})")
    lines += [f"Erken satışla kaçırılan kazanç: ${r.missed_usd:,.0f}",
              f"Panik satış: {r.panic_sells} | FOMO alım: {r.fomo_buys}"]
    if r.score is None:
        lines.append(f"🧻 Kağıt el puanı: yeterli veri yok ({r.measured} işlem ölçülebildi, en az {MIN_SCORED} gerekli)")
    else:
        lines.append(f"🧻 Kağıt el puanı: {r.score}/100")
    if r.unmeasured:
        lines.append(f"({r.unmeasured} işlem son 2 saatte yapıldı ya da fiyatı bulunamadı; değerlendirilmedi)")
    if r.worst:
        lines += ["", "En büyük pişmanlıklar:"]
        for miss, sym, p, hi in r.worst:
            lines.append(f"• {sym}: ${p:.6g}'den sattın, sonra ${hi:.6g} gördü → ${miss:,.0f} kaçtı")
    lines += ["", "Bu bir yatırım tavsiyesi değildir; yalnız geçmiş işlemlerinin ölçümüdür."]
    return "\n".join(lines)
