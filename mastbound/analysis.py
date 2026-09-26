"""Pişmanlık Aynası — saf hesaplar (ağ erişimi yok, test edilebilir).

Girdi: bir cüzdanın işlemleri (Trade listesi) ve her token için günlük USD fiyat serisi.
Çıktı: erken satış kaçırılan kazancı, panik satış ve FOMO alım sayıları, kağıt el puanı.

Tanımlar (hepsi yalnız o işlemden SONRAKİ fiyatla, geriye dönük bilgi kullanmadan ölçülür):
  erken satış  : satıştan sonraki PENCERE gün içinde görülen en yüksek fiyat satış fiyatının
                 en az %MIN_MISS üstüne çıktıysa, kaçırılan = (en yüksek − satış fiyatı) × satılan miktar
  panik satış  : satıştan önceki 3 günde fiyat ≥ %PANIC_DROP düşmüş ve 7 gün içinde satış fiyatının üstüne dönmüş
  FOMO alım    : alımdan önceki 3 günde fiyat ≥ %FOMO_RISE yükselmiş ve 7 gün içinde alım fiyatının
                 %FOMO_FALL altına inmiş
"""
from dataclasses import dataclass, field

WINDOW_DAYS = 30
MIN_MISS = 0.10
PANIC_DROP = 0.15
FOMO_RISE = 0.30
FOMO_FALL = 0.15
DAY = 86400


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
    missed_usd: float = 0.0
    early_sells: int = 0
    panic_sells: int = 0
    fomo_buys: int = 0
    worst: list = field(default_factory=list)     # (kaçırılan $, sembol, satış fiyatı, sonraki zirve)
    score: int = 0

    def summary(self):
        return {k: v for k, v in self.__dict__.items()}


def _day(ts):
    return int(ts // DAY)


def price_at(series, day):
    """series: {gün_no: kapanış_fiyatı}. O gün yoksa en yakın önceki gün."""
    for d in range(day, day - 7, -1):
        if d in series:
            return series[d]
    return None


def max_after(series, day, window):
    vals = [series[d] for d in range(day + 1, day + window + 1) if d in series]
    return max(vals) if vals else None


def min_after(series, day, window):
    vals = [series[d] for d in range(day + 1, day + window + 1) if d in series]
    return min(vals) if vals else None


def analyze(trades, prices, now=None):
    """trades: [Trade]; prices: {mint: {gün_no: fiyat}}"""
    r = Report(trades=len(trades))
    for t in trades:
        s = prices.get(t.mint) or {}
        d = _day(t.ts)
        if t.side == "sell":
            r.sells += 1
            hi = max_after(s, d, WINDOW_DAYS)
            if hi and t.price > 0 and hi >= t.price * (1 + MIN_MISS):
                miss = (hi - t.price) * t.amount
                r.missed_usd += miss
                r.early_sells += 1
                r.worst.append((miss, t.symbol or t.mint[:6], t.price, hi))
            before = price_at(s, d - 3)
            back = max_after(s, d, 7)
            if before and t.price > 0 and t.price <= before * (1 - PANIC_DROP) and back and back > t.price:
                r.panic_sells += 1
        elif t.side == "buy":
            r.buys += 1
            before = price_at(s, d - 3)
            lo = min_after(s, d, 7)
            if before and t.price > 0 and t.price >= before * (1 + FOMO_RISE) and lo and lo <= t.price * (1 - FOMO_FALL):
                r.fomo_buys += 1
    r.worst.sort(reverse=True)
    r.worst = r.worst[:3]
    r.score = paper_hands_score(r)
    return r


def paper_hands_score(r):
    """0 = taş gibi sakin, 100 = tamamen kağıt el. Duygusal işlemlerin oranına göre."""
    if not r.trades:
        return 0
    emo = r.early_sells + r.panic_sells + r.fomo_buys
    base = emo / max(r.trades, 1)
    return int(round(min(1.0, base * 1.5) * 100))


def card_text(addr, r):
    short = f"{addr[:4]}…{addr[-4:]}"
    lines = [f"⚓ MASTBOUND — Pişmanlık Aynası ({short})", "",
             f"İncelenen işlem: {r.trades} ({r.buys} alım, {r.sells} satış)",
             f"Erken satışla kaçırılan kazanç: ${r.missed_usd:,.0f}",
             f"Panik satış: {r.panic_sells} | FOMO alım: {r.fomo_buys}",
             f"🧻 Kağıt el puanı: {r.score}/100"]
    if r.worst:
        lines += ["", "En büyük pişmanlıklar:"]
        for miss, sym, p, hi in r.worst:
            lines.append(f"• {sym}: ${p:.6g}'den sattın, {WINDOW_DAYS} gün içinde ${hi:.6g} gördü → ${miss:,.0f} kaçtı")
    lines += ["", "Bu bir yatırım tavsiyesi değildir; yalnız geçmiş işlemlerinin ölçümüdür."]
    return "\n".join(lines)
