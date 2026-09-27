"""Pişmanlık Aynası — saf hesaplar (ağ erişimi yok, test edilebilir).

Girdi: bir cüzdanın işlemleri (Trade listesi) ve her token için USD fiyat serisi {unix_saniye: kapanış}.
Seri karışık çözünürlükte olabilir (son günler saatlik, eskisi günlük); hesaplar zaman damgasıyla yapılır.

Tanımlar (hepsi yalnız o işlemden SONRAKİ fiyatla ölçülür):
  erken satış  : satıştan sonraki WINDOW içinde görülen en yüksek fiyat satış fiyatının en az %MIN_MISS üstüne
                 çıktıysa, kaçırılan = (en yüksek − satış fiyatı) × satılan miktar
  panik satış  : zararına satış; satış fiyatı önceki 3 günün zirvesinin ≥ %PANIC_DROP altında, geri alım yok
                 ve 7 gün içinde fiyat satış fiyatının en az %MIN_MISS üstüne dönmüş
  FOMO alım    : alım fiyatı önceki 3 günün dibinin ≥ %FOMO_RISE üstünde ve sonraki 7 günde fiyat alımın %MIN_MISS
                 üstüne hiç çıkmadan %FOMO_FALL altına inmiş (tepeden alım)
  Önceki 3 gün, en az MIN_BEFORE kadar fiyat geçmişi varsa kullanılır (yeni çıkmış token'lar da ölçülür).
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
MIN_BEFORE = 6 * HOUR         # panik/FOMO için işlemden önce gereken en az fiyat geçmişi
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
    worst: list = field(default_factory=list)     # (kaçan $, sembol, ort. satış fiyatı, sonraki zirve, satış sayısı)
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

    def before(self, ts, span=3 * DAY):
        """İşlemden önceki `span` içindeki fiyatlar; MIN_BEFORE'dan kısa geçmiş varsa boş."""
        i, j = bisect_left(self.t, ts - span), bisect_left(self.t, ts)
        return self.p[i:j] if j > i and ts - self.t[i] >= MIN_BEFORE else []

    def last_ts(self):
        return self.t[-1] if self.t else 0


def analyze(trades, prices):
    """trades: [Trade]; prices: {mint: {unix_saniye: fiyat}}"""
    r = Report(trades=len(trades))
    series = {m: Series(s) for m, s in prices.items()}
    cost = {}                                    # mint -> [adet, toplam maliyet $]
    trades = sorted(trades, key=lambda x: x.ts)
    # satıştan sonra aynı token'ı geri alan kazancı kaçırmış sayılmaz: sonraki alımlar (FIFO) satışı mahsup eder
    rebuy = {}                                   # mint -> [[ts, kalan adet], ...]
    for t in trades:
        if t.side == "buy":
            rebuy.setdefault(t.mint, []).append([t.ts, t.amount])
    per_token = {}                               # mint -> [kaçan $, sembol, satış $ toplamı, satılan adet, zirve, adet]
    for t in trades:
        s = series.get(t.mint) or Series({})
        after = s.window(t.ts + 1, t.ts + WINDOW)
        week = s.window(t.ts + 1, t.ts + 7 * DAY)
        enough = bool(after) and s.last_ts() - t.ts >= MIN_AFTER and t.price > 0
        pre = s.before(t.ts)
        if t.side == "buy":
            r.buys += 1
            c = cost.setdefault(t.mint, [0.0, 0.0])
            c[0] += t.amount
            c[1] += t.usd
            if not enough:
                r.unmeasured += 1
                continue
            r.measured += 1
            if pre and t.price >= min(pre) * (1 + FOMO_RISE) and bought_top(week, t.price):
                r.fomo_buys += 1
            continue
        r.sells += 1
        c = cost.get(t.mint)
        pnl = None
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
        left = 0.0
        if hi >= t.price * (1 + MIN_MISS):
            left = t.amount
            for b in rebuy.get(t.mint, []):
                if left <= 0:
                    break
                if t.ts < b[0] <= t.ts + WINDOW and b[1] > 0:
                    used = min(left, b[1])
                    b[1] -= used
                    left -= used
            if left > 0:
                miss = (hi - t.price) * left
                r.missed_usd += miss
                r.early_sells += 1
                a = per_token.setdefault(t.mint, [0.0, t.symbol or t.mint[:6], 0.0, 0.0, 0.0, 0])
                a[0] += miss
                a[2] += t.price * left
                a[3] += left
                a[4] = max(a[4], hi)
                a[5] += 1
        # panik: düşüşte ZARARINA sattı, geri almadı ve fiyat 7 gün içinde toparlandı
        if (pre and t.price <= max(pre) * (1 - PANIC_DROP) and left > 0 and (pnl is None or pnl < 0)
                and week and max(week) >= t.price * (1 + MIN_MISS)):
            r.panic_sells += 1
    r.tokens = len({t.mint for t in trades})
    r.worst = sorted(((a[0], a[1], a[2] / a[3], a[4], a[5]) for a in per_token.values()), reverse=True)[:3]
    r.score = paper_hands_score(r)
    return r


def bought_top(week, price):
    """Tepeden alım: sonraki 7 günde fiyat alımın %MIN_MISS üstüne hiç çıkmadı ve en az %FOMO_FALL düştü."""
    return bool(week) and max(week) < price * (1 + MIN_MISS) and min(week) <= price * (1 - FOMO_FALL)


def paper_hands_score(r):
    """0 = taş gibi sakin, 100 = tamamen kağıt el; ölçülebilen işlem azsa None."""
    if r.measured < MIN_SCORED:
        return None
    emo = r.early_sells + r.panic_sells + r.fomo_buys
    return int(round(min(1.0, emo / r.measured * 1.5) * 100))


def caption(addr, r):
    """Görselin altındaki kısa açıklama (ayrıntı görselde)."""
    extra = f" · {r.unmeasured} recent trades not scored yet" if r.unmeasured else ""
    return (f"⚓ Mastbound · {addr[:4]}…{addr[-4:]} · {r.trades} trades, {r.tokens} tokens{extra}\n"
            "Not financial advice — a measurement of past trades only.")


def card_text(addr, r):
    short = f"{addr[:4]}…{addr[-4:]}"
    lines = [f"⚓ MASTBOUND — Regret Mirror ({short})", "",
             f"Trades analyzed: {r.trades} ({r.buys} buys, {r.sells} sells, {r.tokens} tokens)"]
    if r.closed:
        sign = "+" if r.realized_pnl >= 0 else "−"
        lines.append(f"Realized PnL: {sign}${abs(r.realized_pnl):,.0f} | "
                     f"win rate {r.wins / r.closed * 100:.0f}% ({r.wins}/{r.closed})")
    lines += [f"Missed by selling early: ${r.missed_usd:,.0f}",
              f"Panic sells: {r.panic_sells} | FOMO buys: {r.fomo_buys}"]
    if r.score is None:
        lines.append(f"🧻 Paper Hands Score: not enough data ({r.measured} trades measurable, need {MIN_SCORED})")
    else:
        lines.append(f"🧻 Paper Hands Score: {r.score}/100")
    if r.unmeasured:
        lines.append(f"({r.unmeasured} trades are from the last 2 hours or have no price data; not scored)")
    if r.worst:
        lines += ["", "Biggest regrets:"]
        for miss, sym, p, hi, n in r.worst:
            kez = f" ({n} sells)" if n > 1 else ""
            lines.append(f"• {sym}{kez}: sold at avg ${p:.6g}, later hit ${hi:.6g} → ${miss:,.0f} missed")
    lines += ["", "Not financial advice — a measurement of your past trades only."]
    return "\n".join(lines)
