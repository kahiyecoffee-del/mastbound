"""Canlı motor: her mum kapanışında strateji kararı, borsa ile mutabakat, risk limitleri, iPad paneli.

  python -m quant.engine            # QUANT_LIVE=1 değilse kâğıt üstünde (paper) çalışır, emir göndermez

Ayarlar ortam değişkenlerinden okunur (deploy/set_quant_keys.sh → /etc/mastbound-quant.env):
  MEXC_API_KEY, MEXC_API_SECRET        yalnız "vadeli işlem" izni; çekim izni VERME, IP kısıtla
  QUANT_LIVE=1                         gerçek emir (varsayılan 0 = paper)
  QUANT_SYMBOLS=BTC_USDT,ETH_USDT      QUANT_INTERVAL=Min15
  QUANT_LEVERAGE=3  QUANT_RISK_PCT=1   QUANT_MAX_MARGIN_PCT=25  QUANT_MAX_DAILY_LOSS_PCT=3
  QUANT_FAST=20 QUANT_SLOW=50 QUANT_TREND=200 QUANT_ATR=14 QUANT_STOP_ATR=2 QUANT_TRAIL_ATR=3 QUANT_SHORT=1
  QUANT_PANEL_HOST=127.0.0.1 QUANT_PANEL_PORT=8787 QUANT_PANEL_TOKEN=...
  QUANT_STATE_DIR=/opt/mastbound/quant_state   QUANT_PAPER_EQUITY=1000
  TELEGRAM_BOT_TOKEN + QUANT_TG_CHAT   (isteğe bağlı bildirim)
"""
import json
import logging
import os
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal

import requests

from . import mexc as mx
from .strategy import Params, decide, indicators, size

log = logging.getLogger("quant")


# ------------------------------------------------------------------ ayarlar
def _env(name, default, cast=str):
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "evet") if cast is bool else cast(v)


@dataclass
class Config:
    live: bool = False
    symbols: list = field(default_factory=lambda: ["BTC_USDT"])
    interval: str = "Min15"
    leverage: int = 3
    risk_pct: float = 1.0
    max_margin_pct: float = 25.0
    max_daily_loss_pct: float = 3.0
    params: Params = field(default_factory=Params)
    state_dir: str = "quant_state"
    paper_equity: float = 1000.0
    fee_bps: float = 2.0
    panel_host: str = "127.0.0.1"
    panel_port: int = 8787
    panel_token: str = ""
    tg_token: str = ""
    tg_chat: str = ""

    @classmethod
    def from_env(cls):
        return cls(
            live=_env("QUANT_LIVE", False, bool),
            symbols=[s.strip().upper() for s in _env("QUANT_SYMBOLS", "BTC_USDT").split(",") if s.strip()],
            interval=_env("QUANT_INTERVAL", "Min15"),
            leverage=_env("QUANT_LEVERAGE", 3, int),
            risk_pct=_env("QUANT_RISK_PCT", 1.0, float),
            max_margin_pct=_env("QUANT_MAX_MARGIN_PCT", 25.0, float),
            max_daily_loss_pct=_env("QUANT_MAX_DAILY_LOSS_PCT", 3.0, float),
            params=Params(_env("QUANT_FAST", 20, int), _env("QUANT_SLOW", 50, int), _env("QUANT_TREND", 200, int),
                          _env("QUANT_ATR", 14, int), _env("QUANT_STOP_ATR", 2.0, float),
                          _env("QUANT_TRAIL_ATR", 3.0, float), _env("QUANT_SHORT", True, bool)),
            state_dir=_env("QUANT_STATE_DIR", "quant_state"),
            paper_equity=_env("QUANT_PAPER_EQUITY", 1000.0, float),
            fee_bps=_env("QUANT_FEE_BPS", 2.0, float),
            panel_host=_env("QUANT_PANEL_HOST", "127.0.0.1"),
            panel_port=_env("QUANT_PANEL_PORT", 8787, int),
            panel_token=_env("QUANT_PANEL_TOKEN", ""),
            tg_token=_env("TELEGRAM_BOT_TOKEN", ""),
            tg_chat=_env("QUANT_TG_CHAT", ""),
        )

    def validate(self):
        if self.interval not in mx.INTERVALS:
            raise ValueError(f"QUANT_INTERVAL şunlardan biri olmalı: {', '.join(mx.INTERVALS)}")
        if not 1 <= self.leverage <= 20:
            raise ValueError("QUANT_LEVERAGE 1-20 arası olmalı")
        if not 0 < self.risk_pct <= 5:
            raise ValueError("QUANT_RISK_PCT 0-5 arası olmalı (işlem başına sermaye yüzdesi)")
        if self.params.fast >= self.params.slow:
            raise ValueError("QUANT_FAST < QUANT_SLOW olmalı")


def round_price(px, detail):
    unit = Decimal(str(detail.get("priceUnit") or "0.0001"))
    return float((Decimal(str(px)) / unit).quantize(Decimal(1)) * unit)


# ------------------------------------------------------------------ aracılar
class LiveBroker:
    """Gerçek MEXC hesabı. Açılışta borsaya sabit stop (stopLossPrice) iliştirilir."""
    live = True

    def __init__(self, api, cfg):
        self.api, self.cfg = api, cfg

    def equity(self, marks):
        return self.api.equity()[0]

    def position(self, symbol):
        return next(iter(self.api.positions(symbol)), None)

    def open(self, symbol, side, vol, price, stop, detail):
        oid = self.api.market_order(symbol, mx.OPEN_LONG if side == "long" else mx.OPEN_SHORT, vol, price,
                                    self.cfg.leverage, mx.ISOLATED, stop_loss=round_price(stop, detail),
                                    tag=f"q{int(time.time())}")
        for _ in range(5):
            time.sleep(1.5)
            p = self.position(symbol)
            if p:
                return p
        raise mx.MexcError(f"emir {oid} gönderildi ama pozisyon görünmüyor — MEXC'den kontrol et")

    def close(self, symbol, pos, price):
        self.api.market_order(symbol, mx.CLOSE_LONG if pos["side"] == "long" else mx.CLOSE_SHORT, pos["vol"],
                              price, pos.get("leverage") or self.cfg.leverage, pos.get("openType") or mx.ISOLATED,
                              position_id=pos.get("positionId"))
        try:
            self.api.cancel_stops(symbol)
        except mx.MexcError as e:
            log.warning("stop emirleri temizlenemedi %s: %s", symbol, e)

    def on_candle(self, symbol, candle):
        pass


class PaperBroker:
    """Kâğıt üstünde hesap: gerçek fiyatlarla, ücret dahil; durumu state.json'da tutulur."""
    live = False

    def __init__(self, book, detail_fn, fee_bps):
        self.book, self.detail, self.fee = book, detail_fn, fee_bps / 1e4

    def _cs(self, symbol):
        return float(self.detail(symbol).get("contractSize") or 1)

    def equity(self, marks):
        u = 0.0
        for s, p in self.book["positions"].items():
            m = marks.get(s, p["entry"])
            u += (m - p["entry"] if p["side"] == "long" else p["entry"] - m) * p["vol"] * self._cs(s)
        return self.book["cash"] + u

    def position(self, symbol):
        return self.book["positions"].get(symbol)

    def open(self, symbol, side, vol, price, stop, detail):
        self.book["cash"] -= self.fee * price * vol * self._cs(symbol)
        p = {"symbol": symbol, "side": side, "vol": vol, "entry": price, "stop": stop}
        self.book["positions"][symbol] = p
        return p

    def close(self, symbol, pos, price):
        p = self.book["positions"].pop(symbol)
        q = p["vol"] * self._cs(symbol)
        self.book["cash"] += (price - p["entry"] if p["side"] == "long" else p["entry"] - price) * q
        self.book["cash"] -= self.fee * price * q

    def on_candle(self, symbol, candle):
        """Borsa stopunun taklidi: mum içinde stop değdiyse stop fiyatından kapat."""
        p = self.book["positions"].get(symbol)
        if not p:
            return
        _, o, h, l, _, _ = candle
        if p["side"] == "long" and l <= p["stop"]:
            self.close(symbol, p, min(o, p["stop"]))
        elif p["side"] == "short" and h >= p["stop"]:
            self.close(symbol, p, max(o, p["stop"]))


# ------------------------------------------------------------------ motor
class Engine:
    def __init__(self, cfg, api=None):
        cfg.validate()
        self.cfg, self.p = cfg, cfg.params
        self.api = api or mx.Mexc(os.environ.get("MEXC_API_KEY", ""), os.environ.get("MEXC_API_SECRET", ""))
        self.step = mx.INTERVALS[cfg.interval]
        self.lock = threading.RLock()
        self.cmds = queue.Queue()
        self.wake = threading.Event()
        self.marks = {}
        os.makedirs(cfg.state_dir, exist_ok=True)
        self.path = os.path.join(cfg.state_dir, "state.json")
        self.s = self._load()
        self.broker = LiveBroker(self.api, cfg) if cfg.live else PaperBroker(self.s["paper"], self.api.detail,
                                                                             cfg.fee_bps)

    # ---------------------------------------------------------- durum
    def _load(self):
        s = {"paused": False, "positions": {}, "trades": [], "equity_log": [], "events": [], "last_bar": {},
             "day": "", "day_start_equity": 0.0, "halted": False, "equity": 0.0,
             "paper": {"cash": self.cfg.paper_equity, "positions": {}}}
        try:
            with open(self.path) as f:
                s.update(json.load(f))
        except (OSError, ValueError):
            pass
        return s

    def save(self):
        with self.lock:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.s, f)
            os.replace(tmp, self.path)

    def event(self, text, notify=False):
        log.info(text)
        with self.lock:
            self.s["events"] = (self.s["events"] + [[int(time.time()), text]])[-150:]
        if notify and self.cfg.tg_token and self.cfg.tg_chat:
            try:
                requests.post(f"https://api.telegram.org/bot{self.cfg.tg_token}/sendMessage",
                              json={"chat_id": self.cfg.tg_chat, "text": f"[quant{'' if self.cfg.live else ' paper'}] {text}"},
                              timeout=10)
            except requests.RequestException:
                pass

    def status(self):
        with self.lock:
            s = json.loads(json.dumps(self.s))
        s.pop("paper", None)
        s.update(live=self.cfg.live, symbols=self.cfg.symbols, interval=self.cfg.interval,
                 leverage=self.cfg.leverage, risk_pct=self.cfg.risk_pct, params=asdict(self.p), marks=self.marks,
                 max_daily_loss_pct=self.cfg.max_daily_loss_pct, now=int(time.time()))
        return s

    def command(self, cmd):
        if cmd not in ("pause", "resume", "close_all"):
            raise ValueError(cmd)
        self.cmds.put(cmd)
        self.wake.set()

    # ---------------------------------------------------------- ana döngü
    def run_forever(self):
        self.event(f"başladı — {'CANLI' if self.cfg.live else 'PAPER'} {','.join(self.cfg.symbols)} "
                   f"{self.cfg.interval} {self.cfg.leverage}x risk %{self.cfg.risk_pct}", notify=True)
        last_snap, errors = 0, 0
        while True:
            try:
                self._commands()
                now = time.time()
                if now - last_snap >= 60:
                    self.snapshot()
                    last_snap = now
                for sym in self.cfg.symbols:
                    if self.s["last_bar"].get(sym, 0) < self._last_closed(now) and now % self.step >= 3:
                        with self.lock:
                            self.process(sym)
                errors = 0
            except Exception as e:                              # döngü asla ölmez; art arda hatada bildirir
                errors += 1
                log.exception("döngü hatası")
                self.event(f"hata: {e}", notify=errors in (3, 30))
            self.save()
            self.wake.wait(5)
            self.wake.clear()

    def _last_closed(self, now):
        """Son kapanmış mumun açılış zamanı."""
        return int(now // self.step) * self.step - self.step

    def _commands(self):
        while not self.cmds.empty():
            c = self.cmds.get()
            with self.lock:
                if c == "pause":
                    self.s["paused"] = True
                    self.event("duraklatıldı: yeni işlem açılmayacak (açık pozisyonlar yönetilmeye devam)", True)
                elif c == "resume":
                    self.s["paused"] = self.s["halted"] = False
                    self.event("devam ediyor", True)
                elif c == "close_all":
                    self.s["paused"] = True
                    for sym in list(self.cfg.symbols):
                        pos = self.broker.position(sym)
                        if pos:
                            self._close(sym, pos, self._price(sym), "panelden kapat")
                    self.event("tüm pozisyonlar kapatıldı, bot duraklatıldı", True)

    def _price(self, sym):
        try:
            self.marks[sym] = self.api.ticker(sym)
        except mx.MexcError as e:
            log.warning("fiyat alınamadı %s: %s", sym, e)
        return self.marks.get(sym)

    def snapshot(self):
        for sym in self.cfg.symbols:
            self._price(sym)
        eq = self.broker.equity(self.marks)
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self.lock:
            s = self.s
            s["equity"] = eq
            s["equity_log"] = (s["equity_log"] + [[int(time.time()), round(eq, 4)]])[-3000:]
            if s["day"] != day:
                s["day"], s["day_start_equity"] = day, eq
                if s["halted"]:
                    s["halted"] = False
                    self.event("yeni gün: günlük zarar kilidi kalktı", True)
            start = s["day_start_equity"]
            if start > 0 and not s["halted"] and eq <= start * (1 - self.cfg.max_daily_loss_pct / 100):
                s["halted"] = True
                self.event(f"günlük zarar limiti (%{self.cfg.max_daily_loss_pct}) aşıldı: {start:.2f} → {eq:.2f}. "
                           "Bugün yeni işlem yok; açık pozisyonların stopları duruyor.", True)

    # ---------------------------------------------------------- tek sembol
    def process(self, sym):
        now = time.time()
        need = self.p.warmup * 3 + 10
        candles = [c for c in self.api.klines(sym, self.cfg.interval, int(now) - need * self.step, int(now))
                   if c[0] + self.step <= now]
        if len(candles) < self.p.warmup + 2:
            self.event(f"{sym}: yetersiz mum ({len(candles)})")
            return
        self.marks[sym] = candles[-1][4]
        self.broker.on_candle(sym, candles[-1])
        book = self.s["positions"].get(sym)
        pos = self.broker.position(sym)

        if book and not pos:                                   # borsada stop oldu veya elle kapatıldı
            self._record(sym, book, book.get("stop", candles[-1][4]), "borsada kapandı (stop / elle)")
            self.s["positions"].pop(sym, None)
            book = None
            if self.broker.live:
                try:
                    self.api.cancel_stops(sym)
                except mx.MexcError:
                    pass
        if pos and not book:
            self.event(f"{sym}: botun açmadığı pozisyon var ({pos['side']} {pos['vol']}), dokunulmuyor")
            self.s["last_bar"][sym] = candles[-1][0]
            return

        ind = indicators(candles, self.p)
        d = decide(candles, ind, len(candles) - 1, book, self.p)
        price = candles[-1][4]
        if d.action == "close" and pos:
            self._close(sym, pos, price, d.reason)
        elif d.action in ("open_long", "open_short"):
            self._open(sym, "long" if d.action == "open_long" else "short", price, d.stop, ind.atr[-1])
        elif book:
            book["trail"] = d.stop
        self.s["last_bar"][sym] = candles[-1][0]

    def _open(self, sym, side, price, stop, atr_now):
        if self.s["paused"] or self.s["halted"]:
            self.event(f"{sym}: {side} sinyali atlandı ({'duraklatıldı' if self.s['paused'] else 'günlük limit'})")
            return
        detail = self.api.detail(sym)
        eq = self.broker.equity(self.marks)
        vol = size(eq, price, stop, detail, self.cfg.leverage, self.cfg.risk_pct, self.cfg.max_margin_pct)
        if vol <= 0:
            self.event(f"{sym}: {side} sinyali — sermaye minimum kontrat için yetersiz (equity {eq:.2f})")
            return
        try:
            pos = self.broker.open(sym, side, vol, price, stop, detail)
        except mx.MexcError as e:
            self.event(f"{sym} {side.upper()} açılamadı: {e}", True)
            return
        entry = pos.get("entry") or price
        stop += entry - price                                   # stop mesafesini gerçek dolum fiyatına taşı
        self.s["positions"][sym] = {"side": side, "entry": entry, "vol": vol, "peak": entry, "atr0": atr_now,
                                    "stop": stop, "trail": stop, "opened": int(time.time())}
        self.event(f"{sym} {side.upper()} açıldı: {vol} kontrat @ {entry:g}, stop {stop:g}", True)

    def _close(self, sym, pos, price, why):
        self.broker.close(sym, pos, price)
        book = self.s["positions"].pop(sym, None) or {"side": pos["side"], "entry": pos["entry"], "vol": pos["vol"]}
        self._record(sym, book, price, why)

    def _record(self, sym, book, exit_px, why):
        cs = float(self.api.detail(sym).get("contractSize") or 1)
        sgn = 1 if book["side"] == "long" else -1
        pnl = sgn * (exit_px - book["entry"]) * book["vol"] * cs
        t = {"symbol": sym, "side": book["side"], "entry": book["entry"], "exit": exit_px, "vol": book["vol"],
             "pnl": round(pnl, 4), "reason": why, "opened": book.get("opened"), "closed": int(time.time())}
        self.s["trades"] = (self.s["trades"] + [t])[-500:]
        self.event(f"{sym} {book['side'].upper()} kapandı @ {exit_px:g} ({why}) ≈ {pnl:+.2f} USDT", True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config.from_env()
    eng = Engine(cfg)
    if cfg.live:
        eng.api.equity()                                        # anahtar/izin hatasını başta yakala
    from .panel import serve
    threading.Thread(target=serve, args=(eng,), daemon=True).start()
    eng.run_forever()


if __name__ == "__main__":
    main()
