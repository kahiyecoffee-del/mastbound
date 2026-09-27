"""Mastbound Telegram botu — kullanıcı Solana cüzdan adresini gönderir, Pişmanlık Aynası raporunu alır.

Çalıştırma: TELEGRAM_BOT_TOKEN=... HELIUS_API_KEY=... python -m mastbound.bot
Bot yalnız herkese açık zincir verisini OKUR; hiçbir işlem göndermez, anahtar istemez.
"""
import html
import os
import queue
import re
import threading
import time

import requests

from . import social, tiers
from .analysis import card_text
from .card import compact_usd, fmt_price, persona, render
from .chain import DataError, wallet_report

ADDR = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")
WELCOME = ("⚓ <b>Welcome aboard Mastbound</b>\n\n"
           "Paste any Solana wallet address and the <b>Regret Mirror</b> shows what early sells, panic sells and "
           "FOMO buys have cost — with a Paper Hands Score from 0 to 100.\n\n"
           "🔒 Read-only. No wallet connect, no signatures. We will <b>never</b> ask for a seed phrase or private key.")
PROMO = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "promo.png")
COMMANDS = [("regret", "Regret Mirror card for a wallet"), ("tier", "Your tier and limits"),
            ("link", "Link your wallet (unlock tiers)"), ("details", "Trade-by-trade breakdown"), ("about", "What Mastbound is"),
            ("safety", "How to stay safe"), ("token", "$MBOUND info"), ("help", "All commands")]
HELP = ("<b>Commands</b>\n/regret &lt;wallet&gt; — your Regret Mirror card (works in groups too)\n"
        "/details — trade-by-trade breakdown of your last card\n/tier — your tier and limits\n"
        "/link — link your wallet to unlock Standard / Pro\n/about — what Mastbound is\n"
        "/safety — how to stay safe\n/token — $MBOUND info\n\nIn a private chat you can also just paste a wallet address.")
FAQ = {
    "/about": ("⚓ Mastbound is a behavioral mirror for crypto traders.\n\nThe Regret Mirror reads a Solana wallet's public swap "
               "history and shows early sells, panic sells, FOMO buys and a Paper Hands Score (0-100). Next: the Ulysses "
               "Lock — commit to your plan before the storm.\n\nRead-only. Not financial advice."),
    "/safety": ("🔐 Stay safe:\n• We NEVER ask for your seed phrase or private key.\n• Admins never DM you first.\n"
                "• The Regret Mirror only needs a public address — no wallet connect, no signature.\n"
                "• Official token address is only announced in this bot's /token and our official channels."),
}
COOLDOWN = 60          # aynı kullanıcı için saniye


class Bot:
    def __init__(self, token, helius_key):
        self.api = f"https://api.telegram.org/bot{token}"
        self.helius = helius_key
        self.jobs = queue.Queue()
        self.last = {}
        self.username = None
        self.store = tiers.Store(os.path.join(os.environ.get("MASTBOUND_CACHE", "cache"), "users.json"))
        self.reports = {}                    # kullanıcı → (adres, rapor): /details için

    def fetch_username(self):
        try:
            self.username = requests.get(f"{self.api}/getMe", timeout=30).json()["result"]["username"]
            requests.post(f"{self.api}/setMyCommands", timeout=30, json={
                "commands": [{"command": c, "description": d} for c, d in COMMANDS]})
            requests.post(f"{self.api}/setMyShortDescription", timeout=30, json={
                "short_description": "Regret Mirror: what your paper hands cost you. Read-only, no keys."})
        except (requests.RequestException, ValueError, KeyError):
            pass
        return self.username

    def keyboard(self):
        rows = [[{"text": "🧭 How it works", "callback_data": "/about"}, {"text": "🔐 Safety", "callback_data": "/safety"}],
                [{"text": "🪙 $MBOUND", "callback_data": "/token"}]]
        ch = os.environ.get("TELEGRAM_CHANNEL_ID", "")
        if ch.startswith("@"):
            rows[1].append({"text": "📣 Channel", "url": f"https://t.me/{ch[1:]}"})
        return {"inline_keyboard": rows}

    def send(self, chat, text, markup=None):
        body = {"chat_id": chat, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
        if markup:
            body["reply_markup"] = markup
        try:
            requests.post(f"{self.api}/sendMessage", json=body, timeout=30)
        except requests.RequestException:
            pass

    def send_photo(self, chat, png, caption, markup=None):
        import json
        extra = {"reply_markup": json.dumps(markup)} if markup else {}
        try:
            r = requests.post(f"{self.api}/sendPhoto", data={"chat_id": chat, "caption": caption[:1000],
                                                             "parse_mode": "HTML", **extra},
                              files={"photo": ("mastbound.png", png, "image/png")}, timeout=60)
            return r.ok
        except requests.RequestException:
            return False

    def worker(self):
        while True:
            chat, addr, user, tier = self.jobs.get()
            try:
                r = wallet_report(addr, self.helius, days=tiers.LIMITS[tier]["days"])
                if not r.trades:
                    self.send(chat, "No SOL/USDC trades found for this wallet.")
                    continue
                text = card_text(addr, r)
                try:
                    ok = self.send_photo(chat, render(addr, r, self.username), html_caption(addr, r))
                except Exception as e:                   # görsel üretilemezse metinle devam
                    print(f"kart hatası: {type(e).__name__}: {e}", flush=True)
                    ok = False
                if not ok:
                    self.send(chat, html.escape(text))
                self.reports[user] = (addr, r)
                if r.events and tiers.LIMITS[tier]["details"]:
                    self.send(chat, f"📋 {len(r.events)} flagged trades — tap for the breakdown.",
                              {"inline_keyboard": [[{"text": "📋 Trade details", "callback_data": "/details"}]]})
            except DataError as e:
                print(f"veri hatası {addr}: {e}", flush=True)
                self.send(chat, "Could not reach trade data right now (data provider error). Please try again in a few minutes.")
            except Exception as e:                   # kullanıcıya kısa hata, ayrıntı loga
                print(f"hata {addr}: {type(e).__name__}: {e}", flush=True)
                self.send(chat, "Analysis failed right now, please try again in a few minutes.")

    def handle(self, msg):
        chat = msg["chat"]["id"]
        private = msg["chat"].get("type") == "private"
        user = (msg.get("from") or {}).get("id", chat)
        text = (msg.get("text") or "").strip()
        words = text.split()
        cmd = words[0].lower() if words and words[0].startswith("/") else ""
        if "@" in cmd:                          # /regret@BotAdı → yalnız bize yazılmışsa
            cmd, _, target = cmd.partition("@")
            if self.username and target != self.username.lower():
                return None
        if not private and not cmd:
            return None                         # grupta düz sohbete karışma
        if cmd in ("/start", "/help"):
            if cmd == "/help":
                return self.send(chat, HELP)
            return self.welcome(chat)
        if cmd in FAQ:
            return self.send(chat, FAQ[cmd])
        if cmd == "/token":
            return self.send(chat, token_info())
        if cmd in ("/link", "/verify", "/unlink") and not private:
            return self.send(chat, "For your safety, wallet linking only works in a private chat with me.")
        if cmd == "/link":
            return self.link(chat, user)
        if cmd == "/verify":
            return self.verify(chat, user, words[1:])
        if cmd == "/unlink":
            self.store.unlink(user)
            return self.send(chat, "Wallet unlinked.")
        if cmd == "/tier":
            return self.send(chat, self.tier_text(user))
        if cmd == "/details":
            return self.details(chat, user)
        if cmd and cmd != "/regret":
            return None if not private else self.send(chat, HELP)
        addr = words[-1] if words else ""
        if not ADDR.match(addr):
            return self.send(chat, "Usage: /regret &lt;Solana wallet address&gt;" if cmd
                             else "Please send a valid Solana wallet address (32-44 characters).")
        key = (chat, user)
        if time.time() - self.last.get(key, 0) < COOLDOWN:
            return self.send(chat, "Please wait a minute before the next analysis.")
        tier = self.store.tier(user)
        if not self.store.take_card(user, tier):
            return self.send(chat, f"You've used today's free card. Hold $3 of $MBOUND and /link your wallet for "
                                   f"{tiers.LIMITS['standard']['cards']} cards a day — or come back tomorrow. ⚓")
        self.last[key] = time.time()
        self.send(chat, f"🔍 Analyzing… ({self.jobs.qsize()} ahead of you, may take 1-3 minutes)")
        self.jobs.put((chat, addr, user, tier))

    def link(self, chat, user):
        msg = self.store.start_link(user)
        page, phantom = tiers.verify_urls(msg, self.username)
        self.send(chat, "🔗 <b>Link your wallet</b>\n\n1. Open the signing page in Phantom and tap <b>Sign</b>.\n"
                        "2. Copy the <code>/verify …</code> line it shows and send it here.\n\n"
                        "Signing a message is <b>not</b> a transaction: it cannot move funds or approve anything. "
                        "The link expires in 15 minutes.",
                  {"inline_keyboard": [[{"text": "✍️ Sign in Phantom", "url": phantom}],
                                       [{"text": "🌐 Open signing page", "url": page}]]})

    def verify(self, chat, user, args):
        if len(args) != 2 or not ADDR.match(args[0]):
            return self.send(chat, "Usage: /verify &lt;wallet&gt; &lt;signature&gt; — start with /link.")
        res = self.store.finish_link(user, args[0], args[1])
        if res == "expired":
            return self.send(chat, "That link request expired. Send /link to get a new one.")
        if res == "bad":
            return self.send(chat, "Signature check failed. Make sure you signed with the same wallet, then try /link again.")
        self.store.tier(user, force=True)
        self.send(chat, "✅ Wallet linked.\n\n" + self.tier_text(user))

    def tier_text(self, user):
        tier = self.store.tier(user)
        u = self.store.user(user)
        lim = tiers.LIMITS[tier]
        cards = "unlimited" if lim["cards"] is None else f"{lim['cards']} a day"
        hist = "full history" if lim["days"] is None else f"last {lim['days']} days"
        lines = [f"🏷 Tier: <b>{tier.capitalize()}</b>"]
        if u.get("wallet"):
            w = u["wallet"]
            lines.append(f"Wallet: <code>{w[:4]}…{w[-4:]}</code>" + (f" · $MBOUND held ≈ ${u['usd']:,.2f}" if "usd" in u else ""))
        elif os.environ.get("MBOUND_MINT"):
            lines.append("No wallet linked — /link to unlock Standard or Pro.")
        else:
            lines.append("Pre-launch: everyone gets Standard until $MBOUND launches.")
        lines += [f"Cards: {cards} · Analysis: {hist}", "",
                  "<b>Free</b> — 1 card/day, last 30 days",
                  "<b>Standard</b> (hold $3 of $MBOUND) — 10 cards/day, 90 days, trade breakdown",
                  "<b>Pro</b> (hold $25) — unlimited, full history, weekly report",
                  "", "<i>Holdings are re-checked daily. Not financial advice.</i>"]
        return "\n".join(lines)

    def details(self, chat, user):
        tier = self.store.tier(user)
        if not tiers.LIMITS[tier]["details"]:
            return self.send(chat, "The trade-by-trade breakdown is a Standard feature — hold $3 of $MBOUND and /link your wallet.")
        if user not in self.reports:
            return self.send(chat, "Run /regret &lt;wallet&gt; first, then ask for details.")
        addr, r = self.reports[user]
        if not r.events:
            return self.send(chat, "No flagged trades on your last card. Clean sailing. ⚓")
        names = {"early": "🧻 Early sell", "panic": "📉 Panic sell", "fomo": "🚀 FOMO buy"}
        refs = {"early": "later peak", "panic": "3-day high", "fomo": "3-day low"}
        lines = [f"📋 <b>Flagged trades</b> · <code>{addr[:4]}…{addr[-4:]}</code>", ""]
        for ts, kind, sym, price, ref in sorted(r.events, reverse=True)[:15]:
            day = time.strftime("%b %d %H:%M", time.gmtime(ts))
            lines.append(f"{names[kind]} · <b>{html.escape(sym[:12])}</b> · {day} UTC\n"
                         f"   at {fmt_price(price)} · {refs[kind]} {fmt_price(ref)}")
        if len(r.events) > 15:
            lines.append(f"\n…and {len(r.events) - 15} more.")
        self.send(chat, "\n".join(lines))

    def weekly(self):
        """Pro kullanıcılara pazartesi 09:00 UTC'den sonra haftalık kart."""
        while True:
            try:
                t = time.gmtime()
                week = time.strftime("%G-W%V", t)
                meta = self.store.d.setdefault("_meta", {})
                if t.tm_wday == 0 and t.tm_hour >= 9 and meta.get("weekly") != week:
                    meta["weekly"] = week
                    self.store.save()
                    for uid in self.store.pro_users():
                        if self.store.tier(uid, force=True) == "pro":
                            self.send(uid, "🗓 <b>Your weekly Regret Mirror</b>")
                            self.jobs.put((uid, self.store.user(uid)["wallet"], uid, "pro"))
            except Exception as e:
                print(f"haftalık rapor hatası: {e}", flush=True)
            time.sleep(600)

    def welcome(self, chat):
        try:
            png = open(PROMO, "rb").read()
        except OSError:
            png = None
        if not (png and self.send_photo(chat, png, WELCOME, self.keyboard())):
            self.send(chat, WELCOME, self.keyboard())

    def callback(self, q):
        user = (q.get("from") or {}).get("id")
        try:
            requests.post(f"{self.api}/answerCallbackQuery", json={"callback_query_id": q["id"]}, timeout=30)
        except requests.RequestException:
            pass
        chat = ((q.get("message") or {}).get("chat") or {}).get("id")
        data = q.get("data")
        if chat is None:
            return
        if data in FAQ:
            self.send(chat, FAQ[data])
        elif data == "/token":
            self.send(chat, token_info())
        elif data == "/details" and user:
            self.details(chat, user)

    def run(self):
        threading.Thread(target=self.worker, daemon=True).start()
        threading.Thread(target=self.weekly, daemon=True).start()
        offset = None
        while True:
            try:
                r = requests.get(f"{self.api}/getUpdates", params={"timeout": 50, "offset": offset},
                                 timeout=60).json()
            except (requests.RequestException, ValueError) as e:
                print(f"getUpdates bağlantı hatası: {e}", flush=True)
                time.sleep(5)
                continue
            if not r.get("ok"):                 # ör. 409: aynı token'la ikinci bir bot çalışıyor; 401: token geçersiz
                print(f"getUpdates hatası: {r.get('error_code')} {r.get('description')}", flush=True)
                time.sleep(15)
                continue
            for u in r.get("result", []):
                offset = u["update_id"] + 1
                if "message" in u or "callback_query" in u:
                    try:
                        if "message" in u:
                            self.handle(u["message"])
                        else:
                            self.callback(u["callback_query"])
                    except Exception as e:      # tek bir mesaj botu düşürmesin
                        print(f"mesaj işleme hatası: {type(e).__name__}: {e}", flush=True)


def html_caption(addr, r):
    title, _ = persona(r.score)
    score = "not enough data yet" if r.score is None else f"<b>{r.score}/100</b>"
    extra = f" · {r.unmeasured} too recent to score" if r.unmeasured else ""
    return (f"⚓ <b>{title}</b> — Paper Hands Score {score}\n"
            f"<code>{addr[:4]}…{addr[-4:]}</code> · {r.trades} trades · {r.tokens} tokens{extra}\n\n"
            "<i>Not financial advice — a measurement of past trades only.</i>")


def token_info():
    mint = os.environ.get("MBOUND_MINT")
    if not mint:
        return ("$MBOUND has not launched yet. The official contract address will be posted here (/token) and on our "
                "official channels only. Any 'MBOUND' token you see before that is fake.")
    return (f"$MBOUND official contract address:\n{mint}\n\nAlways verify it here before buying. "
            "Holding $MBOUND unlocks Standard / Pro features. Not financial advice.")


def x_reply(addr, helius, bot=None):
    """X etiketine cevap: kısa metin + kart görseli."""
    r = wallet_report(addr, helius)
    if not r.trades:
        return f"⚓ No SOL/USDC trades found for {addr[:4]}…{addr[-4:]}.", None
    title, _ = persona(r.score)
    score = "not enough data yet" if r.score is None else f"{r.score}/100"
    text = (f"⚓ {addr[:4]}…{addr[-4:]} — {title}\nPaper Hands Score: {score}\n"
            f"Missed by selling early: {compact_usd(r.missed_usd)}\nNot financial advice.")
    try:
        return text, render(addr, r, bot)
    except Exception:
        return text, None


def main():
    token, helius = os.environ["TELEGRAM_BOT_TOKEN"], os.environ["HELIUS_API_KEY"]
    bot = Bot(token, helius)
    social.start_from_env(token, lambda a: x_reply(a, helius, bot.username), bot.fetch_username())
    print(f"bot başladı: @{bot.username}", flush=True)
    bot.run()


if __name__ == "__main__":
    main()
