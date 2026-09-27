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

from . import social
from .analysis import card_text
from .card import compact_usd, persona, render
from .chain import DataError, wallet_report

ADDR = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")
WELCOME = ("⚓ <b>Welcome aboard Mastbound</b>\n\n"
           "Paste any Solana wallet address and the <b>Regret Mirror</b> shows what early sells, panic sells and "
           "FOMO buys have cost — with a Paper Hands Score from 0 to 100.\n\n"
           "🔒 Read-only. No wallet connect, no signatures. We will <b>never</b> ask for a seed phrase or private key.")
PROMO = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "promo.png")
COMMANDS = [("regret", "Regret Mirror card for a wallet"), ("about", "What Mastbound is"),
            ("safety", "How to stay safe"), ("token", "$MBOUND info"), ("help", "All commands")]
HELP = ("<b>Commands</b>\n/regret &lt;wallet&gt; — your Regret Mirror card (works in groups too)\n/about — what Mastbound is\n"
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
            chat, addr = self.jobs.get()
            try:
                r = wallet_report(addr, self.helius)
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
        if cmd and cmd != "/regret":
            return None if not private else self.send(chat, HELP)
        addr = words[-1] if words else ""
        if not ADDR.match(addr):
            return self.send(chat, "Usage: /regret &lt;Solana wallet address&gt;" if cmd
                             else "Please send a valid Solana wallet address (32-44 characters).")
        key = (chat, user)
        if time.time() - self.last.get(key, 0) < COOLDOWN:
            return self.send(chat, "Please wait a minute before the next analysis.")
        self.last[key] = time.time()
        self.send(chat, f"🔍 Analyzing… ({self.jobs.qsize()} ahead of you, may take 1-3 minutes)")
        self.jobs.put((chat, addr))

    def welcome(self, chat):
        try:
            png = open(PROMO, "rb").read()
        except OSError:
            png = None
        if not (png and self.send_photo(chat, png, WELCOME, self.keyboard())):
            self.send(chat, WELCOME, self.keyboard())

    def callback(self, q):
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

    def run(self):
        threading.Thread(target=self.worker, daemon=True).start()
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
