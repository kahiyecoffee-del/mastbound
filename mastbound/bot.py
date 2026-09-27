"""Mastbound Telegram botu — kullanıcı Solana cüzdan adresini gönderir, Pişmanlık Aynası raporunu alır.

Çalıştırma: TELEGRAM_BOT_TOKEN=... HELIUS_API_KEY=... python -m mastbound.bot
Bot yalnız herkese açık zincir verisini OKUR; hiçbir işlem göndermez, anahtar istemez.
"""
import os
import queue
import re
import threading
import time

import requests

from . import social
from .analysis import caption, card_text
from .card import compact_usd, persona, render
from .chain import DataError, wallet_report

ADDR = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")
WELCOME = ("⚓ Welcome to Mastbound!\n\nSend a Solana wallet address and I'll show you what early sells, panic sells "
           "and FOMO buys have cost you.\n\nI only read public on-chain data. I will never ask for your private key "
           "or seed phrase — anyone who does is a scammer.")
HELP = ("Commands:\n/regret <wallet> — your Regret Mirror card (works in groups too)\n/about — what Mastbound is\n"
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
        except (requests.RequestException, ValueError, KeyError):
            pass
        return self.username

    def send(self, chat, text):
        try:
            requests.post(f"{self.api}/sendMessage", json={"chat_id": chat, "text": text,
                                                           "disable_web_page_preview": True}, timeout=30)
        except requests.RequestException:
            pass

    def send_photo(self, chat, png, caption):
        try:
            r = requests.post(f"{self.api}/sendPhoto", data={"chat_id": chat, "caption": caption[:1000]},
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
                    ok = self.send_photo(chat, render(addr, r), caption(addr, r))
                except Exception as e:                   # görsel üretilemezse metinle devam
                    print(f"kart hatası: {type(e).__name__}: {e}", flush=True)
                    ok = False
                if not ok:
                    self.send(chat, text)
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
            return self.send(chat, WELCOME + "\n\n" + HELP if cmd == "/start" else HELP)
        if cmd in FAQ:
            return self.send(chat, FAQ[cmd])
        if cmd == "/token":
            return self.send(chat, token_info())
        if cmd and cmd != "/regret":
            return None if not private else self.send(chat, HELP)
        addr = words[-1] if words else ""
        if not ADDR.match(addr):
            return self.send(chat, "Usage: /regret <Solana wallet address>" if cmd
                             else "Please send a valid Solana wallet address (32-44 characters).")
        key = (chat, user)
        if time.time() - self.last.get(key, 0) < COOLDOWN:
            return self.send(chat, "Please wait a minute before the next analysis.")
        self.last[key] = time.time()
        self.send(chat, f"🔍 Analyzing… ({self.jobs.qsize()} ahead of you, may take 1-3 minutes)")
        self.jobs.put((chat, addr))

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
                if "message" in u:
                    try:
                        self.handle(u["message"])
                    except Exception as e:      # tek bir mesaj botu düşürmesin
                        print(f"mesaj işleme hatası: {type(e).__name__}: {e}", flush=True)


def token_info():
    mint = os.environ.get("MBOUND_MINT")
    if not mint:
        return ("$MBOUND has not launched yet. The official contract address will be posted here (/token) and on our "
                "official channels only. Any 'MBOUND' token you see before that is fake.")
    return (f"$MBOUND official contract address:\n{mint}\n\nAlways verify it here before buying. "
            "Holding $MBOUND unlocks Standard / Pro features. Not financial advice.")


def x_reply(addr, helius):
    """X etiketine cevap: kısa metin + kart görseli."""
    r = wallet_report(addr, helius)
    if not r.trades:
        return f"⚓ No SOL/USDC trades found for {addr[:4]}…{addr[-4:]}.", None
    title, _ = persona(r.score)
    score = "not enough data yet" if r.score is None else f"{r.score}/100"
    text = (f"⚓ {addr[:4]}…{addr[-4:]} — {title}\nPaper Hands Score: {score}\n"
            f"Missed by selling early: {compact_usd(r.missed_usd)}\nNot financial advice.")
    try:
        return text, render(addr, r)
    except Exception:
        return text, None


def main():
    token, helius = os.environ["TELEGRAM_BOT_TOKEN"], os.environ["HELIUS_API_KEY"]
    bot = Bot(token, helius)
    social.start_from_env(token, lambda a: x_reply(a, helius), bot.fetch_username())
    print(f"bot başladı: @{bot.username}", flush=True)
    bot.run()


if __name__ == "__main__":
    main()
