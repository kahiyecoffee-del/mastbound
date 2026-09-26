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

from .analysis import caption, card_text
from .card import render
from .chain import DataError, wallet_report

ADDR = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")
WELCOME = ("⚓ Mastbound'a hoş geldin!\n\nSolana cüzdan adresini gönder; erken satışların, panik satışların ve "
           "FOMO alımların sana neye mal olduğunu hesaplayayım.\n\nSadece herkese açık zincir verisini okurum. "
           "Asla özel anahtarını ya da gizli kelimelerini isteme(m) — isteyen biri olursa dolandırıcıdır.")
COOLDOWN = 60          # aynı kullanıcı için saniye


class Bot:
    def __init__(self, token, helius_key):
        self.api = f"https://api.telegram.org/bot{token}"
        self.helius = helius_key
        self.jobs = queue.Queue()
        self.last = {}

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
                    self.send(chat, "Bu cüzdanda SOL/USDC karşılığı bir alım-satım bulamadım.")
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
                self.send(chat, "İşlem verisine şu an ulaşılamadı (veri sağlayıcı hatası). Birkaç dakika sonra tekrar dene.")
            except Exception as e:                   # kullanıcıya kısa hata, ayrıntı loga
                print(f"hata {addr}: {type(e).__name__}: {e}", flush=True)
                self.send(chat, "Şu an analiz yapılamadı, birkaç dakika sonra tekrar dene.")

    def handle(self, msg):
        chat = msg["chat"]["id"]
        text = (msg.get("text") or "").strip()
        if text.startswith("/start") or text.startswith("/help"):
            return self.send(chat, WELCOME)
        addr = text.split()[-1] if text else ""
        if not ADDR.match(addr):
            return self.send(chat, "Geçerli bir Solana cüzdan adresi gönder (32-44 karakter).")
        if time.time() - self.last.get(chat, 0) < COOLDOWN:
            return self.send(chat, "Biraz bekle, bir önceki analiz daha yeni bitti.")
        self.last[chat] = time.time()
        self.send(chat, f"🔍 Analiz ediliyor… (sırada {self.jobs.qsize()} kişi var, 1-2 dakika sürebilir)")
        self.jobs.put((chat, addr))

    def run(self):
        threading.Thread(target=self.worker, daemon=True).start()
        offset = None
        while True:
            try:
                r = requests.get(f"{self.api}/getUpdates", params={"timeout": 50, "offset": offset},
                                 timeout=60).json()
            except (requests.RequestException, ValueError):
                time.sleep(5)
                continue
            for u in r.get("result", []):
                offset = u["update_id"] + 1
                if "message" in u:
                    self.handle(u["message"])


def main():
    Bot(os.environ["TELEGRAM_BOT_TOKEN"], os.environ["HELIUS_API_KEY"]).run()


if __name__ == "__main__":
    main()
