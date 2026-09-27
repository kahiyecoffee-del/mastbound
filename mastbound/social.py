"""Sosyal otomasyon: Telegram kanalına + X'e zamanlanmış paylaşım, X'te bizi etiketleyenlere otomatik cevap.

Hepsi isteğe bağlı, ortam değişkenleriyle açılır (anahtarlar yalnız /etc/mastbound.env içinde):
  SOCIAL_ENABLED=1            zamanlanmış paylaşımı aç
  SOCIAL_HOURS=13,19          paylaşım saatleri (UTC)
  TELEGRAM_CHANNEL_ID=@kanal  paylaşılacak Telegram kanalı (bot kanalda yönetici olmalı)
  X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_SECRET   X uygulaması (OAuth 1.0a, okuma+yazma)
  X_REPLY_ENABLED=1           etiketlere cevap (X'in ücretli API katmanı gerekir: mention okuma)

Kurallar (X spam politikası): yalnız bizi etiketleyene cevap verilir, kullanıcı başına günde en fazla
X_REPLY_PER_USER cevap, istenmemiş cevap/DM yok. Paylaşımlarda fiyat/getiri vaadi yok.
"""
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from urllib.parse import quote

import requests

X_API = "https://api.x.com/2"
POSTS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "content", "posts.json")
ADDR_IN_TEXT = re.compile(r"\b[1-9A-HJ-NP-Za-km-z]{32,44}\b")
X_REPLY_PER_USER = int(os.environ.get("X_REPLY_PER_USER", "3"))


def _pct(s):
    return quote(str(s), safe="~-._")


def oauth1_header(method, url, params, key, secret, token, token_secret, nonce=None, ts=None):
    """OAuth 1.0a HMAC-SHA1 imzası. params: sorgu/form parametreleri (JSON gövde imzaya girmez)."""
    oa = {"oauth_consumer_key": key, "oauth_nonce": nonce or secrets.token_hex(16),
          "oauth_signature_method": "HMAC-SHA1", "oauth_timestamp": str(ts or int(time.time())),
          "oauth_token": token, "oauth_version": "1.0"}
    allp = sorted((_pct(k), _pct(v)) for k, v in {**(params or {}), **oa}.items())
    base = "&".join([method.upper(), _pct(url), _pct("&".join(f"{k}={v}" for k, v in allp))])
    sig = hmac.new(f"{_pct(secret)}&{_pct(token_secret)}".encode(), base.encode(), hashlib.sha1).digest()
    oa["oauth_signature"] = base64.b64encode(sig).decode()
    return "OAuth " + ", ".join(f'{_pct(k)}="{_pct(v)}"' for k, v in sorted(oa.items()))


class XClient:
    def __init__(self, key, secret, token, token_secret):
        self.creds = (key, secret, token, token_secret)
        self.user_id = None

    @classmethod
    def from_env(cls):
        vals = [os.environ.get(k) for k in ("X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_SECRET")]
        return cls(*vals) if all(vals) else None

    def _req(self, method, url, params=None, **kw):
        h = {"Authorization": oauth1_header(method, url, params, *self.creds)}
        r = requests.request(method, url, params=params, headers=h, timeout=60, **kw)
        if not r.ok:
            raise RuntimeError(f"X {r.status_code}: {r.text[:200]}")
        return r.json() if r.content else {}

    def upload_image(self, png):
        """Görsel yükler, media_id döner (yükleme olmazsa None; paylaşım metinle devam eder)."""
        try:
            j = self._req("POST", f"{X_API}/media/upload",
                          data={"media_category": "tweet_image"},
                          files={"media": ("card.png", png, "image/png")})
            return (j.get("data") or {}).get("id") or j.get("media_id_string")
        except Exception as e:
            print(f"X görsel yüklenemedi: {e}", flush=True)
            return None

    def post(self, text, png=None, reply_to=None):
        body = {"text": text[:280]}
        mid = self.upload_image(png) if png else None
        if mid:
            body["media"] = {"media_ids": [mid]}
        if reply_to:
            body["reply"] = {"in_reply_to_tweet_id": reply_to}
        return (self._req("POST", f"{X_API}/tweets", json=body).get("data") or {}).get("id")

    def me(self):
        if not self.user_id:
            self.user_id = self._req("GET", f"{X_API}/users/me")["data"]["id"]
        return self.user_id

    def mentions(self, since_id=None):
        p = {"max_results": "20", "tweet.fields": "author_id"}
        if since_id:
            p["since_id"] = since_id
        j = self._req("GET", f"{X_API}/users/{self.me()}/mentions", params=p)
        return list(reversed(j.get("data") or [])), (j.get("meta") or {}).get("newest_id")


class Telegram:
    def __init__(self, token):
        self.api = f"https://api.telegram.org/bot{token}"

    def post(self, chat, text, png=None):
        try:
            if png:
                r = requests.post(f"{self.api}/sendPhoto", data={"chat_id": chat, "caption": text[:1000]},
                                  files={"photo": ("mastbound.png", png, "image/png")}, timeout=60)
            else:
                r = requests.post(f"{self.api}/sendMessage", json={"chat_id": chat, "text": text,
                                                                   "disable_web_page_preview": True}, timeout=30)
            if not r.ok:
                print(f"Telegram kanal hatası: {r.status_code} {r.text[:200]}", flush=True)
            return r.ok
        except requests.RequestException as e:
            print(f"Telegram kanal hatası: {e}", flush=True)
            return False


class State:
    """Küçük JSON durum dosyası (son paylaşım, son okunan etiket, kullanıcı başına cevap sayısı)."""

    def __init__(self, path):
        self.path = path
        try:
            self.d = json.load(open(path))
        except (OSError, ValueError):
            self.d = {}

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        json.dump(self.d, open(tmp, "w"))
        os.replace(tmp, self.path)


def load_posts(path=POSTS_FILE):
    return [p for p in json.load(open(path)) if p.get("text")]


def due_slot(now, hours):
    """Şu anki UTC gününün geçmiş en son paylaşım saati → 'YYYY-MM-DD:HH' (yoksa None)."""
    t = time.gmtime(now)
    past = [h for h in hours if h <= t.tm_hour]
    return f"{time.strftime('%Y-%m-%d', t)}:{max(past):02d}" if past else None


def next_post(posts, state):
    i = state.d.get("post_index", 0) % len(posts)
    state.d["post_index"] = i + 1
    return posts[i]


class Scheduler:
    def __init__(self, state, posts, hours, telegram=None, channel=None, x=None, bot_ref="our Telegram bot"):
        self.state, self.posts, self.hours = state, posts, sorted(hours)
        self.tg, self.channel, self.x, self.bot_ref = telegram, channel, x, bot_ref

    def tick(self, now=None):
        slot = due_slot(now or time.time(), self.hours)
        if not slot or self.state.d.get("last_slot") == slot:
            return None
        if self.state.d.get("last_slot") is None:
            # ilk çalıştırma: geçmiş saati telafi etme, bir sonrakini bekle
            self.state.d["last_slot"] = slot
            self.state.save()
            return None
        p = next_post(self.posts, self.state)
        self.state.d["last_slot"] = slot
        self.state.save()                     # önce kaydet: hata olsa bile aynı gönderi iki kez gitmez
        text = p["text"].replace("{bot}", self.bot_ref)
        png = None
        try:
            from .brand import png_bytes, post_image
            png = png_bytes(post_image(p["text"], self.bot_ref.lstrip("@") if self.bot_ref.startswith("@") else None,
                                       p.get("headline")))
        except Exception as e:                # görsel olmazsa metinle devam
            print(f"gönderi görseli üretilemedi: {e}", flush=True)
        if self.tg and self.channel:
            self.tg.post(self.channel, (p.get("telegram") or p["text"]).replace("{bot}", self.bot_ref), png)
        if self.x:
            try:
                self.x.post(text, png=png)
            except Exception as e:
                print(f"X paylaşım hatası: {e}", flush=True)
        print(f"paylaşıldı [{slot}]: {text[:60]}", flush=True)
        return p

    def run(self):
        while True:
            try:
                self.tick()
            except Exception as e:
                print(f"zamanlayıcı hatası: {type(e).__name__}: {e}", flush=True)
            time.sleep(60)


class MentionReplier:
    """@hesabı etiketleyip bir Solana adresi yazan kişiye Pişmanlık Kartı ile cevap verir."""

    def __init__(self, x, state, report_fn, interval=120):
        self.x, self.state, self.report_fn, self.interval = x, state, report_fn, interval

    def allowed(self, user, now=None):
        day = time.strftime("%Y-%m-%d", time.gmtime(now or time.time()))
        per = self.state.d.setdefault("replies", {})
        if per.get("day") != day:
            per.clear()
            per["day"] = day
        if per.get(user, 0) >= X_REPLY_PER_USER:
            return False
        per[user] = per.get(user, 0) + 1
        return True

    def handle(self, tw):
        m = ADDR_IN_TEXT.search(tw.get("text", ""))
        if not m or not self.allowed(tw.get("author_id", "?")):
            return None                      # adres yoksa ya da sınır dolduysa sessiz kal (spam yok)
        text, png = self.report_fn(m.group(0))
        return self.x.post(text, png=png, reply_to=tw["id"])

    def poll(self):
        tweets, newest = self.x.mentions(self.state.d.get("since_id"))
        first_run = "since_id" not in self.state.d
        if newest:
            self.state.d["since_id"] = newest
        self.state.save()
        if first_run:
            return 0                         # açılıştan önceki eski etiketlere cevap verme
        n = 0
        for tw in tweets:
            try:
                n += bool(self.handle(tw))
            except Exception as e:
                print(f"X cevap hatası {tw.get('id')}: {e}", flush=True)
        self.state.save()
        return n

    def run(self):
        while True:
            try:
                self.poll()
            except Exception as e:
                print(f"X etiket okuma hatası: {e}", flush=True)
                time.sleep(600)             # sınır/yetki hatasında seyrek dene
            time.sleep(self.interval)


def start_from_env(tg_token, report_fn, bot_username=None):
    """bot.main tarafından çağrılır; açık olan otomasyon iş parçacıklarını başlatır."""
    cache = os.environ.get("MASTBOUND_CACHE", "cache")
    x = XClient.from_env()
    started = []
    if os.environ.get("SOCIAL_ENABLED") == "1":
        hours = [int(h) for h in os.environ.get("SOCIAL_HOURS", "13,19").split(",") if h.strip()]
        channel = os.environ.get("TELEGRAM_CHANNEL_ID")
        s = Scheduler(State(os.path.join(cache, "social_state.json")), load_posts(), hours,
                      Telegram(tg_token) if channel else None, channel, x,
                      f"@{bot_username}" if bot_username else "our Telegram bot")
        threading.Thread(target=s.run, daemon=True).start()
        started.append(f"paylaşım {hours} UTC → " + ", ".join(filter(None, [channel and "Telegram", x and "X"])))
    if x and os.environ.get("X_REPLY_ENABLED") == "1":
        r = MentionReplier(x, State(os.path.join(cache, "x_state.json")), report_fn)
        threading.Thread(target=r.run, daemon=True).start()
        started.append("X etiket cevabı")
    print("sosyal otomasyon: " + ("; ".join(started) or "kapalı"), flush=True)
