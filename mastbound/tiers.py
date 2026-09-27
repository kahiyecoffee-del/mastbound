"""Kademe sistemi: cüzdan sahipliği (imzalı mesaj) + tutulan $MBOUND'un dolar değerine göre Free / Standard / Pro.

Akış: /link → bot tek kullanımlık mesaj üretir → kullanıcı GitHub Pages'teki imza sayfasında Phantom ile imzalar
(işlem değildir, para çekemez) → `/verify <cüzdan> <imza>` → bot Ed25519 imzasını doğrular, cüzdanı kaydeder.
Bakiye günde bir yeniden kontrol edilir.

Ortam değişkenleri:
  MBOUND_MINT             token mint adresi (yoksa herkes PRELAUNCH_TIER kademesinde)
  PRELAUNCH_TIER          token çıkmadan önceki kademe (varsayılan: standard)
  SOLANA_RPC              RPC adresi (varsayılan: Helius mainnet; devnet testi için devnet adresi)
  MBOUND_TEST_PRICE_USD   sabit test fiyatı (devnet'te piyasa fiyatı yoktur)
  TIER_STANDARD_USD / TIER_PRO_USD   eşikler (3 / 25)
  VERIFY_PAGE             imza sayfası adresi
"""
import json
import os
import secrets
import threading
import time
from urllib.parse import quote

import requests

TIERS = ("free", "standard", "pro")
LIMITS = {  # günlük kart, analiz edilen geçmiş (gün, None = tümü), işlem dökümü, haftalık rapor
    "free": {"cards": 1, "days": 30, "details": False, "weekly": False},
    "standard": {"cards": 10, "days": 90, "details": True, "weekly": False},
    "pro": {"cards": None, "days": None, "details": True, "weekly": True},
}
GRACE = 0.9            # kademesi olan kullanıcı, fiyat dalgalanmasında eşiğin %90'ına kadar kademesini korur
RECHECK = 24 * 3600
NONCE_TTL = 15 * 60
VERIFY_PAGE = os.environ.get("VERIFY_PAGE", "https://kahiyecoffee-del.github.io/mastbound/verify/")
B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def b58decode(s):
    n = 0
    for ch in s:
        n = n * 58 + B58.index(ch)
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big") if n else b""
    return b"\0" * (len(s) - len(s.lstrip("1"))) + raw


def verify_signature(wallet, message, signature_b58):
    """Solana cüzdanının (Ed25519 açık anahtarı) mesajı imzaladığını doğrular."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    try:
        pub, sig = b58decode(wallet), b58decode(signature_b58)
        if len(pub) != 32 or len(sig) != 64:
            return False
        Ed25519PublicKey.from_public_bytes(pub).verify(sig, message.encode())
        return True
    except (ValueError, InvalidSignature):
        return False


def link_message(user_id, nonce):
    return (f"Mastbound wallet link\nTelegram user: {user_id}\nNonce: {nonce}\n"
            "Signing this message proves you own this wallet. It is not a transaction and cannot move funds.")


def verify_urls(message, bot=None):
    page = f"{VERIFY_PAGE}?m={quote(message)}" + (f"&bot={bot}" if bot else "")
    phantom = f"https://phantom.app/ul/browse/{quote(page, safe='')}?ref={quote(VERIFY_PAGE, safe='')}"
    return page, phantom


def tier_for_usd(usd, previous="free"):
    std, pro = float(os.environ.get("TIER_STANDARD_USD", 3)), float(os.environ.get("TIER_PRO_USD", 25))
    if usd >= pro or (previous == "pro" and usd >= pro * GRACE):
        return "pro"
    if usd >= std or (previous in ("standard", "pro") and usd >= std * GRACE):
        return "standard"
    return "free"


def rpc_url():
    if os.environ.get("SOLANA_RPC"):
        return os.environ["SOLANA_RPC"]
    return f"https://mainnet.helius-rpc.com/?api-key={os.environ.get('HELIUS_API_KEY', '')}"


def token_balance(wallet, mint):
    """Cüzdandaki toplam token miktarı (ondalıklı)."""
    body = {"jsonrpc": "2.0", "id": 1, "method": "getTokenAccountsByOwner",
            "params": [wallet, {"mint": mint}, {"encoding": "jsonParsed"}]}
    r = requests.post(rpc_url(), json=body, timeout=30)
    r.raise_for_status()
    total = 0.0
    for acc in (r.json().get("result") or {}).get("value", []):
        info = acc["account"]["data"]["parsed"]["info"]["tokenAmount"]
        total += float(info.get("uiAmount") or 0)
    return total


def token_price(mint):
    if os.environ.get("MBOUND_TEST_PRICE_USD"):
        return float(os.environ["MBOUND_TEST_PRICE_USD"])
    j = requests.get(f"https://api.geckoterminal.com/api/v2/simple/networks/solana/token_price/{mint}",
                     timeout=30).json()
    return float(((j.get("data") or {}).get("attributes") or {}).get("token_prices", {}).get(mint) or 0)


class Store:
    """Kullanıcı kayıtları (JSON): bağlı cüzdan, kademe, son kontrol, günlük kart sayacı, bekleyen nonce."""

    def __init__(self, path):
        self.path, self.lock = path, threading.Lock()
        try:
            self.d = json.load(open(path))
        except (OSError, ValueError):
            self.d = {}

    def user(self, uid):
        return self.d.setdefault(str(uid), {})

    def save(self):
        with self.lock:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            json.dump(self.d, open(tmp, "w"))
            os.replace(tmp, self.path)

    def start_link(self, uid):
        u = self.user(uid)
        nonce, ts = u.get("pending") or [None, 0]
        if nonce and time.time() - ts < NONCE_TTL - 120:
            return link_message(uid, nonce)          # art arda /link: aynı bağlantı geçerli kalsın
        nonce = secrets.token_hex(8)
        u["pending"] = [nonce, time.time()]
        self.save()
        return link_message(uid, nonce)

    def finish_link(self, uid, wallet, signature):
        u = self.user(uid)
        nonce, ts = u.get("pending") or [None, 0]
        if not nonce or time.time() - ts > NONCE_TTL:
            return "expired"
        if not verify_signature(wallet, link_message(uid, nonce), signature):
            return "bad"
        u.update(wallet=wallet, linked_at=int(time.time()), checked=0)
        u.pop("pending", None)
        self.save()
        return "ok"

    def unlink(self, uid):
        u = self.user(uid)
        for k in ("wallet", "tier", "usd", "checked"):
            u.pop(k, None)
        self.save()

    def tier(self, uid, force=False, now=None):
        """Kademe (gerekirse bakiyeyi yeniden kontrol eder). Token yoksa herkes PRELAUNCH_TIER."""
        mint = os.environ.get("MBOUND_MINT")
        if not mint:
            return os.environ.get("PRELAUNCH_TIER", "standard")
        u = self.user(uid)
        if not u.get("wallet"):
            return "free"
        now = now or time.time()
        if force or now - u.get("checked", 0) > RECHECK:
            try:
                usd = token_balance(u["wallet"], mint) * token_price(mint)
                u.update(usd=round(usd, 2), tier=tier_for_usd(usd, u.get("tier", "free")), checked=int(now))
                self.save()
            except Exception as e:                  # RPC hatasında eski kademe korunur
                print(f"bakiye kontrol hatası {uid}: {e}", flush=True)
        return u.get("tier", "free")

    def take_card(self, uid, tier, now=None):
        """Günlük kart hakkından bir tane kullanır; hak yoksa False."""
        limit = LIMITS[tier]["cards"]
        day = time.strftime("%Y-%m-%d", time.gmtime(now or time.time()))
        u = self.user(uid)
        if u.get("day") != day:
            u["day"], u["cards"] = day, 0
        if limit is not None and u["cards"] >= limit:
            return False
        u["cards"] += 1
        self.save()
        return True

    def pro_users(self):
        return [int(k) for k, u in self.d.items() if u.get("wallet") and u.get("tier") == "pro"]
