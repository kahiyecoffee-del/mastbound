import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from mastbound import tiers
from mastbound.analysis import Report


def b58encode(b):
    n, out = int.from_bytes(b, "big"), ""
    while n:
        n, r = divmod(n, 58)
        out = tiers.B58[r] + out
    return "1" * (len(b) - len(b.lstrip(b"\0"))) + out


def keypair():
    k = Ed25519PrivateKey.generate()
    return k, b58encode(k.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw))


def test_b58_roundtrip():
    for raw in (b"\0\0abc", bytes(range(32)), b"\xff" * 64):
        assert tiers.b58decode(b58encode(raw)) == raw


def test_link_flow(tmp_path, monkeypatch):
    s = tiers.Store(str(tmp_path / "u.json"))
    k, wallet = keypair()
    msg = s.start_link(42)
    assert "not a transaction" in msg
    other, _ = keypair()
    assert s.finish_link(42, wallet, b58encode(other.sign(msg.encode()))) == "bad"      # başka anahtar
    assert s.finish_link(42, wallet, b58encode(k.sign(b"something else"))) == "bad"      # başka mesaj
    assert s.finish_link(42, wallet, b58encode(k.sign(msg.encode()))) == "ok"
    assert s.user(42)["wallet"] == wallet and "pending" not in s.user(42)
    assert s.finish_link(42, wallet, b58encode(k.sign(msg.encode()))) == "expired"      # tek kullanımlık


def test_link_expires(tmp_path):
    s = tiers.Store(str(tmp_path / "u.json"))
    k, wallet = keypair()
    msg = s.start_link(7)
    s.user(7)["pending"][0][1] -= tiers.NONCE_TTL + 1
    assert s.finish_link(7, wallet, b58encode(k.sign(msg.encode()))) == "expired"


def test_tier_thresholds_and_grace():
    assert tiers.tier_for_usd(2.99) == "free"
    assert tiers.tier_for_usd(3) == "standard"
    assert tiers.tier_for_usd(25) == "pro"
    assert tiers.tier_for_usd(23, "pro") == "pro"            # %10 dalgalanma payı
    assert tiers.tier_for_usd(22, "pro") == "standard"
    assert tiers.tier_for_usd(2.8, "standard") == "standard"
    assert tiers.tier_for_usd(2.8, "free") == "free"


def test_tier_from_balance_and_daily_recheck(tmp_path, monkeypatch):
    monkeypatch.setenv("MBOUND_MINT", "Mint1111111111111111111111111111111111111")
    calls = []
    monkeypatch.setattr(tiers, "token_balance", lambda w, m: calls.append(w) or 3000.0)
    monkeypatch.setattr(tiers, "token_price", lambda m: 0.01)                          # 3000 × 0.01 = $30
    s = tiers.Store(str(tmp_path / "u.json"))
    assert s.tier(1) == "free"                                                          # cüzdan yok
    s.user(1)["wallet"] = "W"
    now = time.time()
    assert s.tier(1, now=now) == "pro" and s.user(1)["usd"] == 30.0
    s.tier(1, now=now + 3600)
    assert len(calls) == 1                                                              # günde bir kontrol
    s.tier(1, now=now + tiers.RECHECK + 1)
    assert len(calls) == 2


def test_prelaunch_and_daily_card_limit(tmp_path, monkeypatch):
    monkeypatch.delenv("MBOUND_MINT", raising=False)
    s = tiers.Store(str(tmp_path / "u.json"))
    assert s.tier(5) == "standard"
    now = time.time()
    assert s.take_card(5, "free", now) and not s.take_card(5, "free", now)
    assert s.take_card(5, "free", now + 86400)                                          # ertesi gün yenilenir
    assert all(s.take_card(6, "pro", now) for _ in range(50))


def test_details_command():
    from tests.test_social import _bot
    b = _bot("/tmp/mb_test_users2.json")
    r = Report(events=[(1700000000, "panic", "WIF", 1.5, 2.0), (1700003600, "fomo", "BONK<x>", 0.00002, 0.00001)])
    b.reports[3] = ("7" * 43, r)
    b.details(3, 3)
    assert "Panic sell" in b.out[-1] and "BONK&lt;x&gt;" in b.out[-1]


def test_repeated_link_keeps_same_message(tmp_path):
    s = tiers.Store(str(tmp_path / "u.json"))
    k, wallet = keypair()
    first = s.start_link(9)
    assert s.start_link(9) == first
    assert s.finish_link(9, wallet, b58encode(k.sign(first.encode()))) == "ok"


def test_older_unexpired_link_still_works(tmp_path):
    s = tiers.Store(str(tmp_path / "u.json"))
    k, wallet = keypair()
    old = s.start_link(8)
    s.user(8)["pending"][0][1] -= tiers.NONCE_TTL - 60        # eski ama süresi dolmamış
    new = s.start_link(8)
    assert new != old
    assert s.finish_link(8, wallet, b58encode(k.sign(old.encode()))) == "ok"


def test_legacy_pending_format(tmp_path):
    s = tiers.Store(str(tmp_path / "u.json"))
    k, wallet = keypair()
    s.user(4)["pending"] = ["abc123", time.time()]
    msg = tiers.link_message(4, "abc123")
    assert s.finish_link(4, wallet, b58encode(k.sign(msg.encode()))) == "ok"
