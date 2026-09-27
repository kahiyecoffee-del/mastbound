import time

from mastbound import social
from mastbound.bot import Bot


def test_oauth1_signature_known_vector():
    # X'in resmi OAuth 1.0a imza örneği (developer.x.com "Creating a signature")
    h = social.oauth1_header(
        "POST", "https://api.twitter.com/1.1/statuses/update.json",
        {"include_entities": "true", "status": "Hello Ladies + Gentlemen, a signed OAuth request!"},
        "xvz1evFS4wEEPTGEFPHBog", "kAcSOqF21Fu85e7zjz7ZN2U4ZRhfV3WpwPAoE3Z7kBw",
        "370773112-GmHxMAgYyLbNEtIKZeRNFsMKPR9EyMZeS9weJAEb", "LswwdoUaIvS8ltyTt5jkRh4J50vUPVVHtR2YPi5kE",
        nonce="kYjzVBB8Y0ZFabxSWbWovY3uYSQ2pTgmZeNu2VS4cg", ts=1318622958)
    assert 'oauth_signature="hCtSmYh%2BiHYCEqBWrE7C7hYmtUk%3D"' in h


def test_due_slot():
    t = time.mktime(time.strptime("2026-01-02 14:30", "%Y-%m-%d %H:%M")) - time.timezone
    assert social.due_slot(t, [13, 19]) == "2026-01-02:13"
    assert social.due_slot(t - 3 * 3600, [13, 19]) is None


class FakeOut:
    def __init__(self):
        self.sent = []

    def post(self, *a, **kw):
        self.sent.append((a, kw))
        return "1"


def test_scheduler_posts_once_per_slot(tmp_path):
    st = social.State(str(tmp_path / "s.json"))
    tg, x = FakeOut(), FakeOut()
    s = social.Scheduler(st, [{"text": "a {bot}"}, {"text": "b"}], [13, 19], tg, "@chan", x, "@MBot")
    day = time.mktime(time.strptime("2026-01-02 00:00", "%Y-%m-%d %H:%M")) - time.timezone
    assert s.tick(day + 14 * 3600) is None          # ilk çalıştırma geçmişi telafi etmez
    assert s.tick(day + 18 * 3600) is None          # aynı dilim
    assert s.tick(day + 19 * 3600 + 60)["text"] == "a {bot}"
    assert s.tick(day + 20 * 3600) is None
    assert tg.sent[0][0] == ("@chan", "a @MBot") and x.sent[0][0] == ("a @MBot",)
    assert social.State(str(tmp_path / "s.json")).d["post_index"] == 1


def test_posts_fit_x_limit():
    for p in social.load_posts():
        assert len(p["text"].replace("{bot}", "@MastboundRegretBot")) <= 280


class FakeX(FakeOut):
    def __init__(self, tweets):
        super().__init__()
        self.tweets = tweets

    def mentions(self, since_id=None):
        return self.tweets, "99"


def test_mention_replier_rules(tmp_path):
    addr = "7" * 43
    x = FakeX([{"id": "5", "author_id": "u1", "text": f"@mastbound {addr}"},
               {"id": "6", "author_id": "u2", "text": "@mastbound gm"}])
    r = social.MentionReplier(x, social.State(str(tmp_path / "x.json")), lambda a: (f"card {a[:4]}", None))
    assert r.poll() == 0                             # ilk açılışta eski etiketlere cevap yok
    assert r.poll() == 1                             # adres olan cevaplanır, "gm" sessiz kalır
    assert x.sent[0][1]["reply_to"] == "5"
    for _ in range(5):
        r.poll()
    assert len(x.sent) == social.X_REPLY_PER_USER    # kullanıcı başına günlük sınır


def _bot():
    b = Bot("T", "H")
    b.username = "MBot"
    b.out = []
    b.send = lambda chat, text: b.out.append(text)
    return b


def test_group_commands():
    b = _bot()
    grp = {"id": -1, "type": "supergroup"}
    addr = "7" * 43
    b.handle({"chat": grp, "from": {"id": 1}, "text": "gm everyone " + addr})
    assert not b.out and b.jobs.empty()              # grupta düz mesaja karışmaz
    b.handle({"chat": grp, "from": {"id": 1}, "text": "/regret@otherbot " + addr})
    assert b.jobs.empty()
    b.handle({"chat": grp, "from": {"id": 1}, "text": "/regret@MBot " + addr})
    assert b.jobs.get_nowait() == (-1, addr)
    b.handle({"chat": grp, "from": {"id": 2}, "text": "/regret " + addr})
    assert b.jobs.get_nowait() == (-1, addr)         # bekleme süresi kullanıcı başına
    b.handle({"chat": grp, "from": {"id": 1}, "text": "/safety"})
    assert "seed phrase" in b.out[-1]
    b.handle({"chat": grp, "from": {"id": 1}, "text": "/token"})
    assert "not launched" in b.out[-1]


def test_private_plain_address():
    b = _bot()
    b.handle({"chat": {"id": 9, "type": "private"}, "from": {"id": 9}, "text": "7" * 43})
    assert b.jobs.get_nowait() == (9, "7" * 43)


def test_card_with_qr_and_caption():
    from mastbound.analysis import Report
    from mastbound.bot import html_caption
    from mastbound.card import render
    r = Report(trades=35, measured=21, unmeasured=14, score=21, tokens=2, closed=11, wins=10, realized_pnl=53000)
    assert render("7" * 43, r, bot="MBot")[:4] == b"\x89PNG"
    c = html_caption("7" * 43, r)
    assert "Steady Sailor" in c and "<b>21/100</b>" in c


def test_inline_button_callback():
    b = _bot()
    import mastbound.bot as botmod
    orig = botmod.requests.post
    botmod.requests.post = lambda *a, **k: None
    try:
        b.callback({"id": "1", "data": "/safety", "message": {"chat": {"id": 5}}})
    finally:
        botmod.requests.post = orig
    assert "seed phrase" in b.out[-1]
