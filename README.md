<p align="center"><img src="assets/banner_x.png" alt="Mastbound" width="100%"></p>

# Mastbound — Regret Mirror

**Mastbound** is a behavioral mirror for crypto traders. The **Regret Mirror** reads a Solana wallet's public swap
history and measures what emotional trading cost: early sells, panic sells, FOMO buys and a
**Paper Hands Score** from 0 (Iron Captain) to 100 (Paper Boat) — delivered as a shareable card.

> Read-only by design. No wallet connect, no signatures, never a seed phrase or private key.

<p align="center"><img src="assets/promo.png" alt="Regret Mirror" width="70%"></p>

## What it measures

| Signal | Definition (price measured only *after* the trade) |
|---|---|
| Early sell | Price rose ≥10% above the sell within 30 days and the tokens were not bought back |
| Panic sell | Loss-making sell ≥15% below the prior 3-day high, not bought back, then a ≥10% rebound within 7 days |
| FOMO buy | Buy ≥30% above the prior 3-day low; price never rose 10% above it and fell ≥15% within 7 days; not exited in profit |
| Realized PnL | Average-cost basis per token |

Trades too recent (under 2 hours of price data) or without a price are reported as *not scored*, never as zero.
The score needs at least 3 scored trades.

## Channels

- **Telegram bot** — paste a wallet in DM, or `/regret <wallet>` in groups. `/about`, `/safety`, `/token`.
- **Telegram channel + X** — scheduled educational posts (`content/posts.json`).
- **X mentions** — optional auto-reply with the card to accounts that tag us with a wallet (paid X API tier).

## Run locally

```bash
pip install -r requirements.txt
export TELEGRAM_BOT_TOKEN=...   # from BotFather
export HELIUS_API_KEY=...       # helius.dev
python -m mastbound.bot
python -m pytest -q             # tests
python -m mastbound.brand       # regenerate banner / avatar / promo images
```

## Deploy (VPS, systemd)

```bash
bash deploy/setup_vps.sh        # clone, venv, service
bash deploy/set_keys.sh         # Telegram + Helius keys (hidden input)
bash deploy/set_social.sh       # optional: channel, posting hours, X keys
```

Keys live only in `/etc/mastbound.env` (chmod 600) — never in the repo.

## Layout

| Path | Purpose |
|---|---|
| `mastbound/analysis.py` | Pure scoring logic (no network) |
| `mastbound/chain.py` | Helius swaps + GeckoTerminal prices, cached |
| `mastbound/card.py` | 1080×1350 share card |
| `mastbound/brand.py`, `logo.py` | Brand assets and token logo |
| `mastbound/bot.py` | Telegram bot |
| `mastbound/social.py` | Scheduled posts, X client, mention replies |

Fonts: Inter and Space Grotesk, SIL Open Font License (`mastbound/fonts/`).

*Not financial advice. Mastbound measures past on-chain behavior only.*
