# Mastbound — Pişmanlık Aynası

Solana cüzdanının işlem geçmişinden erken satış, panik satış ve FOMO alımların maliyetini hesaplayan Telegram botu.
Yalnız herkese açık zincir verisini **okur**; işlem göndermez, özel anahtar istemez.

## Kurulum
```
pip install -r requirements.txt
export TELEGRAM_BOT_TOKEN=...   # BotFather'dan
export HELIUS_API_KEY=...       # helius.dev ücretsiz hesap
python -m mastbound.bot
```

## Testler
```
python -m pytest -q tests
```

## Yapı
- `mastbound/analysis.py` — saf hesaplar (erken satış, panik satış, FOMO, kağıt el puanı, rapor metni)
- `mastbound/chain.py` — Helius (işlemler) + GeckoTerminal (günlük fiyatlar), önbellekli
- `mastbound/bot.py` — Telegram botu (uzun sorgulama, kuyruklu analiz)
