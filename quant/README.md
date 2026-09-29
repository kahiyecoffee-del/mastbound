# Quant — MEXC vadeli botu + iPad paneli

iPadOS arka plandaki uygulamaları birkaç dakika içinde askıya aldığı için bot **VPS'te 7/24** çalışır;
iPad yalnızca **panel**dir (Safari → Paylaş → *Ana Ekrana Ekle* ile uygulama gibi açılır).
API anahtarları iPad'de durmaz.

```
MEXC vadeli API  ⇄  VPS: quant.engine (systemd, 7/24)  ⇄  panel :8787  ⇄  iPad (Tailscale)
                                  └─ Telegram bildirimi (isteğe bağlı)
```

## Strateji: EMA kesişimi + ATR stop

| | Kural (yalnız kapanmış mumda) |
|---|---|
| Long | EMA20, EMA50'yi yukarı keser **ve** kapanış > EMA200 |
| Short | EMA20, EMA50'yi aşağı keser **ve** kapanış < EMA200 (`QUANT_SHORT=0` ile kapatılır) |
| Borsa stopu | Giriş ± 2×ATR — emirle birlikte MEXC'ye konur, **bot/VPS çökse bile çalışır** |
| İz süren stop | En iyi kapanış ∓ 3×ATR; kapanış bunu geçerse piyasa emriyle çıkılır |
| Ters kesişim | Pozisyon kapatılır |
| Boyut | Stopta kaybedilen = sermayenin %1'i; teminat sermayenin %25'ini aşmaz |
| Günlük limit | Gün içi sermaye %3 düşerse o gün yeni işlem yok (açık stoplar durur) |

Pozisyonlar **izole** marjin ve piyasa emriyle açılır. Botun açmadığı bir pozisyona dokunulmaz.

## 1) Önce backtest

MEXC bazı bulut ağlarından erişilemez; backtest'i GitHub Actions'ta veya VPS'te çalıştır:

- GitHub → Actions → **quant-backtest** → *Run workflow* (sembol, aralık, gün, ek parametre)
- veya VPS'te: `venv/bin/python -m quant.backtest --symbol BTC_USDT --interval Min15 --days 180 --trades`

Ücret varsayımı taraf başı 2 bps + 2 bps kayma (`--fee-bps`, `--slip-bps`). MEXC'nin güncel API vadeli
ücretini hesabından kontrol et. PF < 1.2 veya maxDD yüksekse canlıya geçme.

## 2) VPS'e kurulum

```bash
bash /opt/mastbound/deploy/setup_vps.sh     # repo + venv (zaten kuruluysa: git -C /opt/mastbound pull)
bash /opt/mastbound/deploy/set_quant_keys.sh
```

Betik anahtarları gizli sorar, `/etc/mastbound-quant.env` (chmod 600) dosyasına yazar, servisi başlatır ve
panel adresini yazdırır. **Varsayılan PAPER moddur** — gerçek emir göndermez, gerçek fiyatlarla sanal
1000 USDT'lik hesapla çalışır. En az 1-2 hafta paper izle.

MEXC API anahtarı: yalnız **vadeli işlem** izni, **çekim izni yok**, IP kısıtı = VPS IP'si.

## 3) iPad'den erişim (Tailscale önerilir)

1. VPS: `curl -fsSL https://tailscale.com/install.sh | sh && tailscale up` → `tailscale ip -4`
2. iPad: App Store'dan Tailscale, aynı hesapla giriş.
3. `set_quant_keys.sh`'de panel adresi olarak Tailscale IP'sini (100.x.x.x) ver.
4. Safari: `http://100.x.x.x:8787/?t=<token>` → Paylaş → *Ana Ekrana Ekle*.

Panel internete açık portla yayınlanmamalı (0.0.0.0 kullanma). Alternatif: Termius ile SSH tüneli
(`-L 8787:127.0.0.1:8787`) ve `http://127.0.0.1:8787`.

Panelde: sermaye, bugünkü K/Z, sermaye eğrisi, açık pozisyonlar ve stoplar, son işlemler, olay günlüğü;
**Duraklat** (yeni işlem açma), **Devam**, **Hepsini kapat** (piyasa fiyatından kapat + duraklat).

## 4) Canlıya geçiş

`bash deploy/set_quant_keys.sh` → "CANLI işlem?" sorusuna `1`. Küçük sermaye ve 2-3x kaldıraçla başla.

## Ayarlar (`/etc/mastbound-quant.env`)

| Değişken | Varsayılan | |
|---|---|---|
| `QUANT_LIVE` | 0 | 1 = gerçek emir |
| `QUANT_SYMBOLS` | BTC_USDT | virgülle |
| `QUANT_INTERVAL` | Min15 | Min1…Min60, Hour4, Hour8, Day1 |
| `QUANT_LEVERAGE` | 3 | 1-20 |
| `QUANT_RISK_PCT` | 1 | işlem başı risk, % sermaye |
| `QUANT_MAX_MARGIN_PCT` | 25 | pozisyon başı azami teminat |
| `QUANT_MAX_DAILY_LOSS_PCT` | 3 | |
| `QUANT_FAST` / `SLOW` / `TREND` / `ATR` | 20 / 50 / 200 / 14 | `TREND=0` filtreyi kapatır |
| `QUANT_STOP_ATR` / `QUANT_TRAIL_ATR` | 2 / 3 | |
| `QUANT_SHORT` | 1 | |
| `QUANT_PAPER_EQUITY` | 1000 | paper başlangıç sermayesi |

Değiştirdikten sonra: `systemctl restart mastbound-quant`. Durum `quant_state/state.json`'da tutulur,
yeniden başlatmada pozisyon takibi kaldığı yerden devam eder.

*Yatırım tavsiyesi değildir. Kaldıraçlı işlemler sermayenin tamamını kaybettirebilir.*
