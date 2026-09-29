#!/bin/bash
# Quant botunu kurar/ayarlar. Anahtarlar gizli girilir, yalnız /etc/mastbound-quant.env'e yazılır (chmod 600).
# Önce: bash deploy/setup_vps.sh (repo /opt/mastbound'da ve venv hazır olmalı)
set -e
ENV=/etc/mastbound-quant.env
DIR=/opt/mastbound
old() { [ -f "$ENV" ] && grep -E "^$1=" "$ENV" | cut -d= -f2- || true; }
ask() { local cur; cur=$(old "$1"); [ -z "$cur" ] && cur=$2; read -rp "$3 [$cur]: " v; echo "${v:-$cur}"; }

echo "MEXC API anahtarı: yalnız 'Futures/Vadeli işlem' izni ver, ÇEKİM izni verme, IP kısıtına bu VPS'in IP'sini ekle."
read -rsp "MEXC API Key (boş = mevcudu koru): " K; echo
read -rsp "MEXC API Secret (boş = mevcudu koru): " S; echo
K=${K:-$(old MEXC_API_KEY)}; S=${S:-$(old MEXC_API_SECRET)}
[ -n "$K" ] && [ -n "$S" ] || { echo "Anahtar boş olamaz"; exit 1; }
SYM=$(ask QUANT_SYMBOLS BTC_USDT,ETH_USDT "Semboller")
INT=$(ask QUANT_INTERVAL Min15 "Mum aralığı (Min5/Min15/Min30/Min60/Hour4)")
LEV=$(ask QUANT_LEVERAGE 3 "Kaldıraç")
RISK=$(ask QUANT_RISK_PCT 1 "İşlem başı risk (% sermaye)")
DLL=$(ask QUANT_MAX_DAILY_LOSS_PCT 3 "Günlük zarar limiti (%)")
HOST=$(ask QUANT_PANEL_HOST 127.0.0.1 "Panel adresi (Tailscale IP'si önerilir)")
LIVE=$(ask QUANT_LIVE 0 "CANLI işlem? 0=paper 1=gerçek")
TGC=$(ask QUANT_TG_CHAT "" "Telegram sohbet ID (bildirim, boş=kapalı)")
TOK=$(old QUANT_PANEL_TOKEN); [ -n "$TOK" ] || TOK=$(openssl rand -hex 20)
TG=$(grep -E '^TELEGRAM_BOT_TOKEN=' /etc/mastbound.env 2>/dev/null | cut -d= -f2- || true)

umask 077
cat > "$ENV" <<CFG
MEXC_API_KEY=$K
MEXC_API_SECRET=$S
QUANT_SYMBOLS=$SYM
QUANT_INTERVAL=$INT
QUANT_LEVERAGE=$LEV
QUANT_RISK_PCT=$RISK
QUANT_MAX_DAILY_LOSS_PCT=$DLL
QUANT_LIVE=$LIVE
QUANT_PANEL_HOST=$HOST
QUANT_PANEL_PORT=8787
QUANT_PANEL_TOKEN=$TOK
QUANT_STATE_DIR=$DIR/quant_state
QUANT_TG_CHAT=$TGC
TELEGRAM_BOT_TOKEN=$TG
CFG
chmod 600 "$ENV"
cp "$DIR/deploy/quant.service" /etc/systemd/system/mastbound-quant.service
systemctl daemon-reload
systemctl enable -q mastbound-quant
systemctl restart mastbound-quant
sleep 4
if systemctl is-active --quiet mastbound-quant; then
  echo "Quant botu çalışıyor ✓  ($([ "$LIVE" = 1 ] && echo CANLI || echo PAPER))"
  echo "iPad paneli:  http://$HOST:8787/?t=$TOK"
  echo "Log:          journalctl -u mastbound-quant -f"
else
  echo "Başlamadı:"; journalctl -u mastbound-quant -n 30 --no-pager
fi
