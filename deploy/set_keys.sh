#!/bin/bash
# Anahtarları gizli girişle /etc/mastbound.env dosyasına yazar (ekranda görünmez, geçmişe kaydolmaz).
set -e
ENV=/etc/mastbound.env
read -rsp "Telegram bot anahtarı (BotFather): " TG; echo
read -rsp "Helius API anahtarı: " HL; echo
[ -n "$TG" ] && [ -n "$HL" ] || { echo "Boş bırakılamaz"; exit 1; }
umask 077
printf 'TELEGRAM_BOT_TOKEN=%s\nHELIUS_API_KEY=%s\nMASTBOUND_CACHE=/opt/mastbound/cache\n' "$TG" "$HL" > "$ENV"
chmod 600 "$ENV"
systemctl restart mastbound
sleep 3
systemctl is-active --quiet mastbound && echo "Bot çalışıyor ✓  (log: journalctl -u mastbound -n 30 --no-pager)" \
  || { echo "Bot başlamadı:"; journalctl -u mastbound -n 20 --no-pager; }
