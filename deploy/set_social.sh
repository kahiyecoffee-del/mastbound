#!/bin/bash
# Sosyal otomasyon ayarları: Telegram kanalı + (isteğe bağlı) X anahtarları. Anahtarlar gizli girilir,
# /etc/mastbound.env içindeki mevcut Telegram/Helius anahtarlarına dokunulmaz. Boş bırakılan alan değişmez.
set -e
ENV=/etc/mastbound.env
[ -f "$ENV" ] || { echo "Önce deploy/set_keys.sh çalıştırın"; exit 1; }
umask 077
setkey() {  # setkey AD DEĞER  → varsa değiştir, yoksa ekle
  [ -n "$2" ] || return 0
  grep -v "^$1=" "$ENV" > "$ENV.tmp" || true
  printf '%s=%s\n' "$1" "$2" >> "$ENV.tmp"
  mv "$ENV.tmp" "$ENV"
}
read -rp "Telegram kanalı (ör. @mastbound_news, boş = değiştirme): " CH
read -rp "Paylaşım saatleri UTC (ör. 13,19; boş = 13,19): " HRS
read -rsp "X API Key (boş = X yok/değiştirme): " XK; echo
if [ -n "$XK" ]; then
  read -rsp "X API Key Secret: " XS; echo
  read -rsp "X Access Token: " XT; echo
  read -rsp "X Access Token Secret: " XTS; echo
  [ -n "$XS" ] && [ -n "$XT" ] && [ -n "$XTS" ] || { echo "X anahtarlarının dördü de gerekli"; exit 1; }
fi
read -rp "X'te etiketlere otomatik cevap? (ücretli X API gerekir) [e/H]: " RP
setkey TELEGRAM_CHANNEL_ID "$CH"
setkey SOCIAL_HOURS "${HRS:-13,19}"
setkey SOCIAL_ENABLED 1
setkey X_API_KEY "$XK"; setkey X_API_SECRET "$XS"; setkey X_ACCESS_TOKEN "$XT"; setkey X_ACCESS_SECRET "$XTS"
case "$RP" in e|E|y|Y) setkey X_REPLY_ENABLED 1 ;; *) setkey X_REPLY_ENABLED 0 ;; esac
chmod 600 "$ENV"
systemctl restart mastbound
sleep 4
journalctl -u mastbound -n 5 --no-pager | grep -E "sosyal|hata" || true
systemctl is-active --quiet mastbound && echo "Bot çalışıyor ✓" || journalctl -u mastbound -n 20 --no-pager
