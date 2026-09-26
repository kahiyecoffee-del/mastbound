#!/bin/bash
# Mastbound botunu VPS'e kurar (momentum botundan tamamen ayrı: /opt/mastbound, ayrı servis).
# Kullanım (root):  bash <(curl -fsSL https://raw.githubusercontent.com/kahiyecoffee-del/mastbound/main/deploy/setup_vps.sh)
set -e
DIR=/opt/mastbound
ENV=/etc/mastbound.env
if [ -d "$DIR/.git" ]; then git -C "$DIR" pull -q; else git clone -q https://github.com/kahiyecoffee-del/mastbound "$DIR"; fi
cd "$DIR"
command -v fc-list >/dev/null && fc-list | grep -qi dejavu || apt-get install -y -q fonts-dejavu-core >/dev/null
[ -x venv/bin/python ] || python3 -m venv venv
venv/bin/pip install -q --upgrade pip
venv/bin/pip install -q -r requirements.txt
touch "$ENV" && chmod 600 "$ENV"
cp deploy/mastbound.service /etc/systemd/system/mastbound.service
systemctl daemon-reload
systemctl enable -q mastbound
echo
echo "Kurulum tamam. Şimdi anahtarları gir:  bash $DIR/deploy/set_keys.sh"
