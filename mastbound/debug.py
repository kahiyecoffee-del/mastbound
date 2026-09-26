"""Teşhis: python -m mastbound.debug <cüzdan_adresi>  (HELIUS_API_KEY ortamda olmalı)
Her adımda ne kadar veri geldiğini yazar; anahtarları yazdırmaz."""
import os
import sys

from . import chain
from .analysis import analyze, card_text


def main():
    addr = sys.argv[1]
    key = os.environ.get("HELIUS_API_KEY", "")
    print(f"Helius anahtarı tanımlı mı: {'evet' if key else 'HAYIR'}")
    sol = chain.daily_prices(chain.SOL)
    print(f"SOL günlük fiyat sayısı: {len(sol)}  (GeckoTerminal son durum: {chain.LAST['status']})")
    try:
        txs = chain.fetch_transactions(addr, key)
    except chain.DataError as e:
        print(f"HELIUS HATASI: {e}")
        return
    print(f"Helius SWAP işlemi: {len(txs)}  (son durum: {chain.LAST['status']})")
    if txs:
        t = txs[0]
        print(f"  örnek: type={t.get('type')} source={t.get('source')} "
              f"tokenTransfers={len(t.get('tokenTransfers') or [])} nativeTransfers={len(t.get('nativeTransfers') or [])}")
    trades = chain.to_trades(addr, txs, sol)
    print(f"Çıkarılan alım/satım: {len(trades)}  (alım {sum(t.side == 'buy' for t in trades)}, "
          f"satım {sum(t.side == 'sell' for t in trades)})")
    if trades:
        r = chain.wallet_report(addr, key)
        print()
        print(card_text(addr, r))


if __name__ == "__main__":
    main()
