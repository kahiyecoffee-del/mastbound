"""Backtest parametreleri. Tüm ayarlar burada, kodun geri kalanı bunları okur."""

# ---------------------------------------------------------------- Veri
DATA_SOURCE = "vision"              # "vision": data.binance.vision arşivi | "ccxt": borsa API'si
EXCHANGE = "binanceusdm"             # ccxt borsa id'si: "binanceusdm", "mexc", "okx", "bybit" ...
COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "AVAX", "LINK"]
QUOTE = "USDT"                      # USDT perpetual sembolü: "BTC/USDT:USDT"
START = "2025-09-24"                # backtest başlangıcı (UTC, dahil)
END = "2026-09-24"                  # backtest bitişi (UTC, hariç)
WARMUP_DAYS = 30                    # indikatör ısınması için START'tan önce indirilecek gün
DATA_DIR = "data"                   # parquet: data/<EXCHANGE>_<COIN>_1m.parquet
OUTPUT_DIR = "output"
DOWNLOAD_FUNDING = True             # funding geçmişini indirmeyi dene

# ---------------------------------------------------------------- Trend filtresi
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
FILTER_TIMEFRAMES = ["4h", "1h"]
# "pozitif veya pozitife yakın": MACD > 0  VEYA  MACD > sinyal
FILTER_MACD_ABOVE_ZERO = True
FILTER_MACD_ABOVE_SIGNAL = True

# ---------------------------------------------------------------- Kırılım (15dk)
BREAKOUT_TF = "15min"
BREAKOUT_LOOKBACK = 5               # kapanış > önceki 5 mumun en yüksek high'ı
STOP_LOOKBACK = 5                   # stop = son 5 mumun en düşük low'u
STOP_INCLUDES_BREAKOUT_CANDLE = True  # True: son 5 = kırılım mumu + önceki 4; False: önceki 5

# ---------------------------------------------------------------- Giriş (1dk PSAR)
PSAR_STEP, PSAR_MAX = 0.02, 0.2
ENTRY_TIMEOUT_MIN = 60              # PSAR yukarı→aşağı→yukarı sırası bu sürede tamamlanmalı
CANCEL_IF_BELOW_STOP = True         # giriş öncesi low < stop olursa kurulum iptal

# ---------------------------------------------------------------- Çıkış
TP_R = 2.0
BREAKEVEN_R = 1.0                   # 5dk mum bu R'nin üzerinde kapanırsa stop = giriş
BREAKEVEN_TF_MIN = 5

# ---------------------------------------------------------------- Hesap ve maliyetler
INITIAL_BALANCE = 100.0
LEVERAGE = 2.0
TAKER_FEE = 0.0005                  # giriş ve çıkışta, notional üzerinden
SLIPPAGE = 0.0002                   # her dolumda fiyat aleyhine
MAINT_MARGIN_RATE = 0.005           # likidasyon hesabı (isolated, tier-1 yaklaşık)
# Binance minimum notional (USDT); pozisyon bunun altındaysa işlem atlanır
MIN_NOTIONAL = {"BTC": 100.0, "ETH": 20.0}
MIN_NOTIONAL_DEFAULT = 5.0

# ---------------------------------------------------------------- Loglama
LOG_FIRST_N_TRADES = 5              # ayrıntılı log + grafik çizilecek işlem sayısı
