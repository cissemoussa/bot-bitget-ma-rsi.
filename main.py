"""
Bot de trading Spot MA + RSI pour Bitget
Fichier: bot_ma_rsi_bitget.py

Contenu :
- Connexion via CCXT à Bitget (spot)
- Calculs SMA et RSI via pandas
- Mode simulation (DEMO) ou mode réel (LIVE)
- Notifications Telegram simples via API HTTP
- Logging des trades dans trades.csv
- Instructions d'installation et déploiement en README en tête de fichier

Usage :
1) Installer les dépendances :
   pip install ccxt pandas numpy python-dotenv

2) Créer un fichier .env avec :
   TELEGRAM_BOT_TOKEN=xxx
   TELEGRAM_CHAT_ID=yyy       # ton chat id pour recevoir les messages
   BITGET_API_KEY=xxx
   BITGET_API_SECRET=xxx
   BITGET_API_PASSPHRASE=xxx  # si nécessaire
   MODE=DEMO                  # ou LIVE

3) Lancer :
   python bot_ma_rsi_bitget.py

--- NOTE ---
- En MODE=DEMO, le bot ne place aucune order réelle ; il simule les entrées/sorties et enregistre dans trades.csv.
- En MODE=LIVE, vérifie bien tes clés API et permissions (spot trading) avant d'activer.

"""

import os
import time
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import csv
import traceback
from koyeb import Sandbox
from flask import Flask
import threading


app = Flask(__name__)

@app.route("/")
def home():
    return "Bot running"

def run_bot():
    while True:
        print("Bot en cours d'exécution...")
        time.sleep(60)

if __name__ == "__main__":
    t = threading.Thread(target=run_bot)
   
    



sandbox = Sandbox.create(
  image="ubuntu",
  name="hello-world",
  wait_ready=True,
)

result = sandbox.exec("echo 'Hello World'")
print(result.stdout.strip())

sandbox.delete()

load_dotenv()

# Config
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
BITGET_API_KEY = os.getenv('BITGET_API_KEY')
BITGET_API_SECRET = os.getenv('BITGET_API_SECRET')
BITGET_API_PASSPHRASE = os.getenv('BITGET_API_PASSPHRASE')
MODE = os.getenv('MODE', 'DEMO').upper()  # DEMO or LIVE

SYMBOL = os.getenv('SYMBOL', 'PEPE/USDT')   # ex: 'PEPE/USDT'
TIMEFRAME = os.getenv('TIMEFRAME', '1h')    # ex: '1h'
MA_PERIOD = int(os.getenv('MA_PERIOD', '50'))
RSI_PERIOD = int(os.getenv('RSI_PERIOD', '14'))
RSI_BUY = float(os.getenv('RSI_BUY', '30'))
RSI_SELL = float(os.getenv('RSI_SELL', '70'))
TRADE_SIZE_USDT = float(os.getenv('TRADE_SIZE_USDT', '5'))  # montant par trade en USDT (demo)
             # ou LIVE en français'
nombre_str ="60" 


try:
    POLL_INTERVAL = int(os.getenv('POLL_INTERVAL', nombre_str))  # en secondes
except ValueError:
    POLL_INTERVAL = 60  # défaut 60 secondes

# Setup exchange (ccxt)


exchange = ccxt.bitget({
    "apiKey": BITGET_API_KEY,
    "secret": BITGET_API_SECRET,
    "enableRateLimit": True,
    "options": {
        "adjustForTimeDifference": True
    }
})

balance = exchange.load_markets()


symbol = "PEPE/USDT"
timeframe = "5m"
LIMIT = 10


_ =exchange.load_markets()
# 1 heure en secondes
data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=LIMIT)

print(data)

# Note: Bitget ccxt may need extra params for passphrase or subaccount. Ajuste si besoin.²

# Utilities

def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print('Telegram non configuré. Message:', text)
        return
    try:
        import requests
        url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
        payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': text}
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print('Erreur Telegram:', e)

"""
def get_ohlcv(symbol, timeframe, limit=100):
        # ccxt uses symbol format like 'PEPE/USDT'
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print('Erreur fetch_ohlcv,Erreur API:', e)

    return None
"""
def get_ohlcv(symbol, timeframe, limit=100):
    try:
        ohlcv = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            limit=limit,
            params={
                "productType": "USDT-FUTURES"
            }
        )

        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    except Exception as e:
        print("❌ Erreur fetch_ohlcv Bitget:", e)
        
         
        return None


def compute_indicators(df):
    df = df.copy()
    df['close'] = df['close'].astype(float)
    df['ma'] = df['close'].rolling(MA_PERIOD).mean()
    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    return df


# Simple trade record
TRADE_LOG = 'trades.csv'

if not os.path.exists(TRADE_LOG):
    with open(TRADE_LOG, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'symbol', 'side', 'price', 'size_usdt', 'mode'])


# Position state (demo)
pos = {
    'in_position': False,
    'entry_price': None,
    'side': None,
}


def place_order_demo(side, price, size_usdt):
    # Simule taille en quantité = size_usdt / price
    qty = size_usdt / price if price > 0 else 0
    now = datetime.utcnow().isoformat()
    with open(TRADE_LOG, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([now, SYMBOL, side, price, size_usdt, 'DEMO'])
    print(f'[DEMO ORDER] {side} {qty:.6f} @ {price} ({size_usdt} USDT)')
    send_telegram(f'[DEMO ORDER] {side} {qty:.6f} {SYMBOL} @ {price} ({size_usdt} USDT)')


def place_order_live(side, price, size_usdt):
    # Exemple basique pour créer un market order en spot
    try:
        # calcul qty
        qty = size_usdt / price
        symbol_ccxt = SYMBOL
        # ccxt create_order parameters may differ per exchange — vérifie la doc ccxt/bitget
        order = exchange.create_market_buy_order(symbol_ccxt, qty) if side == 'BUY' else exchange.create_market_sell_order(symbol_ccxt, qty)
        now = datetime.utcnow().isoformat()
        with open(TRADE_LOG, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([now, SYMBOL, side, price, size_usdt, 'LIVE'])
        send_telegram(f'[LIVE ORDER] {side} {qty:.6f} {SYMBOL} executed @ {price}')
        return order
    except Exception as e:
        tb = traceback.format_exc()
        print('Erreur place_order_live:', e, tb)
        send_telegram('Erreur order live: ' + str(e))
        return None


# Main loop

def main_loop():
    global pos
    send_telegram(f'Bot démarré en mode {MODE} — stratégie: MA({MA_PERIOD}) + RSI({RSI_PERIOD}) — symbole: {SYMBOL} — timeframe: {TIMEFRAME}')
    while True:
        try:
            df = get_ohlcv(SYMBOL, TIMEFRAME, limit=100)
            if df is None or df.empty:
                time.sleep(POLL_INTERVAL)
                continue

            df = compute_indicators(df)
            last = df.iloc[-1]
            prev = df.iloc[-2]
            price = float(last['close'])
            ma = float(last['ma']) if not np.isnan(last['ma']) else None
            rsi = float(last['rsi']) if not np.isnan(last['rsi']) else None

            print(f"[{datetime.utcnow().isoformat()}] price={price:.6f} ma={ma} rsi={rsi:.2f}")

            # Signal d'achat: RSI < RSI_BUY et prix casse à la hausse la MA (cross up)
            buy_signal = False
            sell_signal = False
            if ma is not None and rsi is not None:
                # cross up detection: prev.close < prev.ma and last.close > last.ma
                cross_up = (float(prev['close']) < float(prev['ma'])) and (float(last['close']) > float(last['ma']))
                cross_down = (float(prev['close']) > float(prev['ma'])) and (float(last['close']) < float(last['ma']))
                if rsi < RSI_BUY and cross_up:
                    buy_signal = True
                if rsi > RSI_SELL and cross_down:
                    sell_signal = True

            # Trading logic simple
            if not pos['in_position'] and buy_signal:
                # Enter long
                if MODE == 'DEMO':
                    place_order_demo('BUY', price, TRADE_SIZE_USDT)
                else:
                    place_order_live('BUY', price, TRADE_SIZE_USDT)
                pos['in_position'] = True
                pos['entry_price'] = price
                pos['side'] = 'LONG'
                send_telegram(f'Entered LONG {SYMBOL} @ {price:.6f} — RSI={rsi:.2f} MA={ma:.6f}')

            elif pos['in_position'] and sell_signal:
                # Exit long
                if MODE == 'DEMO':
                    place_order_demo('SELL', price, TRADE_SIZE_USDT)
                else:
                    place_order_live('SELL', price, TRADE_SIZE_USDT)
                send_telegram(f'Exited LONG {SYMBOL} @ {price:.6f} — entry was {pos.get("entry_price")}')
                pos['in_position'] = False
                pos['entry_price'] = None
                pos['side'] = None

            # Optional: small sleep
            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            print('Arrêt manual (KeyboardInterrupt)')
            send_telegram('Bot stopped by user')
            break
        except Exception as e:
            tb = traceback.format_exc()
            print('Erreur main loop:', e, tb)
            send_telegram('Erreur bot: ' + str(e))
            time.sleep(POLL_INTERVAL)


if __name__ == '__main__':
    main_loop()
