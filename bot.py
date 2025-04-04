# ✅ VERSION CORRIGÉE ET OPÉRATIONNELLE AVEC ENTRAÎnment toutes les 50000 minutes
# ET TEST + PRÉDICTIONS TAKE PROFIT / STOP LOSS

import asyncio
import ccxt
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import time
import os
from dotenv import load_dotenv
from flask import Flask
from threading import Thread
import discord
from discord.ext import commands
import threading
import requests
from datetime import timedelta

# === FLASK APP KEEP-ALIVE ===
app = Flask("")
URL = "https://test-zpdc.onrender.com"

@app.route('/')
def home():
    return "le bot est en ligne"

def run():
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

def keep_alive():
    t = Thread(target=run)
    t.start()

# === VARIABLES GLOBALES ===
load_dotenv()
exchange = ccxt.binance()
SYMBOLS = "SOL/EUR"
TIMEFRAME = "1m"
WINDOW_OHLCV = 250
LIMIT_TRAIN = 250
WEBHOOK_URL = os.getenv('WEBHOOK_URL')
model = None
feature_order = []
predictions_finales = []

# === DONNÉES ET INDICATEURS ===
def fetch_ohlcv(symbol, timeframe, limit):
    try:
        since = int((pd.Timestamp.now() - timedelta(minutes=limit)).timestamp()) * 1000
        data = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        print(f"Erreur fetch_ohlcv: {e}")
        return pd.DataFrame()

def calculate_indicators(df):
    df = df.copy()
    df["rsi"] = calculate_rsi(df)
    df["ema_diff"] = calculate_ema(df, 10) - calculate_ema(df, 50)
    macd, signal = calculate_macd(df)
    df["macd_diff"] = macd - signal
    df["momentum"] = df["close"].pct_change(periods=10) * 100
    df["volume_rel"] = df["volume"] / df["volume"].rolling(window=20).mean()
    df["moving_average"] = calculate_moving_average(df)
    df["atr"] = calculate_atr(df)
    df["volume_change"] = df["volume"].pct_change() * 100
    df["ema_10"] = calculate_ema(df, 10)
    df["ema_50"] = calculate_ema(df, 50)
    df["ema_100"] = calculate_ema(df, 100)
    df["ema_200"] = calculate_ema(df, 200)
    df["close_pct_change_5"] = df["close"].pct_change(periods=5) * 100
    return df

def calculate_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=period).mean()
    avg_loss = loss.ewm(span=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs)).fillna(50)

def calculate_ema(df, window):
    return df["close"].ewm(span=window, adjust=False).mean()

def calculate_macd(df):
    ema_12 = calculate_ema(df, 12)
    ema_26 = calculate_ema(df, 26)
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9).mean()
    return macd, signal

def calculate_moving_average(df, window=5):
    return df["close"].ewm(span=window).mean()

def calculate_atr(df, window=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()

# === ENTRAÎnment ===
async def train_ml_model():
    global model, feature_order
    df = fetch_ohlcv(SYMBOLS, TIMEFRAME, 50000)
    df = calculate_indicators(df)
    df.dropna(inplace=True)
    if df.empty:
        print("Pas assez de données pour entraîner.")
        return
    features = df[[
        "rsi", "ema_diff", "macd_diff", "momentum", "volume_rel", "moving_average",
        "atr", "volume_change", "ema_200", "ema_100", "ema_10", "ema_50", "close_pct_change_5"
    ]]
    target = (df["close"].shift(-1) > df["close"]).astype(int)
    features, target = features.iloc[:-1], target.iloc[:-1]
    model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03)
    model.fit(features, target)
    feature_order = features.columns.tolist()
    print("Modèle entraîné avec succès.")

# === TEST DU MODÈLE ===
def test_model():
    if model is None:
        print("Modèle non disponible.")
        return
    df = fetch_ohlcv(SYMBOLS, TIMEFRAME, LIMIT_TRAIN)
    df = calculate_indicators(df)
    df.dropna(inplace=True)
    features = df[feature_order].iloc[-1:]
    proba = model.predict_proba(features)[0]
    prediction = int(np.argmax(proba))
    close = df["close"].iloc[-1]
    high = df["high"].iloc[-1]
    low = df["low"].iloc[-1]
    stop_loss = close - df["atr"].iloc[-1] * 1.2
    take_profit = close + df["atr"].iloc[-1] * 1.2
    print(f"Prediction: {prediction}, Proba: {proba}, TP: {take_profit}, SL: {stop_loss}")
    predictions_finales.append([
        df["timestamp"].iloc[-1], close, high, low, take_profit, stop_loss, prediction
    ])

# === ÉVALUATION DU MODÈLE ===
def calcul_reussite():
    r_g, r_p, total = 0, 0, 0
    for i in range(len(predictions_finales) - 2):
        pred = predictions_finales[i][6]
        tp = predictions_finales[i][4]
        sl = predictions_finales[i][5]
        highs = [predictions_finales[i+1][2], predictions_finales[i+2][2]]
        lows = [predictions_finales[i+1][3], predictions_finales[i+2][3]]
        if pred == 1:
            total += 1
            if any(h >= tp for h in highs): r_g += 1
            elif any(l <= sl for l in lows): r_p += 1
    if total == 0: return 0, 0, 0, 0, 0, 0
    return (r_g/total)*100, (r_p/total)*100, (total-r_g-r_p)/total*100, r_g, r_p, total

# === DISCORD BOT ===
client = discord.Client(intents=discord.Intents.all())
intents = discord.Intents.all()
bot = commands.Bot(command_prefix="$", intents=intents)

@client.event
async def on_message(message):
    if message.author.bot: return
    g, p, n, r_g, r_p, total = calcul_reussite()
    await message.channel.send(f"✅ Gagnants: {g:.2f}% ({r_g})\n❌ Perdants: {p:.2f}% ({r_p})\n❓ Neutres: {n:.2f}%\nTotal: {total}")

# === ENTRAÎnment en arrière-plan ===
async def run_loop():
    while True:
        await train_ml_model()
        test_model()
        requests.get(URL)
        await asyncio.sleep(60)  # Pour test, sinon 60 * 50000 secondes

def start_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_loop())

# === MAIN ===
if __name__ == "__main__":
    keep_alive()
    threading.Thread(target=start_loop, daemon=True).start()
    client.run(WEBHOOK_URL)
