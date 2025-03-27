#rework des fonction de stop loss et gain qui sont trop abuser
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
import asyncio
import threading

app = Flask("")
@app.route('/')
def home() :
    return "le bot est en ligne"
def run():
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0',port=port)
def keep_alive():
    t=Thread(target=run)
    t.start()
load_dotenv()


exchange = ccxt.binanceus()

# Paramètres
SYMBOLS = "SOL/EUR"
TIMEFRAME = "1m"
WINDOW_OHLCV = 200  # Pour les prédictions dans analyze_market
LIMIT_TRAIN = 1000 # Pour l’entraînement et test total
WEBHOOK_URL = os.getenv('WEBHOOK_URL')


# Récupération des données OHLCV
def fetch_ohlcv(symbol, timeframe, limit):
    all_ohlcv = []
    total_periods = limit
    max_per_call = 1000
    while len(all_ohlcv) < total_periods:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=min(max_per_call, total_periods - len(all_ohlcv)))
        all_ohlcv.extend(ohlcv)
        time.sleep(1)
    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.tail(total_periods)

# Calcul du RSI
def calculate_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

# Calcul de la moyenne mobile optimisée
def calculate_moving_average(df, window=5):
    return df["close"].ewm(span=window, adjust=False).mean()

# Calcul des EMA
def calculate_ema(df, window):
    return df["close"].ewm(span=window, adjust=False).mean()

# Calcul du MACD
def calculate_macd(df):
    ema_12 = calculate_ema(df, 12)
    ema_26 = calculate_ema(df, 26)
    macd = ema_12 - ema_26
    signal_line = macd.ewm(span=9, adjust=False).mean()
    return macd, signal_line

# Calcul du Momentum
def calculate_momentum(df, window=10):
    if len(df) < window + 1:
        return 0.0
    try:
        cloture_actuelle = df["close"].iloc[-1]
        cloture_precedente = df["close"].iloc[-window - 1]
        if cloture_precedente == 0:
            return 0.0
        momentum = (cloture_actuelle - cloture_precedente) / cloture_precedente * 100
        return momentum
    except (IndexError, KeyError, TypeError) as e:
        print(f"Erreur dans calculate_momentum: {e}")
        return 0.0
# Modèle ML avec XGBoost


def calculate_atr(df, window=14):
    tr = pd.concat([df["high"] - df["low"],
                    np.abs(df["high"] - df["close"].shift()),
                    np.abs(df["low"] - df["close"].shift())], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()
# Analyse et décision avec vérification
def analyze_market(df, rsi_series, symbol, model):
    if df.empty or rsi_series.empty or model is None:
        return "Erreur (données manquantes ou modèle non disponible)", 0, None, None, "Neutre", None

    dernier_rsi = rsi_series.iloc[-1]
    ma = calculate_moving_average(df)
    trend_pct = ((ma.iloc[-1] - ma.iloc[0]) / ma.iloc[0]) * 100 if ma.iloc[0] != 0 else 0
    dernier_prix = df["close"].iloc[-1]
    atr = calculate_atr(df).iloc[-1]
    momentum = calculate_momentum(df)
    moving_average = calculate_moving_average(df).iloc[-1]
    if len(df) < 20:
        print(f"Erreur : Pas assez de périodes ({len(df)}) pour calculer volume_rel pour {symbol}")
        volume_rel = 1.0
    else:
        volume_rel = df["volume"].iloc[-1] / df["volume"].rolling(window=20).mean().iloc[-1]

    ema_10 = calculate_ema(df, 10)
    ema_50 = calculate_ema(df, 50)
    macd, signal_line = calculate_macd(df)
    features = pd.DataFrame({
        "rsi": [dernier_rsi],
        "ema_diff": [ema_10.iloc[-1] - ema_50.iloc[-1]],
        "macd_diff": [macd.iloc[-1] - signal_line.iloc[-1]],
        "momentum": [momentum],
        "volume_rel": [volume_rel],
        "moving_average": [moving_average],
        "atr": [atr],
        "volume_change": [df["volume"].pct_change(periods=1).iloc[-1] * 100 if len(df) > 1 else 0],
        "ema_200": [calculate_ema(df, 200).iloc[-1]],
        "ema_100": [calculate_ema(df, 100).iloc[-1]],
        "ema_10": [calculate_ema(df, 10).iloc[-1]],
        "ema_50": [calculate_ema(df, 50).iloc[-1]],
        "close_pct_change_5": [df["close"].pct_change(periods=5).iloc[-1] * 100 if len(df) > 5 else 0]
    })
    # Système de confiance avec un seul seuil
    proba = model.predict_proba(features)[0]
    if max(proba) > 0.6:  # Seuil unique à 53%
        prediction = np.argmax(proba)
        confidence_factor = max(proba)  # Utilisé pour ajuster les seuils
    else:
        prediction = -1

    signal = "Neutre"
    stop_loss = None
    take_profit = None


    if prediction == 1:
        df = df.copy()
        signal = "Haussier (XGBoost Prediction)"
        df.loc[:, "high_low"] = df["high"] - df["low"]
        df.loc[:, "high_close"] = abs(df["high"] - df["close"].shift(1))
        df.loc[:, "low_close"] = abs(df["low"] - df["close"].shift(1))

        df.loc[:, "true_range"] = df[["high_low", "high_close", "low_close"]].max(axis=1)
        atr = df["true_range"].rolling(window=14).mean().iloc[-1]  # ATR 14 périodes

        # Ajustement dynamique plus faible
        atr_dynamic_factor = 0.5 # 🔥 Facteur ATR réduit

        # 🟢 2. Calcul des Supports et Résistances (20 dernières bougies)
        support = df["low"].rolling(window=20).min().iloc[-1]
        resistance = df["high"].rolling(window=20).max().iloc[-1]

        # Dernier prix connu
        dernier_prix = df["close"].iloc[-1]

        # 🔥 Ratio Risk-Reward réduit
        risk_reward_ratio = 2

        # 🟢 3. Calcul du Stop-Loss et du Take-Profit (plus serré)
        stop_loss = max(support, dernier_prix - (atr * atr_dynamic_factor*risk_reward_ratio*confidence_factor))
        take_profit = min(resistance, dernier_prix + (atr * atr_dynamic_factor * risk_reward_ratio*confidence_factor))

        # Affichage des résultats
    elif prediction == 0:
        signal = "Baissier"
    else:
        signal="neutre"

    subtle_prediction = "Aucune indication claire"
    if signal == "Neutre" and prediction == -1:
        last_change = df["close"].iloc[-1] - df["close"].iloc[-2] if len(df) > 1 else 0
        if trend_pct > 0.1 and last_change > 0:
            subtle_prediction = "Légère hausse probable"
        elif trend_pct < -0.1 and last_change < 0:
            subtle_prediction = "Légère baisse probable"

    return signal, trend_pct, stop_loss, take_profit, subtle_prediction, prediction, proba
keep_alive()
print("Modèle chargé avec succès !")
# Ensure correct event loop policy for Windows
client = discord.Client(intents=discord.Intents.all())
intents = discord.Intents.all()
bot = commands.Bot(command_prefix="$", intents=intents)
predictions_finales =[]
model = XGBClassifier()
model.load_model('model_solana_eur.json')
def test_model(model):
    df_all = fetch_ohlcv(SYMBOLS, TIMEFRAME, LIMIT_TRAIN)
    if df_all.empty:
        print("Échec de la récupération des données historiques")
        return
    print(f"Dernière donnée OHLCV : {df_all['timestamp'].iloc[-1]} (UTC)")


    print(f"recuperation du modèle ")
    if model is None:
        print("Échec de l’entraînement du modèle")
        return

    predictions = []

    timer = df_all["timestamp"].iloc[-1]
    close = df_all["close"].iloc[-1]
    rsi_window = calculate_rsi(fetch_ohlcv(SYMBOLS, TIMEFRAME, WINDOW_OHLCV))
    signal, trend_pct, stop_loss, take_profit, subtle_prediction, prediction, proba = analyze_market(
        df_all, rsi_window, SYMBOLS, model)
    predictions.append(timer)
    predictions.append(signal)
    predictions.append(proba)
    predictions.append(take_profit)
    predictions.append(stop_loss)
    print(predictions)
    predictions_finales.append(predictions)
@client.event
async def on_message(message):
    if message.author.bot:
        return
    await message.channel.send("historique")
    await message.channel.send(predictions_finales)
async def run_training_loop():
    while True:
        test_model(model)  # Assure-toi que 'model' est bien défini et accessible
        await asyncio.sleep(60)  # Pause de 60 secondes avant de recommencer l'entraînement
def start_training_in_background():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_training_loop())

# Lancer l'entraînement dans un thread séparé
def start_training_thread():
    training_thread = threading.Thread(target=start_training_in_background)
    training_thread.daemon = True  # Assure-toi que le thread se termine quand le programme principal se termine
    training_thread.start()

# Démarrage du bot et du thread d'entraînement en arrière-plan
if __name__ == "__main__":
    start_training_thread()  # Lancer l'entraînement en arrière-plan
    client.run(token=WEBHOOK_URL)  # Remplace WEBHOOK_URL par ton vrai token
