import ccxt
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import os
import time
from dotenv import load_dotenv
from flask import Flask
from threading import Thread
import discord
from discord.ext import commands
import asyncio
import threading
import requests
from datetime import timedelta

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


load_dotenv()
model = XGBClassifier()
model.load_model('model_solana_eur_minute.json')

exchange = ccxt.binance()

# Paramètres
SYMBOLS = "SOL/EUR"
TIMEFRAME = "1m"
WINDOW_OHLCV = 200  # Pour les prédictions dans analyze_market
LIMIT_TEST = 200  # Pour l’entraînement et test total
WEBHOOK_URL = os.getenv('WEBHOOK_URL')
Rotation = 0


# Récupération des données OHLCV
def fetch_ohlcv(symbol, timeframe, limit):
    all_ohlcv = []
    a = 0
    max_per_call = 1000

    # Déterminer la date de départ (actuelle moins 50 000 minutes)
    now = pd.to_datetime("now")
    since = (now - timedelta(minutes=limit)).timestamp()  # Conversion en secondes
    since = int(since) * 1000
    while len(all_ohlcv) <= limit - 1:
        a = a + 1
        remaining = limit - len(all_ohlcv)
        fetch_limit = min(max_per_call, remaining)

        # Récupération des bougies
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=fetch_limit)
        print(f"Récupéré {len(ohlcv)} bougies.")
        time.sleep(1)

        # Si des doublons sont détectés, on les supprime
        try:
            if len(all_ohlcv) > 0 and len(ohlcv) > 0 and ohlcv[0][0] <= all_ohlcv[-1][0]:
                print("⚠️ Doublon détecté, suppression du premier élément.")
                ohlcv = ohlcv[1:]  # Retirer le doublon

            if len(ohlcv) > 0:
                since = ohlcv[-1][0] + 1  # Met à jour la valeur de `since` pour le prochain appel
            else:
                since = None  # Si aucune donnée n'a été récupérée, on arrête la boucle

        except Exception as e:
            print(f"Problème lors du traitement : {e}")
            break

        if len(ohlcv) == 0:
            print("Aucune donnée récupérée, arrêt de la boucle.")
            break  # Si aucune donnée n'est récupérée, on arrête

        # Ajout des données récupérées à la liste
        all_ohlcv.extend(ohlcv)
        print(f"Total des bougies récupérées : {len(all_ohlcv)}")

        # Si on atteint la limite, on arrête la boucle
        if len(all_ohlcv) >= limit:
            print("Limite atteinte.")
            break

    # Transformation en DataFrame
    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.tail(limit)


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


def train_ml_model(df):
    # Calcul des features
    global model
    features = pd.DataFrame({
        "rsi": 100 - (100 / (1 + (df["close"].diff(1).where(df["close"].diff(1) > 0, 0).rolling(window=14).mean() /
                                  df["close"].diff(1).where(df["close"].diff(1) < 0, 0).rolling(window=14).mean()))),
        "ema_diff": df["close"].ewm(span=10).mean() - df["close"].ewm(span=50).mean(),
        "macd_diff": df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean() -
                     (df["close"].ewm(span=9).mean() - df["close"].ewm(span=26).mean()),
        "momentum": df["close"].pct_change(periods=10) * 100,
        "volume_rel": df["volume"] / df["volume"].rolling(window=20).mean(),
        "moving_average": df["close"].rolling(window=20).mean(),
        "atr": df["high"].rolling(window=14).max() - df["low"].rolling(window=14).min(),
        "volume_change": df["volume"].pct_change(periods=1) * 100,
        "ema_200": df["close"].ewm(span=200).mean(),
        "ema_100": df["close"].ewm(span=100).mean(),
        "ema_10": df["close"].ewm(span=10).mean(),
        "ema_50": df["close"].ewm(span=50).mean(),
        "close_pct_change_5": df["close"].pct_change(periods=5) * 100,
        "stochastic_oscillator": 100 * ((df["close"] - df["low"].rolling(window=14).min()) /
                                        (df["high"].rolling(window=14).max() - df["low"].rolling(window=14).min())),
        "daily_returns": df["close"].pct_change() * 100,
        "volatility": df["close"].rolling(window=14).std(),
        "fractal_dimension": np.log(df["close"].rolling(window=10).std()) / np.log(10),
        "weighted_moving_average": df["close"].rolling(window=10).apply(
            lambda x: np.dot(x, np.arange(1, 11)) / np.sum(np.arange(1, 11)), raw=True),

    }).dropna()

    # Remplacer les valeurs infinies ou NaN par des valeurs par défaut (par exemple, la moyenne ou la médiane)
    features.replace([np.inf, -np.inf], np.nan, inplace=True)  # Remplacer les inf par NaN
    features.fillna(features.mean(), inplace=True)  # Remplacer les NaN par la moyenne des colonnes

    # Ciblage pour prédiction (Next Close > Close actuel = 1 ou non)
    target = (df["close"].shift(-1) > df["close"]).astype(int).reindex(features.index).dropna()
    features = features.iloc[:-1]
    target = target.iloc[:-1]

    if len(features) != len(target) or len(features) < 1:
        print(
            f"Données insuffisantes ou incohérentes pour l’entraînement: features={len(features)}, target={len(target)}")
        return None
    existing_model = model
    model = existing_model
    model.fit(features, target, xgb_model=model)  # Mise à jour du modèle
    print("Modèle mis à jour avec de nouvelles données!")
    return model


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
        "rsi": 100 - (100 / (1 + (df["close"].diff(1).where(df["close"].diff(1) > 0, 0).rolling(window=14).mean() /
                                  df["close"].diff(1).where(df["close"].diff(1) < 0, 0).rolling(window=14).mean()))),
        "ema_diff": df["close"].ewm(span=10).mean() - df["close"].ewm(span=50).mean(),
        "macd_diff": df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean() -
                     (df["close"].ewm(span=9).mean() - df["close"].ewm(span=26).mean()),
        "momentum": df["close"].pct_change(periods=10) * 100,
        "volume_rel": df["volume"] / df["volume"].rolling(window=20).mean(),
        "moving_average": df["close"].rolling(window=20).mean(),
        "atr": df["high"].rolling(window=14).max() - df["low"].rolling(window=14).min(),
        "volume_change": df["volume"].pct_change(periods=1) * 100,
        "ema_200": df["close"].ewm(span=200).mean(),
        "ema_100": df["close"].ewm(span=100).mean(),
        "ema_10": df["close"].ewm(span=10).mean(),
        "ema_50": df["close"].ewm(span=50).mean(),
        "close_pct_change_5": df["close"].pct_change(periods=5) * 100,
        "stochastic_oscillator": 100 * ((df["close"] - df["low"].rolling(window=14).min()) /
                                        (df["high"].rolling(window=14).max() - df["low"].rolling(window=14).min())),
        "daily_returns": df["close"].pct_change() * 100,
        "volatility": df["close"].rolling(window=14).std(),
        "fractal_dimension": np.log(df["close"].rolling(window=10).std()) / np.log(10),
        "weighted_moving_average": df["close"].rolling(window=10).apply(
            lambda x: np.dot(x, np.arange(1, 11)) / np.sum(np.arange(1, 11)), raw=True),

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
        atr_dynamic_factor = 0.5  # 🔥 Facteur ATR réduit

        # 🟢 2. Calcul des Supports et Résistances (20 dernières bougies)
        support = df["low"].rolling(window=20).min().iloc[-1]
        resistance = df["high"].rolling(window=20).max().iloc[-1]

        # Dernier prix connu
        dernier_prix = df["close"].iloc[-1]

        # 🔥 Ratio Risk-Reward réduit
        risk_reward_ratio = 1.25

        # 🟢 3. Calcul du Stop-Loss et du Take-Profit (plus serré)
        stop_loss = max(support, dernier_prix - (atr * atr_dynamic_factor * risk_reward_ratio * confidence_factor))
        take_profit = min(resistance, dernier_prix + (atr * atr_dynamic_factor * risk_reward_ratio * confidence_factor))

        # Affichage des résultats
    elif prediction == 0:
        signal = "Baissier"
    else:
        signal = "neutre"

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
predictions_finales = []


def test_model():
    global Rotation, model
    df_all = fetch_ohlcv(SYMBOLS, TIMEFRAME, LIMIT_TEST)
    if Rotation == 100:
        model = train_ml_model(df_all.iloc[-100:])
        Rotation = 0
    else:
        Rotation += 1

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
    high = df_all["high"].iloc[-1]
    low = df_all["low"].iloc[-1]
    rsi_window = calculate_rsi(fetch_ohlcv(SYMBOLS, TIMEFRAME, WINDOW_OHLCV))
    signal, trend_pct, stop_loss, take_profit, subtle_prediction, prediction, proba = analyze_market(
        df_all, rsi_window, SYMBOLS, model)
    predictions.append(timer)
    predictions.append(close)
    predictions.append(high)
    predictions.append(low)
    predictions.append(take_profit)
    predictions.append(stop_loss)
    predictions.append(prediction)
    print(predictions)
    predictions_finales.append(predictions)


def calcul_reussite():
    resultat_gagnant = 0
    resultat_perdant = 0
    total_resultat = 0
    gain = 0
    for i in range(len(predictions_finales) - 4):
        entry_price = predictions_finales[i][1]
        prediction = predictions_finales[i][6]

        next_high = predictions_finales[i + 1][2]
        next_next_high = predictions_finales[i + 2][2]
        next_high3 = predictions_finales[i + 3][2]
        next_high4 = predictions_finales[i + 4][2]

        next_low = predictions_finales[i + 1][3]
        next_next_low = predictions_finales[i + 2][3]
        next_low3 = predictions_finales[i + 3][3]
        next_low4 = predictions_finales[i + 4][3]

        take_profit = predictions_finales[i][4]
        stop_loss = predictions_finales[i][5]

        if prediction == 1:
            if take_profit is not None and stop_loss is not None:
                if next_high >= take_profit:
                    resultat_gagnant += 1
                    total_resultat += 1
                    gain += take_profit - entry_price
                elif next_low <= stop_loss:
                    resultat_perdant += 1
                    total_resultat += 1
                    gain +=  stop_loss - entry_price
                elif next_next_high >= take_profit:
                    resultat_gagnant += 1
                    total_resultat += 1
                    gain += take_profit - entry_price
                elif next_next_low <= stop_loss:
                    resultat_perdant += 1
                    total_resultat += 1
                    gain += stop_loss - entry_price
                elif next_high3 >= take_profit:
                    resultat_gagnant += 1
                    total_resultat += 1
                    gain += take_profit - entry_price
                elif next_low3 <= stop_loss:
                    resultat_perdant += 1
                    total_resultat += 1
                    gain += stop_loss - entry_price
                elif next_high4 >= take_profit:
                    resultat_gagnant += 1
                    total_resultat += 1
                    gain += take_profit - entry_price
                elif next_low4 <= stop_loss:
                    resultat_perdant += 1
                    total_resultat += 1
                    gain += stop_loss - entry_price
                else:
                    total_resultat += 1
    if total_resultat == 0:
        pourcentage_gagnant = 0
        pourcentage_perdant = 0
        pourcentage_neutre = 0
        resultat_gagnant = 0
        resultat_perdant = 0
        total_resultat = 0
    else:
        pourcentage_gagnant = resultat_gagnant / total_resultat * 100
        pourcentage_perdant = resultat_perdant / total_resultat * 100
        pourcentage_neutre = (total_resultat - resultat_gagnant - resultat_perdant) / total_resultat * 100
        print("🔁 Réentraînement du modèle...")

    return pourcentage_gagnant, pourcentage_perdant, pourcentage_neutre, resultat_gagnant, resultat_perdant, total_resultat,gain


@client.event
async def on_message(message):
    if message.author.bot:
        return
    pourcentage_gagnant, pourcentage_perdant, pourcentage_neutre, resultat_gagnant, resultat_perdant, total_resultat,gain = calcul_reussite()
    await message.channel.send(
        f"✅ Gagnants: {pourcentage_gagnant:.2f}% ({resultat_gagnant})\n❌ Perdants: {pourcentage_perdant:.2f}% ({resultat_perdant})\n❓ Neutres: {pourcentage_neutre:.2f}%\nTotal: {total_resultat}\n GAIN: {gain}")


async def run_trading_loop():
    while True:
        test_model()
        requests.get(URL)
        await asyncio.sleep(60)  # Pause de 60 secondes avant de recommencer l'entraînement


def start_background_tasks():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(run_trading_loop())
    loop.run_forever()


# Lancer l'entraînement dans un thread séparé
def start_training_thread():
    training_thread = threading.Thread(target=start_background_tasks)
    training_thread.daemon = True  # Assure-toi que le thread se termine quand le programme principal se termine
    training_thread.start()


# Démarrage du bot et du thread d'entraînement en arrière-plan
if __name__ == "__main__":
    training_thread = threading.Thread(target=start_background_tasks)
    training_thread.daemon = True
    training_thread.start()

    client.run(token=WEBHOOK_URL)
