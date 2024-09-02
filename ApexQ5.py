##############################################################################
# lorsque le chargement                  #
#de l'ancien echoue c un problem d'incompatibilité scikitlearn                #
# de version different                                                       #
# on resous ce problem en supprimant                #Volgat#mikeamega
#les anciens modèles sauvegardés :                  #ApexQuqntum                         #
#Supprimez tous les fichiers .joblib dans le dossier "apex_quantum_data"     #
#Cela forcera le système à créer de nouveaux modèles compatibles avec        #
# votre version actuelle de scikit-learn.                                    #                                           #
##############################################################################

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import time
from datetime import datetime
import matplotlib.pyplot as plt
import os
import joblib
import json

# Paramètres globaux
PROGRAM_NAME = "APEX Quantum"
SYMBOLS = ['XAUUSD', 'BTCUSD']
TIMEFRAME = mt5.TIMEFRAME_M5
LOOKBACK_PERIOD = 100
INITIAL_BALANCE = 10.0
MAX_RISK_PER_TRADE = 0.05
MAX_DAILY_RISK = 0.20
LEVERAGE = 500
SAVE_FREQUENCY = 100

# Paramètres de connexion MT5 (à remplacer par vos informations)
MT5_ACCOUNT = 179775260
MT5_PASSWORD = ""
MT5_SERVER = ""

DATA_FOLDER = "apex_quantum_data"
INTERFACE_FILE = "apex_quantum_interface.txt"

class APEXQuantumTradingSystem:
    def __init__(self):
        self.models = {symbol: self.create_models() for symbol in SYMBOLS}
        self.performance = {symbol: {'balance': INITIAL_BALANCE, 'equity': [], 'drawdown': []} for symbol in SYMBOLS}
        self.daily_risk = 0
        self.last_reset = datetime.now().date()
        self.scalers = {symbol: StandardScaler() for symbol in SYMBOLS}
        self.is_running = False
        
        if not os.path.exists(DATA_FOLDER):
            os.makedirs(DATA_FOLDER)
        
        self.load_persistent_data()

    def create_models(self):
        rf = RandomForestClassifier(n_estimators=100)
        lstm = Sequential([
            LSTM(50, return_sequences=True, input_shape=(LOOKBACK_PERIOD, 5)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1, activation='sigmoid')
        ])
        lstm.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
        return {'rf': rf, 'lstm': lstm}

    def load_persistent_data(self):
        for symbol in SYMBOLS:
            rf_path = os.path.join(DATA_FOLDER, f"{symbol}_rf_model.joblib")
            lstm_path = os.path.join(DATA_FOLDER, f"{symbol}_lstm_model.h5")
            scaler_path = os.path.join(DATA_FOLDER, f"{symbol}_scaler.joblib")
            
            if os.path.exists(rf_path):
                self.models[symbol]['rf'] = joblib.load(rf_path)
            if os.path.exists(lstm_path):
                self.models[symbol]['lstm'] = load_model(lstm_path)
            if os.path.exists(scaler_path):
                self.scalers[symbol] = joblib.load(scaler_path)

        performance_path = os.path.join(DATA_FOLDER, "performance.json")
        if os.path.exists(performance_path):
            with open(performance_path, 'r') as f:
                self.performance = json.load(f)

    def save_persistent_data(self):
        for symbol in SYMBOLS:
            joblib.dump(self.models[symbol]['rf'], os.path.join(DATA_FOLDER, f"{symbol}_rf_model.joblib"))
            self.models[symbol]['lstm'].save(os.path.join(DATA_FOLDER, f"{symbol}_lstm_model.h5"))
            joblib.dump(self.scalers[symbol], os.path.join(DATA_FOLDER, f"{symbol}_scaler.joblib"))

        with open(os.path.join(DATA_FOLDER, "performance.json"), 'w') as f:
            json.dump(self.performance, f)

    def initialize_mt5(self):
        if not mt5.initialize():
            print("Initialisation de MetaTrader 5 échouée")
            return False

        if not mt5.login(login=MT5_ACCOUNT, server=MT5_SERVER, password=MT5_PASSWORD):
            print("Échec de l'autorisation")
            mt5.shutdown()
            return False

        print("MetaTrader 5 version:", mt5.__version__)
        print("Compte connecté:", mt5.account_info().login)
        print("Balance:", mt5.account_info().balance)
        print("Symboles disponibles:", mt5.symbols_total())

        return True

    def prepare_data(self, symbol):
        bars = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, LOOKBACK_PERIOD * 2)
        df = pd.DataFrame(bars)
        df['body'] = abs(df['close'] - df['open'])
        df['wick_upper'] = df['high'] - np.maximum(df['open'], df['close'])
        df['wick_lower'] = np.minimum(df['open'], df['close']) - df['low']
        return df

    def detect_enhanced_price_action_pattern(self, df):
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        prev_prev_candle = df.iloc[-3]
        
        recent_volatility = df['high'].rolling(10).max() - df['low'].rolling(10).min()
        avg_volatility = recent_volatility.mean()
        current_volatility = recent_volatility.iloc[-1]
        
        avg_volume = df['tick_volume'].rolling(20).mean().iloc[-1]
        
        if (last_candle['close'] > last_candle['open'] and 
            prev_candle['close'] < prev_candle['open'] and
            last_candle['low'] < prev_candle['low'] and 
            last_candle['close'] > prev_candle['open'] and
            last_candle['tick_volume'] > avg_volume * 1.5 and
            current_volatility > avg_volatility * 1.2):
            if prev_candle['low'] < prev_prev_candle['low']:
                return 1  # Signal haussier fort
        
        elif (last_candle['close'] < last_candle['open'] and 
              prev_candle['close'] > prev_candle['open'] and
              last_candle['high'] > prev_candle['high'] and 
              last_candle['close'] < prev_candle['open'] and
              last_candle['tick_volume'] > avg_volume * 1.5 and
              current_volatility > avg_volatility * 1.2):
            if prev_candle['high'] > prev_prev_candle['high']:
                return -1  # Signal baissier fort
        
        elif (last_candle['close'] > last_candle['open'] and
              prev_candle['close'] > prev_candle['open'] and
              last_candle['low'] > prev_candle['low'] and
              last_candle['tick_volume'] > avg_volume):
            return 0.5  # Signal haussier modéré (continuation)
        
        elif (last_candle['close'] < last_candle['open'] and
              prev_candle['close'] < prev_candle['open'] and
              last_candle['high'] < prev_candle['high'] and
              last_candle['tick_volume'] > avg_volume):
            return -0.5  # Signal baissier modéré (continuation)
        
        return 0

    def update_models(self, symbol, df):
        X = df[['open', 'high', 'low', 'close', 'body']].values
        y = (df['close'].shift(-1) > df['close']).astype(int).values
        
        X = X[:-1]
        y = y[:-1]
        
        X_scaled = self.scalers[symbol].fit_transform(X)
        
        X_lstm = []
        for i in range(len(X_scaled) - LOOKBACK_PERIOD):
            X_lstm.append(X_scaled[i:i+LOOKBACK_PERIOD])
        X_lstm = np.array(X_lstm)
        
        y_lstm = y[LOOKBACK_PERIOD:]
        
        X_rf = X_scaled[LOOKBACK_PERIOD:]
        y_rf = y[LOOKBACK_PERIOD:]
        
        self.models[symbol]['rf'].fit(X_rf, y_rf)
        self.models[symbol]['lstm'].fit(X_lstm, y_lstm, epochs=1, verbose=0)

    def get_prediction(self, symbol, X):
        X_scaled = self.scalers[symbol].transform(X)
        X_rf = X_scaled[-1].reshape(1, -1)
        X_lstm = X_scaled[-LOOKBACK_PERIOD:].reshape(1, LOOKBACK_PERIOD, 5)
        
        rf_pred = self.models[symbol]['rf'].predict_proba(X_rf)[0][1]
        lstm_pred = self.models[symbol]['lstm'].predict(X_lstm)[0][0]
        
        return (rf_pred + lstm_pred) / 2

    def optimize_entry_timing(self, df, direction):
        last_candles = df.tail(5)
        
        average_range = (last_candles['high'] - last_candles['low']).mean()
        recent_high = last_candles['high'].max()
        recent_low = last_candles['low'].min()
        
        candle_strength = abs(last_candles['close'] - last_candles['open']) / (last_candles['high'] - last_candles['low'])
        dojis = (abs(last_candles['close'] - last_candles['open']) / (last_candles['high'] - last_candles['low'])) < 0.1
        
        if direction == 1:  # Pour un achat
            if (last_candles['low'].iloc[-1] <= recent_low and
                last_candles['close'].iloc[-1] > last_candles['open'].iloc[-1] and
                candle_strength.iloc[-1] > 0.6 and
                not dojis.iloc[-2:].any()):
                return True
        
        elif direction == -1:  # Pour une vente
            if (last_candles['high'].iloc[-1] >= recent_high and
                last_candles['close'].iloc[-1] < last_candles['open'].iloc[-1] and
                candle_strength.iloc[-1] > 0.6 and
                not dojis.iloc[-2:].any()):
                return True
        
        return False

    def calculate_position_size(self, symbol, stop_loss_pips):
        account_info = mt5.account_info()
        if account_info is None:
            return 0.0

        balance = account_info.balance
        risk_amount = min(balance * MAX_RISK_PER_TRADE, balance * (MAX_DAILY_RISK - self.daily_risk))
        if risk_amount <= 0:
            return 0.0

        tick_value = mt5.symbol_info(symbol).trade_tick_value
        position_size = (risk_amount * LEVERAGE) / (stop_loss_pips * tick_value)
        
        return min(position_size, mt5.symbol_info(symbol).volume_max)

    def execute_trade(self, symbol, operation, lot, stop_loss, take_profit):
        price = mt5.symbol_info_tick(symbol).ask if operation == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": operation,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "magic": 234000,
            "comment": f"{PROGRAM_NAME} Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Ordre non exécuté: {result.comment}")
        else:
            print(f"Ordre exécuté: {symbol}, Type: {'BUY' if operation == mt5.ORDER_TYPE_BUY else 'SELL'}, Volume: {lot}")
            self.daily_risk += (lot * abs(price - stop_loss) * mt5.symbol_info(symbol).trade_tick_value) / LEVERAGE

    def update_performance(self, symbol):
        account_info = mt5.account_info()
        if account_info is None:
            return

        current_equity = account_info.equity
        self.performance[symbol]['equity'].append(current_equity)
        
        peak = max(self.performance[symbol]['equity'])
        current_drawdown = (peak - current_equity) / peak
        self.performance[symbol]['drawdown'].append(current_drawdown)

    def plot_performance(self):
        for symbol in SYMBOLS:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            ax1.plot(self.performance[symbol]['equity'])
            ax1.set_title(f'{symbol} Equity Curve')
            ax1.set_ylabel('Equity')
            
            ax2.plot(self.performance[symbol]['drawdown'])
            ax2.set_title(f'{symbol} Drawdown')
            ax2.set_ylabel('Drawdown %')
            
            plt.tight_layout()
            plt.savefig(os.path.join(DATA_FOLDER, f'{symbol}_apex_quantum_performance.png'))
            plt.close()

    def handle_user_input(self):
        try:
            with open(INTERFACE_FILE, 'r') as file:
                command = file.read().strip()
            
            if command == "START" and not self.is_running:
                self.is_running = True
                print("APEX Quantum démarré")
            elif command == "STOP" and self.is_running:
                self.is_running = False
                print("APEX Quantum arrêté")
            
            # Effacer le contenu du fichier après lecture
            with open(INTERFACE_FILE, 'w') as file:
                file.write("")
        except FileNotFoundError:
            # Le fichier n'existe pas encore, on l'ignore
            pass

    def update_status(self):
        status = "RUNNING" if self.is_running else "STOPPED"
        with open(INTERFACE_FILE, 'w') as file:
            file.write(status)

    def run(self):
        if not self.initialize_mt5():
            print("Échec de l'initialisation de MetaTrader 5. Arrêt du programme.")
            return

        print(f"{PROGRAM_NAME} démarré avec balance initiale: ${INITIAL_BALANCE}")

        tick_count = 0
        try:
            while True:
                self.handle_user_input()
                self.update_status()
                
                if self.is_running:
                    tick_count += 1
                    current_date = datetime.now().date()
                    if current_date != self.last_reset:
                        self.daily_risk = 0
                        self.last_reset = current_date

                    for symbol in SYMBOLS:
                        df = self.prepare_data(symbol)
                        self.update_models(symbol, df)

                        X = df[['open', 'high', 'low', 'close', 'body']].values
                        prediction = self.get_prediction(symbol, X)
                        
                        price_action_signal = self.detect_enhanced_price_action_pattern(df)
                        
                        positions = mt5.positions_get(symbol=symbol)
                        if len(positions) == 0 and abs(price_action_signal) >= 0.5:
                            current_price = mt5.symbol_info_tick(symbol).ask
                            atr = df['high'].rolling(14).max() - df['low'].rolling(14).min()
                            stop_loss_pips = atr.iloc[-1] / mt5.symbol_info(symbol).point
                            lot = self.calculate_position_size(symbol, stop_loss_pips)
                            
                            if lot > 0:
                                if price_action_signal > 0 and prediction > 0.7:
                                    if self.optimize_entry_timing(df, 1):
                                        sl = current_price - stop_loss_pips * mt5.symbol_info(symbol).point
                                        tp = current_price + 2 * stop_loss_pips * mt5.symbol_info(symbol).point
                                        self.execute_trade(symbol, mt5.ORDER_TYPE_BUY, lot, sl, tp)
                                elif price_action_signal < 0 and prediction < 0.3:
                                    if self.optimize_entry_timing(df, -1):
                                        sl = current_price + stop_loss_pips * mt5.symbol_info(symbol).point
                                        tp = current_price - 2 * stop_loss_pips * mt5.symbol_info(symbol).point
                                        self.execute_trade(symbol, mt5.ORDER_TYPE_SELL, lot, sl, tp)

                        self.update_performance(symbol)

                    if tick_count % SAVE_FREQUENCY == 0:
                        self.save_persistent_data()
                        self.plot_performance()
                        print(f"Performance mise à jour et données sauvegardées: {datetime.now()}")

                    account_info = mt5.account_info()
                    if account_info and account_info.equity <= INITIAL_BALANCE * 0.5:
                        print("Alerte: 50% du capital initial perdu. Arrêt du trading.")
                        self.is_running = False

                time.sleep(1)  # Vérification toutes les secondes

        except KeyboardInterrupt:
            print("Arrêt du programme demandé par l'utilisateur.")
        finally:
            self.save_persistent_data()
            self.plot_performance()
            print("Données finales sauvegardées. Arrêt du programme.")
            mt5.shutdown()

# Code principal pour exécuter le système
if __name__ == "__main__":
    trader = APEXQuantumTradingSystem()
    trader.run()