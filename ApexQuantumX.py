#Volgat MA
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor
import time
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
import os
import joblib
import json
import sklearn
import logging

# Configuration du logging
logging.basicConfig(filename='apex_quantumX.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Paramètres globaux
PROGRAM_NAME = "APEX Quantum X"
SYMBOLS = ['BTCUSD', 'XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'EURCAD', 'EURGBP', 'CHFJPY', 'CHFJPY', 'XAGUSD', 'EURJPY', 'USDCHF', 'USDCAD', 'GBPJPY', 'CADJPY']
TIMEFRAME = mt5.TIMEFRAME_M5
LOOKBACK_PERIOD = 100
INITIAL_BALANCE = 10.0
MAX_RISK_PER_TRADE = 0.05
MAX_DAILY_RISK = 0.20
LEVERAGE = 3000
SAVE_FREQUENCY = 100

# Nouveaux seuils ajustables
PRICE_ACTION_THRESHOLD = 0.3
PREDICTION_THRESHOLD_UPPER = 0.65
PREDICTION_THRESHOLD_LOWER = 0.35

# Paramètres de connexion MT5 (à remplacer par vos informations)
MT5_ACCOUNT = 179775260
MT5_PASSWORD = ""
MT5_SERVER = ""

DATA_FOLDER = "apex_quantum_data"

class APEXQuantumTradingSystem:
    def __init__(self):
        self.lstm_units = 50  # Initialisation de lstm_units
        self.models = {symbol: self.create_models() for symbol in SYMBOLS}
        self.performance = {symbol: {'balance': INITIAL_BALANCE, 'equity': [], 'drawdown': []} for symbol in SYMBOLS}
        self.daily_risk = 0
        self.last_reset = datetime.now().date()
        self.scalers = {symbol: StandardScaler() for symbol in SYMBOLS}
        
        # Nouveaux compteurs pour le suivi
        self.total_trades = 0
        self.successful_trades = 0
        self.total_pnl = 0
        
        if not os.path.exists(DATA_FOLDER):
            os.makedirs(DATA_FOLDER)
        
        self.load_persistent_data()
        logging.info("Initialisation du système APEX Quantum X")

    def create_models(self):
        rf = RandomForestClassifier(n_estimators=100)
        lstm = self.create_lstm_model()
        return {'rf': rf, 'lstm': lstm}

    def create_lstm_model(self, units=None, learning_rate=0.001):
        if units is None:
            units = self.lstm_units
        units = int(units)  # Assurez-vous que units est un entier
        model = Sequential([
            Input(shape=(LOOKBACK_PERIOD, 5)),
            LSTM(units, return_sequences=True),
            LSTM(units, return_sequences=False),
            Dense(25),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy')
        logging.info(f"Modèle LSTM créé avec {units} unités et learning rate de {learning_rate}")
        return model

    def load_persistent_data(self):
        for symbol in SYMBOLS:
            model_path = os.path.join(DATA_FOLDER, f"{symbol}_model_data.joblib")
            lstm_path = os.path.join(DATA_FOLDER, f"{symbol}_lstm_model.weights.h5")
            
            try:
                if os.path.exists(model_path):
                    model_data = joblib.load(model_path)
                    if model_data['sklearn_version'] == sklearn.__version__:
                        self.models[symbol]['rf'] = model_data['rf_model']
                        self.scalers[symbol] = model_data['scaler']
                    else:
                        raise ValueError("Version mismatch")
                if os.path.exists(lstm_path):
                    self.models[symbol]['lstm'] = self.create_lstm_model()
                    self.models[symbol]['lstm'].load_weights(lstm_path)
            except (ValueError, KeyError, AttributeError) as e:
                logging.warning(f"Erreur lors du chargement des modèles pour {symbol}. Création de nouveaux modèles.")
                self.models[symbol] = self.create_models()
                self.scalers[symbol] = StandardScaler()

        performance_path = os.path.join(DATA_FOLDER, "performance.json")
        if os.path.exists(performance_path):
            with open(performance_path, 'r') as f:
                self.performance = json.load(f)

    def save_persistent_data(self):
        for symbol in SYMBOLS:
            model_data = {
                'rf_model': self.models[symbol]['rf'],
                'scaler': self.scalers[symbol],
                'sklearn_version': sklearn.__version__
            }
            joblib.dump(model_data, os.path.join(DATA_FOLDER, f"{symbol}_model_data.joblib"))
            self.models[symbol]['lstm'].save_weights(os.path.join(DATA_FOLDER, f"{symbol}_lstm_model.weights.h5"))

        with open(os.path.join(DATA_FOLDER, "performance.json"), 'w') as f:
            json.dump(self.performance, f)

    def initialize_mt5(self):
        if not mt5.initialize():
            logging.error("Initialisation de MetaTrader 5 échouée")
            return False

        if not mt5.login(login=MT5_ACCOUNT, server=MT5_SERVER, password=MT5_PASSWORD):
            logging.error("Échec de l'autorisation")
            mt5.shutdown()
            return False

        logging.info(f"MetaTrader 5 version: {mt5.__version__}")
        logging.info(f"Compte connecté: {mt5.account_info().login}")
        logging.info(f"Balance: {mt5.account_info().balance}")
        logging.info(f"Symboles disponibles: {mt5.symbols_total()}")

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

    def optimize_hyperparameters(self, symbol, X_lstm, y_lstm, X_rf, y_rf):
        rf_param_dist = {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        rf_random = RandomizedSearchCV(self.models[symbol]['rf'], rf_param_dist, n_iter=20, cv=3, random_state=42)
        rf_random.fit(X_rf, y_rf)
        self.models[symbol]['rf'] = rf_random.best_estimator_

        def create_lstm_model(units=50, learning_rate=0.001):
            return self.create_lstm_model(units=int(units), learning_rate=learning_rate)

        lstm_param_dist = {
            'model__units': [30, 50, 70],
            'model__learning_rate': [0.0001, 0.001, 0.01]
        }
        lstm_random = RandomizedSearchCV(KerasRegressor(model=create_lstm_model, verbose=0), lstm_param_dist, n_iter=10, cv=3)
        lstm_random.fit(X_lstm, y_lstm)
        
        best_lstm = create_lstm_model(units=int(lstm_random.best_params_['model__units']), 
                                      learning_rate=lstm_random.best_params_['model__learning_rate'])
        best_lstm.fit(X_lstm, y_lstm, epochs=10, batch_size=32, verbose=0)
        self.models[symbol]['lstm'] = best_lstm
        logging.info(f"Hyperparamètres optimisés pour {symbol}: LSTM units={lstm_random.best_params_['model__units']}, learning rate={lstm_random.best_params_['model__learning_rate']}")

    def update_models(self, symbol, df):
        X = df[['open', 'high', 'low', 'close', 'body']].values
        y = (df['close'].shift(-1) > df['close']).astype(int).values
        
        X = X[:-1]  # Enlève la dernière ligne car nous n'avons pas de 'y' correspondant
        y = y[:-1]  # Enlève le dernier élément qui est NaN à cause du shift
        
        X_scaled = self.scalers[symbol].fit_transform(X)
        
        # Préparation des données pour LSTM
        X_lstm = []
        y_lstm = []
        for i in range(len(X_scaled) - LOOKBACK_PERIOD):
            X_lstm.append(X_scaled[i:i+LOOKBACK_PERIOD])
            y_lstm.append(y[i+LOOKBACK_PERIOD])
        X_lstm = np.array(X_lstm)
        y_lstm = np.array(y_lstm)
        
        # Données pour Random Forest
        X_rf = X_scaled[LOOKBACK_PERIOD:]
        y_rf = y[LOOKBACK_PERIOD:]
        
        if np.random.rand() < 0.01:  # 1% de chance d'optimiser à chaque mise à jour
            logging.info(f"Optimisation des hyperparamètres pour {symbol}")
            self.optimize_hyperparameters(symbol, X_lstm, y_lstm, X_rf, y_rf)
        else:
            self.models[symbol]['rf'].fit(X_rf, y_rf)
            self.models[symbol]['lstm'].fit(X_lstm, y_lstm, epochs=1, verbose=0)
        logging.info(f"Modèles mis à jour pour {symbol}")

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
            logging.warning(f"Ordre non exécuté: {result.comment}")
            print(f"\033[91mOrdre non exécuté: {result.comment}\033[0m")
        else:
            self.total_trades += 1
            trade_type = 'BUY' if operation == mt5.ORDER_TYPE_BUY else 'SELL'
            logging.info(f"Ordre exécuté: {symbol}, Type: {trade_type}, Volume: {lot}")
            print(f"\033[92mOrdre exécuté: {symbol}, Type: {trade_type}, Volume: {lot}\033[0m")
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

        # Mise à jour du PnL total
        self.total_pnl = current_equity - INITIAL_BALANCE

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

    def is_market_open(self, symbol):
        current_time = datetime.now(timezone.utc)
        weekday = current_time.weekday()
        
        # Crypto (ouvert 24/7)
        if symbol == 'BTCUSD':
            return True
        
        # Forex et métaux
        if symbol in ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'AUDUSD', 'NZDUSD', 'EURCAD', 'EURGBP', 'CHFJPY', 'XAGUSD', 'EURJPY', 'USDCHF', 'USDCAD', 'GBPJPY', 'CADJPY']:
            # Fermé le weekend (samedi et dimanche)
            if weekday >= 5:
                return False
            
            # Vérifier si nous sommes entre dimanche 22h00 et vendredi 22h00 (UTC)
            if weekday == 0:  # Dimanche
                return current_time.hour >= 22
            elif weekday == 4:  # Vendredi
                return current_time.hour < 22
            else:  # Lundi à Jeudi
                return True
        
        # Par défaut, considérer le marché comme fermé
        return False

    def run(self):
        if not self.initialize_mt5():
            logging.error("Échec de l'initialisation de MetaTrader 5. Arrêt du programme.")
            return

        logging.info(f"{PROGRAM_NAME} démarré avec balance initiale: ${INITIAL_BALANCE}")
        print(f"\033[94m{PROGRAM_NAME} démarré avec balance initiale: ${INITIAL_BALANCE}\033[0m")

        tick_count = 0
        try:
            while True:
                tick_count += 1
                current_date = datetime.now().date()
                if current_date != self.last_reset:
                    self.daily_risk = 0
                    self.last_reset = current_date

                logging.info(f"--- Itération {tick_count} ---")
                print(f"\n--- Itération {tick_count} ---")
                for symbol in SYMBOLS:
                    current_time = datetime.now(timezone.utc)
                    market_open = self.is_market_open(symbol)
                    logging.info(f"Vérification d'ouverture du marché pour {symbol}: Jour {current_time.weekday()}, Heure {current_time.hour}, Ouvert: {market_open}")
                    
                    if not market_open:
                        logging.info(f"Le marché est fermé pour {symbol}. Pas d'analyse effectuée.")
                        print(f"Le marché est fermé pour {symbol}. Pas d'analyse effectuée.")
                        continue

                    logging.info(f"Traitement de {symbol}")
                    print(f"\nTraitement de {symbol}")
                    df = self.prepare_data(symbol)
                    logging.info(f"Données préparées pour {symbol}, dernière ligne: {df.tail(1).to_dict('records')[0]}")
                    print(f"Données préparées pour {symbol}, dernière ligne:")
                    print(df.tail(1))

                    try:
                        self.update_models(symbol, df)
                    except Exception as e:
                        logging.error(f"Erreur lors de la mise à jour des modèles pour {symbol}: {str(e)}")
                        continue

                    X = df[['open', 'high', 'low', 'close', 'body']].values
                    prediction = self.get_prediction(symbol, X)
                    logging.info(f"Prédiction pour {symbol}: {prediction}")
                    print(f"Prédiction pour {symbol}: {prediction}")
                    
                    price_action_signal = self.detect_enhanced_price_action_pattern(df)
                    logging.info(f"Signal de prix pour {symbol}: {price_action_signal}")
                    print(f"Signal de prix pour {symbol}: {price_action_signal}")
                    
                    positions = mt5.positions_get(symbol=symbol)
                    logging.info(f"Positions ouvertes pour {symbol}: {len(positions)}")
                    print(f"Positions ouvertes pour {symbol}: {len(positions)}")

                    if len(positions) == 0 and abs(price_action_signal) >= PRICE_ACTION_THRESHOLD:
                        current_price = mt5.symbol_info_tick(symbol).ask
                        atr = df['high'].rolling(14).max() - df['low'].rolling(14).min()
                        stop_loss_pips = atr.iloc[-1] / mt5.symbol_info(symbol).point
                        lot = self.calculate_position_size(symbol, stop_loss_pips)
                        
                        logging.info(f"Opportunité de trade potentielle pour {symbol}:")
                        logging.info(f"  - Prix actuel: {current_price}")
                        logging.info(f"  - Stop loss (pips): {stop_loss_pips}")
                        logging.info(f"  - Taille du lot calculée: {lot}")
                        print(f"Opportunité de trade potentielle pour {symbol}:")
                        print(f"  - Prix actuel: {current_price}")
                        print(f"  - Stop loss (pips): {stop_loss_pips}")
                        print(f"  - Taille du lot calculée: {lot}")
                        
                        if lot > 0:
                            if price_action_signal > 0 and prediction > PREDICTION_THRESHOLD_UPPER:
                                if self.optimize_entry_timing(df, 1):
                                    sl = current_price - stop_loss_pips * mt5.symbol_info(symbol).point
                                    tp = current_price + 2 * stop_loss_pips * mt5.symbol_info(symbol).point
                                    logging.info(f"Exécution d'un ordre d'achat pour {symbol}")
                                    print(f"\033[92mExécution d'un ordre d'achat pour {symbol}\033[0m")
                                    self.execute_trade(symbol, mt5.ORDER_TYPE_BUY, lot, sl, tp)
                            elif price_action_signal < 0 and prediction < PREDICTION_THRESHOLD_LOWER:
                                if self.optimize_entry_timing(df, -1):
                                    sl = current_price + stop_loss_pips * mt5.symbol_info(symbol).point
                                    tp = current_price - 2 * stop_loss_pips * mt5.symbol_info(symbol).point
                                    logging.info(f"Exécution d'un ordre de vente pour {symbol}")
                                    print(f"\033[92mExécution d'un ordre de vente pour {symbol}\033[0m")
                                    self.execute_trade(symbol, mt5.ORDER_TYPE_SELL, lot, sl, tp)
                        else:
                            logging.info(f"Taille du lot trop petite pour {symbol}, pas de trade exécuté")
                            print(f"Taille du lot trop petite pour {symbol}, pas de trade exécuté")
                    else:
                        logging.info(f"Pas d'opportunité de trade pour {symbol} à cette itération")
                        print(f"Pas d'opportunité de trade pour {symbol} à cette itération")

                    self.update_performance(symbol)

                if tick_count % SAVE_FREQUENCY == 0:
                    self.save_persistent_data()
                    self.plot_performance()
                    logging.info(f"Performance mise à jour et données sauvegardées: {datetime.now()}")
                    print(f"\033[94mPerformance mise à jour et données sauvegardées: {datetime.now()}\033[0m")
                    for symbol in SYMBOLS:
                        logging.info(f"{symbol} - Balance: ${self.performance[symbol]['balance']:.2f}, "
                                     f"Drawdown: {self.performance[symbol]['drawdown'][-1]:.2%}")
                        print(f"{symbol} - Balance: ${self.performance[symbol]['balance']:.2f}, "
                              f"Drawdown: {self.performance[symbol]['drawdown'][-1]:.2%}")

                    # Résumé périodique
                    logging.info(f"\n--- Résumé après {tick_count} itérations ---")
                    logging.info(f"Nombre total de trades: {self.total_trades}")
                    logging.info(f"Profit/Perte total: ${self.total_pnl:.2f}")
                    logging.info("------------------------\n")
                    print(f"\n\033[93m--- Résumé après {tick_count} itérations ---\033[0m")
                    print(f"Nombre total de trades: {self.total_trades}")
                    print(f"Profit/Perte total: ${self.total_pnl:.2f}")
                    print("\033[93m------------------------\033[0m\n")

                account_info = mt5.account_info()
                if account_info:
                    logging.info(f"État du compte - Balance: ${account_info.balance:.2f}, Equity: ${account_info.equity:.2f}")
                    print(f"État du compte - Balance: ${account_info.balance:.2f}, Equity: ${account_info.equity:.2f}")
                    if account_info.equity <= INITIAL_BALANCE * 0.5:
                        logging.warning("Alerte: 50% du capital initial perdu. Arrêt du trading.")
                        print("\033[91mAlerte: 50% du capital initial perdu. Arrêt du trading.\033[0m")
                        break
                else:
                    logging.warning("Impossible d'obtenir les informations du compte")
                    print("\033[91mImpossible d'obtenir les informations du compte\033[0m")

                time.sleep(1)  # Vérification toutes les secondes

        except KeyboardInterrupt:
            logging.info("Arrêt du programme demandé par l'utilisateur.")
            print("\033[93mArrêt du programme demandé par l'utilisateur.\033[0m")
        except Exception as e:
            logging.error(f"Erreur inattendue: {str(e)}")
            print(f"\033[91mErreur inattendue: {str(e)}\033[0m")
        finally:
            self.save_persistent_data()
            self.plot_performance()
            logging.info("Données finales sauvegardées. Arrêt du programme.")
            print("\033[94mDonnées finales sauvegardées. Arrêt du programme.\033[0m")
            mt5.shutdown()

if __name__ == "__main__":
    trader = APEXQuantumTradingSystem()
    trader.run()
     