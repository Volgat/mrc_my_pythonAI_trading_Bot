#Volgat MA
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from scikeras.wrappers import KerasRegressor
import time
from datetime import datetime
import matplotlib.pyplot as plt
import os
import joblib
import json
import logging
import multiprocessing
from scipy.stats import pearsonr
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Configuration du logging
logging.basicConfig(filename='apexQX3_4_pure_price_action.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Paramètres globaux
PROGRAM_NAME = "APEX Quantum X v3.4 (Pure Price Action)"
VERSION = "3.4"
SYMBOLS = ['BTCUSD', 'XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'EURCAD', 'EURGBP', 'CHFJPY', 'XAGUSD', 'EURJPY', 'USDCHF', 'USDCAD', 'GBPJPY', 'CADJPY']
TIMEFRAME = mt5.TIMEFRAME_M1
LOOKBACK_PERIOD = 100
INITIAL_BALANCE = 10.0
MAX_RISK_PER_TRADE = 0.10
LEVERAGE = 500
SAVE_FREQUENCY = 100

# Paramètres de connexion MT5
MT5_ACCOUNT = 179775260
MT5_PASSWORD = ""
MT5_SERVER = ""

DATA_FOLDER = "apex_quantum_data_v3_4"

def create_lstm_model(units=50, learning_rate=0.001):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units, return_sequences=True, input_shape=(LOOKBACK_PERIOD, 5)),
        tf.keras.layers.LSTM(units, return_sequences=False),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy')
    return model

def process_symbol(symbol, models, scalers, performance, verbose=False):
    logging.info(f"Traitement de {symbol}")
    if verbose:
        print(f"\nTraitement de {symbol}")
    df = prepare_data(symbol)
    
    if df is None or df.empty:
        logging.warning(f"Pas de données disponibles pour {symbol}")
        return symbol, False, None
    
    logging.info(f"Données préparées pour {symbol}, dernière ligne: {df.tail(1).to_dict('records')[0]}")
    if verbose:
        print(f"Données préparées pour {symbol}, dernière ligne:")
        print(df.tail(1))

    try:
        update_models(symbol, df, models, scalers)
    except Exception as e:
        logging.error(f"Erreur lors de la mise à jour des modèles pour {symbol}: {str(e)}")
        return symbol, False, None

    price_action_signal = detect_price_action_pattern(df)
    prediction = get_prediction(symbol, df, models, scalers)
    
    logging.info(f"Signal de prix pour {symbol}: {price_action_signal}")
    logging.info(f"Prédiction pour {symbol}: {prediction}")
    if verbose:
        print(f"\nAnalyse pour {symbol}:")
        print(f"Signal de prix: {price_action_signal}")
        print(f"Prédiction: {prediction:.4f}")
    
    positions = mt5.positions_get(symbol=symbol)
    logging.info(f"Positions ouvertes pour {symbol}: {len(positions)}")
    if verbose:
        print(f"Positions ouvertes pour {symbol}: {len(positions)}")

    # Vérifier et gérer les positions ouvertes
    for position in positions:
        profit = position.profit
        if profit > 0 and (prediction < 0.45 or prediction > 0.55):  # Condition de sortie
            close_position(position)
            return symbol, True, profit

    if len(positions) == 0 and price_action_signal != 0:
        current_price = mt5.symbol_info_tick(symbol).ask
        stop_loss_pips = calculate_dynamic_stop_loss(df)
        lot = calculate_position_size(symbol, stop_loss_pips)
        
        logging.info(f"Opportunité de trade potentielle pour {symbol}:")
        logging.info(f"  - Prix actuel: {current_price}")
        logging.info(f"  - Stop loss (pips): {stop_loss_pips}")
        logging.info(f"  - Taille du lot calculée: {lot}")
        if verbose:
            print(f"Opportunité de trade potentielle pour {symbol}:")
            print(f"  - Prix actuel: {current_price}")
            print(f"  - Stop loss (pips): {stop_loss_pips}")
            print(f"  - Taille du lot calculée: {lot}")
        
        if lot > 0:
            if (price_action_signal > 0 and prediction > 0.55) or (price_action_signal < 0 and prediction < 0.45):
                if optimize_entry_timing(df, price_action_signal):
                    sl = current_price - stop_loss_pips * mt5.symbol_info(symbol).point if price_action_signal > 0 else \
                         current_price + stop_loss_pips * mt5.symbol_info(symbol).point
                    tp = current_price + 3 * stop_loss_pips * mt5.symbol_info(symbol).point if price_action_signal > 0 else \
                         current_price - 3 * stop_loss_pips * mt5.symbol_info(symbol).point
                    operation = mt5.ORDER_TYPE_BUY if price_action_signal > 0 else mt5.ORDER_TYPE_SELL
                    ordre_type = "d'achat" if price_action_signal > 0 else "de vente"
                    logging.info(f"Exécution d'un ordre {ordre_type} pour {symbol}")
                    print(f"\033[92mExécution d'un ordre {ordre_type} pour {symbol}\033[0m")
                    execute_trade(symbol, operation, lot, sl, tp)
                    return symbol, True, None
                else:
                    logging.info(f"Conditions d'entrée non optimales pour {symbol}, trade non exécuté")
                    if verbose:
                        print(f"Conditions d'entrée non optimales pour {symbol}, trade non exécuté")
            else:
                logging.info(f"Divergence entre signal de prix et prédiction pour {symbol}, trade non exécuté")
                if verbose:
                    print(f"Divergence entre signal de prix et prédiction pour {symbol}, trade non exécuté")
        else:
            logging.info(f"Taille du lot trop petite pour {symbol}, pas de trade exécuté")
            if verbose:
                print(f"Taille du lot trop petite pour {symbol}, pas de trade exécuté")
    else:
        logging.info(f"Pas d'opportunité de trade pour {symbol} à cette itération")
        if verbose:
            print(f"Pas d'opportunité de trade pour {symbol} à cette itération")

    return symbol, False, None

def prepare_data(symbol):
    bars = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, LOOKBACK_PERIOD)
    if bars is None or len(bars) < LOOKBACK_PERIOD:
        logging.warning(f"Pas assez de données pour {symbol}")
        return None
    
    df = pd.DataFrame(bars)
    df['body'] = abs(df['close'] - df['open'])
    df['wick_upper'] = df['high'] - np.maximum(df['open'], df['close'])
    df['wick_lower'] = np.minimum(df['open'], df['close']) - df['low']
    
    return df

def detect_price_action_pattern(df):
    last_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]
    prev_prev_candle = df.iloc[-3]
    
    # Détection de pattern haussier
    if (last_candle['close'] > last_candle['open'] and 
        prev_candle['close'] < prev_candle['open'] and
        last_candle['low'] < prev_candle['low'] and 
        last_candle['close'] > prev_candle['high']):
        return 1  # Signal haussier
    
    # Détection de pattern baissier
    elif (last_candle['close'] < last_candle['open'] and 
          prev_candle['close'] > prev_candle['open'] and
          last_candle['high'] > prev_candle['high'] and 
          last_candle['close'] < prev_candle['low']):
        return -1  # Signal baissier
    
    return 0  # Pas de signal clair

def update_models(symbol, df, models, scalers):
    X = df[['open', 'high', 'low', 'close', 'body']].values
    y = (df['close'].shift(-1) > df['close']).astype(int).values
    
    X = X[:-1]
    y = y[:-1]
    
    X_scaled = scalers[symbol].fit_transform(X)
    
    X_lstm = []
    y_lstm = []
    for i in range(len(X_scaled) - LOOKBACK_PERIOD):
        X_lstm.append(X_scaled[i:i+LOOKBACK_PERIOD])
        y_lstm.append(y[i+LOOKBACK_PERIOD])
    X_lstm = np.array(X_lstm)
    y_lstm = np.array(y_lstm)
    
    models[symbol]['rf'].fit(X_scaled, y)
    models[symbol]['gb'].fit(X_scaled, y)
    models[symbol]['lstm'].fit(X_lstm, y_lstm, epochs=5, verbose=0)
    
    logging.info(f"Modèles mis à jour pour {symbol}")

def get_prediction(symbol, df, models, scalers):
    X = df[['open', 'high', 'low', 'close', 'body']].values
    X_scaled = scalers[symbol].transform(X)
    X_rf = X_scaled[-1].reshape(1, -1)
    X_lstm = X_scaled[-LOOKBACK_PERIOD:].reshape(1, LOOKBACK_PERIOD, 5)
    
    rf_pred = models[symbol]['rf'].predict_proba(X_rf)[0][1]
    gb_pred = models[symbol]['gb'].predict_proba(X_rf)[0][1]
    lstm_pred = models[symbol]['lstm'].predict(X_lstm)[0][0]
    
    combined_pred = (rf_pred + gb_pred + lstm_pred) / 3
    return combined_pred

def calculate_dynamic_stop_loss(df):
    volatility = df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()
    return volatility.iloc[-1] * 1.5

def calculate_position_size(symbol, stop_loss_pips):
    account_info = mt5.account_info()
    if account_info is None:
        return 0.0

    balance = account_info.balance
    risk_amount = balance * MAX_RISK_PER_TRADE
    
    tick_value = mt5.symbol_info(symbol).trade_tick_value
    position_size = (risk_amount * LEVERAGE) / (stop_loss_pips * tick_value)
    
    return min(position_size, mt5.symbol_info(symbol).volume_max)

def optimize_entry_timing(df, direction):
    last_candles = df.tail(5)
    
    if direction > 0:  # Pour un achat
        if (last_candles['close'].iloc[-1] > last_candles['open'].iloc[-1] and
            last_candles['close'].iloc[-1] > last_candles['high'].iloc[-2]):
            return True
    
    elif direction < 0:  # Pour une vente
        if (last_candles['close'].iloc[-1] < last_candles['open'].iloc[-1] and
            last_candles['close'].iloc[-1] < last_candles['low'].iloc[-2]):
            return True
    
    return False

def execute_trade(symbol, operation, lot, stop_loss, take_profit):
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
        logging.info(f"Ordre exécuté: {symbol}, Type: {'BUY' if operation == mt5.ORDER_TYPE_BUY else 'SELL'}, Volume: {lot}")
        print(f"\033[92mOrdre exécuté: {symbol}, Type: {'BUY' if operation == mt5.ORDER_TYPE_BUY else 'SELL'}, Volume: {lot}\033[0m")

def close_position(position):
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
        "position": position.ticket,
        "price": mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
        "magic": 234000,
        "comment": f"{PROGRAM_NAME} Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.warning(f"Fermeture de position non exécutée: {result.comment}")
        print(f"\033[91mFermeture de position non exécutée: {result.comment}\033[0m")
    else:
        logging.info(f"Position fermée: {position.symbol}, Profit: {position.profit}")
        print(f"\033[92mPosition fermée: {position.symbol}, Profit: {position.profit}\033[0m")

class APEXQuantumTradingSystem:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.models = {symbol: self.create_models() for symbol in SYMBOLS}
        self.performance = {symbol: {'balance': INITIAL_BALANCE, 'equity': [], 'drawdown': []} for symbol in SYMBOLS}
        self.scalers = {symbol: StandardScaler() for symbol in SYMBOLS}
        
        self.total_trades = 0
        self.successful_trades = 0
        self.total_pnl = 0
        
        if not os.path.exists(DATA_FOLDER):
            os.makedirs(DATA_FOLDER)
        
        self.load_persistent_data()
        logging.info(f"Initialisation du système {PROGRAM_NAME}")

    def create_models(self):
        return {
            'rf': RandomForestClassifier(n_estimators=100),
            'lstm': create_lstm_model(),
            'gb': GradientBoostingClassifier()
        }

    def load_persistent_data(self):
        for symbol in SYMBOLS:
            model_path = os.path.join(DATA_FOLDER, f"{symbol}_model_data.joblib")
            lstm_path = os.path.join(DATA_FOLDER, f"{symbol}_lstm_model.weights.h5")
            
            try:
                if os.path.exists(model_path):
                    model_data = joblib.load(model_path)
                    self.models[symbol]['rf'] = model_data['rf_model']
                    self.models[symbol]['gb'] = model_data['gb_model']
                    self.scalers[symbol] = model_data['scaler']
                
                if os.path.exists(lstm_path):
                    self.models[symbol]['lstm'] = create_lstm_model()
                    self.models[symbol]['lstm'].load_weights(lstm_path)
                logging.info(f"Modèles chargés avec succès pour {symbol}")
            except Exception as e:
                logging.error(f"Erreur lors du chargement des modèles pour {symbol}: {str(e)}. Création de nouveaux modèles.")
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
                'gb_model': self.models[symbol]['gb'],
                'scaler': self.scalers[symbol]
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

    def update_performance(self, symbol):
        account_info = mt5.account_info()
        if account_info is None:
            return

        current_equity = account_info.equity
        self.performance[symbol]['equity'].append(current_equity)
        
        peak = max(self.performance[symbol]['equity'])
        current_drawdown = (peak - current_equity) / peak
        self.performance[symbol]['drawdown'].append(current_drawdown)

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

    def handle_profitable_trade(self, symbol, profit):
        self.total_pnl += profit
        self.successful_trades += 1
        
        # Mettre à jour les statistiques
        self.update_performance(symbol)
        
        # Réévaluer les paramètres de trading si nécessaire
        self.adjust_trading_parameters(symbol)
        
        # Logging et affichage
        logging.info(f"Trade profitable sur {symbol}. Profit: ${profit:.2f}")
        print(f"\033[92mTrade profitable sur {symbol}. Profit: ${profit:.2f}\033[0m")
        
        # Vérifier si l'objectif est atteint
        if self.total_pnl >= 1000000 - INITIAL_BALANCE:
            logging.info("Objectif atteint! Le programme a transformé $10 en $1,000,000.")
            print("\033[92mObjectif atteint! Le programme a transformé $10 en $1,000,000.\033[0m")
            return True  # Signal pour arrêter le programme
        
        return False

    def adjust_trading_parameters(self, symbol):
        # Ajuster les paramètres de trading en fonction de la performance récente
        recent_performance = self.performance[symbol]['equity'][-10:]
        if len(recent_performance) > 1:
            performance_change = (recent_performance[-1] - recent_performance[0]) / recent_performance[0]
            
            if performance_change > 0.05:  # Si la performance a augmenté de plus de 5%
                global MAX_RISK_PER_TRADE
                MAX_RISK_PER_TRADE = min(MAX_RISK_PER_TRADE * 1.1, 0.15)  # Augmenter le risque, max 15%
            elif performance_change < -0.05:  # Si la performance a diminué de plus de 5%
                MAX_RISK_PER_TRADE = max(MAX_RISK_PER_TRADE * 0.9, 0.05)  # Diminuer le risque, min 5%
            
            logging.info(f"Ajustement des paramètres de trading pour {symbol}. Nouveau MAX_RISK_PER_TRADE: {MAX_RISK_PER_TRADE:.2f}")

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
                logging.info(f"--- Itération {tick_count} ---")
                print(f"\n--- Itération {tick_count} ---")

                with multiprocessing.Pool() as pool:
                    results = pool.starmap(process_symbol, [(symbol, self.models, self.scalers, self.performance, self.verbose) for symbol in SYMBOLS])

                trades_this_iteration = 0
                for symbol, trade_executed, profit in results:
                    if trade_executed:
                        trades_this_iteration += 1
                        self.total_trades += 1
                        if profit is not None:
                            stop_program = self.handle_profitable_trade(symbol, profit)
                            if stop_program:
                                return
                        else:
                            self.update_performance(symbol)

                print(f"\nRésumé de l'itération {tick_count}:")
                print(f"Trades exécutés: {trades_this_iteration}")
                print(f"Total des trades: {self.total_trades}")
                print(f"Trades réussis: {self.successful_trades}")
                print(f"Profit/Perte total: ${self.total_pnl:.2f}")

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

                account_info = mt5.account_info()
                if account_info:
                    logging.info(f"État du compte - Balance: ${account_info.balance:.2f}, Equity: ${account_info.equity:.2f}")
                    print(f"État du compte - Balance: ${account_info.balance:.2f}, Equity: ${account_info.equity:.2f}")
                    if account_info.equity >= 1000000:
                        logging.info("Objectif atteint! Le programme a transformé $10 en $1,000,000.")
                        print("\033[92mObjectif atteint! Le programme a transformé $10 en $1,000,000.\033[0m")
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
    trader = APEXQuantumTradingSystem(verbose=True)  # Set to True for detailed output
    trader.run()