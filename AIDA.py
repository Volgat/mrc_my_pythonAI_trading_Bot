import numpy as np
import pandas as pd
from datetime import datetime
import time
import logging
import MetaTrader5 as mt5

class AIDA:
    def __init__(self):
        self.name = "AIDA"
        self.version = "2.00"
        self.description = "Adaptive Intelligent Decision Assistant for Forex Trading - Price Action Focus"
        self.risk_per_trade = 0.01  # 1% risk per trade
        self.profit_factor = 10.0  # Objectif de profit ambitieux
        self.q_table = {}
        
        # MetaTrader 5 connection parameters
        self.mt5_login = 179775260  # Your MT5 account number
        self.mt5_password = ""  # Your MT5 password
        self.mt5_server = ""  # Your broker's server name
        self.symbol = "EURUSD"  # Example symbol, change as needed

        self.setup_logging()
        self.load_q_learning_data()
        self.setup_mt5_connection()

    def setup_logging(self):
        logging.basicConfig(filename=f'{self.name}_log.txt', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s: %(message)s')

    def setup_mt5_connection(self):
        if not mt5.initialize():
            logging.error("MetaTrader5 initialization failed")
            raise Exception("MetaTrader5 initialization failed")
        
        authorized = mt5.login(self.mt5_login, password=self.mt5_password, server=self.mt5_server)
        
        if authorized:
            logging.info(f"Connected to account #{self.mt5_login}")
        else:
            logging.error(f"Failed to connect to account #{self.mt5_login}, error code: {mt5.last_error()}")
            raise Exception(f"Failed to connect to account #{self.mt5_login}")
        
        selected = mt5.symbol_select(self.symbol, True)
        if not selected:
            logging.error(f"Failed to select symbol {self.symbol}")
            raise Exception(f"Failed to select symbol {self.symbol}")
        
        logging.info("MetaTrader5 connection established successfully")

    def load_q_learning_data(self):
        try:
            self.q_table = np.load('aida_q_table.npy', allow_pickle=True).item()
            logging.info("Q-table loaded successfully")
        except FileNotFoundError:
            logging.warning("No existing Q-table found. Starting with a new one.")

    def save_q_learning_data(self):
        np.save('aida_q_table.npy', self.q_table)
        logging.info("Q-table saved successfully")

    def determine_market_state(self, data):
        last_candles = data.tail(5)
        body_sizes = abs(last_candles['close'] - last_candles['open'])
        wick_sizes = last_candles['high'] - last_candles['low'] - body_sizes
        
        avg_body_size = body_sizes.mean()
        avg_wick_size = wick_sizes.mean()
        
        if avg_body_size > avg_wick_size * 2:
            return 0  # Strong trend
        elif avg_wick_size > avg_body_size * 2:
            return 1  # High volatility
        else:
            return 2  # Ranging market

    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0]  # [Buy, Sell, Hold]
        
        if np.random.random() < 0.1:  # Exploration
            return np.random.choice([0, 1, 2])
        else:  # Exploitation
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, new_state):
        if new_state not in self.q_table:
            self.q_table[new_state] = [0, 0, 0]
        
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[new_state])
        
        new_value = (1 - 0.1) * old_value + 0.1 * (reward + 0.95 * next_max)
        self.q_table[state][action] = new_value

    def check_entry_conditions(self, data):
        last_candles = data.tail(3)
        
        # Check for bullish engulfing pattern
        if (last_candles['open'].iloc[-2] > last_candles['close'].iloc[-2] and
            last_candles['close'].iloc[-1] > last_candles['open'].iloc[-2] and
            last_candles['open'].iloc[-1] < last_candles['close'].iloc[-2]):
            return 'BUY'
        
        # Check for bearish engulfing pattern
        elif (last_candles['close'].iloc[-2] > last_candles['open'].iloc[-2] and
              last_candles['open'].iloc[-1] > last_candles['close'].iloc[-2] and
              last_candles['close'].iloc[-1] < last_candles['open'].iloc[-2]):
            return 'SELL'
        
        else:
            return 'HOLD'

    def calculate_position_size(self, account_balance, stop_loss):
        risk_amount = account_balance * self.risk_per_trade
        position_size = risk_amount / stop_loss
        return position_size

    def execute_trade(self, action, data):
        current_price = data['close'].iloc[-1]
        account_balance = self.get_account_balance()
        
        if action == 0:  # Buy
            stop_loss = min(data['low'].tail(5))
            take_profit = current_price + self.profit_factor * (current_price - stop_loss)
            position_size = self.calculate_position_size(account_balance, current_price - stop_loss)
            self.place_buy_order(position_size, stop_loss, take_profit)
            logging.info(f"Buy order placed. Size: {position_size}, SL: {stop_loss}, TP: {take_profit}")
        elif action == 1:  # Sell
            stop_loss = max(data['high'].tail(5))
            take_profit = current_price - self.profit_factor * (stop_loss - current_price)
            position_size = self.calculate_position_size(account_balance, stop_loss - current_price)
            self.place_sell_order(position_size, stop_loss, take_profit)
            logging.info(f"Sell order placed. Size: {position_size}, SL: {stop_loss}, TP: {take_profit}")

    def manage_open_positions(self, data):
        for position in self.get_open_positions():
            if self.should_close_position(position, data):
                self.close_position(position)
                logging.info(f"Position closed. Ticket: {position.ticket}")

    def should_close_position(self, position, data):
        current_price = data['close'].iloc[-1]
        if position.type == mt5.ORDER_TYPE_BUY:
            return current_price <= position.sl or current_price >= position.tp
        elif position.type == mt5.ORDER_TYPE_SELL:
            return current_price >= position.sl or current_price <= position.tp

    def run(self):
        while True:
            try:
                current_data = self.get_latest_data()
                
                state = self.determine_market_state(current_data)
                action = self.choose_action(state)
                
                entry_condition = self.check_entry_conditions(current_data)
                if entry_condition != 'HOLD' and action in [0, 1]:
                    self.execute_trade(action, current_data)
                
                self.manage_open_positions(current_data)
                
                reward = self.calculate_reward()
                new_state = self.determine_market_state(self.get_latest_data())
                self.update_q_table(state, action, reward, new_state)
                
                self.save_q_learning_data()
                
                time.sleep(60)  # Wait for 1 minute before next iteration
            except Exception as e:
                logging.error(f"An error occurred: {str(e)}")
                time.sleep(300)  # Wait for 5 minutes before retrying

    def get_latest_data(self):
        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, 0, 100)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def calculate_reward(self):
        total_profit = 0
        for position in self.get_open_positions():
            total_profit += position.profit + position.swap
        return total_profit

    def get_account_balance(self):
        return mt5.account_info().balance

    def place_buy_order(self, position_size, stop_loss, take_profit):
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": position_size,
            "type": mt5.ORDER_TYPE_BUY,
            "price": mt5.symbol_info_tick(self.symbol).ask,
            "sl": stop_loss,
            "tp": take_profit,
            "magic": 234000,
            "comment": "AIDA Buy",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Buy order failed: {result.comment}")
        else:
            logging.info(f"Buy order placed successfully. Ticket: {result.order}")

    def place_sell_order(self, position_size, stop_loss, take_profit):
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": position_size,
            "type": mt5.ORDER_TYPE_SELL,
            "price": mt5.symbol_info_tick(self.symbol).bid,
            "sl": stop_loss,
            "tp": take_profit,
            "magic": 234000,
            "comment": "AIDA Sell",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Sell order failed: {result.comment}")
        else:
            logging.info(f"Sell order placed successfully. Ticket: {result.order}")

    def get_open_positions(self):
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            logging.error("Failed to get open positions")
            return []
        return positions

    def close_position(self, position):
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_BUY if position.type == mt5.ORDER_TYPE_SELL else mt5.ORDER_TYPE_SELL,
            "position": position.ticket,
            "price": mt5.symbol_info_tick(self.symbol).ask if position.type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(self.symbol).bid,
            "magic": 234000,
            "comment": "AIDA Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Failed to close position: {result.comment}")
        else:
            logging.info(f"Position closed successfully. Ticket: {position.ticket}")

if __name__ == "__main__":
    try:
        aida = AIDA()
        aida.run()
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
    finally:
        mt5.shutdown()
        logging.info("MetaTrader5 connection closed.")