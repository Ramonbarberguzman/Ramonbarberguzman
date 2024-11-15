import random
import time
import math

class ExelaStockBot:
    def __init__(self):
        self.capital = 1000
        self.stocks = {
            "XELA": {"price": 2, "volatility": 0.02, "sector": "Tech", "dividend_yield": 0.05},
            "AAPL": {"price": 150, "volatility": 0.015, "sector": "Tech", "dividend_yield": 0.02},
            "GOOG": {"price": 2800, "volatility": 0.01, "sector": "Tech", "dividend_yield": 0.03}
        }
        self.portfolio = {"XELA": 0, "AAPL": 0, "GOOG": 0}
        self.options_portfolio = {}
        self.difficulty = 1
        self.turns_taken = 0
        self.history = {'capital': [], 'portfolio_value': [], 'profit': []}
        self.game_over = False
        self.transaction_fee = 0.005
        self.tax_rate = 0.15
        self.stop_loss_percentage = 0.10
        self.take_profit_percentage = 0.20
        self.target_capital = 5000000
        self.market_news_impact = 0.05
        self.diversification_threshold = 0.1

    def simulate_market(self):
        """Simulate market fluctuations based on volatility and random news events."""
        for stock, data in self.stocks.items():
            volatility_factor = data["volatility"]
            fluctuation = random.uniform(-volatility_factor, volatility_factor) * self.difficulty
            data["price"] += data["price"] * fluctuation
            news_impact = random.uniform(-self.market_news_impact, self.market_news_impact)
            data["price"] += data["price"] * news_impact
            data["price"] = max(data["price"], 1)

    def buy_stock(self, stock):
        """Buy stock if funds are available."""
        stock_price = self.stocks[stock]["price"]
        if self.capital >= stock_price:
            self.capital -= stock_price * (1 + self.transaction_fee)
            self.portfolio[stock] += 1
            print(f"Bought 1 share of {stock} at ${stock_price:.2f}")
        else:
            print(f"Not enough capital to buy {stock}.")

    def sell_stock(self, stock):
        """Sell stock and return capital."""
        stock_price = self.stocks[stock]["price"]
        if self.portfolio[stock] > 0:
            self.capital += stock_price * (1 - self.transaction_fee)
            self.portfolio[stock] -= 1
            print(f"Sold 1 share of {stock} at ${stock_price:.2f}")
        else:
            print(f"No shares of {stock} to sell.")

    def buy_option(self, stock):
        """Buy options (call) for the given stock."""
        option_price = self.simulate_option_prices(stock)
        if self.capital >= option_price:
            self.capital -= option_price
            if stock not in self.options_portfolio:
                self.options_portfolio[stock] = 0
            self.options_portfolio[stock] += 1
            print(f"Bought 1 call option for {stock} at ${option_price:.2f}")
        else:
            print(f"Not enough funds to buy option for {stock}.")

    def sell_option(self, stock):
        """Sell the call option for the given stock."""
        if stock in self.options_portfolio and self.options_portfolio[stock] > 0:
            option_price = self.simulate_option_prices(stock) * 1.5
            self.capital += option_price
            self.options_portfolio[stock] -= 1
            print(f"Sold 1 call option for {stock} at ${option_price:.2f}")
        else:
            print(f"No options to sell for {stock}.")

    def simulate_option_prices(self, stock):
        """Simulate option pricing for the stock using a basic model."""
        stock_price = self.stocks[stock]["price"]
        volatility = self.stocks[stock]["volatility"]
        time_to_expiration = 30
        strike_price = stock_price * 1.05
        d1 = (math.log(stock_price / strike_price) + (0.5 * volatility**2) * time_to_expiration) / (volatility * math.sqrt(time_to_expiration))
        d2 = d1 - volatility * math.sqrt(time_to_expiration)
        call_option_price = stock_price * 0.5 * (1 + math.erf(d1 / math.sqrt(2))) - strike_price * 0.5 * (1 + math.erf(d2 / math.sqrt(2)))
        return call_option_price

    def risk_management(self):
        """Risk management to protect from major losses and lock profits."""
        for stock, data in self.stocks.items():
            if self.portfolio[stock] > 0 and data["price"] < self.stocks[stock]["price"] * (1 - self.stop_loss_percentage):
                print(f"Stop-loss triggered for {stock}. Selling...")
                self.sell_stock(stock)
            elif self.portfolio[stock] > 0 and data["price"] > self.stocks[stock]["price"] * (1 + self.take_profit_percentage):
                print(f"Take-profit triggered for {stock}. Selling...")
                self.sell_stock(stock)

    def update_performance(self):
        """Update performance and track portfolio value."""
        total_value = self.capital + sum([self.stocks[stock]["price"] * self.portfolio[stock] for stock in self.stocks])
        total_option_value = sum([self.simulate_option_prices(stock) * self.options_portfolio.get(stock, 0) for stock in self.stocks])
        total_value += total_option_value
        profit = total_value - 1000
        self.history['capital'].append(self.capital)
        self.history['portfolio_value'].append(total_value)
        self.history['profit'].append(profit)
        print(f"Portfolio value: {total_value:.2f} | Capital: {self.capital:.2f}")

    def strategy(self):
        """Smart strategy to decide buy and sell actions."""
        diversification_factor = sum(self.portfolio.values()) / len(self.stocks)  # Diversification check
        for stock, data in self.stocks.items():
            if diversification_factor > self.diversification_threshold:
                if self.portfolio[stock] > 0:
                    self.sell_stock(stock)
            if random.random() < 0.5 and data["price"] < 10:
                self.buy_stock(stock)
            elif self.portfolio[stock] > 0 and data["price"] > self.stocks[stock]["price"] * 1.2:
                self.sell_stock(stock)
            if random.random() < 0.3 and data["volatility"] > 0.02:
                self.buy_option(stock)
            if stock in self.options_portfolio and self.options_portfolio[stock] > 0:
                self.sell_option(stock)

    def calculate_tax(self):
        """Calculate and deduct taxes on profits."""
        profit = sum([self.stocks[stock]["price"] * self.portfolio[stock] for stock in self.stocks]) - 1000
        tax = profit * self.tax_rate
        self.capital -= tax
        print(f"Tax deducted: {tax:.2f}")

    def display_performance(self):
        """Display performance summary."""
        total_value = self.capital + sum([self.stocks[stock]["price"] * self.portfolio[stock] for stock in self.stocks])
        total_option_value = sum([self.simulate_option_prices(stock) * self.options_portfolio.get(stock, 0) for stock in self.stocks])
        total_value += total_option_value
        print("\nPerformance Summary:")
        print(f"Total Capital: {self.capital:.2f}")
        print(f"Total Portfolio Value: {total_value:.2f}")
        print(f"Profit/Loss: {self.capital + total_value - 1000:.2f}")

    def run(self):
        """Run the trading bot until $5 million is reached."""
        while self.capital < self.target_capital and not self.game_over:
            self.turns_taken += 1
            print(f"\nTurn {self.turns_taken}")
            self.simulate_market()
            self.strategy()
            self.risk_management()
            self.update_performance()
            self.display_performance()
            self.calculate_tax()
            if self.capital >= self.target_capital:
                print(f"\nGame Over! Target Capital of ${self.target_capital} reached!")
                self.game_over = True
            time.sleep(1)

# Instantiate the bot and run it
bot = ExelaStockBot()
bot.run()
import random
import time

# Simulate Exela stock price movement
class ExelaStockSimulator:
    def __init__(self):
        self.current_price = 10.00  # Start at $10 per share
        self.previous_price = self.current_price
        self.buy_price = 0.0
        self.total_profit = 0.0

    def update_price(self):
        # Random fluctuation between -5% and +5%
        change = random.uniform(-0.05, 0.05)
        self.previous_price = self.current_price
        self.current_price += self.current_price * change
        self.current_price = round(self.current_price, 2)
        
    def buy(self):
        self.buy_price = self.current_price
        print(f"Bought at {self.buy_price}")
        
    def sell(self):
        if self.buy_price > 0:
            profit = self.current_price - self.buy_price
            self.total_profit += profit
            print(f"Sold at {self.current_price}, Profit: {profit}")
            self.buy_price = 0.0

    def get_profit(self):
        return round(self.total_profit, 2)

# Simulate Exela stock trading
class ExelaStockTrader:
    def __init__(self):
        self.stock_simulator = ExelaStockSimulator()
        self.trading_history = []
        
    def trade(self):
        # Simulate trading strategy: Buy if price goes up, sell if it goes down
        if self.stock_simulator.current_price > self.stock_simulator.previous_price and self.stock_simulator.buy_price == 0:
            self.stock_simulator.buy()
            self.trading_history.append(f"Bought at {self.stock_simulator.buy_price}")
        elif self.stock_simulator.current_price < self.stock_simulator.previous_price and self.stock_simulator.buy_price > 0:
            self.stock_simulator.sell()
            self.trading_history.append(f"Sold at {self.stock_simulator.current_price}")

    def display_performance(self):
        print(f"Total Profit: {self.stock_simulator.get_profit()}")

    def display_stock_price(self):
        print(f"Current Price: {self.stock_simulator.current_price}")

    def display_trading_history(self):
        print("\nTrading History:")
        for trade in self.trading_history:
            print(trade)

# Main Execution
def main():
    trader = ExelaStockTrader()
    
    while True:
        trader.stock_simulator.update_price()  # Update stock price
        trader.trade()  # Simulate trading action
        
        # Display current stock price, performance, and trading history
        trader.display_stock_price()
        trader.display_performance()
        trader.display_trading_history()
        
        time.sleep(1)  # Simulate real-time (1-second intervals)

if __name__ == "__main__":
    main()
def display_graph(profit):
    bars = int(profit / 10)  # scale profit to number of bars
    print(f"Performance: {'#' * bars} ({profit})")
import random
import time
import math

class ExelaStockBot:
    def __init__(self):
        self.capital = 1000
        self.stocks = {
            "XELA": {"price": 2, "volatility": 0.02, "sector": "Tech", "dividend_yield": 0.05},
            "AAPL": {"price": 150, "volatility": 0.015, "sector": "Tech", "dividend_yield": 0.02},
            "GOOG": {"price": 2800, "volatility": 0.01, "sector": "Tech", "dividend_yield": 0.03}
        }
        self.portfolio = {"XELA": 0, "AAPL": 0, "GOOG": 0}
        self.options_portfolio = {}
        self.difficulty = 1
        self.turns_taken = 0
        self.history = {'capital': [], 'portfolio_value': [], 'profit': []}
        self.game_over = False
        self.transaction_fee = 0.005
        self.tax_rate = 0.15
        self.stop_loss_percentage = 0.10
        self.take_profit_percentage = 0.20
        self.target_capital = 5000000
        self.market_news_impact = 0.05
        self.diversification_threshold = 0.1
        self.trading_strategy_log = []  # Track trading strategy decisions

    def simulate_market(self):
        """Simulate market fluctuations based on volatility and random news events."""
        for stock, data in self.stocks.items():
            volatility_factor = data["volatility"]
            fluctuation = random.uniform(-volatility_factor, volatility_factor) * self.difficulty
            data["price"] += data["price"] * fluctuation
            news_impact = random.uniform(-self.market_news_impact, self.market_news_impact)
            data["price"] += data["price"] * news_impact
            data["price"] = max(data["price"], 1)

    def buy_stock(self, stock):
        """Buy stock if funds are available."""
        stock_price = self.stocks[stock]["price"]
        if self.capital >= stock_price:
            self.capital -= stock_price * (1 + self.transaction_fee)
            self.portfolio[stock] += 1
            self.trading_strategy_log.append(f"Bought 1 share of {stock} at ${stock_price:.2f}")
            print(f"Bought 1 share of {stock} at ${stock_price:.2f}")
        else:
            print(f"Not enough capital to buy {stock}.")

    def sell_stock(self, stock):
        """Sell stock and return capital."""
        stock_price = self.stocks[stock]["price"]
        if self.portfolio[stock] > 0:
            self.capital += stock_price * (1 - self.transaction_fee)
            self.portfolio[stock] -= 1
            self.trading_strategy_log.append(f"Sold 1 share of {stock} at ${stock_price:.2f}")
            print(f"Sold 1 share of {stock} at ${stock_price:.2f}")
        else:
            print(f"No shares of {stock} to sell.")

    def update_performance(self):
        """Update performance and track portfolio value."""
        total_value = self.capital + sum([self.stocks[stock]["price"] * self.portfolio[stock] for stock in self.stocks])
        total_option_value = sum([self.simulate_option_prices(stock) * self.options_portfolio.get(stock, 0) for stock in self.stocks])
        total_value += total_option_value
        profit = total_value - 1000
        self.history['capital'].append(self.capital)
        self.history['portfolio_value'].append(total_value)
        self.history['profit'].append(profit)
        print(f"Portfolio value: {total_value:.2f} | Capital: {self.capital:.2f}")

    def strategy(self):
        """Smart strategy to decide buy and sell actions."""
        diversification_factor = sum(self.portfolio.values()) / len(self.stocks)  # Diversification check
        for stock, data in self.stocks.items():
            if diversification_factor > self.diversification_threshold:
                if self.portfolio[stock] > 0:
                    self.sell_stock(stock)
            if random.random() < 0.5 and data["price"] < 10:
                self.buy_stock(stock)
            elif self.portfolio[stock] > 0 and data["price"] > self.stocks[stock]["price"] * 1.2:
                self.sell_stock(stock)

    def display_performance(self):
        """Display performance summary."""
        total_value = self.capital + sum([self.stocks[stock]["price"] * self.portfolio[stock] for stock in self.stocks])
        total_option_value = sum([self.simulate_option_prices(stock) * self.options_portfolio.get(stock, 0) for stock in self.stocks])
        total_value += total_option_value
        print("\nPerformance Summary:")
        print(f"Total Capital: {self.capital:.2f}")
        print(f"Total Portfolio Value: {total_value:.2f}")
        print(f"Profit/Loss: {self.capital + total_value - 1000:.2f}")

    def display_trading_strategy(self):
        """Display the full trading strategy after hitting target capital."""
        print("\nTrading Strategy Summary:")
        for trade in self.trading_strategy_log:
            print(trade)

    def display_graph(self):
        """Display a simple graph of profit over time."""
        print("\nPerformance Graph:")
        profit = sum(self.history['profit'])
        bars = int(profit / 100)  # Scale profit to number of bars
        print(f"Performance: {'#' * bars} ({profit:.2f})")

    def run(self):
        """Run the trading bot until $5 million is reached."""
        while self.capital < self.target_capital and not self.game_over:
            self.turns_taken += 1
            print(f"\nTurn {self.turns_taken}")
            self.simulate_market()
            self.strategy()
            self.update_performance()
            self.display_performance()
            if self.capital >= self.target_capital:
                print(f"\nGame Over! Target Capital of ${self.target_capital} reached!")
                self.game_over = True
                self.display_trading_strategy()  # Show the strategy when the target is reached
                self.display_graph()  # Show a simple graph of performance
            time.sleep(1)

# Instantiate the bot and run it
bot = ExelaStockBot()
bot.run()
import requests
import pandas as pd
import time

# Set up API keys and endpoints (Alpha Vantage for this example)
API_KEY = "your_alpha_vantage_api_key"
BASE_URL = "https://www.alphavantage.co/query"

def get_stock_data(symbol, interval="5min"):
    url = f"{BASE_URL}?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    if "Time Series (5min)" in data:
        df = pd.DataFrame(data["Time Series (5min)"]).transpose()
        df["timestamp"] = pd.to_datetime(df.index)
        df["close"] = pd.to_numeric(df["4. close"])
        return df
    return None
import math
import random
import numpy as np

class TradingBotRL:
    def __init__(self, data, risk_tolerance=0.02, learning_rate=0.01):
        self.data = data
        self.risk_tolerance = risk_tolerance
        self.learning_rate = learning_rate
        self.action_space = ["buy", "hold", "sell"]
        self.model = self.build_model()

    def build_model(self):
        # Placeholder for building a reinforcement learning model (e.g., Q-Learning or DQN)
        return {}

    def train(self, episodes=1000):
        for episode in range(episodes):
            state = self.get_current_state()
            action = self.select_action(state)
            reward = self.execute_action(action)
            next_state = self.get_current_state()
            self.update_model(state, action, reward, next_state)

    def get_current_state(self):
        # Extract market state based on features (like technical indicators, price trends)
        features = self.data[-50:]  # Placeholder for real features
        return np.array(features)

    def select_action(self, state):
        # Placeholder for model prediction to select action (buy, sell, hold)
        return random.choice(self.action_space)

    def execute_action(self, action):
        # Simulate the action (buy/sell) and calculate reward (profit/loss)
        reward = 0  # Placeholder for profit/loss calculation
        return reward

    def update_model(self, state, action, reward, next_state):
        # Use reinforcement learning techniques to update the model (e.g., Q-learning)
        pass

class RiskManagement:
    def __init__(self, portfolio_value, max_risk=0.02):
        self.portfolio_value = portfolio_value
        self.max_risk = max_risk

    def calculate_risk(self, trade_size, stop_loss_percent):
        potential_loss = trade_size * stop_loss_percent
        risk = potential_loss / self.portfolio_value
        return risk

    def adjust_position(self, trade_size, stop_loss_percent):
        risk = self.calculate_risk(trade_size, stop_loss_percent)
        if risk > self.max_risk:
            # Adjust position to meet risk tolerance
            trade_size = self.portfolio_value * self.max_risk / stop_loss_percent
        return trade_size

class OptionPricing:
    def black_scholes_call(self, S, K, T, r, sigma):
        # Black-Scholes formula for European call option
        d1 = (math.log(S / K) + (r + (sigma**2) / 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        call_price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
        return call_price

    def calculate_greeks(self, S, K, T, r, sigma):
        d1 = (math.log(S / K) + (r + (sigma**2) / 2) * T) / (sigma * math.sqrt(T))
        delta = norm_cdf(d1)
        gamma = norm_pdf(d1) / (S * sigma * math.sqrt(T))
        theta = -(S * norm_pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm_cdf(d1 - sigma * math.sqrt(T))
        vega = S * norm_pdf(d1) * math.sqrt(T)
        rho = K * T * math.exp(-r * T) * norm_cdf(d1 - sigma * math.sqrt(T))
        return delta, gamma, theta, vega, rho

def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def norm_pdf(x):
    return math.exp(-x**2 / 2.0) / math.sqrt(2.0 * math.pi)

class SelfImprovement:
    def __init__(self, model):
        self.model = model

    def retrain_model(self, data):
        # Retrain the model with new data
        self.model.train(data)

    def evaluate_performance(self):
        # Placeholder for evaluating bot performance
        return 0  # Example: Sharpe ratio or other metrics

class ContinuousImprovement:
    def __init__(self, model):
        self.model = model

    def retrain_on_new_data(self, new_data):
        self.model.train(new_data)

    def performance_metrics(self, performance_data):
        # Placeholder for calculating performance metrics
        return 0  # Example: Sharpe ratio or other metrics

class PortfolioOptimization:
    def optimize(self, returns, risk_free_rate=0.02):
        n = len(returns)
        expected_returns = [sum(r) / len(r) for r in returns]
        covariance_matrix = [[np.cov(returns[i], returns[j]) for j in range(n)] for i in range(n)]

        # Define variables
        w = [0] * n
        ret = sum([expected_returns[i] * w[i] for i in range(n)])
        risk = sum([sum([covariance_matrix[i][j] * w[i] * w[j] for j in range(n)]) for i in range(n)])

        # Simple optimization: maximize returns - risk
        # Placeholder for actual optimization
        return w

class TradingBot:
    def __init__(self, data, portfolio_value):
        self.data = data
        self.portfolio_value = portfolio_value
        self.risk_management = RiskManagement(portfolio_value)
        self.trading_model = TradingBotRL(data)
        self.option_pricing = OptionPricing()
        self.self_improvement = SelfImprovement(self.trading_model)
        self.continuous_improvement = ContinuousImprovement(self.trading_model)
        self.portfolio_optimization = PortfolioOptimization()

    def execute_trade(self):
        # Placeholder for trade execution logic
        state = self.trading_model.get_current_state()
        action = self.trading_model.select_action(state)
        reward = self.trading_model.execute_action(action)
        return action, reward

    def optimize_portfolio(self):
        optimized_weights = self.portfolio_optimization.optimize(self.data)
        return optimized_weights

    def manage_risk(self, trade_size, stop_loss_percent):
        adjusted_trade_size = self.risk_management.adjust_position(trade_size, stop_loss_percent)
        return adjusted_trade_size

    def calculate_option_price(self, S, K, T, r, sigma):
        return self.option_pricing.black_scholes_call(S, K, T, r, sigma)

    def evaluate_performance(self):
        return self.self_improvement.evaluate_performance()

    def retrain_model(self, new_data):
        self.continuous_improvement.retrain_on_new_data(new_data)

# Sample market data (e.g., closing prices for 50 periods)
market_data = [random.randint(100, 200) for _ in range(50)]

# Initialize the bot
bot = TradingBot(market_data, portfolio_value=100000)

# Execute a trade and manage risk
trade_action, reward = bot.execute_trade()
print(f"Trade Action: {trade_action}, Reward: {reward}")

# Optimize portfolio
optimized_weights = bot.optimize_portfolio()
print(f"Optimized Portfolio Weights: {optimized_weights}")

# Calculate option price
option_price = bot.calculate_option_price(150, 160, 30, 0.05, 0.2)
print(f"Option Price: {option_price}")
