import numpy as np
from exchange import MarketExchange
from agent import Agent
import random

# 1. Create the exchange
exchange = MarketExchange(underlying_price=100, interval=5, option_depth=6)

# 2. Filter to just the option tickers (exclude the underlying)
option_ticker_names = [ticker for ticker in exchange.market_books if ticker != 'STOCK_UNDERLYING']


agent = Agent()

for i in range(0,100):
    print(i, "run")
    agent.exchange = exchange
    agent.collect(200)
    agent.sample_batch()
    agent.learn()

print(agent.get_action(98, 99)) 