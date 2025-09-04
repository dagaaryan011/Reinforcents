# Save as: agents/sac/env_sac.py
import random
from market.orderbook import Order, Side
from .agent_sac import MarketMakerAgent

class MarketMakerEnv:
    def __init__(self):
        self.exchange = None # Will be connected by main.py

    def get_orderbook(self):
        option_ticker_names = [t[0] for t in self.exchange.tickers if t[0] != 'STOCK_UNDERLYING']
        random_ticker = random.choice(option_ticker_names)
        return self.exchange.get_book(random_ticker)
    
    def get_highestbid_lowestask(self, ob, agent_id):
        bids = ob.get_bids(agent_id); asks = ob.get_asks(agent_id)
        highest_bid = bids[0][0] if bids else ob.market_price
        lowest_ask = asks[0][0] if asks else ob.market_price
        return highest_bid, lowest_ask
    
    def update_book(self, ob, b_p, a_p, b_s, a_s, agent_id):
        buy_order = Order(Side.BUY, b_p, b_s, owner_id=agent_id)
        sell_order = Order(Side.SELL, a_p, a_s, owner_id=agent_id)
        executed_trades_buy = ob.add_order(buy_order)
        executed_trades_sell = ob.add_order(sell_order)
        
        done_buy_price = sum(t.price for t in executed_trades_buy)
        done_buy_size = sum(t.size for t in executed_trades_buy)
        done_sell_price = sum(t.price for t in executed_trades_sell)
        done_sell_size = sum(t.size for t in executed_trades_sell)
        
        return done_buy_price, done_sell_price, done_buy_size, done_sell_size
    
    def get_reward(self, executed_ask_price, executed_ask_size, executed_bid_price, executed_bid_size, highest_bid, lowest_ask):
        PL = executed_ask_price * executed_ask_size - executed_bid_price * executed_bid_size
        diff_bid = abs(executed_bid_price - highest_bid)
        diff_ask = abs(lowest_ask - executed_ask_price)
        reward = PL - diff_bid - diff_ask
        size_diff = executed_bid_size - executed_ask_size
        return PL, size_diff, reward

def initialize_sac_agent(agent_id):
    agent = MarketMakerAgent(id=agent_id)
    env = MarketMakerEnv()
    agent.env = env 
    return agent, env

def run_sac_step(agent, env, central_market):
    env.exchange = central_market 
    agent.collect()
    if agent.t > agent.batch_size and agent.t % 200 == 0:
        agent.learn()