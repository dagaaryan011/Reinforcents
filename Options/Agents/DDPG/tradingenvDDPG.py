import numpy as np
from collections import defaultdict


from market.orderbook import Order, Side
from .agent import AgentDDPG 

class Space:
    def __init__(self, low, high, shape):
        self.low = np.array([low])
        self.high = np.array([high])
        self.shape = shape


class DDPG_TradingEnvironment:
    def __init__(self, agent_id, start_cash=100000, active_ticker_count=11):
        self.agent_id = agent_id
        #kaunse orderbooks ko chhedna hai
        self.active_ticker_count = active_ticker_count
        # list for orderbooks jinko chhedna hai
        self.active_tickers = []
        self.initial_cash = start_cash
        self.cash_balance = start_cash
        # A dictionary to track the quantity of each ticker held by the agent.
        self.portfolio = defaultdict(int)

        #state = cash + orderbook data
        self.state_dimensions = 1 + (self.active_ticker_count * 5)
        self.n_tickers = self.active_ticker_count
        self.n_actions = self.n_tickers + 1 
        self.action_space = Space(low=-1.0, high=1.0, shape=(self.n_actions,))

    def get_state(self, central_market):
        
        self._update_active_tickers(central_market)
        
        state = np.full(self.state_dimensions, -1.0)
        state[0] = self.cash_balance
        pointer = 1
        # Loop through the active tickers to populate the state vector.
        for ticker_name in self.active_tickers:
            position_size = self.portfolio.get(ticker_name, 0)
            book = central_market.get_book(ticker_name)
            if book:
                bids = book.get_bids('retail'); asks = book.get_asks('retail')
                best_bid_price = bids[0][0] if bids else -1.0
                best_bid_size = bids[0][1] if bids else -1.0
                best_ask_price = asks[0][0] if asks else -1.0
                best_ask_size = asks[0][1] if asks else -1.0
            else:
                best_bid_price, best_bid_size, best_ask_price, best_ask_size = -1.0, -1.0, -1.0, -1.0
            
            
            state[pointer:pointer+5] = [position_size, best_bid_price, best_bid_size, best_ask_price, best_ask_size]
            pointer += 5
        return state

    def step(self, raw_action, central_market):
        
        old_portfolio_value = self._calculate_portfolio_value(central_market)
        
        ticker_probs = raw_action[:self.n_tickers] # Probabilities for which ticker to trade.
        size_dir = raw_action[self.n_tickers]      # Value from -1 to 1 for volume/direction.
        chosen_idx = np.argmax(ticker_probs)       #  highest probability.
        ticker_to_trade = self.active_tickers[chosen_idx]
        side = Side.BUY if size_dir > 0 else Side.SELL
        
        req_size = 10 
        lot_size = 50 
        trade_size = req_size
        #jitna hai utna hi bech sakta hai
        if side == Side.SELL:
            trade_size = min(req_size, self.portfolio.get(ticker_to_trade, 0))
        
    
        if trade_size > 0:
            book = central_market.get_book(ticker_to_trade)
            if book:
                bids = book.get_bids(); asks = book.get_asks()
                price = -1
                #buy - best bid , sell - best aks
                if side == Side.BUY and asks: price = asks[0][0]
                elif side == Side.SELL and bids: price = bids[0][0]

                if price != -1:
                    order = Order(side, price, trade_size, self.agent_id)
                    executed_trades = book.add_order(order)
                    #portfolio 
                    for trade in executed_trades:
                        cost = trade.price * trade.size * lot_size
                        if trade.taker_side == Side.BUY: self.cash_balance -= cost
                        else: self.cash_balance += cost
                        self.portfolio[trade.ticker_id] += trade.size if trade.taker_side == Side.BUY else -trade.size
        
        #reward
        new_portfolio_value = self._calculate_portfolio_value(central_market)
        reward = new_portfolio_value - old_portfolio_value
        next_state = self.get_state(central_market)
        return reward, next_state

    def _calculate_portfolio_value(self, central_market):
        
        value = self.cash_balance
        for name, qty in self.portfolio.items():
            if qty > 0:
                book = central_market.get_book(name)
                if book:
                    bids = book.get_bids('institutional')
                    if bids:
                        current_price = bids[0][0]
                        value += qty * current_price * 50 
        return value

    def _update_active_tickers(self, central_market):
        
        price = central_market.underlying_price
        all_opts = [t for t in central_market.tickers if t[0] != 'STOCK_UNDERLYING']
        # Sort options by how close their strike is to the current underlying price.
        all_opts.sort(key=lambda t: abs(t[1] - price))
        # Select the closest N-1 options.
        closest = [t[0] for t in all_opts[:self.active_ticker_count - 1]]
        # The agent always considers the underlying stock plus the closest options.
        self.active_tickers = ['STOCK_UNDERLYING'] + closest
    # In tradingenvDDPG.py, add this method to the DDPG_TradingEnvironment class

    def settle_expired_positions(self, final_underlying_price):
        print("settling")

        lot_size = 50
        options_to_remove = []
        total_pnl_from_settlement = 0

        for ticker_name, quantity in self.portfolio.items():
            if 'STOCK_UNDERLYING' in ticker_name:
                continue

            options_to_remove.append(ticker_name)

            #ticker ka naam is STOCK_Price_CE , so last 2 char apne ko side batyga , value btw _ _ is strike price , in dono se nikaal lenge ki kitna p/l hua hai
            option_type = ticker_name[-2:]
            underscore_index1 = ticker_name.find('_')
            underscore_index2 = ticker_name.find('_', underscore_index1 + 1)
            strike_price_str = ticker_name[underscore_index1 + 1 : underscore_index2]
            strike_price = int(strike_price_str)
            
            
            
            settlement_per_share = 0
            if option_type == 'CE':
                settlement_per_share = max(0, final_underlying_price - strike_price)
            elif option_type == 'PE':
                settlement_per_share = max(0, strike_price - final_underlying_price)

            if settlement_per_share > 0 and quantity != 0:
                total_settlement_cash = settlement_per_share * quantity * lot_size
                self.cash_balance += total_settlement_cash
                total_pnl_from_settlement += total_settlement_cash

                if quantity > 0:
                    print(f"  ₹{total_settlement_cash:,.2f} ({quantity} units @ {ticker_name})")
                else:
                    print(f"   ₹{abs(total_settlement_cash):,.2f} ({abs(quantity)} units @ {ticker_name})")

        for ticker_name in options_to_remove:
            del self.portfolio[ticker_name]
        
        print(f"  Settlement P&L: ₹{total_pnl_from_settlement:,.2f} | Final Cash Balance: ₹{self.cash_balance:,.2f}")
        




def run_ddpg_step(agent, env, central_market, done_flag):
    observation = env.get_state(central_market)
    action = agent.choose_action(observation)
    reward, new_observation = env.step(action.numpy(), central_market)
    
    agent.remember(observation, action.numpy(), reward, new_observation, done_flag)
    agent.learn()

#initialize constructor type function
def initialize_ddpg_agent(agent_id='DDPG_01', start_cash=10000000, active_tickers=11):
    env = DDPG_TradingEnvironment(
        agent_id=agent_id,
        start_cash=start_cash,
        active_ticker_count=active_tickers
    )
    agent = AgentDDPG(
        agent_id=agent_id,
        input_dims=[env.state_dimensions],
        n_actions=env.n_actions,
        n_tickers=env.n_tickers,
        env=env 
    )
    return agent, env