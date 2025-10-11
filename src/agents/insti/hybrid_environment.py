import numpy as np
from collections import defaultdict

# Import components from other modules
from ...market.blackscholes import BlaScho
from ...tools.functions import calculate_historical_volatility
from .agent_insti import Agent_Insti
from ...market.exchange import MarketExchange
from ...market.orderbook import Order, Side
from config import LOT_SIZE, NUM_LONG_TERM_OUTPUTS

class HybridAgentEnvironment:
    def __init__(self, agent: Agent_Insti, exchange: MarketExchange, initial_capital: float = 1000000.0):
        """
        Initializes the environment that manages the institutional agent's trading activity.
        """
        self.agent = agent
        self.exchange = exchange
        self.settlement_reward = 0
        
        # --- State Tracking ---
        self.portfolio = defaultdict(int)
        self.cash_balance = initial_capital
        self.portfolio_value = initial_capital
        
        # Define the set of market data this RL agent will observe
        self.n_tickers_to_observe = 5
        self.active_tickers = []

        # Define the structure of the state space for consistency
        self.market_data_size = self.n_tickers_to_observe * 4
        self.portfolio_size = 1 + self.n_tickers_to_observe
        self.rnn_context_size = NUM_LONG_TERM_OUTPUTS
        self.total_state_size = self.rnn_context_size + self.portfolio_size + self.market_data_size

    def settle_at_expiry(self, final_stock_price: float):
        """
        Calculates the final P&L for all open option positions at expiry and
        updates the agent's cash balance.
        """
        print(f"\n--- SETTLEMENT FOR AGENT {self.agent.agent_id} ---")
        print(f"    Initial Cash Balance: {self.cash_balance:,.2f}")
        total_settlement_pnl = 0

        for ticker, quantity in list(self.portfolio.items()):
            try:
                parts = ticker.split('_')
                strike_price = int(parts[1])
                option_type = parts[2]
            except (IndexError, ValueError):
                continue

            intrinsic_value = 0
            if option_type == 'CE': # Call Option
                intrinsic_value = max(0, final_stock_price - strike_price)
            elif option_type == 'PE': # Put Option
                intrinsic_value = max(0, strike_price - final_stock_price)

            position_final_value = intrinsic_value * quantity * LOT_SIZE
            total_settlement_pnl += position_final_value
            print(f"    - Settling {quantity} units of {ticker}: Final Value = {position_final_value:,.2f}")

        self.settlement_reward = total_settlement_pnl
        self.cash_balance += total_settlement_pnl
        self.portfolio.clear()
        
        print(f"    Total P&L from Settlement: {total_settlement_pnl:,.2f}")
        print(f"    Final Cash Balance After Settlement: {self.cash_balance:,.2f}")
        print("------------------------------------")

    def get_state(self) -> np.ndarray:
        """
        Constructs the complete state vector for the DDPG agent.
        """
        rnn_context, _ = self.agent.get_rnn_context()
        if rnn_context is None:
            return np.zeros(self.total_state_size, dtype=np.float32)

        self._update_active_tickers()
        portfolio_state = self._get_portfolio_state()
        raw_market_data = self._get_raw_market_data()
        
        state = np.concatenate([
            rnn_context.flatten(),
            portfolio_state.flatten(),
            raw_market_data.flatten()
        ], dtype=np.float32)
        
        return state

    def step(self, action: np.ndarray):
        """
        Executes one full decision-making and trading cycle for the agent.
        """
        old_portfolio_value = self._calculate_portfolio_value()
        
        executed_trade_info = self._execute_trade(action) 
        self._check_trade_confirmations()
        
        _, rnn_signal = self.agent.get_rnn_context()
        reward = self._calculate_shaped_reward(old_portfolio_value, rnn_signal, executed_trade_info)
        
        new_state = self.get_state()
        done = self.cash_balance < 0.0 or self._is_episode_done() 
        
        return new_state, reward, done

    def _get_raw_market_data(self) -> np.ndarray:
        market_data = []
        for ticker in self.active_tickers:
            book = self.exchange.get_book(ticker)
            if book and book.get_bids() and book.get_asks():
                best_bid_price, best_bid_size = book.get_bids()[0]
                best_ask_price, best_ask_size = book.get_asks()[0]
                market_data.extend([best_bid_price, best_bid_size, best_ask_price, best_ask_size])
            else:
                market_data.extend([0, 0, 0, 0])
        return np.array(market_data)

    def _get_portfolio_state(self) -> np.ndarray:
        portfolio_state = [self.cash_balance]
        for ticker in self.active_tickers:
            portfolio_state.append(self.portfolio.get(ticker, 0))
        return np.array(portfolio_state)

    def _execute_trade(self, action: np.ndarray):
        """
        MINIMAL CHANGE: Just add confidence threshold to existing working logic
        """
        self._update_active_tickers()

        # ORIGINAL LOGIC (that was working)
        max_action_index = np.argmax(np.abs(action))
        trade_strength = action[max_action_index]
        ticker_to_trade = self.active_tickers[max_action_index]
        current_position = self.portfolio.get(ticker_to_trade, 0)
        order_side = Side.BUY if trade_strength > 0 else Side.SELL
        print(f"DEBUG: Agent {self.agent.agent_id} - Ticker: {ticker_to_trade}, Current Pos: {current_position}, Side: {order_side}, Strength: {trade_strength:.3f}")
        # NEW: Only trade if we have sufficient conviction
        if abs(trade_strength) < 0.2:
            return None

        # REST OF ORIGINAL LOGIC...
        book = self.exchange.get_book(ticker_to_trade)
        if not book: 
            print("idhar error  - 1")
            return None
        current_position = self.portfolio.get(ticker_to_trade, 0)
        order_side = Side.BUY if trade_strength > 0 else Side.SELL

        # Check if we can actually sell what we don't own
        order_side = Side.BUY if trade_strength > 0 else Side.SELL
    
        
        if order_side == Side.BUY and book.get_asks():
            price = book.get_asks()[0][0]
        elif order_side == Side.SELL and book.get_bids():
            price = book.get_bids()[0][0]
        else:
            print("error -2")
            return None

        base_size = 50
    
        if order_side == Side.BUY:
            potential_new_position = current_position + base_size
            if potential_new_position > 50:  # Would exceed long limit
                base_size = min(50 - current_position,base_size)
                
        else:  # SELL
            potential_new_position = current_position - base_size  
            if potential_new_position < -50:  # Would exceed short limit
                base_size = max(-50 - current_position , base_size )
                
    
        # Create and submit order
        order = Order(order_side, price, base_size, self.agent.agent_id)
        executed_trades = book.add_order(order)

        # Process immediate fills
        executed_action_info = None
        if executed_trades:
            executed_action_info = {'ticker': ticker_to_trade, 'side': order_side}
            for trade in executed_trades:
                print(f"    [TRADE CONFIRMED] Agent {self.agent.agent_id} -> {trade.taker_side.name} {trade.size} of {trade.ticker_id} @ {trade.price:.2f}")

                # Update portfolio and cash immediately
                cost = trade.price * trade.size * LOT_SIZE
                if trade.taker_side == Side.BUY:
                    self.portfolio[trade.ticker_id] += trade.size
                    self.cash_balance -= cost
                elif trade.taker_side == Side.SELL:
                    self.portfolio[trade.ticker_id] -= trade.size
                    self.cash_balance += cost

                if self.portfolio.get(trade.ticker_id) == 0:
                    del self.portfolio[trade.ticker_id]

        return executed_action_info

    def _is_fairly_priced(self, ticker: str) -> bool:
        """
        NEW: Simple fair value check similar to retail trader
        """
        book = self.exchange.get_book(ticker)
        if not book or not book.get_asks():
            return False
            
        # For now, just check if there's liquidity
        # You can add the full Black-Scholes fair value calculation here later
        best_ask = book.get_asks()[0][0]
        return best_ask > 0.01  # Basic check - option has some value

    def _submit_market_order(self, ticker: str, side: Side, strength: float):
        """
        NEW: Submit a market order based on signal strength
        """
        book = self.exchange.get_book(ticker)
        if not book:
            return None
            
        # Calculate position size based on signal strength
        base_size = max(1, int(strength * 20))  # Scale with conviction (1-20 lots)
        
        # Determine price based on side
        if side == Side.BUY and book.get_asks():
            price = book.get_asks()[0][0]
        elif side == Side.SELL and book.get_bids():
            price = book.get_bids()[0][0]
        else:
            return None  # No liquidity
            
        # Create and submit order
        order = Order(side, price, base_size, self.agent.agent_id)
        executed_trades = book.add_order(order)
        
        # Process immediate fills
        executed_action_info = None
        if executed_trades:
            executed_action_info = {'ticker': ticker, 'side': side}
            for trade in executed_trades:
                print(f"    [TRADE CONFIRMED] Agent {self.agent.agent_id} -> {trade.taker_side.name} {trade.size} of {trade.ticker_id} @ {trade.price:.2f}")
                
                # Update portfolio and cash immediately
                cost = trade.price * trade.size * LOT_SIZE
                if trade.taker_side == Side.BUY:
                    self.portfolio[trade.ticker_id] += trade.size
                    self.cash_balance -= cost
                elif trade.taker_side == Side.SELL:
                    self.portfolio[trade.ticker_id] -= trade.size
                    self.cash_balance += cost
                
                if self.portfolio.get(trade.ticker_id) == 0:
                    del self.portfolio[trade.ticker_id]
                    
        return executed_action_info

    # def _check_trade_confirmations(self):
    #     """
    #     Checks the notification mailbox for any PASSIVE orders that were filled
    #     by other market participants.
    #     """
    #     for ticker_tuple in self.exchange.tickers:
    #         book = self.exchange.get_book(ticker_tuple[0])
    #         if book:
    #             # This logic remains the same, handling passive fills
    #             notifications = book.collect_notifications_for(self.agent.agent_id)
    #             for trade in notifications:
    #                 print(f"    [PASSIVE FILL] Agent {self.agent.agent_id}'s resting order was hit: {trade}")
    #                 # Update portfolio and cash for these fills as well
    #                 cost = trade.price * trade.size * LOT_SIZE
    #                 # Note: For a passive fill, the side is opposite the order
    #                 # e.g., a resting BUY order is filled by a Taker's SELL
    #                 if trade.taker_side == Side.SELL: # Our resting BUY was hit
    #                     self.portfolio[trade.ticker_id] += trade.size
    #                     self.cash_balance -= cost
    #                 elif trade.taker_side == Side.BUY: # Our resting SELL was hit
    #                     self.portfolio[trade.ticker_id] -= trade.size
    #                     self.cash_balance += cost    

    def _check_trade_confirmations(self):
        """ Checks all active order books for trade notifications for this agent. """
        for ticker_tuple in self.exchange.tickers:
            book = self.exchange.get_book(ticker_tuple[0])
            if book:
                notifications = book.collect_notifications_for(self.agent.agent_id)
                for trade in notifications:
                    print(f"    CONFIRMED: Agent {self.agent.agent_id} trade executed: {trade}")
                    cost = trade.price * trade.size * LOT_SIZE
                    
                    # SIMPLE FIX: If we're getting notified, we were the MAKER
                    # (Only makers get passive fill notifications)
                    if trade.maker_side == Side.BUY:  # Our BUY order was hit
                        self.portfolio[trade.ticker_id] += trade.size
                        self.cash_balance -= cost
                    elif trade.maker_side == Side.SELL:  # Our SELL order was hit  
                        self.portfolio[trade.ticker_id] -= trade.size
                        self.cash_balance += cost
                    
                    if self.portfolio.get(trade.ticker_id) == 0:
                        del self.portfolio[trade.ticker_id]
    # In src/agents/insti/hybrid_environment.py



    def _calculate_portfolio_value(self) -> float:
        """
        Calculates portfolio value using live book prices, with a fallback
        to the theoretical Black-Scholes price if the book is illiquid.
        """
        value = self.cash_balance
        for ticker, quantity in self.portfolio.items():
            if quantity == 0:
                continue
            
            book = self.exchange.get_book(ticker)
            price = 0

            # --- START: ROBUST VALUATION LOGIC ---

            # 1. First, try to get a live market price from the order book
            if quantity > 0 and book and book.get_bids():
                price = book.get_bids()[0][0]
            elif quantity < 0 and book and book.get_asks():
                price = book.get_asks()[0][0]
            else:
                # 2. FALLBACK: If no live price, calculate the theoretical price
                try:
                    parts = ticker.split('_')
                    strike_price = int(parts[1])
                    option_type = 'call' if parts[2] == 'CE' else 'put'

                    # Use the agent's trend history to calculate volatility
                    volatility = calculate_historical_volatility(self.agent.trend)

                    bs_model = BlaScho(
                        spot=self.exchange.underlying_price,
                        strike=strike_price,
                        time=self.exchange.time_to_expiry,
                        ret=0.05, # Assuming a risk-free rate of 5%
                        vol=volatility,
                        opt=option_type
                    )
                    theoretical_price, _, _, _, _ = bs_model.calculate()

                    if theoretical_price and theoretical_price > 0:
                        price = theoretical_price
                    else:
                        # If BS model fails, we cannot value this position
                        continue
                except (ValueError, IndexError, ZeroDivisionError):
                    # Catch potential errors during calculation and skip position
                    continue

            # --- END: ROBUST VALUATION LOGIC ---

            # This calculation now works for both long and short positions
            value += quantity * price * LOT_SIZE

        self.portfolio_value = value
        return value

    def _calculate_shaped_reward(self, old_value: float, rnn_signal: int, executed_action: dict) -> float:
        pnl_reward = self._calculate_portfolio_value() - old_value
        
        consistency_reward = 0.2
        if executed_action and rnn_signal is not None:
            action_is_bullish = (executed_action['side'] == Side.BUY and 'CE' in executed_action['ticker']) or \
                                (executed_action['side'] == Side.SELL and 'PE' in executed_action['ticker'])
            action_is_bearish = (executed_action['side'] == Side.BUY and 'PE' in executed_action['ticker']) or \
                                (executed_action['side'] == Side.SELL and 'CE' in executed_action['ticker'])

            if rnn_signal == 1 and action_is_bullish: consistency_reward = 0.1
            elif rnn_signal == 1 and action_is_bearish: consistency_reward = -0.1
            elif rnn_signal == -1 and action_is_bearish: consistency_reward = 0.1
            elif rnn_signal == -1 and action_is_bullish: consistency_reward = -0.1
        
        settle_reward = self.settlement_reward
        self.settlement_reward = 0
        return pnl_reward + consistency_reward + settle_reward

    def _update_active_tickers(self):
        price = self.exchange.underlying_price
        all_opts = [t[0] for t in self.exchange.tickers if t[0] != 'STOCK_UNDERLYING']
        all_opts.sort(key=lambda ticker: abs(int(ticker.split('_')[1]) - price))
        self.active_tickers = all_opts[:self.n_tickers_to_observe]

    def _is_episode_done(self):
        # Placeholder for episode termination logic (e.g., end of trading day)
        return False