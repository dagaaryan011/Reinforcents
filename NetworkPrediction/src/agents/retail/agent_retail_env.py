# In src/agents/retail/agent_retail_env.py

import numpy as np
from collections import defaultdict

# Import the core components
from .agent_retail import Agent
from ...market.exchange import MarketExchange
from ...market.orderbook import Order, Side
from ...market.blackscholes import BlaScho
from ...tools.functions import calculate_historical_volatility
from config import LOT_SIZE

class AgentEnvironment:
    def __init__(self, agent: Agent, exchange: MarketExchange, initial_capital: float = 100000.0):
        """
        Initializes the environment that manages a single agent's trading activity.
        """
        self.agent = agent
        self.exchange = exchange
        
        # --- State Tracking ---
        self.portfolio = defaultdict(int)
        self.cash_balance = initial_capital
        self.order_size = 10

    def update_state(self, price: float):
        """
        Updates the agent's internal state by preprocessing the latest price.
        This is called on EVERY step of the simulation.
        """
        self.agent.preprocess_input(price)

    def make_decision(self):
        """
        Executes one full decision-making and trading cycle for the agent.
        This is called only on periodic decision steps.
        """
        # 1. First, check for any confirmed trades from the previous step
        self._check_trade_confirmations()

        # 2. Get a Signal from the Agent's "Brain"
        probabilities = self.agent.inference()
        signal = Agent.predict(probabilities)

        if signal is None:  # Agent is still in its warmup period
            return

        # 3. Manage Existing Positions & Look for New Trades
        if self.portfolio:
            self._handle_exit_logic(signal)
    
        self._handle_entry_logic(signal)

    # def _check_trade_confirmations(self):
    #     """ Checks all active order books for trade notifications for this agent. """
    #     for ticker_tuple in self.exchange.tickers:
    #         book = self.exchange.get_book(ticker_tuple[0])
    #         if book:
    #             notifications = book.collect_notifications_for(self.agent.agent_id)
    #             for trade in notifications:
    #                 print(f"    CONFIRMED: Agent {self.agent.agent_id} trade executed: {trade}")
    #                 cost = trade.price * trade.size * LOT_SIZE
                    
    #                 if trade.taker_side == Side.BUY:
    #                     self.portfolio[trade.ticker_id] += trade.size
    #                     self.cash_balance -= cost
    #                 elif trade.taker_side == Side.SELL:
    #                     self.portfolio[trade.ticker_id] -= trade.size
    #                     self.cash_balance += cost
                    
    #                 if self.portfolio.get(trade.ticker_id) == 0:
    #                     del self.portfolio[trade.ticker_id]
    def _check_trade_confirmations(self):
        """ Checks all active order books for trade notifications for this agent. """
        for ticker_tuple in self.exchange.tickers:
            book = self.exchange.get_book(ticker_tuple[0])
            if book:
                notifications = book.collect_notifications_for(self.agent.agent_id)
                for trade in notifications:
                    print(f"    CONFIRMED: Agent {self.agent.agent_id} trade executed: {trade}")
                    
                    # DEBUG: Check what data we actually have
                    print(f"      DEBUG: taker_id={trade.taker_id}, maker_id={trade.maker_id}")
                    print(f"      DEBUG: taker_side={trade.taker_side}, maker_side={trade.maker_side}")
                    print(f"      DEBUG: our_id={self.agent.agent_id}")
                    
                    cost = trade.price * trade.size * LOT_SIZE
                    
                    # Determine if we bought or sold
                    if trade.maker_id == self.agent.agent_id:
                        # We were the MAKER (our resting order got filled)
                        if trade.maker_side == Side.BUY:
                            # Our BUY order was filled - we bought
                            self.portfolio[trade.ticker_id] += trade.size
                            self.cash_balance -= cost
                            print(f"      ACTION: Our BUY order filled: +{trade.size} {trade.ticker_id}")
                        else:  # trade.maker_side == Side.SELL
                            # Our SELL order was filled - we sold
                            self.portfolio[trade.ticker_id] -= trade.size
                            self.cash_balance += cost
                            print(f"      ACTION: Our SELL order filled: -{trade.size} {trade.ticker_id}")
                    
                    # Clean up zero positions
                    if self.portfolio.get(trade.ticker_id) == 0:
                        del self.portfolio[trade.ticker_id]
                    
                    print(f"      UPDATED: Cash={self.cash_balance}, Portfolio={dict(self.portfolio)}")
    
    def _handle_exit_logic(self, signal: int):
        """ Submits orders to close open positions if the signal is opposite. """
        for ticker, quantity in list(self.portfolio.items()):
            is_call_position = 'CE' in ticker
            
            # Exit a BUY trade (a Call) if signal flips to SELL
            if is_call_position and signal == -1:
                self._submit_exit_order(ticker, quantity)

            # Exit a SELL trade (a Put) if signal flips to BUY
            elif not is_call_position and signal == 1:
                self._submit_exit_order(ticker, quantity)

    def _submit_exit_order(self, ticker, quantity):
        print(f"Agent {self.agent.agent_id}: Signal flipped. Submitting exit order for {quantity} of {ticker}.")
        book = self.exchange.get_book(ticker)
        if book and book.get_bids():
            exit_price = book.get_bids()[0][0]
            exit_order = Order(side=Side.SELL, price=exit_price, size=quantity, owner_id=self.agent.agent_id)
            book.add_order(exit_order)

    def _handle_entry_logic(self, signal: int):
        """ Checks for opportunities to enter a new trade based on the RNN signal and a value check. """
        # print(f"Agent{self.agent.agent_id}:signal={signal}")
        if signal == 0:
            return

        option_type = 'CE' if signal == 1 else 'PE'
        ticker_to_trade = f"STOCK_{self.exchange.atm}_{option_type}"
        book = self.exchange.get_book(ticker_to_trade)
        
        if book and book.get_asks():
            fair_value = self._calculate_fair_value(ticker_to_trade)
            best_ask_price = book.get_asks()[0][0]
            
            
            if best_ask_price <= fair_value:
                print(f"Agent {self.agent.agent_id}: Signal={signal}, Price is favorable. Submitting BUY order for {ticker_to_trade}.")
                entry_order = Order(side=Side.BUY, 
                                    price=best_ask_price, 
                                    size=self.order_size, 
                                    owner_id=self.agent.agent_id)
                book.add_order(entry_order)

    def _calculate_fair_value(self, ticker: str) -> float:
        """ Calculates a theoretical "fair value" for an option using the Black-Scholes model. """
        try:
            parts = ticker.split('_'); strike_price = int(parts[1]); option_type = 'call' if parts[2] == 'CE' else 'put'
        except (IndexError, ValueError):
            return 0.0

        volatility = calculate_historical_volatility(self.agent.trend)
        underlying_price = self.exchange.underlying_price
        time_to_expiry = self.exchange.time_to_expiry
        risk_free_rate = 0.05
        
        bs_price = BlaScho(
            spot=underlying_price, strike=strike_price, time=time_to_expiry,
            ret=risk_free_rate, vol=volatility, opt=option_type
        )
        b_price,_,_,_,_= bs_price.calculate()
        if b_price is None or b_price <= 0:
            return 0.01
            
        return b_price * 1.05
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
