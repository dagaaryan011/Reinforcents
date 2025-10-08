# In src/agent/agent_env.py

import numpy as np

# Import the core components this environment needs to interact with
from .agent_retail import Agent
from ..market.exchange import MarketExchange
from ..market.orderbook import Order, Side

class AgentEnvironment:
    def __init__(self, agent: Agent, exchange: MarketExchange, initial_capital: float = 10000.0):
        """
        Initializes the environment that manages a single agent's trading activity.

        Args:
            agent (Agent): The agent instance that will make predictions.
            exchange (MarketExchange): The market exchange where trades will be placed.
            initial_capital (float): The starting capital for the agent.
        """
        self.agent = agent
        self.exchange = exchange
        
        # --- State Tracking ---
        self.open_position = None  # e.g., {'ticker': 'STOCK_105_CE', 'side': 'BUY'}
        self.order_size = 10       # Fixed order size for simplicity
        self.portfolio_value = initial_capital

    def step(self):
        """
        Executes one full decision-making and trading cycle for the agent.
        This is the main function to be called from the simulation loop.
        """
        # --- 1. Get a Signal from the Agent's "Brain" ---
        current_price = self.exchange.underlying_price
        self.agent.preprocess_input(current_price)
        probabilities = self.agent.inference()
        signal = Agent.predict(probabilities)
        print(f"    Agent {self.agent.agent_id}: RNN Signal = {signal}")
        if signal is None:  # Agent is still in its warmup period
            return

        # --- 2. Manage Existing Positions (Exit Logic) ---
        if self.open_position:
            self._handle_exit_logic(signal)

        # --- 3. Look for New Trades (Entry Logic) ---
        if self.open_position is None:
            self._handle_entry_logic(signal)

    def _handle_exit_logic(self, signal: int):
        """ Checks if an open position should be closed based on the new signal. """
        position_ticker = self.open_position['ticker']
        position_side = self.open_position['side']

        # Exit a BUY trade (a Call) if signal flips to SELL
        if position_side == 'BUY' and signal == -1:
            print(f"Agent {self.agent.agent_id}: Signal flipped to SELL. Exiting Call position on {position_ticker}.")
            book = self.exchange.get_book(position_ticker)
            if book and book.get_bids():
                exit_price = book.get_bids()[0][0] # Sell at the best available bid
                exit_order = Order(side=Side.SELL, price=exit_price, size=self.order_size, owner_id=self.agent.agent_id)
                book.add_order(exit_order)
                self.open_position = None
        
        # Exit a SELL trade (a Put) if signal flips to BUY
        elif position_side == 'SELL' and signal == 1:
            print(f"Agent {self.agent.agent_id}: Signal flipped to BUY. Exiting Put position on {position_ticker}.")
            book = self.exchange.get_book(position_ticker)
            if book and book.get_bids():
                exit_price = book.get_bids()[0][0] # Sell the put at the best available bid
                exit_order = Order(side=Side.SELL, price=exit_price, size=self.order_size, owner_id=self.agent.agent_id)
                book.add_order(exit_order)
                self.open_position = None

    

    def _handle_entry_logic(self, signal: int):
        """
        Checks for opportunities to enter a new trade and PRINTS DETAILED DEBUG INFO.
        """
        if signal == 0:
            return

        # Determine which option type to look for based on the signal
        option_type = 'CE' if signal == 1 else 'PE'
        ticker_to_trade = f"STOCK_{self.exchange.atm}_{option_type}"
        book = self.exchange.get_book(ticker_to_trade)

        # --- DETAILED DEBUG BLOCK ---
        print(f"\n--- AGENT {self.agent.agent_id} EVALUATING TRADE ---")
        print(f"    - RNN Signal: {signal} ({'Bullish' if signal == 1 else 'Bearish'})")
        print(f"    - Target Ticker: {ticker_to_trade}")

        if not book or not book.get_asks():
            print(f"    - Market Status: Order book is empty or has no sellers. CANNOT TRADE.")
            print("    -----------------------------------\n")
            return

        fair_value = self._calculate_fair_value(ticker_to_trade)
        best_ask_price = book.get_asks()[0][0]
        condition_met = best_ask_price <= fair_value

        print(f"    - Best Ask Price: {best_ask_price}")
        print(f"    - Calculated Fair Value: {fair_value:.2f}")
        print(f"    - Trade Condition (best_ask <= fair_value): {condition_met}")

        if condition_met:
            print(f"    - ACTION: Condition met! Submitting BUY order.")
            entry_order = Order(side=Side.BUY, 
                                price=best_ask_price, 
                                size=self.order_size, 
                                owner_id=self.agent.agent_id)
            book.add_order(entry_order)
            # Portfolio is updated via notifications, so we don't update it here.
        else:
            print("    - ACTION: Price is not favorable. Not trading.")

        print("    -----------------------------------\n")

    def _calculate_fair_value(self, ticker: str) -> float:
        """
        Calculates a theoretical "fair value" for an option.
        This is a placeholder for a more advanced pricing model like Black-Scholes.
        """
        strike_price = self.exchange.get_book(ticker).ticker_id.split('_')[1]
        premium = self.exchange.calculate_initial_premium((ticker, int(strike_price)))
        # Agent is willing to pay up to a 2% premium over the simple calculated value
        return premium * 2