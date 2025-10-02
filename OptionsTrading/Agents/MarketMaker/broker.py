from Market.orderbook import Order, Side

class Broker:
    def __init__(self):
        
        self.start_cash = 100000
        self.temp_capital = 100000
        self.capital = 100000

        self.start_amount = 100
        self.temp_inventory = 100
        self.inventory = 100

        self.portfolio = {}
        self.cash_settlement = {}

        self.env = None    # one common connected to the market

    def get_actual_state(self,ticker):
        state = self.env.get_state(ticker)
        state.append(self.capital)
        state.append(self.inventory)
        state.append(self.portfolio[ticker])

        return state
    
    def get_all_states(self):
        states = []
        for ticker in self.env.tickers_list:
            states.append(self.get_actual_state(ticker))

        return states
    
    def get_notifications(self, agent_id):
        for ticker in self.env.tickers_list: 
            ob = self.env.exchange.get_book(ticker)
            notifs = ob.collect_notifications_for(agent_id)
            for notif in notifs:
                if notif.maker_side == Side.BUY:
                    self.portfolio[notif.ticker_id] += notif.size
                    self.inventory += notif.size
                    self.temp_inventory += notif.size
                    self.capital -= notif.price * notif.size
                    self.temp_capital -= notif.price * notif.size
                    #print(notif.size)
                if notif.maker_side == Side.SELL:
                    self.portfolio[notif.ticker_id] -= notif.size
                    self.inventory -= notif.size
                    self.temp_inventory -= notif.size
                    self.capital += notif.price * notif.size
                    self.temp_capital += notif.price * notif.size
                    #print(notif.size)
    
    def update_book(self, ticker, b_p, a_p, b_s, a_s, agent_id):     # palces the actual orders which get executed or not
        buy_order = Order(Side.BUY, b_p, b_s, owner_id=agent_id)
        sell_order = Order(Side.SELL, a_p, a_s, owner_id=agent_id)
        ob = self.env.exchange.get_book(ticker)
        executed_trades_buy = ob.add_order(buy_order)
        executed_trades_sell = ob.add_order(sell_order)

        done_buy_price = sum(t.price for t in executed_trades_buy)
        done_buy_size = sum(t.size for t in executed_trades_buy)
        done_sell_price = sum(t.price for t in executed_trades_sell)
        done_sell_size = sum(t.size for t in executed_trades_sell)

        self.capital += done_sell_price - done_buy_price
        self.temp_capital += a_p - b_p + done_sell_price - done_buy_price
        self.inventory += done_buy_size - done_sell_size
        self.temp_inventory += b_s - a_s + done_buy_size - done_sell_size
        self.portfolio[ticker] += done_buy_size - done_sell_size

        print(self.capital)
        print(self.temp_capital)
        print(self.inventory)
        print(self.temp_inventory)
        print(self.portfolio[ticker])

        # return done_buy_price, done_sell_price, done_buy_size, done_sell_size
        
            
    def reset_portfolio(self):
        self.portfolio = {}
        self.capital = 100000
        self.temp_capital = 100000
        self.inventory = 100
        self.temp_inventory = 100

    def set_portfolio(self):
        for ticker in self.env.tickers_list:
            self.portfolio[ticker] = 0
        print("portfolio set")

    def settlement(self, final):   #settlement action called at end of expiry
        for ticker in self.env.tickers_list:
            strike = self.env.strikes_dict[ticker]
            amount = self.portfolio[ticker]
            PL = 0
            if amount < 0 :
                if 'PE' in ticker :
                    PL = final - strike
                else :
                    PL = strike - final
                PL = min(0, PL) * abs(amount)
            if amount > 0 :
                if 'PE' in ticker :
                    PL = strike - final
                else:
                    PL = final-strike
                PL = max(0, PL) * abs(amount)
            self.cash_settlement[ticker] = PL

    def settle(self):   #settlement action called at end of expiry
        for ticker in self.env.tickers_list:
            self.capital += self.cash_settlement[ticker]


    def get_PL(self, initial, final):
        PL = ( self.capital + self.inventory * final ) - ( self.start_cash + self.start_amount * initial)
        return PL

    def new_option(self):
        self.temp_capital = self.capital
        self.temp_inventory = self.inventory


