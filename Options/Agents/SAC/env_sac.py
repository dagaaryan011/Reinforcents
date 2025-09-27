
# Save as: agents/sac/env_sac.py
import numpy as np
import random
import csv
from market.orderbook import Order, Side
from .agent_sac import MarketMakerAgent
from .blackscholes import BlaScho

# black = BlaScho()

class MarketMakerEnv:
    def __init__(self):
        self.exchange = None  # Will be connected by main.py

        self.tickers_list = []
        
        self.inventory = 100
        self.capital = 100000
        self.start_amount = 100
        self.start_cash = 100000
        self.portfolio = {}

        # data dictionaries for each ticker
        self.highest_bids = {}
        self.lowest_asks = {}
        self.spreads = {}
        self.mids = {}
        self.premiums = {}
        self.deltas = {}
        self.gammas = {}
        self.thetas = {}
        self.vegas = {}

        self.MFM = {}
        self.volumes = {}
        self.MFV = {}
        self.CMF = {}

        self.min_prices = []  # minute-by-minute price
        self.opens = []
        self.closes = []
        self.highs = []
        self.lows = []

        self.time_to_expiry = None
        self.volatility = None

    def get_current_state(self,ticker):    #gives the current state for the ticker asked by agent
        c = self.capital
        q = self.portfolio[ticker]
        hb = self.highest_bids[ticker]
        la = self.lowest_asks[ticker]
        s = self.spreads[ticker]
        m = self.mids[ticker]
        p = self.premiums[ticker]
        d = self.deltas[ticker]
        g = self.gammas[ticker]
        t = self.thetas[ticker]
        v = self.vegas[ticker]
        ti = self.time_to_expiry
        vo = self.volatility
        cm = self.CMF[ticker]

        return [c, q, hb, la, s, m, p, d, g, t, v, ti, vo, cm]

    def run_step_min(self, price, agent_id):
        self.get_min_prices(price) #append current minute underlying price
        self.get_highestbid_dict(agent_id) # get the highest bids in all tickers
        self.get_lowestask_dict(agent_id)  # get the lowest asks in all tickers
        self.get_spread_and_mid_dict()     # get the spreads and mid values for all tickers

    def run_step_day(self, open, close, high, low, expiry):
        self.get_ticker_list()    # gets the option tickers for today
        self.set_mfm_and_volumes_and_mfv_dicts()   # resets the cmf calculation for today
        self.volatility = self.get_volatility()   #gets volatility in price till today
        self.get_day_prices_and_cmf_related(open, close, high, low, expiry) #does all calcs of getting cmf though balckscholes on openclosehighlow of prev day for all orderbooks 
        self.time_to_expiry = expiry    #constant for today
        self.get_premium_and_greek_dict(open)   #gets the premiums and greeks of all options/ tickers using blackscholes
        self.get_CMF_dict()      #gets CMF for today 
        # all these once calculated for today will be constant throughout the day

    def reset_portfolio(self):   # at new expiry option
        for ticker in self.tickers_list:
            self.portfolio[ticker] = 0

    def reset_minute_data(self):
        self.highest_bids = {}
        self.lowest_asks = {}
        self.spreads = {}
        self.mids = {}

    def get_ticker_list(self):
        self.tickers_list = [t[0] for t in self.exchange.tickers if t[0] != 'STOCK_UNDERLYING']
        for t in self.tickers_list:
            self.portfolio[t] = 0
        #print(self.tickers_list)

    def set_mfm_and_volumes_and_mfv_dicts(self):
        for ticker in self.tickers_list:
            self.volumes.setdefault(ticker, [])
            self.MFV.setdefault(ticker, [])
            self.MFM.setdefault(ticker,[])

    def get_min_prices(self, price):
        self.min_prices.append(price)

    def get_highestbid_dict(self, agent_id):
        for ticker in self.tickers_list:
            ob = self.exchange.get_book(ticker)
            bids = ob.get_bids(agent_id)
            highest_bid = bids[0][0] if bids else ob.market_price
            self.highest_bids[ticker] = highest_bid

    def get_lowestask_dict(self, agent_id):
        for ticker in self.tickers_list:
            ob = self.exchange.get_book(ticker)
            asks = ob.get_asks(agent_id)
            lowest_ask = asks[0][0] if asks else ob.market_price
            self.lowest_asks[ticker] = lowest_ask

    def get_spread_and_mid_dict(self):
        for ticker in self.tickers_list:
            bid = self.highest_bids.get(ticker, 0)
            ask = self.lowest_asks.get(ticker, 0)
            self.spreads[ticker] = ask - bid
            self.mids[ticker] = (ask + bid) / 2 if (ask + bid) != 0 else 0

    def get_premium_and_greek_dict(self, open):
        for ticker in self.tickers_list:
            parts = ticker.split('_')
            
            if len(self.min_prices)<=1:
                Spot = open
            else:
                Spot = self.min_prices[-1]
            Strike = float(parts[1])
            Time = self.time_to_expiry
            if "PE" in ticker :
                Option = "put"
            else :
                Option = "call"
            black = BlaScho(Spot, Strike, Time, Option)
            premium, delta, gamma, theta, vega = black.calculate()

            self.premiums[ticker] = premium
            self.deltas[ticker] = delta
            self.gammas[ticker] = gamma
            self.thetas[ticker] = theta
            self.vegas[ticker] = vega

    def get_day_prices_and_cmf_related(self, open, close, high, low, Time):
        self.opens.append(open)    #these lists are not used anywhere
        self.closes.append(close)
        self.highs.append(high)
        self.lows.append(low)

        for ticker in self.tickers_list:
            # MFM calc
            parts = ticker.split('_')
            Strike = float(parts[1])
            if 'CE' in ticker:
                Option = "call"
            else:
                Option = "put"
            # print(open, close, high, low, Strike, Time, Option)
            black_open = BlaScho(open, Strike, Time, Option)
            black_close = BlaScho(close, Strike, Time, Option)
            black_high = BlaScho(high, Strike, Time, Option)
            black_low = BlaScho(low, Strike, Time, Option)
            open_premium, _, _, _, _= black_open.calculate()
            close_premium, _, _, _, _= black_close.calculate()
            high_premium, _, _, _, _= black_high.calculate()
            low_premium, _, _, _, _= black_low.calculate()
            mfm = ((close_premium - low_premium) - (high_premium - close_premium)) / (high_premium - low_premium + 1e-6)  # Avoid division by zero
            self.MFM[ticker].append(mfm)

            # Volume calc
            total_volume = self.get_executed_count(ticker)
            prev_total = self.volumes[ticker][-1] if self.volumes[ticker] else 0
            daily_volume = total_volume - prev_total
            self.volumes[ticker].append(total_volume)

            # MFV calc
            self.MFV[ticker].append(mfm * daily_volume)

    def get_CMF_dict(self):     #uses calculate_CMF for each ticker
        for ticker in self.tickers_list:
            cmf = self.calculate_CMF(self.MFV[ticker], self.volumes[ticker])
            self.CMF[ticker] = cmf

    def calculate_CMF(self, mfv_list, vol_list, n=10):
        mfv_array = np.array(mfv_list[-n:]) if len(mfv_list) >= n else np.array(mfv_list)
        vol_array = np.array(vol_list[-n:]) if len(vol_list) >= n else np.array(vol_list)

        sum_vol = np.sum(vol_array)
        if sum_vol == 0:
            return 0.0
        return float(np.sum(mfv_array)) / float(sum_vol)

    def get_volatility(self):
        data = self.min_prices[-50:] if len(self.min_prices) > 50 else self.min_prices
        volatility = float(np.std(data)) if data else 0.0
        return volatility

    def get_orderbook(self, ticker):   #returns the book for the ticker
        return self.exchange.get_book(ticker)

    def get_highestbid_lowestask(self, ticker, agent_id):   #returns highest_bid and lowest_ask of a ticker
        ob = self.exchange.get_boook(ticker)
        bids = ob.get_bids(agent_id)
        asks = ob.get_asks(agent_id)
        highest_bid = bids[0][0] if bids else ob.market_price
        lowest_ask = asks[0][0] if asks else ob.market_price
        return highest_bid, lowest_ask

    def update_book(self, ticker, b_p, a_p, b_s, a_s, agent_id, buy_id, sell_id):     # palces the actual orders which get executed or not
        buy_order = Order(Side.BUY, b_p, b_s, owner_id=agent_id, order_id=buy_id)
        sell_order = Order(Side.SELL, a_p, a_s, owner_id=agent_id, order_id=sell_id)
        ob = self.get_orderbook(ticker)
        executed_trades_buy = ob.add_order(buy_order)
        executed_trades_sell = ob.add_order(sell_order)

        done_buy_price = sum(t.price for t in executed_trades_buy)
        done_buy_size = sum(t.size for t in executed_trades_buy)
        done_sell_price = sum(t.price for t in executed_trades_sell)
        done_sell_size = sum(t.size for t in executed_trades_sell)

        return done_buy_price, done_sell_price, done_buy_size, done_sell_size

    def get_reward(self, ticker, executed_ask_price, executed_ask_size, executed_bid_price, executed_bid_size, highest_bid, lowest_ask):
        PL = executed_ask_price * executed_ask_size - executed_bid_price * executed_bid_size
        diff_bid = abs(executed_bid_price - highest_bid)
        diff_ask = abs(lowest_ask - executed_ask_price)
        reward = PL - diff_bid - diff_ask
        size_diff = executed_bid_size - executed_ask_size
        self.capital+=PL
        self.inventory += size_diff
        self.portfolio[ticker] += size_diff
        return reward

    def get_executed_count(self, target, column="Ticker", filepath="C:\ProjectX\OptionsMulti\Options\market\master_trades.csv"):   #for calulating vol (no of matched/executed trades) for each day
        count = 0
        with open(filepath, 'r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get(column) == target:
                    count += float(row.get("Size"))
        return count

    def old_action_at_expiry(self, final, agent_id):
        cash_settlement = 0
        filepath = "C:\ProjectX\OptionsMulti\Options\market\master_trades.csv"
        with open(filepath, 'r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                PL = 0
                column_1 = row.get("Incoming")
                column_2 = row.get("Matched With")
                if agent_id in column_1 :
                    order = column_1
                    parts = order.split('_')
                    strike = float(parts[1])
                    if 'SELL' in order :
                        if 'PE' in order :
                            PL = final - strike
                        else :
                            PL = strike - final

                    if 'BUY' in order :
                        if 'PE' in order :
                            if final<strike:
                                PL = strike - final
                            else :
                                PL = 0
                        
                        else:
                            if final>strike:
                                PL = final-strike
                            else :
                                PL=0
                    cash_settlement+=PL
                    PL=0
                        
                if agent_id in column_2 :
                    order = column_2
                    parts = order.split('_')
                    strike = float(parts[1])
                    if 'SELL' in order :
                        if 'PE' in order :
                            PL = final - strike
                        else :
                            PL = strike - final

                    if 'BUY' in order :
                        if 'PE' in order :
                            if final<strike:
                                PL = strike - final
                            else :
                                PL = 0
                        
                        else:
                            if final>strike:
                                PL = final-strike
                            else :
                                PL=0
                    cash_settlement+=PL
                    PL=0
 
    def action_at_expiry(self, final):    #settlement action calle dat end of expiry
        cash_settlement = 0 
        for ticker in self.tickers_list:
            parts = ticker.split('_')
            strike = float(parts[1])
            amount = self.portfolio[ticker]
            PL = 0
            if amount < 0 :
                if 'PE' in ticker :
                    PL += final - strike
                else :
                    PL += strike - final

            if amount > 0 :
                if 'PE' in ticker :
                    if final<strike:
                        PL += strike - final
                    else :
                        PL += 0
                else:
                    if final>strike:
                        PL += final-strike
                    else :
                        PL += 0
            cash_settlement+=PL
            #print(cash_settlement)
    
        self.capital += cash_settlement

    def get_PL(self, final, initial):      #get the final profit and loss at end of expiry 
        PL = (self.capital + final * self.inventory) - (self.start_cash + initial * self.start_amount)
                                                        
        new_start_capital = self.capital
        new_start_inventory = self.inventory
        self.start_cash = new_start_capital
        self.start_amount = new_start_inventory
        return PL
    
    def notifications(self, ticker, agent_id):     # to check for trades which took place later on
        ob = self.get_orderbook(ticker)
        notifs = ob.collect_notifications_for(agent_id)
        for notif in notifs:
            if notif.maker_side == Side.BUY:
                self.portfolio[notif.ticker_id] += notif.size
                self.inventory += notif.size
                self.capital -= notif.price * notif.size
                #print(notif.size)
            if notif.maker_side == Side.SELL:
                self.portfolio[notif.ticker_id] -= notif.size
                self.inventory -= notif.size
                self.capital += notif.price * notif.size
                #print(notif.size)
        
    def _calculate_portfolio_value(self, central_market):
        
        value = self.env.capital
        for ticker_name,quantity in self.inventory.items():
            if quantity == 0:
                continue
            book = central_market.get_book(ticker_name)
            if book:
                price  = 0
                if quantity>0:
                    bids=book.get_bids('MarketMaker')
                    if bids:
                        price = bids[0][0]
                    if price > 0:
                        value += quantity * price 
        return value

def initialize_sac_agent(agent_id):        #to make an agent
    agent = MarketMakerAgent(id=agent_id)
    env = MarketMakerEnv()
    agent.env = env
    return agent, env

def run_sac_step(agent, price, agent_id):   # step for each min
    agent.env.run_step_min(price, agent_id)
    agent.collect()
    if agent.t > agent.memory_size and agent.t % 2 == 0:
        agent.learn()
