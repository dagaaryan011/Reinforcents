import numpy as np
import pandas as pd
import csv
from src.market.blackscholes import BlaScho
from src.market.orderbook import Order, Side


class Env:
    def __init__(self):

        self.exchange = None

        self.trend = []
        self.tickers_list = []
        self.strikes_dict = {}
        self.MFM = []
        self.volumes = {}
        self.daily_volumes = []
        self.total_volumes = {}
        self.MFV = []

        self.highest_bids = {}
        self.lowest_asks = {}
        self.spreads = {}
        self.mids = {}
        self.premiums = {}
        self.deltas = {}
        self.gammas = {}
        self.thetas = {}
        self.vegas = {}

        self.time_to_expiry = 0
        self.CMF = 0
        self.days_passed = 0
        self.mins_passed = 0

    def get_state(self, ticker):
        cmf = self.CMF
        hb = self.highest_bids[ticker]
        la = self.lowest_asks[ticker]
        s = self.spreads[ticker]
        m = self.mids[ticker]
        pr = self.premiums[ticker]
        d = self.deltas[ticker]
        g = self.gammas[ticker]
        t = self.thetas[ticker]
        v = self.vegas[ticker]
        ti = self.time_to_expiry
        return [cmf, hb, la, s, m, pr, d, g, t, v, ti]


    def get_trend(self, price):
        self.trend.append(price)
        self.mins_passed += 1

    def get_open_high_close_low(self):
        open = self.trend[0]
        high = max(self.trend)
        close = self.trend[-1]
        low = min(self.trend)

        return open, high, close, low
    
    def get_tickers_strikes_list(self):
        self.tickers_list = [t[0] for t in self.exchange.tickers if t[0] != 'STOCK_UNDERLYING']
        for t in self.exchange.tickers:
            if t[0] != 'STOCK_UNDERLYING':
                self.strikes_dict[t[0]] = t[1]

    def set_total_volumes_dict(self):
        for ticker in self.tickers_list:
            self.total_volumes[ticker] = 0
            
    def reset_mfm_and_volumes_and_mfv_dicts(self):
        for ticker in self.tickers_list:
            self.volumes.setdefault(ticker, [])
        self.MFV = []
        self.MFM = []

    def get_volumes(self):
        for ticker in self.tickers_list:
            daily_volume = self.get_executed_count(ticker)
            self.volumes[ticker].append(daily_volume)
            self.total_volumes[ticker] += daily_volume/1000


    def get_executed_count(self, target, column="Ticker", filepath="C:\ProjectX\OptionsTrading\Market\master_trades.csv"):   #for calulating vol (no of matched/executed trades) for each day
        count = 0
        with open(filepath, 'r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get(column) == target:
                    count += float(row.get("Size"))
        return count

    def get_day_prices_and_cmf_related(self,open, high, close, low, Time):     

        for ticker in self.tickers_list:
            # MFM calc
            mfm = ((close - low) - (high - close)) / (high - low + 1e-6)
            self.MFM.append(mfm)
            daily_volume = 0
            for ticker in self.tickers_list:
                daily_volume+=self.volumes[ticker][-1]
            mfv = mfm*daily_volume
            self.MFV.append(mfv)
            self.daily_volumes.append(daily_volume)
            

    def calculate_CMF(self, mfv_list, vol_list, n=10):
        mfv_array = np.array(mfv_list[-n:]) if len(mfv_list) >= n else np.array(mfv_list)
        vol_array = np.array(vol_list[-n:]) if len(vol_list) >= n else np.array(vol_list)

        sum_vol = np.sum(vol_array)
        if sum_vol == 0:
            return 0.0
        return float(np.sum(mfv_array)) / float(sum_vol)
    
    
    def get_CMF(self):     #uses calculate_CMF for each ticker
        self.CMF=self.calculate_CMF(self.MFV, self.daily_volumes)

    def get_premium_and_greek_dict(self, open): #calculates premium and greeks for all tickers
        for ticker in self.tickers_list:
            parts = ticker.split('_')
            
            if len(self.trend)<=1:
                Spot = open
            else:
                Spot = self.trend[-1]
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

    def get_highestbid_lowestask_dict(self, agent_id="MM"):
        for ticker in self.tickers_list:
            ob = self.exchange.get_book(ticker)
            bids = ob.get_bids(agent_id)
            asks = ob.get_asks(agent_id)
            highest_bid = bids[0][0] if bids else ob.market_price
            lowest_ask = asks[0][0] if asks else ob.market_price
            self.highest_bids[ticker] = highest_bid
            self.lowest_asks[ticker] = lowest_ask

    def get_spread_and_mid_dict(self):
        for ticker in self.tickers_list:
            bid = self.highest_bids.get(ticker, 0)
            ask = self.lowest_asks.get(ticker, 0)
            self.spreads[ticker] = ask - bid
            self.mids[ticker] = (ask + bid) / 2 if (ask + bid) != 0 else 0


    def indicator(self):         # calcculates cmf
        self.reset_mfm_and_volumes_and_mfv_dicts()
        self.get_volumes()
        open, high, close, low = self.get_open_high_close_low()
        self.get_day_prices_and_cmf_related(open, high, close, low, self.time_to_expiry)
        self.get_CMF()
        self.mins_passed = 0
        self.days_passed += 1

    def every_min(self):
        self.get_highestbid_lowestask_dict()
        self.get_spread_and_mid_dict()
        self.get_premium_and_greek_dict(self.trend[0])

    def action_at_expiry(self):
        self.exchange = None

        self.trend = []
        self.tickers_list = []
        self.strikes_dict = {}
        self.MFM = {}
        self.volumes = {}
        self.total_volumes = {}
        self.MFV = {}
        self.CMF = {}

        self.highest_bids = {}
        self.lowest_asks = {}
        self.spreads = {}
        self.mids = {}
        self.premiums = {}
        self.deltas = {}
        self.gammas = {}
        self.thetas = {}
        self.vegas = {}

        self.time_to_expiry = 0
        self.days_passed = 0
        self.mins_passed = 0

    def run(self, time, price):
        self.time_to_expiry = time
        if self.days_passed <= 0 and self.mins_passed <= 0 :
            self.get_tickers_strikes_list()
            self.set_total_volumes_dict()
            self.get_trend(price)
            # print("first, first", self.days_passed, self.mins_passed)
        elif self.days_passed > 0 and self.mins_passed <= 0 :
            self.trend = []
            self.get_trend(price)
            self.every_min()
            # print("other, first", self.days_passed, self.mins_passed)
        elif self.days_passed <= 0 :
            self.get_trend(price)
            # print("first, other", self.days_passed, self.mins_passed)
        else :
            self.every_min()
            self.get_trend(price)
            # print("other, other", self.days_passed, self.mins_passed)


def run(agent, env):
    if env.days_passed <= 0 and env.mins_passed <= 1 :    #first day first min
        agent.broker.set_portfolio()
        # print(" agent, first, first", env.days_passed, env.mins_passed)
    elif env.days_passed > 0 and env.mins_passed <= 1 :     # other day first min
        agent.broker.new_day() 
        # print(" agent, other, first", env.days_passed, env.mins_passed)    
    elif env.days_passed <= 0 :  
        pass                          # first day other mins
        # print(" agent, first, other", env.days_passed, env.mins_passed)
    else :                                                   # other day other mins
        agent.collect()
        # print(" agent, other, other", env.days_passed, env.mins_passed)

    