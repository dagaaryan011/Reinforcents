import numpy as np
from orderbook import OrderBook, Order, Side
from datetime import datetime, timedelta
from noise2 import path
import matplotlib.pyplot as plt

class Env:
    trends = {
        0: "up",
        1: "down",
        2: "neutral"
    }

    actions = {
        0: "buy",
        1: "sell",
        2: "hold",
    }



class Agent:
    def __init__(self):
        self.Qtable = np.zeros((3,3))
        self.ob = OrderBook()
        self.trend = []
        self.capital = 100000
        self.lot = 100
        self.buys = 0
        self.sells = 0
        self.size = 10
        self.times = []
        self.trends = []
        self.market_prices = []
        self.actions = []
        self.executing_prices = []
        self.alpha = 0.001
        self.epsilon = 0.2
        self.gamma = 0.7

    def trade(self,close):
        for i in range(1000, 21000, 2000):
            self.time = i
            self.see()
            self.get_action()
            self.act()
        print("lot", self.lot)
        print("buys", self.buys)
        print("sells", self.sells)
        lot_change = self.lot - 100
        PL = lot_change * close
        print("capital", self.capital + PL)
        print("PL", PL)
        
    def do(self):
        for i in range(1000, 21000, 2000):
            self.time = i
            self.see()
            self.action()
            self.act()


    def see(self):
        diff = []
        for i in range(0,1000):
            diff.append(self.trend[self.time+i+1] - self.trend[self.time+i])
        avg = sum(diff)
        threshold = 0.3

        if avg > threshold:
            self.t = 0
        elif avg < -threshold:
            self.t = 1
        else:
            self.t = 2
        
        self.times.append(self.time)
        self.market_prices.append(self.trend[self.time])
        self.trends.append(self.t)

    def action(self):
        if np.random.rand() < self.epsilon:
            self.a=np.random.randint(0,3)
        else:
            self.get_action()
        self.actions.append(self.a)
         
    def get_action(self):
        self.a = np.argmax(self.Qtable[self.t])

    def act(self):

        if self.a == 0 :
            if self.capital >= 3000 :
                self.price = self.trend[self.time] + 3
                self.executing_prices.append(self.price)
                self.side = Side.BUY
                order= Order(self.side, self.price, self.size)
                self.ob.process_order(order)
                trade_prices = np.array(order.tradeprices)
                t_p = np.sum(trade_prices) * self.size
                self.capital -= t_p
                self.lot += self.size
                self.buys += 1
            else :
                self.a = 2
                self.actions.pop()
                self.actions.append(self.a)
                self.price = self.trend[self.time]
                self.executing_prices.append(self.price)

        
        elif self.a == 1 :
            if self.lot >= 30 :
                self.price = self.trend[self.time] - 3
                self.executing_prices.append(self.price)
                self.side = Side.SELL
                order= Order(self.side, self.price, self.size)
                self.ob.process_order(order)
                trade_prices = np.array(order.tradeprices)
                t_p = np.sum(trade_prices) * self.size
                self.capital += t_p
                self.lot -= self.size
                self.sells += 1
            else :
                self.a = 2
                self.actions.pop()
                self.actions.append(self.a)
                self.price = self.trend[self.time]
                self.executing_prices.append(self.price)

        elif self.a == 2 :
            self.price = self.trend[self.time]
            self.executing_prices.append(self.price)
        
    def clear(self):
        self.times = []
        self.trends = []
        self.market_prices = []
        self.actions = []
        self.executing_prices = []
        self.capital = 100000
        self.lot = 100
        self.buys = 0
        self.sells = 0

    def update(self):
        
        for  i  in range(0,len(self.times)):
            time = self.times[i]
            trend = self.trends[i]
            market_price = self.market_prices[i]
            action = self.actions[i]
            executing_price = self.executing_prices[i]

            PL = self.trend[time+1000] - market_price
            PL*=10

            if market_price == executing_price :
                self.Qtable[trend, action] -= self.alpha 
            else:
                self.Qtable[trend, action] += self.alpha * (PL + self.gamma * np.max(self.Qtable[trend, action]  -  self.Qtable[trend, action]))

    


        


    

if __name__ == "__main__":
    
    

    bot = Agent()
    j=1
    for i in range(0,1000):
        bot.ob = OrderBook()
        ob = bot.ob
        # BUY Orders (bot can sell to these)
        ob.bids[190].append(Order(Side.BUY, 190, 120))
        ob.bids[191].append(Order(Side.BUY, 191, 150))
        ob.bids[192].append(Order(Side.BUY, 192, 180))
        ob.bids[193].append(Order(Side.BUY, 193, 90))
        ob.bids[194].append(Order(Side.BUY, 194, 100))
        ob.bids[195].append(Order(Side.BUY, 195, 140))
        ob.bids[196].append(Order(Side.BUY, 196, 160))
        ob.bids[197].append(Order(Side.BUY, 197, 130))
        ob.bids[190].append(Order(Side.BUY, 190, 200))
        ob.bids[191].append(Order(Side.BUY, 191, 85))
        ob.bids[192].append(Order(Side.BUY, 192, 175))
        ob.bids[193].append(Order(Side.BUY, 193, 90))
        ob.bids[194].append(Order(Side.BUY, 194, 110))
        ob.bids[195].append(Order(Side.BUY, 195, 95))
        ob.bids[196].append(Order(Side.BUY, 196, 150))

        # SELL Orders (bot can buy from these)
        ob.offers[190].append(Order(Side.SELL, 190, 100))
        ob.offers[191].append(Order(Side.SELL, 191, 160))
        ob.offers[192].append(Order(Side.SELL, 192, 180))
        ob.offers[193].append(Order(Side.SELL, 193, 140))
        ob.offers[194].append(Order(Side.SELL, 194, 90))
        ob.offers[195].append(Order(Side.SELL, 195, 120))
        ob.offers[196].append(Order(Side.SELL, 196, 170)) 
        ob.offers[197].append(Order(Side.SELL, 197, 130))
        ob.offers[190].append(Order(Side.SELL, 190, 150))
        ob.offers[191].append(Order(Side.SELL, 191, 110))
        ob.offers[192].append(Order(Side.SELL, 192, 145))
        ob.offers[193].append(Order(Side.SELL, 193, 125))
        ob.offers[194].append(Order(Side.SELL, 194, 100))
        ob.offers[195].append(Order(Side.SELL, 195, 190))
        ob.offers[196].append(Order(Side.SELL, 196, 85))
        

        start_date = datetime.strptime('2021-07-01', '%Y-%m-%d').date()
        end_date = datetime.strptime('2021-07-02', '%Y-%m-%d').date()
        
        p=path(start_date, end_date)
        p.do()
        t=p.get_path()

        bot.trend = t
        bot.do()
        bot.update()
        if i%100 == 0 :
            bot.trade(p.close_price)
            print(f"open price {p.open_price}")
            print(f"close price {p.close_price}")
            plt.plot(t)
            plt.show()

        bot.clear()
        
        start_date += timedelta(days=1)  # Add 1 day
        end_date += timedelta(days=1)  # Add 1 day
        
        
    
    bot.ob = OrderBook()
    ob = bot.ob
        # BUY Orders (bot can sell to these)
    ob.bids[190].append(Order(Side.BUY, 190, 120))
    ob.bids[191].append(Order(Side.BUY, 191, 150))
    ob.bids[192].append(Order(Side.BUY, 192, 180))
    ob.bids[193].append(Order(Side.BUY, 193, 90))
    ob.bids[194].append(Order(Side.BUY, 194, 100))
    ob.bids[195].append(Order(Side.BUY, 195, 140))
    ob.bids[196].append(Order(Side.BUY, 196, 160))
    ob.bids[197].append(Order(Side.BUY, 197, 130))
    ob.bids[190].append(Order(Side.BUY, 190, 200))
    ob.bids[191].append(Order(Side.BUY, 191, 85))
    ob.bids[192].append(Order(Side.BUY, 192, 175))
    ob.bids[193].append(Order(Side.BUY, 193, 90))
    ob.bids[194].append(Order(Side.BUY, 194, 110))
    ob.bids[195].append(Order(Side.BUY, 195, 95))
    ob.bids[196].append(Order(Side.BUY, 196, 150))

        # SELL Orders (bot can buy from these)
    ob.offers[190].append(Order(Side.SELL, 190, 100))
    ob.offers[191].append(Order(Side.SELL, 191, 160))
    ob.offers[192].append(Order(Side.SELL, 192, 180))
    ob.offers[193].append(Order(Side.SELL, 193, 140))
    ob.offers[194].append(Order(Side.SELL, 194, 90))
    ob.offers[195].append(Order(Side.SELL, 195, 120))
    ob.offers[196].append(Order(Side.SELL, 196, 170))
    ob.offers[197].append(Order(Side.SELL, 197, 130))
    ob.offers[190].append(Order(Side.SELL, 190, 150))
    ob.offers[191].append(Order(Side.SELL, 191, 110))
    ob.offers[192].append(Order(Side.SELL, 192, 145))
    ob.offers[193].append(Order(Side.SELL, 193, 125))
    ob.offers[194].append(Order(Side.SELL, 194, 100))
    ob.offers[195].append(Order(Side.SELL, 195, 190))
    ob.offers[196].append(Order(Side.SELL, 196, 85))


    start_date = datetime.strptime('2021-07-01', '%Y-%m-%d').date()
    end_date = datetime.strptime('2021-07-02', '%Y-%m-%d').date()
        
    p=path(start_date, end_date)
    p.do()
    t=p.get_path()

    bot.trend = t
    print("\nfinal trade")
    bot.trade(p.close_price)
    print(f"open price {p.open_price}")
    print(f"close price {p.close_price}")
    print (bot.Qtable)
    plt.plot(t)
    plt.show()











