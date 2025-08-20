import numpy as np
from orderbook import OrderBook, Order, Side
from noise import samples as trend


class BotOrderBook(object):
    def __init__(self):

        #stocks bought by bot earlier  (will sell them now)
        self.bid_prices = [80, 120, 200, 260, 300, 310]    
        self.bid_sizes  = [30, 25, 15, 40, 20, 10]

        #stocks sold by bot earlier   (will buy them now)
        self.offer_prices = [150, 180, 100, 60, 70, 50]
        self.offer_sizes  = [10, 15, 20, 25, 30, 12]
        

class Bot:
    def __init__(self):
        self.book = pob
        self.obook = ob
        self.t = 0

    def see(self):
        diff = []
        for i in range(0,15):
            diff.append(trend[self.t+i+1] - trend[self.t+i])
        avg = sum(diff)
        threshold = 2

        if avg > threshold:
            self.buy()
        elif avg < -threshold:
            self.sell()
        else:
            self.side = "NONE"


    def sell(self):
        if not pob.bid_prices:
            self.side = "NONE"
            return
        i = np.argmin(pob.bid_prices)
        self.side = Side.SELL
        self.price = pob.bid_prices[i]
        self.size = pob.bid_sizes[i]
        pob.bid_prices.pop(i)
        pob.bid_sizes.pop(i)

    def buy(self):
        if not pob.offer_prices:
            self.side = "NONE"
            return
        i = np.argmax(pob.offer_prices)
        self.side = Side.BUY
        self.price = pob.offer_prices[i]
        self.size = pob.offer_sizes[i]
        pob.offer_prices.pop(i)
        pob.offer_sizes.pop(i)


    def execute(self):
        if self.side == "NONE":
            return
        order= Order(self.side, self.price, self.size)
        ob.process_order(order)
        trade_prices = np.array(order.tradeprices)
        trade_sizes = np.array(order.tradesizes)
        price = trade_prices * trade_sizes
        final_price = sum (price)
        original = self.price * self.size

        if self.side == Side.BUY:
            self.change = original - final_price
        
        elif self.side == Side.SELL:
            self.change = final_price - original

        print (f"You earned {self.change}\n")

ob = OrderBook()    #main order book
def preload_mixed_orders(ob):
    # BUY orders (market wants to buy) → bot may SELL to these
    ob.bids[90].append(Order(Side.BUY, 90, 20))     # Profitable for bot (bought at 80)
    ob.bids[115].append(Order(Side.BUY, 115, 10))   # Profitable (bought at 80, 120)
    ob.bids[250].append(Order(Side.BUY, 250, 30))   # Profitable (bought at 200)
    ob.bids[220].append(Order(Side.BUY, 220, 10))   # Loss (bot bought at 260+)
    ob.bids[305].append(Order(Side.BUY, 305, 10))   # Slight loss or break-even (bought at 310)
    ob.bids[150].append(Order(Side.BUY, 150, 50))   # Partial match for mid-priced assets

    # SELL orders (market wants to sell) → bot may BUY from these
    ob.offers[70].append(Order(Side.SELL, 70, 15))    # Buy back at profit (sold at 100, 150)
    ob.offers[55].append(Order(Side.SELL, 55, 20))    # Loss (sold at 50)
    ob.offers[85].append(Order(Side.SELL, 85, 25))    # Profit (sold at 150)
    ob.offers[175].append(Order(Side.SELL, 175, 15))  # Buy high again → loss (sold at 180)
    ob.offers[65].append(Order(Side.SELL, 65, 10))    # Mix
    ob.offers[200].append(Order(Side.SELL, 200, 5))   # Expensive — probably won't match

    # Extra book depth
    ob.bids[60].append(Order(Side.BUY, 60, 50))     
    ob.bids[130].append(Order(Side.BUY, 130, 10))    
    ob.offers[300].append(Order(Side.SELL, 300, 40))
    ob.offers[320].append(Order(Side.SELL, 320, 20))


pob = BotOrderBook()   #personal order book (just lists of previous)

preload_mixed_orders(ob)

bot = Bot()

print("before")
ob.show_book()

for x in range(30, 100):
    bot.t = x
    bot.see()
    bot.execute()
    if x % 100 == 0:
        ob.show_book()

print("after")
ob.show_book()
