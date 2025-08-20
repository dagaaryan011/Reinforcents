import enum
import queue
import time
from collections import defaultdict


class Side(enum.Enum):
    BUY = 0
    SELL = 1


def get_timestamp():
    """ Microsecond timestamp """ 
    return int(1e6 * time.time())


class OrderBook(object):
    def __init__(self):
        self.bid_prices = []
        self.bid_sizes = []
        self.offer_prices = []
        self.offer_sizes = []
        self.bids = defaultdict(list)
        self.offers = defaultdict(list)
        self.unprocessed_orders = queue.Queue()
        self.trades = queue.Queue()
        self.order_id = 0

    def new_order_id(self):
        self.order_id += 1
        return self.order_id

    @property
    def max_bid(self):
        return max(self.bids.keys()) if self.bids else 0.

    @property
    def min_offer(self):
        return min(self.offers.keys()) if self.offers else float('inf')

    def process_order(self, incoming_order):
        incoming_order.timestamp = get_timestamp()
        incoming_order.order_id = self.new_order_id()
        if incoming_order.side == Side.BUY:
            if incoming_order.price >= self.min_offer and self.offers:
                self.process_match(incoming_order)
            else:
                self.bids[incoming_order.price].append(incoming_order)
        else:
            if incoming_order.price <= self.max_bid and self.bids:
                self.process_match(incoming_order)
            else:
                self.offers[incoming_order.price].append(incoming_order)

    def process_match(self, incoming_order):
        levels = self.bids if incoming_order.side == Side.SELL else self.offers
        prices = sorted(levels.keys(), reverse=(incoming_order.side == Side.SELL))

        def price_doesnt_match(book_price):
            if incoming_order.side == Side.BUY:
                return incoming_order.price < book_price
            else:
                return incoming_order.price > book_price

        for price in prices:
            if (incoming_order.size == 0) or (price_doesnt_match(price)):
                break
            orders_at_level = levels[price]
            for book_order in orders_at_level:
                if incoming_order.size == 0:
                    break
                trade, t, s, p = self.execute_match(incoming_order, book_order)
                incoming_order.size = max(0, incoming_order.size - trade.size)
                book_order.size = max(0, book_order.size - trade.size)
                self.trades.put(trade)
            levels[price] = [o for o in orders_at_level if o.size > 0]
            if len(levels[price]) == 0:
                levels.pop(price)
        if incoming_order.size > 0:
            same_side = self.bids if incoming_order.side == Side.BUY else self.offers
            same_side[incoming_order.price].append(incoming_order)

    def execute_match(self, incoming_order, book_order):
        trade_size = min(incoming_order.size, book_order.size)
        
        # Determine who is the buyer and who is the seller
        if incoming_order.side == Side.BUY:
            
            o_t = "BUY"
            action_str = f"Done {o_t} {trade_size} units at price {book_order.price}"
            incoming_order.tradeprices.append(book_order.price)
            incoming_order.tradesizes.append(trade_size)
            print (action_str)
        else:
            
            o_t = "SELL"
            action_str = f"Done {o_t} {trade_size} units at price {book_order.price}"
            incoming_order.tradeprices.append(book_order.price)
            incoming_order.tradesizes.append(trade_size)
            print (action_str)
        
        return Trade(incoming_order.side, book_order.price, trade_size,
                    incoming_order.order_id, book_order.order_id), o_t, trade_size, book_order.price


    def book_summary(self):
        self.bid_prices = sorted(self.bids.keys(), reverse=True)
        self.offer_prices = sorted(self.offers.keys())
        self.bid_sizes = [sum(o.size for o in self.bids[p]) for p in self.bid_prices]
        self.offer_sizes = [sum(o.size for o in self.offers[p]) for p in self.offer_prices]

    def show_book(self):
        self.book_summary()
        print("\n===== ORDER BOOK =====")
        print('Sell side:')
        if len(self.offer_prices) == 0:
            print('EMPTY')
        else:
            for i, price in reversed(list(enumerate(self.offer_prices))):
                print(f"{i+1}) Price={self.offer_prices[i]}, Total units={self.offer_sizes[i]}")

        print('Buy side:')
        if len(self.bid_prices) == 0:
            print('EMPTY')
        else:
            for i, price in enumerate(self.bid_prices):
                print(f"{i+1}) Price={self.bid_prices[i]}, Total units={self.bid_sizes[i]}")
        print("======================\n")


class Order(object):
    def __init__(self, side, price, size, timestamp=None, order_id=None):
        self.side = side
        self.price = price
        self.size = size
        self.timestamp = timestamp
        self.order_id = order_id
        self.tradeprices=[]
        self.tradesizes=[]

    def __repr__(self):
        return f"{self.side.name} {self.size} units at {self.price}"


class Trade(object):
    def __init__(self, incoming_side, incoming_price, trade_size, incoming_order_id, book_order_id):
        self.side = incoming_side
        self.price = incoming_price
        self.size = trade_size
        self.incoming_order_id = incoming_order_id
        self.book_order_id = book_order_id

    def __repr__(self):
        return f"Executed: {self.side.name} {self.size} units at {self.price}"


# if __name__ == '__main__':
#     ob = OrderBook()
#     while True:
#         print("Menu:")
#         print("1. Place BUY order")
#         print("2. Place SELL order")
#         print("3. Show Order Book")
#         print("4. Exit")
#         choice = input("Enter choice: ")

#         if choice == "1":
#             price = float(input("Enter BUY price: "))
#             size = int(input("Enter quantity: "))
#             order = Order(Side.BUY, price, size)
#             ob.process_order(order)
#         elif choice == "2":
#             price = float(input("Enter SELL price: "))
#             size = int(input("Enter quantity: "))
#             order = Order(Side.SELL, price, size)
#             ob.process_order(order)
#         elif choice == "3":
#             ob.show_book()
#         elif choice == "4":
#             print("Exiting...")
#             break
#         else:
#             print("Invalid choice, try again.")