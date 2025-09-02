import numpy as np
from exchange import MarketExchange
from agent import Agent
import random
import numpy as np
from orderbook import OrderBook, Order, Side
from datetime import datetime, timedelta
from noise2 import path
import matplotlib.pyplot as plt
from environment import Env


def SMA_n(trend, time_step, n):
    prices = []
    t=time_step
    if t<n:
        while t>=0:
            prices.append(trend[t])
            t-=1
    else:
        i=0
        while i<n:
            prices.append(trend[time_step-i])
            i+=1

    price_list = np.array(prices)
    sum = np.sum(price_list)
    if time_step<n:
        SMA = sum/ (time_step + 1)
    else:
        SMA = sum/n
    mean = np.mean(price_list)
    sd = np.std(price_list, ddof=1)

    return SMA, sd


def EMA_n(trend, time_step, n):

    if time_step<n:
        sma , sd = SMA_n(trend, time_step, n)
        return sma

    else:
        sma, sd = SMA_n(trend, time_step - n,  n)
        mas = []
        mas.append(sma)
        alpha = 1/(n+1)
        i=0
        while i<n :
            mas.append( trend[time_step - n + i+1]*alpha +  mas[i] * (1 - alpha))
            i+=1
    
    #print (mas)
    return mas[-1]

def MACD(trend, time_step):

    EMA_26 = []
    EMA_12 = []
    for i in range(0,9):
        EMA_26.append(EMA_n(trend, time_step - 8 + i , 26))
        EMA_12.append(EMA_n(trend, time_step - 8 + i , 12))

    print(EMA_12)
    print(EMA_26)
    MACD = np.array(EMA_12) - np.array(EMA_26)
    signal = EMA_n(MACD, len(MACD)-1, 9)

    return MACD, signal

# trend = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 123, 124, 125, 126, 127, 128, 129, 130]
# n = 6
# t = 28

# print("SMA:", SMA_n(trend, t, n)) 
# print("EMA:", EMA_n(trend, t, n))  
# print("MACD and signal : ", MACD(trend, t))


start_date = datetime.strptime('2021-07-01', '%Y-%m-%d').date()
end_date = datetime.strptime('2021-07-02', '%Y-%m-%d').date()

p=path(start_date, end_date)
p.do()
trend=p.get_path()


env = Env()
agent = Agent()
agent.env = env

for i in range(0,len(trend)):
    #print(i, "run")
    #agent.exchange = exchange
    agent.collect()
    if agent.t==2000:
        #agent.sample_batch()
        agent.learn()
    if agent.t > 2000 and agent.t%100 == 0:
        #agent.sample_batch()
        agent.learn()


#print(agent.get_action(98, 99)) 