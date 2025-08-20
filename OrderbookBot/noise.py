# import numpy as np
# import matplotlib.pyplot as plt
import yfinance as yf

# def jittery_noise(n_points=20, low=10, high=2000, step_size=0.0000000000002,start = 0, end = 2000):
    
#     values = [start]  # start randomly within range
#     peak = np.random.randint(1,3)

#     for _ in range(1, n_points):

#         step = np.random.uniform(-step_size, step_size)
        
#         new_val = values[-1] +step
        
#         # clamp to bounds
#         new_val = max(low, min(high, new_val))
#         # if values[-1] == high :


#         values.append(new_val)
    
#     return np.array(values)pip in



ticker = "RELIANCE.BO"
stock = yf.Ticker(ticker)
data = stock.history(start="2008-08-25", end="2008-08-26")  
open_price = data["Open"].iloc[0]
close_price = data["Close"].iloc[0]
high_price = data["High"].iloc[0]
low_price = data["Low"].iloc[0]
# low = low_price
# high = high_price
# data = jittery_noise(6000, low, high, step_size=0.2 , start = open_price , end = close_price)

# plt.plot(data)

# # 






# # Fetch data for 25 Aug 2008





print(f"Stock: {ticker} (25 Aug 2008)")
print(f"Open: {open_price}")
print(f"Close: {close_price}")
print(f"High: {high_price}")
print(f"Low: {low_price}")

# plt.show()
import numpy
import matplotlib.pyplot as plt
import random as r
import math

mean = 0
std = 0.05
num_samples = 599
samples = numpy.random.normal(mean, std, size=num_samples)
samples = [r.uniform(open_price, close_price) +( math.sin(i*math.e))  for i in range(num_samples)]
samples[0] = open_price
samples[-1] =(close_price)
# plt.plot(samples)
# plt.axhline(low_price, color="red", linestyle="--")
# plt.axhline(high_price, color="green", linestyle="--")
# plt.axhline(open_price, color="black", linestyle="--", linewidth=0.5)

# plt.ylim(180,198)
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
import random as r
import math




























# # Example data
# num_samples = 100
# open_price = 50
# close_price = 60
# strike_price = 55

# samples = [r.uniform(open_price, close_price) + math.sin(i * math.e) for i in range(num_samples)]
# x = np.arange(num_samples)
# y = np.array(samples)

# # Plot where y > strike_price (green)
# plt.plot(x[y > strike_price], y[y > strike_price], color='green', linewidth=2, label='Above strike')

# # Plot where y <= strike_price (red)
# plt.plot(x[y <= strike_price], y[y <= strike_price], color='red', linewidth=2, label='Below strike')

# plt.axhline(strike_price, color='gray', linestyle='--', label='Strike price')  # optional reference line
# plt.legend()
# plt.show()