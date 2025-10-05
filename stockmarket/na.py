import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Define the stock ticker and the date for simulation
ticker = 'RELIANCE.NS'
start_date = '2025-07-15'
end_date = '2025-07-16'

# Fetch daily OHLC data from yfinance
try:
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty:
        print("No data found")
        exit()
    ohlc = data.iloc[0]
    #y finance returns a Pandas Series, so we need to extract the values
    open_price = ohlc['Open'].item()
    high_price = ohlc['High'].item()
    low_price = ohlc['Low'].item()
    close_price = ohlc['Close'].item()
except Exception as e:
    print(f"Error fetching data: {e}")
    exit()


#defining parameters for the GBM

# drift coefficient
mu = 0.05
# volatility coefficient
sigma = 0.3
# initial value of the stock
S0 = 100
# no. of steps
n = 21600
# time period
T = 6
# no. of sims
M = 1

dt = T / n

# Generate a GBM-like random walk
dW = np.random.normal(0, np.sqrt(dt), size=n)
log_returns = (mu - 0.5 * sigma**2) * dt + sigma * dW
simulated_log_path = log_returns.cumsum()

# Rescale the path to start at Open and end at Close
simulated_log_path -= simulated_log_path[0]
scaling_factor = (np.log(close_price) - np.log(open_price)) / simulated_log_path[-1]
simulated_log_path = simulated_log_path * scaling_factor

# Convert log path to price path
simulated_path = open_price * np.exp(simulated_log_path)

# Apply high and low constraints
path_high = np.max(simulated_path)
path_low = np.min(simulated_path)

scale = (high_price - low_price) / (path_high - path_low)
shift = low_price - (path_low * scale)

constrained_path = simulated_path * scale + shift

# Manually set start and end points to ensure accuracy
constrained_path[0] = open_price
constrained_path[-1] = close_price

print(open_price, close_price, high_price, low_price)

# Plotting the result
time = np.linspace(0, T, n)

plt.figure(figsize=(12, 6))
plt.plot(time, constrained_path, label='Simulated Intraday Path (GBM-based)', color='blue')
plt.axhline(y=high_price, color='gray', linestyle=':', label=f'High: ₹{high_price:.2f}')
plt.axhline(y=low_price, color='gray', linestyle=':', label=f'Low: ₹{low_price:.2f}')

plt.title(f"Simulated Constrained Intraday Path for {ticker} on {start_date}")
plt.xlabel("Time (hours)")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

def givedata():
    return constrained_path.tolist(), open_price, close_price, high_price, low_price