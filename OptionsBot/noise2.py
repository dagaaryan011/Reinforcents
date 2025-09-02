import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Define the stock ticker and the date for simulation
class path():
    def __init__(self, s, e):
        self.ticker = 'RELIANCE.NS'
        self.start_date = s
        self.end_date = e


    # Fetch daily OHLC data from yfinance
    def do(self):
        try:
            data = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False, auto_adjust=False)
            if data.empty:
                print("No data found")
                exit()
            ohlc = data.iloc[0]
            self.open_price = ohlc['Open'].item()
            high_price = ohlc['High'].item()
            low_price = ohlc['Low'].item()
            self.close_price = ohlc['Close'].item()
        except Exception as e:
            print(f"Error fetching data: {e}")
            exit()

        # Defining parameters for the GBM
        mu = 0.06
        sigma = 0.3
        S0 = self.open_price # Start from the open price
        n = 21600
        T = 6
        M = 1

        dt = T / n

        # Generate a GBM-like random walk
        dW = np.random.normal(0, np.sqrt(dt), size=n)
        simulated_log_returns = (mu - 0.5 * sigma**2) * dt + sigma * dW
        simulated_log_path = simulated_log_returns.cumsum()

        # Adjust the simulated path to fit the OHLC range
        # The logic here is key to a smooth fit

        # 1. Scale and shift the log path to start at 0 and end at the log return from open to close
        log_open_to_close = np.log(self.close_price / self.open_price)
        simulated_log_path_scaled = simulated_log_path * (log_open_to_close / simulated_log_path[-1])

        # 2. Convert the log path to a price path starting at the open price
        simulated_price_path = self.open_price * np.exp(simulated_log_path_scaled)

        # 3. Scale and shift the price path to fit the high and low constraints
        sim_high = np.max(simulated_price_path)
        sim_low = np.min(simulated_price_path)

        # Handle cases where path is flat to avoid division by zero
        if sim_high == sim_low:
            self.constrained_path = np.full_like(simulated_price_path, self.open_price)
        else:
            scale = (high_price - low_price) / (sim_high - sim_low)
            shift = low_price - (sim_low * scale)
            self.constrained_path = simulated_price_path * scale + shift

        # The path should naturally start and end at the desired prices with this method.
        #print(self.open_price, self.close_price, high_price, low_price)

        # Plotting the result
        time = np.linspace(0, T, n)

        # plt.figure(figsize=(12, 6))
        # plt.plot(time, self.constrained_path, label='Simulated Intraday Path (GBM-based)', color='blue')
        # plt.axhline(y=high_price, color='gray', linestyle=':', label=f'High: ₹{high_price:.2f}')
        # plt.axhline(y=low_price, color='gray', linestyle=':', label=f'Low: ₹{low_price:.2f}')

        # plt.title(f"Simulated Constrained Intraday Path for {self.ticker} on {self.start_date}")
        # plt.xlabel("Time (hours)")
        # plt.ylabel("Price")
        # plt.legend()
        # plt.grid(True)
        # plt.show()

    def get_path(self):
        return  self.constrained_path