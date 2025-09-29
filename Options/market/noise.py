import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import timedelta  ,datetime

def generate_daily_price_path(current_date):
    """
    Generates a simulated intraday price path based on GBM and daily OHLC.
    Returns a NumPy array of price samples.
    """
    # Define the stock ticker and the date for simulation
    ticker = 'RELIANCE.NS'
#     start_date_str =

# # Convert the string to a datetime object
#     
    start_date = current_date
# # Add 50 days to the datetime object, not the string
    end_date = current_date + timedelta(days=1)
    try:
        
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            # This happens on weekends or holidays
            return None
        else:
            ohlc = data.iloc[0]
            open_price = ohlc['Open'].item()
            high_price = ohlc['High'].item()
            low_price = ohlc['Low'].item()
            close_price = ohlc['Close'].item()
    except Exception as e:
        print(f"Error fetching data: {e}. Using default values.")
        open_price, high_price, low_price, close_price = 100, 105, 95, 102

    # defining parameters for the GBM
    mu = 0.06
    sigma = 0.3
    n = 375 # Number of steps in a trading day (e.g., one per minute)
    T = 1
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
    constrained_path[0] = open_price
    constrained_path[-1] = close_price

    return constrained_path
    
# This block will only run when you execute `python noise.py` directly
if __name__ == '__main__':
    # Generate the price path by calling the function
    price_samples = generate_daily_price_path()
    
    print("Generated daily price samples:")
    print(price_samples)
    
    # Plotting the result for visualization
    plt.figure(figsize=(12, 6))
    plt.plot(price_samples, label='Simulated Intraday Path', color='blue')
    plt.title("Simulated Intraday Price Path")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()