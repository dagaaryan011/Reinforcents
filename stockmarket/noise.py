import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

def generate_daily_price_path():
    """
    Generates a simulated intraday price path based on GBM and daily OHLC.
    Returns the price path and the OHLC values used for generation.
    """
    # Define the stock ticker and the date for simulation
    ticker = 'RELIANCE.NS'
    start_date = '2025-07-15'
    end_date = '2025-07-16'

    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            print("No data found for the given date, using default values.")
            open_price, high_price, low_price, close_price = 100.0, 105.0, 95.0, 102.0
        else:
            ohlc = data.iloc[0]
            open_price = ohlc['Open'].item()
            high_price = ohlc['High'].item()
            low_price = ohlc['Low'].item()
            close_price = ohlc['Close'].item()
    except Exception as e:
        print(f"Error fetching data: {e}. Using default values.")
        open_price, high_price, low_price, close_price = 100.0, 105.0, 95.0, 102.0

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
    # Avoid division by zero if the path is flat
    if simulated_log_path[-1] == 0:
        scaling_factor = 0
    else:
        scaling_factor = (np.log(close_price) - np.log(open_price)) / simulated_log_path[-1]
    simulated_log_path = simulated_log_path * scaling_factor

    # Convert log path to price path
    simulated_path = open_price * np.exp(simulated_log_path)

    # Apply high and low constraints
    path_high = np.max(simulated_path)
    path_low = np.min(simulated_path)

    # Avoid division by zero if the generated path is flat
    if (path_high - path_low) == 0:
        scale = 1
    else:
        scale = (high_price - low_price) / (path_high - path_low)
    
    shift = low_price - (path_low * scale)

    constrained_path = simulated_path * scale + shift
    constrained_path[0] = open_price
    constrained_path[-1] = close_price

    return constrained_path, open_price, close_price, high_price, low_price

def generate_samples():
    """
    This is the new function for your Streamlit UI.
    It calls the main generator and returns the values in the correct order.
    """
    # The values are returned in the order: path, open, close, high, low
    # This matches the unpacking in your Streamlit code.
    return generate_daily_price_path()

# This block will only run when you execute `python noise.py` directly
if __name__ == '__main__':
    # Generate the data by calling the new function
    samples, open_p, close_p, high_p, low_p = generate_samples()
    
    print("Generated daily price samples:")
    print(samples)
    print(f"\nOpen: {open_p}, High: {high_p}, Low: {low_p}, Close: {close_p}")
    
    # Plotting the result for visualization
    plt.figure(figsize=(12, 6))
    plt.plot(samples, label='Simulated Intraday Path', color='blue')
    plt.axhline(y=high_p, color='g', linestyle='--', label=f'High: {high_p:.2f}')
    plt.axhline(y=low_p, color='r', linestyle='--', label=f'Low: {low_p:.2f}')
    plt.title("Simulated Intraday Price Path")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()