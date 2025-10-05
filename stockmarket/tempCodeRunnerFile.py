import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def generate_samples():
    """
    Generates a simulated intraday stock price path. The base is a straight line
    from the open price to the close price, with noise added using GBM.

    Returns:
        tuple: A tuple containing:
            - samples (list): The list of simulated stock prices.
            - open_price (float): The actual opening price.
            - close_price (float): The actual closing price.
            - high_price (float): The actual high price.
            - low_price (float): The actual low price.
    """
    # Define the stock ticker and the date for simulation
    ticker = 'RELIANCE.NS'
    
    # Fetch data for the last 7 days to ensure we find a valid trading day
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    ohlc = None
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # Iterate backwards through the data to find the first day with price movement
        for _, row in data.iloc[::-1].iterrows():
            if row['High'] != row['Low']:
                ohlc = row
                break
        
        if ohlc is None:
            # If no day with price movement is found, use fallback data
            raise ValueError("No valid trading day found in the last 7 days.")
            
        # Explicitly convert the pandas Series values to floats to avoid conflicts
        open_price = float(ohlc['Open'])
        high_price = float(ohlc['High'])
        low_price = float(ohlc['Low'])
        close_price = float(ohlc['Close'])
        print(open_price)
        print(close_price)
        print(high_price)
        print(low_price)
    except Exception as e:
        print(f"Error fetching data: {e}. Using hardcoded values.")
        return [100.0] * 600, 100.0, 100.0, 101.0, 99.0

    # Defining parameters for the GBM noise
    mu = 0.05
    sigma = 0.3
    n = 600  # Number of steps
    T = 6    # Trading hours in a day (e.g., 6 hours)
    dt = T / n

    # Step 1: Create a straight-line base path from Open to Close
    base_path = np.linspace(open_price, close_price, n)

    # Step 2: Generate GBM-like noise
    dW = np.random.normal(0, np.sqrt(dt), size=n)
    noise = sigma * dW

    # Apply noise to the base path
    # We use a cumulative sum to make the noise a "random walk"
    noisy_path = base_path + (noise.cumsum() - noise.cumsum()[0])

    # Apply High and Low constraints to the noisy path
    path_high = np.max(noisy_path)
    path_low = np.min(noisy_path)
    
    # Use a small epsilon to avoid division by zero if path_high equals path_low
    if abs(path_high - path_low) < 1e-9:
        constrained_path = noisy_path
    else:
        scale = (high_price - low_price) / (path_high - path_low)
        shift = low_price - (path_low * scale)
        constrained_path = noisy_path * scale + shift
    
    # Manually set start and end points to ensure accuracy
    constrained_path[0] = open_price
    constrained_path[-1] = close_price

    return constrained_path.tolist(), open_price, close_price, high_price, low_price


if __name__ == "__main__":
    # Call the function and unpack the results
    samples, open_price, close_price, high_price, low_price = generate_samples()

    # Plot the simulated path
    plt.figure(figsize=(12, 6))
    plt.plot(samples, label='Simulated Intraday Path', color='blue')
    plt.axhline(y=open_price, color='gray', linestyle='--', label=f'Open: ₹{open_price:.2f}')
    plt.axhline(y=close_price, color='purple', linestyle='--', label=f'Close: ₹{close_price:.2f}')
    plt.axhline(y=high_price, color='green', linestyle=':', label=f'High: ₹{high_price:.2f}')
    plt.axhline(y=low_price, color='red', linestyle=':', label=f'Low: ₹{low_price:.2f}')

    plt.title("Simulated Intraday Stock Price Path")
    plt.xlabel("Time Step (e.g., minutes)")
    plt.ylabel("Price (₹)")
    plt.legend()
    plt.grid(True)
    plt.show()