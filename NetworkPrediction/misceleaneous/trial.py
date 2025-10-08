import os
import pandas as pd
from datetime import date, timedelta
from tqdm import tqdm
from src.market.noise import generate_price_samples # Make sure noise.py is in the same folder

def create_test_csv(filename="test_price_data.csv", days_to_fetch=60):
    """
    Fetches a small amount of price data and saves it to a CSV file for testing.
    """
    # 1. Check if the file already exists to avoid re-downloading
    if os.path.exists(filename):
        print(f"'{filename}' already exists. No new data fetched.")
        return

    print(f"Creating a new test dataset with {days_to_fetch} days of data...")
    
    # 2. Set up a short date range for fetching
    end_date = date.today()
    start_date = end_date - timedelta(days=days_to_fetch)
    current_date = start_date
    
    test_price_path = []
    total_days = (end_date - start_date).days

    # 3. Fetch the data using the tqdm loop
    with tqdm(total=total_days, desc="Fetching Test Data") as pbar:
        while current_date < end_date:
            daily_path = generate_price_samples(current_date)
            
            if daily_path is not None:
                test_price_path.extend(daily_path.tolist())
            
            current_date += timedelta(days=1)
            pbar.update(1)
            
    print("\nTest data fetching complete.")

    # 4. Save the fetched data to the specified CSV file
    print(f"Saving data to '{filename}'...")
    df_to_save = pd.DataFrame(test_price_path, columns=['price'])
    df_to_save.to_csv(filename, index=False)
    print("Test data saved successfully.")


if __name__ == '__main__':
    # You can change the filename or number of days here if you want
    create_test_csv(filename="test_price_data.csv", days_to_fetch=365)