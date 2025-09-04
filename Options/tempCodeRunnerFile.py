import requests
from bs4 import BeautifulSoup
import time

def get_stock_price(ticker='500180'):
    
    # 1. Define the URL
    url = f'https://www.google.com/finance/quote/{ticker}:BOM'
    
    # 2. Make the request to the website
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # 3. Parse the HTML with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 4. Find the HTML element containing the price
        # The class name "YMlKec fxKbKc" is what the video identifies for the price.
        price_element = soup.find(class_='YMlKec fxKbKc')
        
        if price_element:
            # 5. Extract and print the price text
            price = price_element.text
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"--- Data for {ticker}:BOM ---")
            print(f"Price: {price}")
            print(f"Scrape Timestamp: {timestamp}")
        else:
            print("Could not find the price element. The website's HTML structure may have changed.")
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")

# Run the function
get_stock_price()