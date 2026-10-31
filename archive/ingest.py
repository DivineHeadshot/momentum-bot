import yfinance as yf
import pandas as pd
import os

# 1. Make sure we are saving to the right place
# This creates the 'data' folder if it doesn't already exist
os.makedirs("data", exist_ok=True)

# 2. Define our 10 "dummy" stocks
tickers = ["AAPL", "MSFT", "F", "JPM", "JNJ", "WMT", "PG", "XOM", "AMZN", "GOOGL"]

print("Downloading historical data... this might take a few seconds.")

# 3. Download 5 years of daily historical data
data = yf.download(tickers, period="5y")

# 4. Filter for just the 'Close' prices 
# (yfinance now automatically adjusts these for splits/dividends by default)
closing_prices = data['Close']

# 5. Save the data table to a CSV file
csv_path = "data/dummy_prices.csv"
closing_prices.to_csv(csv_path)

print(f"Success! 5 years of data saved to {csv_path}")