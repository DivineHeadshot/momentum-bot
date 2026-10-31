import pandas as pd
import yfinance as yf
import os

# 1. Define the 50 sector-diversified tickers
TICKERS = [
    "VZ", "CMCSA", "DIS",              # Communication
    "AMZN", "HD", "MCD", "NKE", "TGT", # Consumer Disc
    "WMT", "KO", "PEP", "PG", "CL", "COST", # Consumer Staples
    "XOM", "CVX", "COP",               # Energy
    "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "SCHW", # Financials
    "JNJ", "UNH", "PFE", "MRK", "ABT", "AMGN", "LLY",    # Healthcare
    "CAT", "DE", "UNP", "MMM", "HON", "LMT",             # Industrials
    "AAPL", "MSFT", "INTC", "CSCO", "IBM", "QCOM", "TXN", "ADBE", # Tech
    "APD", "ECL", "DD",                # Materials
    "O", "SPG",                        # Real Estate
    "NEE", "SO", "DUK"                 # Utilities
]

print(f"Initializing download for {len(TICKERS)} diversified tickers...")

# 2. Fetch 25 years of data (2001 to 2026)
# We pull 'Close' because it accounts for splits and dividend distributions
start_date = "2001-01-01"
end_date = "2026-01-01"

try:
    raw_data = yf.download(TICKERS, start=start_date, end=end_date)
    
    # Extract only the Close prices
    close_prices = raw_data['Close']
    
    # 3. Handle any initial data cleanliness checks
    # Drop columns that have more than 10% missing values (just in case)
    max_missing_pct = 0.10
    allowed_missing = len(close_prices) * max_missing_pct
    cleaned_prices = close_prices.dropna(thresh=len(close_prices) - allowed_missing, axis=1)
    
    # Forward-fill and backward-fill minor holiday/alignment gaps
    cleaned_prices = cleaned_prices.ffill().bfill()
    
    # 4. Save to CSV
    os.makedirs("data", exist_ok=True)
    csv_path = "data/real_prices.csv"
    cleaned_prices.to_csv(csv_path)
    
    print("\n--- DOWNLOAD COMPLETE ---")
    print(f"Dataset shape: {cleaned_prices.shape} (Days x Tickers)")
    print(f"Data ranges from {cleaned_prices.index.min().date()} to {cleaned_prices.index.max().date()}")
    print(f"Successfully saved to {csv_path}")

except Exception as e:
    print(f"An error occurred during data ingestion: {e}")