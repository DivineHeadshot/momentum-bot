import pandas as pd
import os

print("Generating momentum signals and candidate holdings...")

# 1. Load the real 50-stock data
prices = pd.read_csv("data/real_prices.csv", index_col="Date", parse_dates=True)
monthly_prices = prices.resample('ME').last()

# 2. Calculate Momentum (12-month lookback return)
momentum_scores = monthly_prices / monthly_prices.shift(12) - 1.0

# 3. Select the Top Decile candidates each month
# Initialize a blank dataframe of False values with the exact same shape
candidates = pd.DataFrame(False, index=momentum_scores.index, columns=momentum_scores.columns)

for date in momentum_scores.index:
    row = momentum_scores.loc[date].dropna()
    
    # We need at least enough stocks to form a valid decile
    if len(row) < 10:
        continue
        
    # Dynamically calculate the top 10% of the available universe
    decile_count = max(1, int(len(row) * 0.10))
    top_decile = row.nlargest(decile_count).index
    
    candidates.loc[date, top_decile] = True
# 4. Save the outputs to overwrite the old dummy files
os.makedirs("strategy", exist_ok=True)
momentum_scores.to_csv("strategy/momentum_scores.csv")
candidates.to_csv("strategy/candidate_holdings.csv")

print(f"Success! Generated signals for {len(prices.columns)} tickers.")