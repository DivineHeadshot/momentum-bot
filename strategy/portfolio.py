import pandas as pd

# 1. Load the candidate holdings (the True/False matrix from Stage 4)
holdings = pd.read_csv("strategy/filtered_holdings.csv", index_col="Date", parse_dates=True)

# 2. Convert True/False into mathematical numbers (True = 1.0, False = 0.0)
weights = holdings.astype(float)

# 3. Calculate equal weights
# First, count how many stocks we are buying each month by summing the row
# Since we chose top_n=2, this will equal 2
num_stocks = weights.sum(axis=1)

# Divide each 1.0 by the total number of stocks (1.0 / 2 = 0.5)
# .div(axis=0) ensures we divide row-by-row
target_weights = weights.div(num_stocks, axis=0)

# If there were any months where we bought 0 stocks (like the first year of NaNs), 
# the math does 0 / 0, which creates an error. We fill those errors with 0.0.
target_weights = target_weights.fillna(0.0)

# 4. Save the target weights for the Backtest Engine
csv_path = "strategy/target_weights.csv"
target_weights.to_csv(csv_path)

print("Target weights for the last 5 months (0.5 = 50% allocation):")
print(target_weights.tail())
print(f"\nSuccess! Target weights saved to {csv_path}")