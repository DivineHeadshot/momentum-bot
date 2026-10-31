import pandas as pd

# 1. Load the momentum scores we generated in Stage 3
scores = pd.read_csv("strategy/momentum_scores.csv", index_col="Date", parse_dates=True)

# 2. Rank the stocks for each month
# 'axis=1' tells Python to rank across the columns (the stocks) for each row (the date)
# 'ascending=False' means the highest positive return gets rank #1
ranks = scores.rank(axis=1, ascending=False)

# 3. Select the winners
# We want the top performers. Since we only have 10 stocks, we will select the top 2.
top_n = 5

# This creates a table of True/False values. 
# True if the stock's rank is 1 or 2. False if it is 3 through 10.
winners = ranks <= top_n

# 4. Save the candidate holdings for Stage 5
csv_path = "strategy/candidate_holdings.csv"
winners.to_csv(csv_path)

print("Candidate holdings for the last 5 months (True means we buy/hold):")
print(winners.tail())
print(f"\nSuccess! Candidate holdings saved to {csv_path}")