import pandas as pd
import vectorbt as vbt
import os

# 1. Load prices, target weights, and regime scalars
prices = pd.read_csv("data/real_prices.csv", index_col="Date", parse_dates=True)
weights = pd.read_csv("strategy/target_weights.csv", index_col="Date", parse_dates=True)
scalars_df = pd.read_csv("strategy/regime_scalars.csv", index_col="Date", parse_dates=True)
scalars = scalars_df['Scalar'] 

# 2. Align the timeframes
monthly_prices = prices.resample('ME').last()
weights, monthly_prices = weights.align(monthly_prices, join='inner', axis=0)
scalars, _ = scalars.align(weights, join='inner', axis=0)

# --- THE OOS SLICE ---
# Force the backtester to only simulate the 5-year holdout period
oos_start = "2021-01-01"
monthly_prices = monthly_prices.loc[oos_start:]
weights = weights.loc[oos_start:]
scalars = scalars.loc[oos_start:]
# ---------------------

# 3. Apply the Machine Learning Overlay
scaled_weights = weights.multiply(scalars, axis=0)

print(f"Running historical simulation strictly on OOS period starting {oos_start}...")

# 4. Initialize the Vectorbt Portfolio
portfolio = vbt.Portfolio.from_orders(
    monthly_prices,
    scaled_weights,
    size_type='targetpercent',
    init_cash=100000,
    fees=0.001,
    freq='30d',
    cash_sharing=True,
    group_by=True,
    size_granularity=1.0  
)

# 5. Extract and Print the Tear Sheets
print("\n" + "="*40)
print("🏆 STRATEGY PERFORMANCE METRICS 🏆")
print("="*40)
# portfolio.stats() generates a comprehensive list of all major quant metrics
print(portfolio.stats())


print("\n" + "="*40)
print("📈 BENCHMARK PERFORMANCE METRICS 📈")
print("="*40)

# 1. Create a matrix where every stock gets an equal slice of the portfolio
# For 50 stocks, this commands the engine to hold exactly 2% of each stock every month
equal_weights = pd.DataFrame(
    1.0 / len(monthly_prices.columns), 
    index=monthly_prices.index, 
    columns=monthly_prices.columns
)

# 2. Run the benchmark through the exact same execution engine as your strategy
# This completely bypasses the pandas 'ME' bug because we hardcode freq='30d'
benchmark_portfolio = vbt.Portfolio.from_orders(
    monthly_prices,
    equal_weights,
    size_type='targetpercent',
    init_cash=100000,
    fees=0.001,
    freq='30d',
    cash_sharing=True,
    group_by=True
)

# 3. Print the clean stats
print(benchmark_portfolio.stats())

# 6. Render the interactive HTML graph
print("\nGenerating simulated equity curve and drawdowns...")

# Explicitly request the Cumulative Returns and the Underwater (Drawdown) chart
fig_portfolio = portfolio.plot(subplots=[
    'cum_returns',
    'underwater'  
])
fig_portfolio.show()

# 7. View the Actual Trade Executions (Trade Blotter)
# Instead of trying to plot a fused column, we pull the exact order records
print("\n--- RECENT TRADE EXECUTIONS ---")
print(portfolio.orders.records_readable.tail(15))