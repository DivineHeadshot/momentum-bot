import pandas as pd
import numpy as np
import xgboost as xgb
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

print("Building walk-forward Meta-Label Filter with Volatility...")

# Load stock data
prices = pd.read_csv("data/real_prices.csv", index_col="Date", parse_dates=True)
candidates = pd.read_csv("strategy/candidate_holdings.csv", index_col="Date", parse_dates=True)
momentum = pd.read_csv("strategy/momentum_scores.csv", index_col="Date", parse_dates=True)

# Building features and labels
monthly_prices = prices.resample('ME').last()
fwd_returns = monthly_prices.shift(-1) / monthly_prices - 1.0
labels = (fwd_returns > 0).astype(int)

# Feature 1: Distance from 3-month MA (Lagging / Trend)
ma_3 = monthly_prices.rolling(3).mean()
dist_from_ma = (monthly_prices - ma_3) / ma_3

# Feature 2: 20-Day Annualized Volatility (Leading / Fast-Twitch)
daily_returns = prices.pct_change()
daily_vol = daily_returns.rolling(window=20).std() * np.sqrt(252)
monthly_vol = daily_vol.resample('ME').last()

# Gather historical data
train_data = []
for date in candidates.index:
    for ticker in candidates.columns:
        if candidates.loc[date, ticker] == True:
            mom_score = momentum.loc[date, ticker]
            ma_dist = dist_from_ma.loc[date, ticker]
            vol_score = monthly_vol.loc[date, ticker]
            trade_label = labels.loc[date, ticker]

            if pd.isna(mom_score) or pd.isna(ma_dist) or pd.isna(vol_score) or pd.isna(trade_label):
                continue

            train_data.append({
                'Date': date,
                'Ticker': ticker,
                'Momentum': mom_score,
                'MA_Dist': ma_dist,
                'Volatility': vol_score,
                'Label': trade_label
            })

train_df = pd.DataFrame(train_data)

# Out-Of-Sample Walk-Forward Loop
oos_start = pd.to_datetime("2021-01-01")
filtered_candidates = candidates.copy()
oos_dates = [d for d in candidates.index if d >= oos_start]

print(f"Running XGBoost strictly on out-of-sample dates from {oos_start.date()}...")

for current_date in tqdm(oos_dates):
    historical_trades = train_df[train_df['Date'] < current_date]
    
    if len(historical_trades) < 50: 
        continue
        
    X_train = historical_trades[['Momentum', 'MA_Dist', 'Volatility']]
    y_train = historical_trades['Label']
    
    # Train model with Volatility included
    model = xgb.XGBClassifier(n_estimators=50, max_depth=2, random_state=42)
    model.fit(X_train, y_train)
    
    for ticker in candidates.columns:
        if candidates.loc[current_date, ticker] == True:
            mom_score = momentum.loc[current_date, ticker]
            ma_dist = dist_from_ma.loc[current_date, ticker]
            vol_score = monthly_vol.loc[current_date, ticker]
            
            if pd.isna(mom_score) or pd.isna(ma_dist) or pd.isna(vol_score):
                filtered_candidates.loc[current_date, ticker] = False
                continue
            
            # Predict using all three features
            features = pd.DataFrame({'Momentum': [mom_score], 'MA_Dist': [ma_dist], 'Volatility': [vol_score]})
            prob_win = model.predict_proba(features)[0][1]
            
            # The Confidence Gate
            if prob_win < 0.50:
                filtered_candidates.loc[current_date, ticker] = False

# Save candidates to csv
filtered_candidates.loc[filtered_candidates.index < oos_start] = False

os.makedirs("strategy", exist_ok=True)
csv_path = "strategy/filtered_holdings.csv"
filtered_candidates.to_csv(csv_path)

print(f"\nSuccess! Walk-forward OOS filtered candidates saved to {csv_path}")