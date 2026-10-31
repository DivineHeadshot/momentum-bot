import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import os
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# 1. Load the pricing data
prices = pd.read_csv("data/real_prices.csv", index_col="Date", parse_dates=True)
market_returns = prices.pct_change().mean(axis=1).dropna()

features = pd.DataFrame({
    'return': market_returns,
    'volatility': market_returns.rolling(window=20).std()
}).dropna()

# 2. Setup the Walk-Forward Parameters
# 252 trading days = 1 year. We use a 3-year "burn-in" period to initially train the model.
min_train_days = 252 * 3 
step_size = 20 # Refit the model every 20 trading days (roughly one month) to optimize compute time

# Create a blank series to hold our mathematically honest predictions
out_of_sample_scalars = pd.Series(index=features.index, dtype=float)

print("Running forward-only expanding window HMM...")

# 3. The Walk-Forward Loop
# Step through time, training only on the past, to predict the next block of unseen days.
for i in tqdm(range(min_train_days, len(features), step_size)):
    
    # PAST DATA: Strictly the days before our current step
    train_features = features.iloc[:i]
    
    # FUTURE DATA: The next 20 days we want to predict
    end_idx = min(i + step_size, len(features))
    test_features = features.iloc[i:end_idx]
    
    # Initialize a fresh model so it doesn't remember previous loops
    hmm_model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
    
    try:
        # Fit strictly on the past
        hmm_model.fit(train_features)
        
        # Predict the past regimes to map which one is Bull, Bear, and Neutral
        train_regimes = hmm_model.predict(train_features)
        
        train_features_copy = train_features.copy()
        train_features_copy['Regime'] = train_regimes
        state_means = train_features_copy.groupby('Regime')['return'].mean().sort_values()
        
        bear_state = state_means.index[0]
        neutral_state = state_means.index[1]
        bull_state = state_means.index[2]
        
        # Now, predict the regimes for the UNSEEN future data
        test_regimes = hmm_model.predict(test_features)
        
        # Translate those unseen regimes into exposure scalars
        for j, current_date in enumerate(test_features.index):
            current_regime = test_regimes[j]
            if current_regime == bear_state:
                out_of_sample_scalars.loc[current_date] = 0.5
            elif current_regime == neutral_state:
                out_of_sample_scalars.loc[current_date] = 0.75
            else:
                out_of_sample_scalars.loc[current_date] = 1.0
                
    except Exception as e:
        # HMMs can occasionally fail to converge on highly volatile data subsets. 
        # If the math fails, default to a conservative 0.75 exposure.
        out_of_sample_scalars.iloc[i:end_idx] = 0.75

# 4. Clean up and format
# Drop the initial 3-year burn-in period since we couldn't make predictions yet
out_of_sample_scalars = out_of_sample_scalars.dropna()

# Align to end-of-month for the portfolio backtester
monthly_scalars = out_of_sample_scalars.resample('ME').last()
monthly_scalars.name = 'Scalar'

# 5. Save the output
os.makedirs("strategy", exist_ok=True)
csv_path = "strategy/regime_scalars.csv"
monthly_scalars.to_csv(csv_path)

print(f"\nSuccess! Out-of-sample regime scalars saved to {csv_path}")