"""
run_ablations.py -- single source of truth for the momentum pipeline.

WHY THIS EXISTS
---------------
1. Run-order safety. signals.py / regime.py / meta_label.py each write CSVs that
   later stages read. If you run them out of order, results silently change with
   no error. This script runs those config-independent stages ONCE, in the correct
   order, so every ablation below is built from the same canonical outputs.

2. Ablations. On their own, a good backtest number is meaningless -- you can't tell
   whether it came from momentum, from the regime overlay parking you in cash during
   a drawdown, or from the meta-label filter. This script builds all four configs
   from the same ingredients and compares them against the benchmark, so you can say
   which component actually earns its keep.

ABLATIONS (each row adds one component to the row above it)
    1. Momentum only        top-decile momentum, equal weight
    2. + Regime             scale total exposure by the HMM regime scalar
    3. + Meta-label         keep only XGBoost high-confidence names
    4. Full stack           both overlays

USAGE
    Place at the project root (same folder as download_data.py), then:
        python run_ablations.py                 # uses existing data if present
        python run_ablations.py --refresh-data  # re-download prices first
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

# vectorbt is only needed to actually RUN the backtest. Guard the import so the
# pure helper functions below stay unit-testable in an environment without it.
try:
    import vectorbt as vbt
except ImportError:
    vbt = None

# ---- project layout -------------------------------------------------------
# This file lives at the project root. Anchor to its folder so the pipeline
# works no matter which directory you launch from. Your stage scripts write
# paths relative to the root (e.g. "data/real_prices.csv"), so everything must
# run with the root as the working directory -- which is what this guarantees.
ROOT = os.path.dirname(os.path.abspath(__file__))

# Where each stage script lives, relative to the root. Adjust if your layout
# differs. download_data.py is at the root; the rest are in strategy/.
DATA_SCRIPT      = "download_data.py"
SIGNALS_SCRIPT   = os.path.join("strategy", "signals.py")
REGIME_SCRIPT    = os.path.join("strategy", "regime.py")
METALABEL_SCRIPT = os.path.join("strategy", "meta_label.py")

# ---- shared backtest settings (match backtest.py) -------------------------
OOS_START = "2021-01-01"
INIT_CASH = 100_000
FEES = 0.001
FREQ = "30d"          # matches backtest.py; sidesteps a vectorbt 'ME' inference issue
ANN = 12              # ~monthly rebalance, for annualizing Sharpe
RF = 0.0              # risk-free rate assumed 0 for the comparison


# ===========================================================================
# Stage 1 -- run the config-independent stages IN ORDER
# ===========================================================================
def run_stage(name, script):
    print(f"\n=== {name}  ({script}) ===")
    if not os.path.exists(script):
        sys.exit(f"Missing {script}. Run this from the project root.")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        sys.exit(f"\nPipeline FAILED at {script} -- fix it before comparing configs.")


def build_ingredients(refresh_data):
    if refresh_data or not os.path.exists("data/real_prices.csv"):
        run_stage("Data ingestion", DATA_SCRIPT)
    else:
        print("Using existing data/real_prices.csv  (pass --refresh-data to re-download)")
    run_stage("Signals + candidates", SIGNALS_SCRIPT)
    run_stage("Regime HMM", REGIME_SCRIPT)
    run_stage("Meta-label filter", METALABEL_SCRIPT)


# ===========================================================================
# Stage 2 -- pure helpers (unit-tested; no vectorbt needed)
# ===========================================================================
def equal_weight(holdings):
    """True/False holdings -> equal target weights per row (rows of all-False -> 0)."""
    w = holdings.astype(float)
    n = w.sum(axis=1)
    return w.div(n, axis=0).fillna(0.0)


def one_way_turnover(weights):
    """Mean one-way turnover across rebalances, as a percent.

    one-way turnover_t = 0.5 * sum_i |w_t,i - w_{t-1},i|
    """
    delta = weights.diff().abs().sum(axis=1) * 0.5
    return float(delta.iloc[1:].mean() * 100)


def metrics_from_value(value):
    """Total return %, annualized Sharpe (rf=0), and max drawdown % from an equity curve."""
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    ret = value.pct_change().dropna()
    total = value.iloc[-1] / value.iloc[0] - 1.0
    sharpe = ((ret.mean() - RF) / ret.std() * np.sqrt(ANN)) if ret.std() > 0 else np.nan
    max_dd = (value / value.cummax() - 1.0).min()
    return total * 100, sharpe, max_dd * 100


# ===========================================================================
# Stage 3 -- assemble and run one config
# ===========================================================================
def run_config(holdings, scalars, use_regime, monthly_prices):
    if vbt is None:
        sys.exit("vectorbt is not installed. `pip install vectorbt` to run the backtest.")

    weights = equal_weight(holdings)
    weights, mp = weights.align(monthly_prices, join="inner", axis=0)

    if use_regime:
        s, _ = scalars.align(weights, join="inner", axis=0)
        weights = weights.multiply(s.fillna(1.0), axis=0)

    # OOS slice -- every config is judged on the same holdout window
    weights = weights.loc[OOS_START:]
    mp = mp.loc[OOS_START:]

    pf = vbt.Portfolio.from_orders(
        mp, weights,
        size_type="targetpercent",
        init_cash=INIT_CASH,
        fees=FEES,
        freq=FREQ,
        cash_sharing=True,
        group_by=True,
    )

    total, sharpe, max_dd = metrics_from_value(pf.value())
    invested_months = int((weights.sum(axis=1) > 1e-9).sum())
    avg_holdings = float(holdings.loc[OOS_START:].astype(bool).sum(axis=1).mean())

    return {
        "Total Return %": round(total, 1),
        "Sharpe": round(sharpe, 2) if pd.notna(sharpe) else np.nan,
        "Max DD %": round(max_dd, 1),
        "Turnover %": round(one_way_turnover(weights), 1),
        "Months Invested": invested_months,
        "Avg Holdings": round(avg_holdings, 1),
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh-data", action="store_true",
                    help="re-run download_data.py before comparing")
    args = ap.parse_args()

    # Run from the project root regardless of where the user launched this.
    os.chdir(ROOT)
    if not os.path.isdir("strategy") or not os.path.isdir("data"):
        sys.exit(
            f"Expected 'strategy/' and 'data/' folders next to this script.\n"
            f"Put run_ablations.py at the project root (beside download_data.py).\n"
            f"Currently looking in: {ROOT}"
        )

    build_ingredients(args.refresh_data)

    prices = pd.read_csv("data/real_prices.csv", index_col="Date", parse_dates=True)
    monthly_prices = prices.resample("ME").last()

    raw = pd.read_csv("strategy/candidate_holdings.csv", index_col="Date", parse_dates=True)
    filtered = pd.read_csv("strategy/filtered_holdings.csv", index_col="Date", parse_dates=True)
    scalars = pd.read_csv("strategy/regime_scalars.csv", index_col="Date", parse_dates=True)["Scalar"]

    # Benchmark: equal-weight every name, fully invested, no overlays
    bench = pd.DataFrame(True, index=monthly_prices.index, columns=monthly_prices.columns)

    configs = {
        "Benchmark (EW all)": (bench, False),
        "1. Momentum only":   (raw, False),
        "2. + Regime":        (raw, True),
        "3. + Meta-label":    (filtered, False),
        "4. Full stack":      (filtered, True),
    }

    rows = {name: run_config(h, scalars, ur, monthly_prices)
            for name, (h, ur) in configs.items()}

    table = pd.DataFrame(rows).T[
        ["Total Return %", "Sharpe", "Max DD %", "Turnover %", "Months Invested", "Avg Holdings"]
    ]

    print("\n" + "=" * 80)
    print(f"ABLATION COMPARISON -- out-of-sample from {OOS_START}")
    print("=" * 80)
    print(table.to_string())
    print("\nEach row (1->4) adds one component to the row above it.")
    print("If a component doesn't improve Sharpe out-of-sample, that's a finding to")
    print("report honestly -- not a failure to hide.\n")

    table.to_csv("strategy/ablation_results.csv")
    print("Saved -> strategy/ablation_results.csv")


if __name__ == "__main__":
    main()