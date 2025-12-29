!pip install optuna
import yfinance as yf
import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def get_data(tickers, start_date, end_date):
    """
    Fetches data and filters out 'bad ticks' (e.g., stock splits or data errors).
    """
    print(f"Fetching data for {tickers}...")
    # Fetch Data
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    data = data.ffill().dropna()

    # --- OUTLIER FILTERING ---
    # Calculate daily percent changes
    pct_changes = data.pct_change()

    # Identify rows where ANY stock moves more than 20% in a single day
    # (Unless it's a penny stock, >20% usually means a split or bad data)
    outlier_mask = (pct_changes.abs() > 0.20).any(axis=1)

    # Keep only clean rows
    data_clean = data[~outlier_mask]

    if len(data) != len(data_clean):
        print(f"Removed {len(data) - len(data_clean)} outlier days from data.")

    # Normalize to start at 1.0 so weights apply equally
    return data_clean / data_clean.iloc[0]


def calculate_metrics_dynamic(prices, weights, entry_z, exit_z, stop_loss_z):
    """
    Calculates returns with Volatility Targeting to prevent unrealistic leverage explosions.
    """
    # 1. Construct Spread
    spread = (prices * weights).sum(axis=1)
    
    # 2. Z-Score Calculation
    window = 20
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    
    # Avoid division by zero
    rolling_std.replace(0, np.nan, inplace=True)
    z_score = (spread - rolling_mean) / rolling_std
    
    # 3. Signal Logic (Vectorized)
    positions = pd.Series(np.nan, index=z_score.index)
    
    # Entry
    positions[z_score < -entry_z] = 1.0  # Long
    positions[z_score > entry_z] = -1.0  # Short
    
    # Exit (Take Profit)
    positions[abs(z_score) < exit_z] = 0.0
    
    # Stop Loss (Risk Management)
    positions[abs(z_score) > stop_loss_z] = 0.0
    
    # Forward Fill Positions
    positions = positions.ffill().fillna(0.0)
    
    # Re-enforce Stop Loss (Priority over fill)
    positions[abs(z_score) > stop_loss_z] = 0.0
    
    # 4. Volatility Targeting
    target_vol = 0.15 
    pct_changes = spread.pct_change()
    realized_vol = pct_changes.rolling(20).std() * np.sqrt(252)
    realized_vol = realized_vol.replace(0, 0.01) 
    leverage = (target_vol / realized_vol).clip(0, 2.0)
    
    # 5. Apply Leverage
    weighted_positions = positions.shift(1) * leverage
    
    # 6. Returns
    strategy_returns = weighted_positions * pct_changes
    trades_occurred = weighted_positions.diff().abs().fillna(0)
    costs = trades_occurred * 0.0005 
    net_returns = strategy_returns - costs
    
    # --- THE FIX IS HERE ---
    if net_returns.std() == 0:
        # Must return a Tuple (Sharpe, EquityCurve) even on failure
        # We return a flat line (all 1.0s) for the equity curve
        return -10.0, pd.Series(1.0, index=prices.index)
    
    sharpe = (net_returns.mean() / net_returns.std()) * np.sqrt(252)
    
    return sharpe, (1 + net_returns).cumprod()

def objective_full_system(trial, data):
    # --- A. Suggest Weights ---
    w = []
    for i in range(len(data.columns)):
        # Weights between -1.0 and 1.0
        w.append(trial.suggest_float(f'w_{i}', -1.0, 1.0))
    
    weights = np.array(w)
    # Normalize absolute sum to 1.0
    if np.sum(np.abs(weights)) == 0: return -10.0
    weights = weights / np.sum(np.abs(weights))
    
    # --- B. Suggest Thresholds ---
    entry_z = trial.suggest_float('entry_z', 1.0, 3.5)
    exit_z = trial.suggest_float('exit_z', 0.0, 0.8)
    stop_loss_z = trial.suggest_float('stop_loss_z', 3.6, 5.0)
    
    # --- C. Logical Constraints (Pruning) ---
    # 1. Entry must be > Exit
    if entry_z <= (exit_z + 0.2):
        raise optuna.exceptions.TrialPruned()
    
    # 2. Stop Loss must be > Entry
    if stop_loss_z <= (entry_z + 0.5):
        raise optuna.exceptions.TrialPruned()
        
    # 3. Stationarity Check (ADF Test)
    # We only want to trade Mean Reverting spreads, not trends.
    spread = (data * weights).sum(axis=1)
    
    try:
        p_value = adfuller(spread.dropna())[1]
    except:
        return -10.0
        
    if p_value > 0.05:
        # Penalize non-stationary spreads heavily
        return -1.0 
        
    # --- D. Run Strategy ---
    sharpe, _ = calculate_metrics_dynamic(data, weights, entry_z, exit_z, stop_loss_z)
    
    return sharpe


def generate_tearsheet(equity_curve, title="Strategy Performance"):
    # Calculate Max Drawdown
    hwm = equity_curve.cummax()
    drawdown = (equity_curve - hwm) / hwm
    max_dd = drawdown.min()
    
    # Calculate Total Return
    total_ret = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    
    print(f"--- {title} ---")
    print(f"Total Return: {total_ret*100:.2f}%")
    print(f"Max Drawdown: {max_dd*100:.2f}%")
    
    # Plotting
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot 1: Equity Curve
    ax[0].plot(equity_curve, label='BayesOpt Strategy', color='blue')
    ax[0].set_title(f"{title} - Equity Curve")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()
    
    # Plot 2: Drawdown
    ax[1].fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    ax[1].set_title("Drawdown (Risk)")
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # --- FIX FOR OPTUNA VERBOSITY ---
    # Set logging to WARNING to stop the flood of messages
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # 1. Config
    # Pick correlated assets
    tickers = ['XOM', 'CVX'] 
    start_date = '2018-01-01'
    end_date = '2024-01-01'
    
    # 2. Load Data
    data = get_data(tickers, start_date, end_date)
    
    # 3. Setup Walk-Forward Loop
    train_window = 252 # 1 Year Training
    test_window = 63   # 3 Months Testing
    
    all_oos_returns = [] # Store out-of-sample results
    
    print("\nStarting Rolling Walk-Forward Optimization...")
    print("This may take a few minutes depending on your CPU.\n")
    
    # 4. Run Loop
    for t in range(train_window, len(data), test_window):
        # Slice Data
        train_data = data.iloc[t-train_window : t]
        test_data = data.iloc[t : t+test_window]
        
        if len(test_data) < 10: break
        
        # Optimize on TRAIN data
        # REMOVED 'verbose=False' which caused the error
        study = optuna.create_study(direction='maximize')
        
        # Optimize
        study.optimize(lambda trial: objective_full_system(trial, train_data), n_trials=30)
        
        best_params = study.best_params
        
        # Reconstruct Weights from Best Params
        w_best = []
        for i in range(len(tickers)):
            w_best.append(best_params[f'w_{i}'])
        weights_best = np.array(w_best)
        weights_best /= np.sum(np.abs(weights_best))
        
        # Test on UNSEEN (Test) Data
        _, oos_equity = calculate_metrics_dynamic(
            test_data, 
            weights_best, 
            best_params['entry_z'], 
            best_params['exit_z'], 
            best_params['stop_loss_z']
        )
        
        # Stitch results
        oos_period_returns = oos_equity.pct_change().fillna(0)
        all_oos_returns.extend(oos_period_returns.values)
        
        print(f"Window {data.index[t].date()}: Sharpe {study.best_value:.2f} | StopLoss: {best_params['stop_loss_z']:.2f}")

    # 5. Final Report
    full_equity_curve = pd.Series(all_oos_returns).fillna(0)
    full_equity_curve = (1 + full_equity_curve).cumprod()
    
    # Re-align dates (approximate)
    full_equity_curve.index = data.iloc[train_window : train_window+len(all_oos_returns)].index
    
    generate_tearsheet(full_equity_curve, title="Walk-Forward BayesOpt Strategy")

