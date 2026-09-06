import os
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
import yfinance as yf


def prepare_econometric_data(ticker_symbol: str, start_date: str = '2022-01-01') -> pd.DataFrame:
    df = yf.download(ticker_symbol, start=start_date, end=pd.Timestamp.today().strftime('%Y-%m-%d'))

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df.columns]

    df = df.reset_index()
    close_col = f'Close_{ticker_symbol}' if f'Close_{ticker_symbol}' in df.columns else 'Close'
    df = df[['Date', close_col]].dropna().rename(columns={close_col: 'Close'})

    # Target variable: Stationary log-returns at time t
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Autoregressive components (strictly t-1, t-2, ...)
    for lag in [1, 2, 3, 5, 10]:
        df[f'Return_Lag_{lag}'] = df['Log_Return'].shift(lag)

    # Shifted by 1 step prior to rolling window to prevent contemporaneous lookahead bias
    df['Rolling_Vol_10'] = df['Log_Return'].shift(1).rolling(window=10).std()
    df['Rolling_Mean_10'] = df['Log_Return'].shift(1).rolling(window=10).mean()

    # Ex-ante calendar features
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month

    return df.dropna().reset_index(drop=True)


def main():
    ticker_symbol = input("Enter stock symbol (e.g. AAPL, NVDA): ").strip().upper()
    forecast_horizon = 45

    data = prepare_econometric_data(ticker_symbol)

    feature_cols = [
        'Return_Lag_1',
        'Return_Lag_2',
        'Return_Lag_3',
        'Return_Lag_5',
        'Return_Lag_10',
        'Rolling_Vol_10',
        'Rolling_Mean_10',
        'Day_of_Week',
        'Month',
    ]

    X = data[feature_cols]
    y = data['Log_Return']

    # Chronological split preserving time-series structure
    split_idx = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train_scaled, y_train)

    pred_returns_test = model.predict(X_test_scaled)

    # Dynamic reconstruction of price paths from predicted log-returns
    base_price_test = data['Close'].iloc[split_idx - 1]
    reconstructed_test_prices = [base_price_test]
    actual_test_prices = data['Close'].iloc[split_idx:].values

    for r in pred_returns_test:
        reconstructed_test_prices.append(reconstructed_test_prices[-1] * np.exp(r))
    reconstructed_test_prices = np.array(reconstructed_test_prices[1:])

    test_r2 = r2_score(actual_test_prices, reconstructed_test_prices)
    test_rmse = np.sqrt(mean_squared_error(actual_test_prices, reconstructed_test_prices))
    test_mae = mean_absolute_error(actual_test_prices, reconstructed_test_prices)
    test_mape = mean_absolute_percentage_error(actual_test_prices, reconstructed_test_prices) * 100

    print("\n" + "=" * 50)
    print(f"Stationary Model Evaluation ({ticker_symbol})")
    print("=" * 50)
    print(f"Test R2 Score : {test_r2:.4f}")
    print(f"Test RMSE     : ${test_rmse:.2f}")
    print(f"Test MAE      : ${test_mae:.2f}")
    print(f"Test MAPE     : {test_mape:.2f}%")
    print("=" * 50)

    # Multi-step recursive out-of-sample projection
    last_known_features = X.iloc[-1].copy()
    recent_returns = list(data['Log_Return'].iloc[-10:].values)
    current_price = data['Close'].iloc[-1]
    future_dates = [data['Date'].iloc[-1] + timedelta(days=i) for i in range(1, forecast_horizon + 1)]

    future_prices = []
    running_price = current_price

    for date in future_dates:
        feat_df = pd.DataFrame([last_known_features])[feature_cols]
        feat_scaled = scaler.transform(feat_df)
        pred_return = model.predict(feat_scaled)[0]
        
        running_price = running_price * np.exp(pred_return)
        future_prices.append(running_price)

        recent_returns.append(pred_return)
        rolling_window = recent_returns[-10:]

        last_known_features['Return_Lag_10'] = last_known_features['Return_Lag_5']
        last_known_features['Return_Lag_5'] = last_known_features['Return_Lag_3']
        last_known_features['Return_Lag_3'] = last_known_features['Return_Lag_2']
        last_known_features['Return_Lag_2'] = last_known_features['Return_Lag_1']
        last_known_features['Return_Lag_1'] = pred_return
        last_known_features['Rolling_Mean_10'] = np.mean(rolling_window)
        last_known_features['Rolling_Vol_10'] = np.std(rolling_window, ddof=1) if len(rolling_window) > 1 else 0.0
        last_known_features['Day_of_Week'] = date.dayofweek
        last_known_features['Month'] = date.month

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast': future_prices,
        'Lower_Bound': np.array(future_prices) * 0.98,
        'Upper_Bound': np.array(future_prices) * 1.02,
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

    ax1.plot(data['Date'].iloc[:split_idx], data['Close'].iloc[:split_idx], label='Train Data (Actual)', color='#1f77b4')
    ax1.plot(data['Date'].iloc[split_idx:], actual_test_prices, label='Out-of-Sample Test (Actual)', color='#2ca02c', lw=1.5)
    ax1.plot(data['Date'].iloc[split_idx:], reconstructed_test_prices, label='Model Reconstruction (Predicted)', color='#ff7f0e', linestyle='--', lw=1.5)
    ax1.set_title(f'{ticker_symbol} Out-of-Sample Backtesting | R²: {test_r2:.2f} | MAPE: {test_mape:.2f}% | MAE: ${test_mae:.2f}', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontweight='bold')
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(25))
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.plot(data['Date'], data['Close'], label='Historical Price', color='#1f77b4')
    ax2.plot(forecast_df['Date'], forecast_df['Forecast'], label='45-Day Out-of-Sample Projection', color='#d62728', lw=2)
    ax2.fill_between(forecast_df['Date'], forecast_df['Lower_Bound'], forecast_df['Upper_Bound'], color='#d62728', alpha=0.2, label='±2% Confidence Interval')
    ax2.set_title(f'{ticker_symbol} 45-Day Horizon Forecast (Base Price: ${current_price:.2f})', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontweight='bold')
    ax2.set_ylabel('Price ($)', fontweight='bold')
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(25))
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()

    output_path = os.path.join(os.path.expanduser('~'), f'{ticker_symbol}_forecast_evaluation.csv')
    forecast_df.to_csv(output_path, index=False)
    print(f"\nForecast output saved to: {output_path}")


if __name__ == '__main__':
    main()
    
