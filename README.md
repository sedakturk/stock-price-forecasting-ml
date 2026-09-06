<img width="1496" height="815" alt="Figure_1" src="https://github.com/user-attachments/assets/a02ac093-9e57-4b42-b4c8-6cecfa9bcf67" />

# Stock Forecasting Pipeline with Random Forest & Log-Returns

An econometric machine learning pipeline designed to forecast short-term stock prices using stationary daily log-returns, autoregressive lagged features, and rolling volatility indicators.

## Overview

Tree-based models (such as Random Forest) cannot extrapolate trends when trained directly on non-stationary price series ($I(1)$). This pipeline transforms price data into stationary log-returns ($I(0)$), trains a regression model on lagged returns and calendar features, and dynamically reconstructs the price levels for out-of-sample backtesting and 45-day horizon projections.

## Methodology

1. **Target Transformation:** Daily log-returns $r_t = \ln(P_t / P_{t-1})$.
2. **Feature Engineering:**
   - Autoregressive lags: $t-1, t-2, t-3, t-5, t-10$
   - 10-day rolling mean and standard deviation strictly lagged by $t-1$ (`shift(1)`) to eliminate contemporaneous target leakage (lookahead bias).
   - Calendar features: Day of week, month
3. **Validation:** Chronological 80/20 train-test split for out-of-sample backtesting (preserving temporal order).
4. **Reconstruction:** Recursive compounding of predicted returns onto base prices: $\hat{P}_t = \hat{P}_{t-1} \cdot \exp(\hat{r}_t)$.

## Model Performance (AAPL Example)

| Metric | Raw Price Baseline (v1) | Stationary Log-Return Model (v2) |
| :--- | :--- | :--- |
| Test R² | -3.08 | 0.0028 |
| Test MAPE | 14.57% | 6.30% |
| Test MAE | $42.09 | $18.53 |
| Test RMSE | $47.70 | $24.09 |

*Note: In asset pricing under the Efficient Market Hypothesis (EMH), returns exhibit random-walk characteristics, yielding an $R^2 \approx 0$ out of sample. However, dynamic log-return compounding significantly reduces out-of-sample level errors (MAPE 6.30%) compared to raw price extrapolation.*

<img width="1496" height="815" alt="Figure_1" src="https://github.com/user-attachments/assets/a02ac093-9e57-4b42-b4c8-6cecfa9bcf67" />

## Installation & Usage 

Clone the repository:
git clone https://github.com/sedakturk/stock-price-forecasting-ml.git

Install dependencies:
pip install -r requirements.txt

Run the forecast:
python predict.py
