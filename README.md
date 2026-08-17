# 📈 Stock Price Forecasting with Random Forest & Technical Indicators

An end-to-end Machine Learning project forecasting stock prices using historical market data, technical indicators (45-day MA & RSI), and a **Random Forest Regressor**.

---

## 🚀 Overview & Features
* **Automated Data Retrieval:** Fetches live historical price data via `yfinance`.
* **Feature Engineering:** Calculates 45-day Moving Average (MA), 45-day Relative Strength Index (RSI), and date seasonality features.
* **ML Model:** Trains a `RandomForestRegressor` with feature normalization using `StandardScaler`.
* **Forward Projection:** Produces a 45-day forward price prediction interval with upper/lower bounds.
* **Trading Decision Logic:** Identifies potential buy/sell timing based on predicted extrema.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Data Source:** yfinance
* **Visualization:** Matplotlib

---

## 📦 How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/sedakturk/stock-price-forecasting-ml.git]
