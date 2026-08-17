import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# Get the user's home directory
home_dir = os.path.expanduser("~")

stock = input("Enter the stock symbol: ").strip().upper()

# Get company stock data from Yahoo Finance
print(f"Downloading data for {stock}...")
company = yf.download(stock, start='2022-01-01', end=pd.Timestamp.today().strftime('%Y-%m-%d'))

# Flatten columns if they are multi-level
if isinstance(company.columns, pd.MultiIndex):
    company.columns = ['_'.join(col).strip() if col[1] else col[0] for col in company.columns]

# Reset index to move the date into a column
company = company.reset_index()

company.head()

# Select only the Date and Close columns
close_column = f'Close_{stock}'
company = company[['Date', close_column]].dropna()
day=45
# Calculate 45-day Moving Average (MA)
print("Calculating Moving Average...")
company['MA_45'] = company[close_column].rolling(window=day).mean()

# Calculate RSI (Relative Strength Index) with a 45-day period
print("Calculating RSI...")
delta = company[close_column].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=day).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=day).mean()
RS = gain / loss
company['RSI_45'] = 100 - (100 / (1 + RS))

# Drop any rows that have NaN values
company = company.dropna()

# Create features for the model
company['day_of_week'] = company['Date'].dt.dayofweek
company['month'] = company['Date'].dt.month
company['day'] = company['Date'].dt.day

# Prepare features and target
features = ['MA_45', 'RSI_45', 'day_of_week', 'month', 'day']
X = company[features]
y = company[close_column]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the Random Forest model
print("Training the Random Forest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Create future dates for prediction
print("Making predictions...")
last_date = company['Date'].iloc[-1]
future_dates = [last_date + timedelta(days=x) for x in range(1, day + 1)]

# Prepare future features
future_data = []
for date in future_dates:
    # Use the last known MA and RSI for future predictions
    last_ma = company['MA_45'].iloc[-1]
    last_rsi = company['RSI_45'].iloc[-1]
    future_data.append({
        'MA_45': last_ma,
        'RSI_45': last_rsi,
        'day_of_week': date.dayofweek,
        'month': date.month,
        'day': date.day
    })

future_df = pd.DataFrame(future_data)
future_scaled = scaler.transform(future_df)

# Make predictions
predictions = model.predict(future_scaled)

# Create a DataFrame for the forecast
forecast = pd.DataFrame({
    'ds': future_dates,
    'yhat': predictions,
    'yhat_lower': predictions - predictions * 0.02,  # 2% lower bound
    'yhat_upper': predictions + predictions * 0.02   # 2% upper bound
})

# Plot the forecast
print("Plotting the forecast...")
plt.figure(figsize=(12, 6))
plt.plot(company['Date'], company[close_column], label='Historical')
plt.plot(forecast['ds'], forecast['yhat'], label='Predicted', color='red')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                 color='red', alpha=0.2)
plt.title(f'{stock} 45-Day Forecast with Random Forest')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()

# Save the forecast to a CSV file in the user's home directory
output_file = os.path.join(home_dir, f'{stock}_45day_forecast_with_indicators.csv')
print(f"Saving forecast to '{output_file}'...")
forecast.to_csv(output_file, index=False)

print("All done! Forecast is ready and saved.")

# Trading advice
order = input("Do you want to 'buy' or 'sell'? ").strip().lower()

if order == 'buy':
    best_day = forecast.loc[forecast['yhat'].idxmin()]
    print(f"Best day to BUY: {best_day['ds'].date()} (Predicted price: ${best_day['yhat']:.2f})")
elif order == 'sell':
    best_day = forecast.loc[forecast['yhat'].idxmax()]
    print(f"Best day to SELL: {best_day['ds'].date()} (Predicted price: ${best_day['yhat']:.2f})")
else:
    print("I didn't understand your order. Please type 'buy' or 'sell'.")
