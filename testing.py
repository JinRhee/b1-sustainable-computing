import pandas as pd
import os
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


## Get data file path
file_name = 'df_fuel_ckan.csv'
file_path = os.path.join(os.getcwd(), 'data', file_name)

def split_data(data: pd.DataFrame, timescale: str):
    timescale_carbon = data.resample(timescale).mean()
    train_data = timescale_carbon.truncate(after=pd.Timestamp('2023-12-31 23:59:59').tz_localize('UTC'))
    test_data = timescale_carbon.truncate(before=pd.Timestamp('2024-01-01 00:00:00').tz_localize('UTC'))
    return train_data, test_data

## Get data file path
scale = 'h'
file_name = 'df_fuel_ckan.csv'
file_path = os.path.join(os.getcwd(), 'data', file_name)

## Read data
df = pd.read_csv(file_path)
carbon_data = pd.DataFrame(df['CARBON_INTENSITY'].values, index=pd.to_datetime(df['DATETIME']))

train_data, test_data = split_data(carbon_data, scale)

# Fit the ARIMA model
model = ARIMA(train_data, order=(5, 1, 0))
model_fit = model.fit()

# Forecast for the year 2024
forecast = model_fit.forecast(steps=len(test_data))

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(train_data, label='Training Data')
plt.plot(test_data.index, forecast, label='Forecast for 2024', color='red')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Carbon Intensity')
plt.title('Carbon Intensity Forecast for 2024')
plt.show()

# Fit the Exponential Smoothing model
model = ExponentialSmoothing(train_data, seasonal='add', seasonal_periods=24*7)
model_fit = model.fit()

# Forecast for the year 2024
forecast = model_fit.forecast(steps=len(test_data))

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(train_data, label='Training Data')
plt.plot(test_data.index, forecast, label='Forecast for 2024', color='red')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Carbon Intensity')
plt.title('Carbon Intensity Forecast for 2024')
plt.show()