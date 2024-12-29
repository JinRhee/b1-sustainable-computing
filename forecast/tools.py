import os
import numpy as np
import pandas as pd
import time
import sys
import select
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from river import time_series
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from pmdarima.arima import auto_arima

# Check stationarity (from https://www.geeksforgeeks.org/sarima-seasonal-autoregressive-integrated-moving-average/)
def check_stationarity(timeseries):
    # Perform the Dickey-Fuller test
    result = adfuller(timeseries, autolag='AIC')
    p_value = result[1]
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {p_value}')
    print('Stationary' if p_value < 0.05 else 'Non-Stationary')

def diff(timeseries, num):
    diff_timeseries = timeseries.diff(periods=num)
    diff_timeseries.dropna(inplace=True)
    return diff_timeseries

def diff_n(timeseries, num, n):
    # carry out diff() n times
    for _ in range(n):
        diff_timeseries = diff(timeseries, num)
    return diff_timeseries

def split_data(data: pd.DataFrame, timescale: str):
    timescale_carbon = data.resample(timescale).mean()
    train_data = timescale_carbon.truncate(after=pd.Timestamp('2023-12-31 23:59:59').tz_localize('UTC'))
    test_data = timescale_carbon.truncate(before=pd.Timestamp('2024-01-01 00:00:00').tz_localize('UTC'))
    return train_data, test_data


def add_dates(data, start_date, timescale):
    date_range = pd.date_range(start=start_date, periods=len(data), freq=timescale)
    return pd.DataFrame(data, index=date_range)


def calculate_errors(observed, forecast_mean):
    mae = mean_absolute_error(observed, forecast_mean)
    mse = mean_squared_error(observed, forecast_mean)
    rmse = root_mean_squared_error(observed, forecast_mean)
    return mae, mse, rmse

def plot_forecast(train_data, test_data, forecast, scale, forecast_periods, m, method):
    # Plot the forecast
    plt.figure(figsize=(20, 7))
    plt.plot(train_data, 'bo-', label='Observed')
    plt.plot(forecast, 'rx-', label='Forecast')
    plt.plot(test_data, 'ko--', label='Actual')
    #plt.fill_between(forecast.conf_int().index, forecast.conf_int().iloc[:, 0], forecast.conf_int().iloc[:, 1], color='pink')
    plt.title(f"Carbon Intensity Forecast for 2024 in {scale}")
    plt.xlabel("Date")
    plt.ylabel("Carbon Intensity")
    plt.legend()
    plt.savefig(f"plots/{scale}_{forecast_periods}_{m}_{method}.png")
    #plt.show()


def holt_winter(train_data, scale, forecast_periods, m):
    # Fit the Exponential Smoothing model
    model = ExponentialSmoothing(train_data, seasonal='add', seasonal_periods=m)
    model_fit = model.fit()

    # Forecast for 2024 and add dates
    forecast = model_fit.forecast(steps=forecast_periods)
    forecast = add_dates(forecast, pd.Timestamp('2024-01-01 00:00:00').tz_localize('UTC'), scale)
    forecast.dropna(inplace=True)

    return forecast


def sarima(train_data, scale, forecast_periods, m, params, seasonal_params):
    # Unpack arguments
    (p, d, q) = params
    (P, D, Q) = seasonal_params

    # Fit the SNARIMAX model
    model = time_series.SNARIMAX(p, d, q, m, P, D, Q)
    for intensity in train_data.to_numpy():
        model.learn_one(intensity)
    
    # Forecast for 2024 and add dates
    forecast = model.forecast(horizon = forecast_periods)
    forecast = add_dates(forecast, pd.Timestamp('2024-01-01 00:00:00').tz_localize('UTC'), scale)
    forecast.dropna(inplace=True)

    return forecast

def log_results(forecast, scale, forecast_periods, m, method):
    forecast.to_csv(f'results/{scale}_{forecast_periods}_{m}_{method}.csv', header=['FORECAST_INTENSITY'], index_label=['DATETIME'])

